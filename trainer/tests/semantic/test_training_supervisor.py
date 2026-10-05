import json
from pathlib import Path

import pytest
from fakes import FakeTrainingRunner

from yolox_trainer.core.datasets import DatasetCatalog
from yolox_trainer.core.errors import FailedPrecondition
from yolox_trainer.core.event_log import EventLog
from yolox_trainer.core.events import (
    EpochEval,
    Event,
    IterProgress,
    JobFinished,
    JobStarted,
    JobState,
    WorkerDone,
    WorkerStarted,
)
from yolox_trainer.core.registry import ModelRegistry
from yolox_trainer.core.supervisor import TrainingSupervisor

ITER = IterProgress(1, 1, 1, 2, 0.001, {"total_loss": 3.0}, 416, 10.0)
EVAL = EpochEval(1, 0.6, 0.4, 0.4)
WORK_FILES = {"weights.pth": "weights", "train_log.txt": "log"}


def done(outcome: str, weights: str | None = "weights.pth", error: str = "") -> WorkerDone:
    return WorkerDone(outcome, weights, 1, 0.6, 0.4, 1, error)  # type: ignore[arg-type]


@pytest.fixture
def env(tmp_path: Path) -> dict:
    ann = tmp_path / "datasets" / "ds" / "annotations"
    ann.mkdir(parents=True)
    for split in ("train", "val"):
        (ann / f"{split}.json").write_text(json.dumps({"images": [], "categories": [{"id": 1, "name": "cat"}]}))
    pretrained = tmp_path / "yolox_tiny.pth"
    pretrained.write_bytes(b"pretrained")
    return {
        "registry": ModelRegistry(tmp_path / "models"),
        "catalog": DatasetCatalog(tmp_path / "datasets"),
        "pretrained": pretrained,
    }


def make_supervisor(env: dict, runner: FakeTrainingRunner) -> TrainingSupervisor:
    return TrainingSupervisor(lambda: runner, env["registry"], env["catalog"], env["pretrained"])


def wait_finished(log: EventLog) -> list[Event]:
    closed = False
    for _ in range(50):
        _, closed = log.read_after(log.last_seq, timeout=0.1)
        if closed:
            break
    assert closed, "job did not finish"
    return log.snapshot()


def test_second_start_while_running_is_rejected(env: dict) -> None:
    supervisor = make_supervisor(env, FakeTrainingRunner([WorkerStarted(1, 2)], wait_for_stop=True))
    supervisor.start("ds", {}, "")

    with pytest.raises(FailedPrecondition):
        supervisor.start("ds", {}, "")


def test_missing_pretrained_is_rejected(env: dict) -> None:
    env["pretrained"].unlink()
    supervisor = make_supervisor(env, FakeTrainingRunner())

    with pytest.raises(FailedPrecondition):
        supervisor.start("ds", {}, "")
    assert supervisor.current() is None


def test_successful_job_emits_events_and_registers_model(env: dict) -> None:
    runner = FakeTrainingRunner(
        [WorkerStarted(1, 2), ITER, EVAL, done("succeeded")], work_files=WORK_FILES
    )
    supervisor = make_supervisor(env, runner)

    job_id, hp = supervisor.start("ds", {"max_epoch": 20}, "my model")
    events = wait_finished(supervisor.event_log(job_id))

    assert runner.spec is not None and runner.spec.class_names == ["cat"]
    payloads = [e.payload for e in events]
    assert payloads[0] == JobStarted("ds", hp, 1, 2)
    assert payloads[1:3] == [ITER, EVAL]
    assert payloads[3] == JobFinished(JobState.SUCCEEDED, job_id, "")
    record = env["registry"].get(job_id)
    assert record.status == "completed"
    assert record.display_name == "my model"
    assert record.hyperparameters.max_epoch == 20
    assert record.eval_history == [EVAL]
    assert (env["registry"].weights_path(job_id)).read_text() == "weights"
    assert (env["registry"].weights_path(job_id).parent / "train_log.txt").read_text() == "log"
    summary = supervisor.current()
    assert summary is not None and summary.state is JobState.SUCCEEDED


def test_stop_cancels_job_and_registers_stopped_model(env: dict) -> None:
    runner = FakeTrainingRunner(
        [WorkerStarted(1, 2)],
        wait_for_stop=True,
        stop_script=[done("cancelled")],
        work_files=WORK_FILES,
    )
    supervisor = make_supervisor(env, runner)
    job_id, _ = supervisor.start("ds", {}, "")

    supervisor.stop(job_id)
    events = wait_finished(supervisor.event_log(job_id))

    assert runner.stop_requested.is_set()
    assert events[-1].payload == JobFinished(JobState.CANCELLED, job_id, "")
    record = env["registry"].get(job_id)
    assert record.status == "stopped"
    assert record.display_name == job_id


def test_process_exit_without_done_fails_job(env: dict) -> None:
    supervisor = make_supervisor(env, FakeTrainingRunner([WorkerStarted(1, 2)], exit_code=1))

    job_id, _ = supervisor.start("ds", {}, "")
    events = wait_finished(supervisor.event_log(job_id))

    assert events[-1].payload == JobFinished(JobState.FAILED, "", "training process exited with code 1")
    assert env["registry"].list() == []
    summary = supervisor.current()
    assert summary is not None and summary.state is JobState.FAILED
