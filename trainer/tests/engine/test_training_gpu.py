import time
from pathlib import Path

import pytest
from training_run import RUN_TIMEOUT, Recorder, TrainedJob, make_spec

from yolox_trainer.core.datasets import DatasetInfo
from yolox_trainer.core.events import EpochEval, IterProgress, WorkerDone, WorkerStarted

pytestmark = [pytest.mark.engine, pytest.mark.gpu, pytest.mark.needs_weights]

STOP_LIMIT_SECONDS = 30


def test_two_epoch_run_streams_progress_and_exports_weights(trained_job: TrainedJob, balloon: DatasetInfo) -> None:
    import torch

    spec, rec = trained_job.spec, trained_job.recorder

    assert rec.exit_code == 0
    started, *middle, done = rec.messages
    assert isinstance(started, WorkerStarted) and started.max_epoch == 2
    assert isinstance(done, WorkerDone)
    assert (done.outcome, done.epochs_completed, done.error) == ("succeeded", 2, "")
    iters = [m for m in middle if isinstance(m, IterProgress)]
    evals = [m for m in middle if isinstance(m, EpochEval)]
    assert len(iters) == 2 * started.iters_per_epoch
    assert [m.global_iter for m in iters] == list(range(1, len(iters) + 1))
    assert [e.epoch for e in evals] == [1, 2]
    assert middle.index(evals[0]) > middle.index(iters[0])
    assert evals[-1].best_ap50_95 == max(e.ap50_95 for e in evals)
    assert (Path(spec.work_dir) / "train_log.txt").is_file()

    assert done.weights_path == str(Path(spec.work_dir) / "weights.pth")
    state = torch.load(done.weights_path, map_location="cpu", weights_only=True)["model"]
    assert state["head.cls_preds.0.weight"].shape[0] == len(balloon.class_names)


def test_stop_after_first_iteration_cancels_quickly(
    tmp_path: Path, balloon: DatasetInfo, pretrained_weights: Path
) -> None:
    from yolox_trainer.engine.process_runner import ProcessTrainingRunner

    runner = ProcessTrainingRunner()
    rec = Recorder()
    requested: list[float] = []

    def on_message(message: object) -> None:
        rec.on_message(message)
        if isinstance(message, IterProgress) and not requested:
            requested.append(time.monotonic())
            runner.request_stop()

    runner.start(make_spec(tmp_path, balloon, pretrained_weights), on_message, rec.on_exit)

    assert rec.exited.wait(RUN_TIMEOUT)
    done = rec.messages[-1]
    assert isinstance(done, WorkerDone)
    assert (done.outcome, done.weights_path, done.error) == ("cancelled", None, "")
    assert requested and rec.done_at is not None
    assert rec.done_at - requested[0] < STOP_LIMIT_SECONDS
