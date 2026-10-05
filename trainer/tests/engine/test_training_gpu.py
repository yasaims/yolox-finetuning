import os
import threading
import time
from dataclasses import replace
from pathlib import Path

import pytest

from yolox_trainer.core.datasets import DatasetCatalog, DatasetInfo
from yolox_trainer.core.errors import NotFound
from yolox_trainer.core.events import EpochEval, IterProgress, WorkerDone, WorkerStarted
from yolox_trainer.core.hyperparams import DEFAULTS
from yolox_trainer.core.supervisor import JobSpec

pytestmark = [pytest.mark.engine, pytest.mark.gpu, pytest.mark.needs_weights]

DATASET = "balloon"
RUN_TIMEOUT = 900
STOP_LIMIT_SECONDS = 30


@pytest.fixture
def balloon() -> DatasetInfo:
    root = Path(os.environ.get("TRAINER_DATASETS_DIR", "/data/datasets"))
    try:
        return DatasetCatalog(root).get(DATASET)
    except NotFound:
        pytest.skip(f"dataset {DATASET!r} not found under {root}")


def _spec(tmp_path: Path, dataset: DatasetInfo, pretrained: Path) -> JobSpec:
    hp = replace(DEFAULTS, max_epoch=2, batch_size=4, input_size=320, warmup_epochs=0, no_aug_epochs=1)
    return JobSpec(
        job_id="gpu-test",
        dataset_name=dataset.name,
        data_dir=str(dataset.root),
        class_names=list(dataset.class_names),
        hyperparameters=hp,
        display_name="gpu-test",
        work_dir=str(tmp_path / ".work" / "gpu-test"),
        pretrained_path=str(pretrained),
    )


class _Recorder:
    def __init__(self) -> None:
        self.messages: list[object] = []
        self.exit_code: int | None = None
        self.exited = threading.Event()
        self.done_at: float | None = None

    def on_message(self, message: object) -> None:
        self.messages.append(message)
        if isinstance(message, WorkerDone):
            self.done_at = time.monotonic()

    def on_exit(self, code: int | None) -> None:
        self.exit_code = code
        self.exited.set()


def test_two_epoch_run_streams_progress_and_exports_weights(
    tmp_path: Path, balloon: DatasetInfo, pretrained_weights: Path
) -> None:
    import torch

    from yolox_trainer.engine.process_runner import ProcessTrainingRunner

    spec = _spec(tmp_path, balloon, pretrained_weights)
    rec = _Recorder()

    ProcessTrainingRunner().start(spec, rec.on_message, rec.on_exit)

    assert rec.exited.wait(RUN_TIMEOUT)
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
    rec = _Recorder()
    requested: list[float] = []

    def on_message(message: object) -> None:
        rec.on_message(message)
        if isinstance(message, IterProgress) and not requested:
            requested.append(time.monotonic())
            runner.request_stop()

    runner.start(_spec(tmp_path, balloon, pretrained_weights), on_message, rec.on_exit)

    assert rec.exited.wait(RUN_TIMEOUT)
    done = rec.messages[-1]
    assert isinstance(done, WorkerDone)
    assert (done.outcome, done.weights_path, done.error) == ("cancelled", None, "")
    assert requested and rec.done_at is not None
    assert rec.done_at - requested[0] < STOP_LIMIT_SECONDS
