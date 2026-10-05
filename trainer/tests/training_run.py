"""Helpers for engine tests that run real training jobs."""
import threading
import time
from dataclasses import dataclass, field, replace
from pathlib import Path

from yolox_trainer.core.datasets import DatasetInfo
from yolox_trainer.core.events import WorkerDone
from yolox_trainer.core.hyperparams import DEFAULTS
from yolox_trainer.core.supervisor import JobSpec

RUN_TIMEOUT = 900


def make_spec(work_root: Path, dataset: DatasetInfo, pretrained: Path) -> JobSpec:
    """A 2-epoch balloon job at input size 320."""
    hp = replace(DEFAULTS, max_epoch=2, batch_size=4, input_size=320, warmup_epochs=0, no_aug_epochs=1)
    return JobSpec(
        job_id="gpu-test",
        dataset_name=dataset.name,
        data_dir=str(dataset.root),
        class_names=list(dataset.class_names),
        hyperparameters=hp,
        display_name="gpu-test",
        work_dir=str(work_root / ".work" / "gpu-test"),
        pretrained_path=str(pretrained),
    )


@dataclass
class Recorder:
    messages: list[object] = field(default_factory=list)
    exit_code: int | None = None
    exited: threading.Event = field(default_factory=threading.Event)
    done_at: float | None = None

    def on_message(self, message: object) -> None:
        self.messages.append(message)
        if isinstance(message, WorkerDone):
            self.done_at = time.monotonic()

    def on_exit(self, code: int | None) -> None:
        self.exit_code = code
        self.exited.set()


@dataclass(frozen=True)
class TrainedJob:
    spec: JobSpec
    recorder: Recorder
