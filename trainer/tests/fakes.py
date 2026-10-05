from __future__ import annotations

import dataclasses
import threading
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

from yolox_trainer.core.coco_metrics import ClassAp, CocoStats, EvaluationMetrics
from yolox_trainer.core.events import WorkerDone
from yolox_trainer.core.inference import Detection, PredictResult
from yolox_trainer.core.supervisor import JobSpec


class FakeTrainingRunner:
    """Plays scripted worker messages on a background thread, like a real runner's pump thread.

    Files in `work_files` are written into the job's work_dir at start. A relative
    `WorkerDone.weights_path` is resolved against the work_dir. With `wait_for_stop`,
    the thread pauses after `script` until `request_stop()` and then plays `stop_script`.
    """

    def __init__(
        self,
        script: Sequence[object] = (),
        *,
        wait_for_stop: bool = False,
        stop_script: Sequence[object] = (),
        exit_code: int | None = 0,
        work_files: Mapping[str, str] | None = None,
    ) -> None:
        self.script = list(script)
        self.wait_for_stop = wait_for_stop
        self.stop_script = list(stop_script)
        self.exit_code = exit_code
        self.work_files = dict(work_files or {})
        self.spec: JobSpec | None = None
        self.stop_requested = threading.Event()
        self.thread: threading.Thread | None = None

    def start(
        self,
        spec: JobSpec,
        on_message: Callable[[object], None],
        on_exit: Callable[[int | None], None],
    ) -> None:
        self.spec = spec
        work_dir = Path(spec.work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
        for name, content in self.work_files.items():
            (work_dir / name).write_text(content)
        self.thread = threading.Thread(target=self._play, args=(work_dir, on_message, on_exit), daemon=True)
        self.thread.start()

    def request_stop(self) -> None:
        self.stop_requested.set()

    def _play(
        self,
        work_dir: Path,
        on_message: Callable[[object], None],
        on_exit: Callable[[int | None], None],
    ) -> None:
        for message in self.script:
            on_message(self._resolve(message, work_dir))
        if self.wait_for_stop:
            self.stop_requested.wait()
            for message in self.stop_script:
                on_message(self._resolve(message, work_dir))
        on_exit(self.exit_code)

    @staticmethod
    def _resolve(message: object, work_dir: Path) -> object:
        if isinstance(message, WorkerDone) and message.weights_path and not Path(message.weights_path).is_absolute():
            return dataclasses.replace(message, weights_path=str(work_dir / message.weights_path))
        return message


class FakeInferenceService:
    """Returns canned results and records the arguments it was called with."""

    def __init__(self) -> None:
        self.evaluate_calls: list[tuple[str, str, str]] = []
        self.predict_calls: list[tuple[str, int, float]] = []
        self.forgotten: list[str] = []

    def evaluate(self, model_id: str, dataset_name: str, split: str) -> EvaluationMetrics:
        self.evaluate_calls.append((model_id, dataset_name, split))
        return EvaluationMetrics(CocoStats(*([0.5] * 12)), [ClassAp("cat", 0.4, 0.6)], 3)

    def predict(self, model_id: str, image: bytes, score_threshold: float) -> PredictResult:
        self.predict_calls.append((model_id, len(image), score_threshold))
        return PredictResult(640, 480, [Detection(1.0, 2.0, 30.0, 40.0, 0.9, 0, "cat")])

    def forget(self, model_id: str) -> None:
        self.forgotten.append(model_id)
