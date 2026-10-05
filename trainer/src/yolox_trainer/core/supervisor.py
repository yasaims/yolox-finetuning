from __future__ import annotations

import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

from .datasets import DatasetCatalog
from .errors import FailedPrecondition, NotFound
from .event_log import EventLog
from .events import (
    EpochEval,
    IterProgress,
    JobFinished,
    JobStarted,
    JobState,
    WorkerDone,
    WorkerStarted,
)
from .hyperparams import Hyperparameters, resolve
from .ids import new_model_id
from .registry import ModelRecord, ModelRegistry

BASE_MODEL = "yolox_tiny"
_ACTIVE = (JobState.STARTING, JobState.RUNNING, JobState.STOPPING)


@dataclass(frozen=True)
class JobSpec:
    job_id: str
    dataset_name: str
    data_dir: str
    class_names: list[str]
    hyperparameters: Hyperparameters
    display_name: str
    work_dir: str
    pretrained_path: str


class TrainingRunner(Protocol):
    def start(
        self,
        spec: JobSpec,
        on_message: Callable[[object], None],
        on_exit: Callable[[int | None], None],
    ) -> None: ...

    def request_stop(self) -> None: ...


@dataclass(frozen=True)
class JobSummary:
    job_id: str
    state: JobState
    dataset_name: str
    hyperparameters: Hyperparameters
    last_seq: int
    started_unix_ms: int


@dataclass
class _Job:
    spec: JobSpec
    log: EventLog
    started_unix_ms: int
    state: JobState = JobState.STARTING
    runner: TrainingRunner | None = None
    done_received: bool = False
    eval_history: list[EpochEval] = field(default_factory=list)
    finished: threading.Event = field(default_factory=threading.Event)


class TrainingSupervisor:
    """Owns the single active training job and turns worker messages into public events.

    Runner callbacks arrive on the runner's thread; the runner and registry are never
    called while the lock is held.
    """

    def __init__(
        self,
        runner_factory: Callable[[], TrainingRunner],
        registry: ModelRegistry,
        catalog: DatasetCatalog,
        pretrained_path: Path,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._runner_factory = runner_factory
        self._registry = registry
        self._catalog = catalog
        self._pretrained_path = Path(pretrained_path)
        self._clock = clock
        self._lock = threading.Lock()
        self._job: _Job | None = None

    def start(
        self, dataset_name: str, overrides: Mapping[str, object], display_name: str
    ) -> tuple[str, Hyperparameters]:
        with self._lock:
            if self._job is not None and self._job.state in _ACTIVE:
                raise FailedPrecondition(f"training job {self._job.spec.job_id} is still active")
            if not self._pretrained_path.is_file():
                raise FailedPrecondition(f"pretrained weights not found: {self._pretrained_path}")
            hp = resolve(overrides)
            dataset = self._catalog.get(dataset_name)
            now = self._clock()
            job_id = new_model_id(dataset.name, datetime.fromtimestamp(now, tz=timezone.utc))
            spec = JobSpec(
                job_id=job_id,
                dataset_name=dataset.name,
                data_dir=str(dataset.root),
                class_names=list(dataset.class_names),
                hyperparameters=hp,
                display_name=display_name or job_id,
                work_dir=str(self._registry.work_dir(job_id)),
                pretrained_path=str(self._pretrained_path),
            )
            job = _Job(spec, EventLog(job_id, clock=self._clock), int(now * 1000))
            self._job = job

        try:
            runner = self._runner_factory()
            runner.start(spec, lambda msg: self._on_message(job, msg), lambda code: self._on_exit(job, code))
        except Exception as e:
            self._finish(job, JobState.FAILED, "", f"failed to start training process: {e}")
            raise
        with self._lock:
            job.runner = runner
            stop_pending = job.state is JobState.STOPPING
        if stop_pending:
            runner.request_stop()
        return job_id, hp

    def stop(self, job_id: str) -> None:
        with self._lock:
            job = self._require(job_id)
            if job.state not in (JobState.STARTING, JobState.RUNNING):
                return
            job.state = JobState.STOPPING
            runner = job.runner
        # A runner still being started picks up the STOPPING state in start().
        if runner is not None:
            runner.request_stop()

    def current(self) -> JobSummary | None:
        with self._lock:
            job = self._job
            if job is None:
                return None
            return JobSummary(
                job_id=job.spec.job_id,
                state=job.state,
                dataset_name=job.spec.dataset_name,
                hyperparameters=job.spec.hyperparameters,
                last_seq=job.log.last_seq,
                started_unix_ms=job.started_unix_ms,
            )

    def event_log(self, job_id: str) -> EventLog:
        with self._lock:
            return self._require(job_id).log

    def shutdown(self, timeout: float) -> None:
        with self._lock:
            job = self._job
        if job is None:
            return
        self.stop(job.spec.job_id)
        job.finished.wait(timeout)

    def _require(self, job_id: str) -> _Job:
        if self._job is None or self._job.spec.job_id != job_id:
            raise NotFound(f"training job not found: {job_id}")
        return self._job

    def _on_message(self, job: _Job, message: object) -> None:
        if isinstance(message, WorkerDone):
            with self._lock:
                if job.done_received or job.finished.is_set():
                    return
                job.done_received = True
            self._handle_done(job, message)
            return
        with self._lock:
            if job.finished.is_set():
                return
            if isinstance(message, WorkerStarted):
                if job.state is JobState.STARTING:
                    job.state = JobState.RUNNING
                job.log.append(
                    JobStarted(
                        job.spec.dataset_name,
                        job.spec.hyperparameters,
                        message.max_epoch,
                        message.iters_per_epoch,
                    )
                )
            elif isinstance(message, (IterProgress, EpochEval)):
                if isinstance(message, EpochEval):
                    job.eval_history.append(message)
                job.log.append(message)

    def _on_exit(self, job: _Job, code: int | None) -> None:
        with self._lock:
            if job.done_received:
                return
        self._finish(job, JobState.FAILED, "", f"training process exited with code {code}")

    def _handle_done(self, job: _Job, done: WorkerDone) -> None:
        if done.outcome == "failed":
            self._finish(job, JobState.FAILED, "", done.error)
        elif done.weights_path:
            try:
                model_id = self._register(job, done, Path(done.weights_path))
            except Exception as e:
                self._finish(job, JobState.FAILED, "", f"model registration failed: {e}")
            else:
                state = JobState.SUCCEEDED if done.outcome == "succeeded" else JobState.CANCELLED
                self._finish(job, state, model_id, "")
        elif done.outcome == "cancelled":
            self._finish(job, JobState.CANCELLED, "", "")
        else:
            self._finish(job, JobState.FAILED, "", "training finished without producing weights")

    def _register(self, job: _Job, done: WorkerDone, weights: Path) -> str:
        spec = job.spec
        with self._lock:
            eval_history = list(job.eval_history)
        record = ModelRecord(
            id=spec.job_id,
            display_name=spec.display_name,
            status="completed" if done.outcome == "succeeded" else "stopped",
            created_unix_ms=job.started_unix_ms,
            finished_unix_ms=int(self._clock() * 1000),
            base_model=BASE_MODEL,
            dataset_name=spec.dataset_name,
            class_names=list(spec.class_names),
            hyperparameters=spec.hyperparameters,
            epochs_trained=done.epochs_completed,
            best_epoch=done.best_epoch,
            best_ap50=done.best_ap50,
            best_ap50_95=done.best_ap50_95,
            eval_history=eval_history,
        )
        log_path = Path(spec.work_dir) / "train_log.txt"
        self._registry.register(
            record, weights, job.log.snapshot(), log_path if log_path.is_file() else None
        )
        return spec.job_id

    def _finish(self, job: _Job, state: JobState, model_id: str, error: str) -> None:
        with self._lock:
            if job.finished.is_set():
                return
            job.state = state
            job.log.append(JobFinished(state, model_id, error))
            job.log.close()
            job.finished.set()
