from __future__ import annotations

import logging
import multiprocessing
import multiprocessing.process
import multiprocessing.queues
import multiprocessing.synchronize
import queue
import threading
from collections.abc import Callable

from ..core.supervisor import JobSpec
from .worker import run_job

log = logging.getLogger(__name__)

WorkerTarget = Callable[
    [JobSpec, "multiprocessing.queues.Queue[object]", multiprocessing.synchronize.Event], None
]

POLL_SECONDS = 0.2
TERMINATE_TIMEOUT_SECONDS = 5.0


class ProcessTrainingRunner:
    """Runs one training job in a spawned child process.

    A pump thread forwards every child message to `on_message`, then calls `on_exit` with the
    exit code once the child has exited and the queue is drained. A stop sets the child's stop
    event; a child still alive after `stop_timeout` is terminated, then killed.
    """

    def __init__(self, target: WorkerTarget = run_job, stop_timeout: float = 30.0) -> None:
        self._ctx = multiprocessing.get_context("spawn")
        self._target = target
        self._stop_timeout = stop_timeout
        self._stop_event = self._ctx.Event()
        self._exited = threading.Event()
        self._lock = threading.Lock()
        self._process: multiprocessing.process.BaseProcess | None = None
        self._watchdog: threading.Thread | None = None

    def start(
        self,
        spec: JobSpec,
        on_message: Callable[[object], None],
        on_exit: Callable[[int | None], None],
    ) -> None:
        messages: multiprocessing.queues.Queue[object] = self._ctx.Queue()
        # Not a daemon: the child's DataLoader spawns worker processes of its own.
        process = self._ctx.Process(
            target=self._target, args=(spec, messages, self._stop_event), name=f"train-{spec.job_id}"
        )
        process.start()
        with self._lock:
            self._process = process
        threading.Thread(
            target=self._pump, args=(process, messages, on_message, on_exit), name="train-pump", daemon=True
        ).start()

    def request_stop(self) -> None:
        self._stop_event.set()
        with self._lock:
            if self._process is None or self._watchdog is not None:
                return
            self._watchdog = threading.Thread(
                target=self._enforce_stop, args=(self._process,), name="train-stop", daemon=True
            )
            self._watchdog.start()

    def _pump(
        self,
        process: multiprocessing.process.BaseProcess,
        messages: multiprocessing.queues.Queue[object],
        on_message: Callable[[object], None],
        on_exit: Callable[[int | None], None],
    ) -> None:
        try:
            while process.is_alive():
                try:
                    on_message(messages.get(timeout=POLL_SECONDS))
                except queue.Empty:
                    pass
            process.join()
            # Everything the child put before exiting is already in the pipe.
            while True:
                try:
                    on_message(messages.get_nowait())
                except queue.Empty:
                    break
        except Exception:
            log.exception("training message pump failed; killing the training process")
            process.kill()
            process.join()
        finally:
            self._exited.set()
            messages.close()
            on_exit(process.exitcode)

    def _enforce_stop(self, process: multiprocessing.process.BaseProcess) -> None:
        # Only the pump thread reaps the child; this thread just waits for its signal.
        if self._exited.wait(self._stop_timeout):
            return
        log.warning("training process ignored the stop request; terminating")
        process.terminate()
        if self._exited.wait(TERMINATE_TIMEOUT_SECONDS):
            return
        log.warning("training process survived terminate; killing")
        process.kill()
