import threading
import time
from pathlib import Path

from fakes import SCRIPTED_ITERS, scripted_worker, stop_ignoring_worker

from yolox_trainer.core.events import IterProgress, WorkerDone, WorkerStarted
from yolox_trainer.core.hyperparams import DEFAULTS
from yolox_trainer.core.supervisor import JobSpec
from yolox_trainer.engine.process_runner import ProcessTrainingRunner

EXIT_TIMEOUT = 60


class _Recorder:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []
        self.threads: set[int] = set()
        self.started = threading.Event()
        self.exited = threading.Event()

    def on_message(self, message: object) -> None:
        self.threads.add(threading.get_ident())
        self.calls.append(("message", message))
        if isinstance(message, WorkerStarted):
            self.started.set()

    def on_exit(self, code: int | None) -> None:
        self.threads.add(threading.get_ident())
        self.calls.append(("exit", code))
        self.exited.set()


def _spec(tmp_path: Path) -> JobSpec:
    return JobSpec("job-1", "ds", str(tmp_path), ["cat"], DEFAULTS, "job", str(tmp_path / "work"), "")


def test_delivers_every_message_then_exit_on_one_thread(tmp_path: Path) -> None:
    rec = _Recorder()

    ProcessTrainingRunner(target=scripted_worker).start(_spec(tmp_path), rec.on_message, rec.on_exit)

    assert rec.exited.wait(EXIT_TIMEOUT)
    kinds = [type(payload) for _, payload in rec.calls[:-1]]
    assert kinds == [WorkerStarted] + [IterProgress] * SCRIPTED_ITERS + [WorkerDone]
    assert rec.calls[-1] == ("exit", 0)
    assert len(rec.threads) == 1


def test_stop_terminates_a_worker_that_ignores_the_stop_event(tmp_path: Path) -> None:
    rec = _Recorder()
    runner = ProcessTrainingRunner(target=stop_ignoring_worker, stop_timeout=0.5)
    runner.start(_spec(tmp_path), rec.on_message, rec.on_exit)
    assert rec.started.wait(EXIT_TIMEOUT)

    requested = time.monotonic()
    runner.request_stop()

    assert rec.exited.wait(10)
    assert time.monotonic() - requested < 10
    code = rec.calls[-1][1]
    assert isinstance(code, int) and code != 0
