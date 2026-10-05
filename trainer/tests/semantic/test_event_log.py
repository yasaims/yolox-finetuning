import threading
import time

from yolox_trainer.core.event_log import EventLog
from yolox_trainer.core.events import EpochEval


def payload(n: int) -> EpochEval:
    return EpochEval(n, 0.0, 0.0, 0.0)


def test_read_after_replays_only_newer_events() -> None:
    log = EventLog("job", clock=lambda: 1.5)
    for n in range(1, 4):
        log.append(payload(n))

    events, closed = log.read_after(1, timeout=0)

    assert [e.seq for e in events] == [2, 3]
    assert not closed
    assert events[0].job_id == "job"
    assert events[0].time_unix_ms == 1500
    assert log.last_seq == 3
    assert [e.seq for e in log.snapshot()] == [1, 2, 3]


def test_blocked_reader_wakes_on_append() -> None:
    log = EventLog("job")
    result: list = []
    started = threading.Event()

    def reader() -> None:
        started.set()
        result.append(log.read_after(0, timeout=10))

    t = threading.Thread(target=reader)
    t.start()
    started.wait()
    time.sleep(0.05)
    log.append(payload(1))
    t.join(timeout=5)

    assert not t.is_alive()
    events, closed = result[0]
    assert [e.seq for e in events] == [1]
    assert not closed


def test_read_after_times_out_when_idle() -> None:
    log = EventLog("job")
    assert log.read_after(0, timeout=0.05) == ([], False)


def test_closed_log_returns_immediately_at_end() -> None:
    log = EventLog("job")
    log.append(payload(1))
    log.close()

    assert log.closed
    start = time.monotonic()
    assert log.read_after(1, timeout=10) == ([], True)
    assert time.monotonic() - start < 1
    events, closed = log.read_after(0, timeout=10)
    assert [e.seq for e in events] == [1] and closed


def test_close_wakes_blocked_reader() -> None:
    log = EventLog("job")
    result: list = []
    t = threading.Thread(target=lambda: result.append(log.read_after(0, timeout=10)))
    t.start()
    time.sleep(0.05)
    log.close()
    t.join(timeout=5)
    assert result == [([], True)]


def test_concurrent_appends_get_unique_contiguous_seqs() -> None:
    log = EventLog("job")

    def work() -> None:
        for n in range(100):
            log.append(payload(n))

    threads = [threading.Thread(target=work) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert [e.seq for e in log.snapshot()] == list(range(1, 401))
