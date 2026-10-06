from __future__ import annotations

import threading
import time
from collections.abc import Callable

from .events import Event


class EventLog:
    """Append-only, thread-safe event list with blocking reads by sequence number."""

    def __init__(self, job_id: str, clock: Callable[[], float] = time.time) -> None:
        self._job_id = job_id
        self._clock = clock
        self._events: list[Event] = []
        self._closed = False
        self._cond = threading.Condition()

    @property
    def closed(self) -> bool:
        with self._cond:
            return self._closed

    @property
    def last_seq(self) -> int:
        with self._cond:
            return len(self._events)

    def append(self, payload: object) -> Event:
        with self._cond:
            if self._closed:
                raise RuntimeError("event log is closed")
            event = Event(self._job_id, len(self._events) + 1, int(self._clock() * 1000), payload)  # type: ignore[arg-type]
            self._events.append(event)
            self._cond.notify_all()
            return event

    def close(self) -> None:
        with self._cond:
            self._closed = True
            self._cond.notify_all()

    def snapshot(self) -> list[Event]:
        with self._cond:
            return list(self._events)

    def read_after(self, after_seq: int, timeout: float) -> tuple[list[Event], bool]:
        """Events with seq > after_seq, waiting up to timeout if none exist yet.

        The flag is True when the log is closed and nothing follows the returned events.
        """
        start = max(after_seq, 0)
        with self._cond:
            self._cond.wait_for(lambda: len(self._events) > start or self._closed, timeout)
            return self._events[start:], self._closed
