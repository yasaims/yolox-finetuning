from __future__ import annotations

import dataclasses
import enum
from dataclasses import dataclass
from typing import Any, Literal

from .hyperparams import Hyperparameters


class JobState(enum.Enum):
    STARTING = 1
    RUNNING = 2
    STOPPING = 3
    SUCCEEDED = 4
    FAILED = 5
    CANCELLED = 6


@dataclass(frozen=True)
class JobStarted:
    dataset_name: str
    hyperparameters: Hyperparameters
    max_epoch: int
    iters_per_epoch: int


@dataclass(frozen=True)
class IterProgress:
    epoch: int
    iter: int
    global_iter: int
    total_iters: int
    lr: float
    losses: dict[str, float]
    input_size: int
    eta_seconds: float


@dataclass(frozen=True)
class EpochEval:
    epoch: int
    ap50: float
    ap50_95: float
    best_ap50_95: float


@dataclass(frozen=True)
class JobFinished:
    state: JobState
    model_id: str
    error: str


@dataclass(frozen=True)
class Event:
    job_id: str
    seq: int
    time_unix_ms: int
    payload: JobStarted | IterProgress | EpochEval | JobFinished


@dataclass(frozen=True)
class WorkerStarted:
    max_epoch: int
    iters_per_epoch: int


@dataclass(frozen=True)
class WorkerDone:
    outcome: Literal["succeeded", "cancelled", "failed"]
    weights_path: str | None
    best_epoch: int
    best_ap50: float
    best_ap50_95: float
    epochs_completed: int
    error: str


_TYPE_NAMES: dict[type, str] = {
    JobStarted: "started",
    IterProgress: "iter",
    EpochEval: "eval",
    JobFinished: "finished",
}


def event_to_dict(event: Event) -> dict[str, Any]:
    kind = _TYPE_NAMES.get(type(event.payload))
    if kind is None:
        raise TypeError(f"unsupported event payload: {type(event.payload).__name__}")
    data = dataclasses.asdict(event.payload)
    if isinstance(event.payload, JobFinished):
        data["state"] = event.payload.state.name
    return {
        "job_id": event.job_id,
        "seq": event.seq,
        "time_unix_ms": event.time_unix_ms,
        "type": kind,
        "payload": data,
    }


def event_from_dict(d: dict[str, Any]) -> Event:
    kind = d["type"]
    data = dict(d["payload"])
    payload: JobStarted | IterProgress | EpochEval | JobFinished
    if kind == "started":
        data["hyperparameters"] = Hyperparameters(**data["hyperparameters"])
        payload = JobStarted(**data)
    elif kind == "iter":
        payload = IterProgress(**data)
    elif kind == "eval":
        payload = EpochEval(**data)
    elif kind == "finished":
        data["state"] = JobState[data["state"]]
        payload = JobFinished(**data)
    else:
        raise ValueError(f"unknown event type: {kind!r}")
    return Event(d["job_id"], d["seq"], d["time_unix_ms"], payload)
