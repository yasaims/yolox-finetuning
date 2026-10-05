import json
import pickle

import pytest

from yolox_trainer.core.events import (
    EpochEval,
    Event,
    IterProgress,
    JobFinished,
    JobStarted,
    JobState,
    WorkerDone,
    WorkerStarted,
    event_from_dict,
    event_to_dict,
)
from yolox_trainer.core.hyperparams import DEFAULTS

PAYLOADS = [
    JobStarted("balloon", DEFAULTS, 50, 12),
    IterProgress(1, 10, 10, 600, 0.001, {"total_loss": 3.5, "iou_loss": 1.25}, 416, 12.5),
    EpochEval(2, 0.5, 0.25, 0.3),
    JobFinished(JobState.SUCCEEDED, "model-1", ""),
    JobFinished(JobState.FAILED, "", "boom"),
]


@pytest.mark.parametrize("payload", PAYLOADS, ids=lambda p: type(p).__name__ + getattr(getattr(p, "state", None), "name", ""))
def test_event_dict_roundtrip(payload: object) -> None:
    event = Event("job-1", 7, 1_700_000_000_000, payload)
    d = event_to_dict(event)
    assert event_from_dict(json.loads(json.dumps(d))) == event


def test_job_state_serialized_as_name() -> None:
    d = event_to_dict(Event("j", 1, 0, JobFinished(JobState.CANCELLED, "", "")))
    assert "CANCELLED" in json.dumps(d)


def test_unknown_payload_rejected() -> None:
    with pytest.raises(TypeError):
        event_to_dict(Event("j", 1, 0, object()))


@pytest.mark.parametrize(
    "message",
    [
        WorkerStarted(50, 12),
        WorkerDone("succeeded", "/w/best.pth", 3, 0.9, 0.6, 50, ""),
        WorkerDone("failed", None, 0, 0.0, 0.0, 0, "oom"),
        IterProgress(1, 1, 1, 10, 0.1, {"a": 1.0}, 416, 1.0),
        EpochEval(1, 0.1, 0.1, 0.1),
    ],
)
def test_worker_messages_pickle_roundtrip(message: object) -> None:
    assert pickle.loads(pickle.dumps(message)) == message
