import json
from typing import Any

import grpc
import pytest
from conftest import ClientFactory
from fakes import FakeTrainerClient


def parse_frames(body: str) -> list[dict[str, str]]:
    frames = []
    for block in body.replace("\r\n", "\n").split("\n\n"):
        fields = dict(line.split(": ", 1) for line in block.splitlines() if not line.startswith(":") and line)
        if fields:
            frames.append(fields)
    return frames


def strict_json(text: str) -> Any:
    def reject(constant: str) -> None:
        raise ValueError(f"non-standard JSON constant {constant}")

    return json.loads(text, parse_constant=reject)


def events() -> list[tuple[str, int, dict[str, Any]]]:
    return [
        ("started", 6, {"seq": 6, "time_unix_ms": 1, "max_epoch": 2}),
        ("iter", 7, {"seq": 7, "time_unix_ms": 2, "losses": {"total_loss": float("inf"), "iou_loss": float("nan")}}),
        ("finished", 8, {"seq": 8, "time_unix_ms": 3, "state": "SUCCEEDED", "model_id": "m1", "error": ""}),
        ("iter", 9, {"seq": 9, "time_unix_ms": 4}),
    ]


async def test_frames_carry_id_event_and_finite_json_and_end_at_finished(http: ClientFactory) -> None:
    fake = FakeTrainerClient()
    fake.events = events()
    response = await http(fake).get("/api/trainings/j1/events")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    frames = parse_frames(response.text)
    assert [(f["id"], f["event"]) for f in frames] == [("6", "started"), ("7", "iter"), ("8", "finished")]
    assert strict_json(frames[0]["data"]) == {"seq": 6, "time_unix_ms": 1, "max_epoch": 2}
    assert strict_json(frames[1]["data"])["losses"] == {"total_loss": None, "iou_loss": None}
    assert strict_json(frames[2]["data"])["state"] == "SUCCEEDED"


@pytest.mark.parametrize(
    ("headers", "query", "expected"),
    [
        ({"Last-Event-ID": "5"}, "", 5),
        ({}, "?after_seq=3", 3),
        ({"Last-Event-ID": "5"}, "?after_seq=3", 5),
        ({}, "", 0),
        ({"Last-Event-ID": "junk"}, "?after_seq=3", 3),
    ],
)
async def test_resume_position(http: ClientFactory, headers: dict[str, str], query: str, expected: int) -> None:
    fake = FakeTrainerClient()
    await http(fake).get(f"/api/trainings/j1/events{query}", headers=headers)
    assert fake.calls == [("stream_events", ("j1", expected))]


async def test_unknown_job_is_http_404(http: ClientFactory) -> None:
    response = await http(FakeTrainerClient(error=grpc.StatusCode.NOT_FOUND)).get("/api/trainings/nope/events")
    assert response.status_code == 404
