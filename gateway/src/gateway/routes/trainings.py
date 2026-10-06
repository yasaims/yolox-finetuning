from __future__ import annotations

import asyncio
import contextlib
import json
import math
from collections.abc import AsyncGenerator, AsyncIterator
from typing import Annotated, Any

from fastapi import APIRouter, Header, Query, Response
from sse_starlette import EventSourceResponse, ServerSentEvent

from ..deps import Client
from ..schemas import CurrentJobResponse, StartTrainingRequest, StartTrainingResult
from ..trainer_client import EventTuple

PING_SECONDS = 15
# How long to wait for the first event so a missing job still yields an HTTP 404.
FIRST_EVENT_PROBE_SECONDS = 1.0

router = APIRouter(prefix="/trainings")


def _finite(value: Any) -> Any:
    """Replaces non-finite floats with None so the JSON is valid for JSON.parse."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: _finite(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_finite(v) for v in value]
    return value


def _resume_seq(last_event_id: str | None, after_seq: int) -> int:
    if last_event_id is not None:
        with contextlib.suppress(ValueError):
            return max(int(last_event_id), 0)
    return after_seq


@router.post("", status_code=201)
async def start_training(body: StartTrainingRequest, client: Client) -> StartTrainingResult:
    overrides = body.hyperparameters.model_dump(exclude_none=True)
    return await client.start_training(body.dataset_name, overrides, body.display_name)


@router.get("/current")
async def current_training(client: Client) -> CurrentJobResponse:
    return CurrentJobResponse(job=await client.get_current_job())


@router.post("/{job_id}/stop", status_code=202)
async def stop_training(job_id: str, client: Client) -> Response:
    await client.stop_training(job_id)
    return Response(status_code=202)


@router.get("/{job_id}/events")
async def training_events(
    job_id: str,
    client: Client,
    last_event_id: Annotated[str | None, Header()] = None,
    after_seq: Annotated[int, Query(ge=0)] = 0,
) -> EventSourceResponse:
    stream = client.stream_events(job_id, _resume_seq(last_event_id, after_seq))
    first = asyncio.ensure_future(anext(stream))
    try:
        await asyncio.wait({first}, timeout=FIRST_EVENT_PROBE_SECONDS)
        if first.done() and not first.cancelled():
            exc = first.exception()
            if exc is not None and not isinstance(exc, StopAsyncIteration):
                raise exc
    except BaseException:
        await _discard(first, stream)
        raise

    async def frames() -> AsyncIterator[ServerSentEvent]:
        try:
            try:
                item: EventTuple | None = await first
            except StopAsyncIteration:
                return
            while item is not None:
                event_type, seq, payload = item
                yield ServerSentEvent(data=json.dumps(_finite(payload)), event=event_type, id=str(seq))
                if event_type == "finished":
                    return
                item = await anext(stream, None)
        finally:
            await _discard(first, stream)

    return EventSourceResponse(frames(), ping=PING_SECONDS)


async def _discard(first: asyncio.Future[EventTuple], stream: AsyncGenerator[EventTuple, None]) -> None:
    """Cancels the pending first read and closes the stream, which cancels the gRPC call."""
    if not first.done():
        first.cancel()
    await asyncio.wait({first})
    if not first.cancelled():
        first.exception()
    await stream.aclose()
