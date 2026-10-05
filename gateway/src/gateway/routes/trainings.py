from __future__ import annotations

from fastapi import APIRouter, Response

from ..deps import Client
from ..schemas import CurrentJobResponse, StartTrainingRequest, StartTrainingResult

router = APIRouter(prefix="/trainings")


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
