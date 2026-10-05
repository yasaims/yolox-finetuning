from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, File, Form, HTTPException, Response, UploadFile

from ..deps import Client
from ..schemas import EvaluateRequest, EvaluationResult, Model, PredictResponse

MAX_IMAGE_BYTES = 20 << 20
DEFAULT_SCORE_THRESHOLD = 0.3

router = APIRouter(prefix="/models")


@router.get("")
async def list_models(client: Client) -> list[Model]:
    return await client.list_models()


@router.get("/{model_id}")
async def get_model(model_id: str, client: Client) -> Model:
    return await client.get_model(model_id)


@router.delete("/{model_id}", status_code=204)
async def delete_model(model_id: str, client: Client) -> Response:
    await client.delete_model(model_id)
    return Response(status_code=204)


@router.post("/{model_id}/evaluate")
async def evaluate_model(model_id: str, body: EvaluateRequest, client: Client) -> EvaluationResult:
    return await client.evaluate_model(model_id, body.dataset_name, body.split)


@router.post("/{model_id}/predict")
async def predict(
    model_id: str,
    client: Client,
    file: Annotated[UploadFile, File()],
    score_threshold: Annotated[float, Form(ge=0, le=1)] = DEFAULT_SCORE_THRESHOLD,
) -> PredictResponse:
    image = await file.read(MAX_IMAGE_BYTES + 1)
    if len(image) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail=f"image exceeds {MAX_IMAGE_BYTES >> 20} MB")
    return await client.predict(model_id, image, score_threshold)
