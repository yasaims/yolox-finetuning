from __future__ import annotations

from fastapi import APIRouter, Response

from ..deps import Client
from ..schemas import Model

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
