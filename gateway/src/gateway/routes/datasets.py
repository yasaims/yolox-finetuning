from __future__ import annotations

from fastapi import APIRouter

from ..deps import Client
from ..schemas import Dataset, Hyperparameters

router = APIRouter()


@router.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/datasets")
async def list_datasets(client: Client) -> list[Dataset]:
    return await client.list_datasets()


@router.get("/hyperparameters/defaults")
async def default_hyperparameters(client: Client) -> Hyperparameters:
    return await client.get_default_hyperparameters()
