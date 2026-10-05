from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI

from .errors import install_error_handlers
from .routes import datasets, models, trainings
from .settings import Settings
from .trainer_client import GrpcTrainerClient, TrainerClient

ClientFactory = Callable[[], TrainerClient]


def _grpc_client() -> TrainerClient:
    return GrpcTrainerClient(Settings().trainer_addr)


def create_app(client_factory: ClientFactory = _grpc_client) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        yield
        if app.state.client is not None:
            await app.state.client.close()

    app = FastAPI(title="YOLOX finetuning gateway", lifespan=lifespan)
    app.state.client_factory = client_factory
    app.state.client = None
    install_error_handlers(app)
    for module in (datasets, models, trainings):
        app.include_router(module.router, prefix="/api")
    return app


app = create_app()
