from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Request

from .trainer_client import TrainerClient


def get_client(request: Request) -> TrainerClient:
    """The trainer client, created on first use inside the running event loop."""
    app = request.app
    if app.state.client is None:
        app.state.client = app.state.client_factory()
    client: TrainerClient = app.state.client
    return client


Client = Annotated[TrainerClient, Depends(get_client)]
