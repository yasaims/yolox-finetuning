from __future__ import annotations

import grpc
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from .trainer_client import TrainerError

_STATUS: dict[grpc.StatusCode, int] = {
    grpc.StatusCode.NOT_FOUND: 404,
    grpc.StatusCode.FAILED_PRECONDITION: 409,
    grpc.StatusCode.INVALID_ARGUMENT: 422,
    grpc.StatusCode.UNAVAILABLE: 503,
}
_DEFAULT_STATUS = 502


def http_status(code: grpc.StatusCode) -> int:
    return _STATUS.get(code, _DEFAULT_STATUS)


def install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(TrainerError)
    async def _trainer_error(request: Request, exc: TrainerError) -> JSONResponse:
        return JSONResponse(status_code=http_status(exc.code), content={"detail": exc.message})
