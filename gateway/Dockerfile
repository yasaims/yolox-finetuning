# syntax=docker/dockerfile:1.7
# Build context: repository root.
FROM ghcr.io/astral-sh/uv:python3.12-bookworm AS builder

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/app/.venv

WORKDIR /app/gateway

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=gateway/pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=gateway/uv.lock,target=uv.lock \
    uv sync --locked --no-install-project

COPY proto/ /app/proto/
RUN mkdir -p /app/gateway/src \
    && /app/.venv/bin/python -m grpc_tools.protoc \
        -I /app/proto \
        --python_out=/app/gateway/src \
        --grpc_python_out=/app/gateway/src \
        --pyi_out=/app/gateway/src \
        /app/proto/yoloxft/v1/trainer.proto \
    && touch /app/gateway/src/yoloxft/__init__.py /app/gateway/src/yoloxft/v1/__init__.py

COPY gateway/ /app/gateway/

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev

FROM python:3.12-slim-bookworm

COPY --from=builder /app /app

ENV PATH=/app/.venv/bin:$PATH \
    PYTHONUNBUFFERED=1

WORKDIR /app/gateway

EXPOSE 8000

CMD ["uvicorn", "gateway.main:app", "--host", "0.0.0.0", "--port", "8000"]
