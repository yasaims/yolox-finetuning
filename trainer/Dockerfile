# syntax=docker/dockerfile:1.7
# Build context: repository root.
FROM ghcr.io/astral-sh/uv:python3.10-bookworm AS builder

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/app/.venv

WORKDIR /app/trainer

# torch and other third-party deps first; yolox builds against them without isolation.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=trainer/pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=trainer/uv.lock,target=uv.lock \
    uv sync --locked --extra engine --no-install-project --no-install-package yolox

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=trainer/pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=trainer/uv.lock,target=uv.lock \
    uv sync --locked --extra engine --no-install-project

# proto/ is optional until the service contract exists.
RUN --mount=type=bind,source=.,target=/ctx \
    if [ -d /ctx/proto ]; then cp -r /ctx/proto /app/proto; fi
# gRPC stub generation (grpc_tools.protoc) goes here.

COPY trainer/ /app/trainer/

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --extra engine

FROM python:3.10-slim-bookworm

COPY --from=builder /app /app

ENV PATH=/app/.venv/bin:$PATH \
    PYTHONUNBUFFERED=1

WORKDIR /app/trainer

CMD ["python", "-c", "import yolox"]
