#!/usr/bin/env bash
# Generates gRPC stubs from proto/ into trainer/src and gateway/src.
set -euo pipefail

cd "$(dirname "$0")/.."

for project in trainer gateway; do
    uv run --project "$project" python -m grpc_tools.protoc \
        -I proto \
        --python_out="$project/src" \
        --grpc_python_out="$project/src" \
        --pyi_out="$project/src" \
        proto/yoloxft/v1/trainer.proto
    touch "$project/src/yoloxft/__init__.py" "$project/src/yoloxft/v1/__init__.py"
done
