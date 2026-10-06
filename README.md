# YOLOX Finetune platform

Local platform for finetuning YOLOX-Tiny on a small COCO-format dataset. Start a training run from the browser, watch loss and AP live, then evaluate and run inference with the resulting models.

## Architecture

```
browser ──> nginx + React UI (:8080) ──> gateway (FastAPI: REST + SSE) ──gRPC──> trainer (YOLOX, GPU)
                                                                                    │
                                                           datasets/  models/  pretrained/  (bind mounts)
```

- `frontend/`: React UI, served by nginx, which also proxies `/api` to the gateway.
- `gateway/`: translates REST/SSE to the trainer's gRPC API.
- `trainer/`: runs training jobs (one at a time), keeps the model registry, evaluates and predicts.
- `proto/`: gRPC contract. `scripts/`: dataset and weights tools and the e2e smoke test.

## Prerequisites

- Docker Desktop with NVIDIA GPU support (WSL2 backend)
- At least 30 GB free on the drive holding Docker's disk image. The trainer image is about 7 GB, and Docker Desktop stops responding if its disk image cannot grow during a build. Old trainer builds can be removed with `docker image prune`.
- [uv](https://docs.astral.sh/uv/)
- Node 22 (frontend development only)

## Setup

```bash
uv run --project scripts prepare-balloon --out datasets/balloon
uv run --project scripts download-weights --out pretrained/yolox_tiny.pth
bash scripts/gen_proto.sh   # only for host development and tests
```

The Balloon dataset comes from the matterport Mask_RCNN v2.1 release and is converted to COCO format.

## Run

```bash
docker compose up -d --build
```

Open http://localhost:8080.

## Using your own dataset

Place a COCO-format dataset under `datasets/`:

```
datasets/<name>/
  train/                 # training images
  val/                   # validation images
  annotations/
    train.json
    val.json
```

`<name>` must match `^[a-z0-9][a-z0-9_-]{0,63}$`. Category ids start from 1. It appears in the UI after a refresh; no restart is needed.

## Tests

Fresh checkouts need `bash scripts/gen_proto.sh` before the host tests of `trainer` and `gateway`.

| Component | Command |
| --- | --- |
| scripts | `cd scripts && uv run pytest` |
| trainer (host, no GPU) | `cd trainer && uv run pytest` |
| trainer engine (GPU, in container) | `docker compose run --rm trainer pytest -m engine` |
| gateway | `cd gateway && uv run pytest` |
| frontend | `cd frontend && npm test` |

### End-to-end smoke test

With the stack running and the Balloon dataset prepared:

```bash
uv run --project scripts e2e-smoke --base http://localhost:8080 --epochs 5
```

It trains, checks the SSE stream, evaluates, predicts, cancels a second job, and deletes the models it created. It exits non-zero on failure.
