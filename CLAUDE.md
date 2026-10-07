## purpose

local platform for finetuning yolox object detection on small dataset

## dev rules

- TDD
- commit on develop branch after each tasks

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

## commands

- Host tests need stubs first on a fresh checkout: `bash scripts/gen_proto.sh`
- scripts: `cd scripts && uv run pytest`
- trainer (host): `cd trainer && uv run pytest`; engine/GPU: `docker compose run --rm trainer pytest -m engine`
- gateway: `cd gateway && uv run pytest`
- frontend: `cd frontend && npm test`
- stack: `docker compose up -d --build` (UI at http://localhost:8080)
- e2e: `uv run --project scripts e2e-smoke --base http://localhost:8080 --epochs 5`

## Tests

Fresh checkouts need `bash scripts/gen_proto.sh` before the host tests of `trainer` and `gateway`.

| Component                          | Command                                            |
| ---------------------------------- | -------------------------------------------------- |
| scripts                            | `cd scripts && uv run pytest`                      |
| trainer (host, no GPU)             | `cd trainer && uv run pytest`                      |
| trainer engine (GPU, in container) | `docker compose run --rm trainer pytest -m engine` |
| gateway                            | `cd gateway && uv run pytest`                      |
| frontend                           | `cd frontend && npm test`                          |

### End-to-end smoke test

With the stack running and the Balloon dataset prepared:

```bash
uv run --project scripts e2e-smoke --base http://localhost:8080 --epochs 5
```

It trains, checks the SSE stream, evaluates, predicts, cancels a second job, and deletes the models it created. It exits non-zero on failure.
