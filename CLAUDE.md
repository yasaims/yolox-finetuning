## purpose

local platform for finetuning yolox object detection on small dataset

## dev rules

- TDD
- commit on develop branch after each tasks

## commands

- Host tests need stubs first on a fresh checkout: `bash scripts/gen_proto.sh`
- scripts: `cd scripts && uv run pytest`
- trainer (host): `cd trainer && uv run pytest`; engine/GPU: `docker compose run --rm trainer pytest -m engine`
- gateway: `cd gateway && uv run pytest`
- frontend: `cd frontend && npm test`
- stack: `docker compose up -d --build` (UI at http://localhost:8080)
- e2e: `uv run --project scripts e2e-smoke --base http://localhost:8080 --epochs 5`
