# YOLOX-Tiny Finetuning Platform — Implementation Plan

Spec: `docs/superpowers/specs/2026-10-05-yolox-finetune-platform-design.md` (binding authority; Appendices A–D hold the proto, dataset layout, hyperparameter table and YOLOX facts).

## Global Constraints

- Repo root: `C:\Users\didib\git\yolox-finetuning` (Windows 11, Git Bash). Work on branch `develop`; **commit after each task** with a message ending in `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- **TDD**: write the failing test first, watch it fail, then implement. Only tests that guarantee behavior; no tests for the sake of tests.
- Python test layout per project: `tests/mechanical/` (contract-level, derivable from the code/API) and `tests/semantic/` (project-meaning behavior). Keep semantic tests minimal.
- Python: **uv only** (no pip/conda on host), type hints on all new code. Each Python project is its own uv project (`trainer/` Python 3.10; `gateway/` and `scripts/` Python 3.12), src layout.
- Comments: short, describe current behavior only; no history, no references to plans/tasks/docs.
- Don't reinvent: reuse YOLOX (`yolox.core.Trainer`, `yolox.exp.default.yolox_tiny.Exp`, `yolox.utils.load_ckpt`, `COCOEvaluator`, `yolox.tools.demo.Predictor` / `ValTransform` + `postprocess`), pycocotools, grpcio, FastAPI, sse-starlette.
- YOLOX upstream: `https://github.com/Megvii-BaseDetection/YOLOX` pinned at `6ddff4824372906469a7fae2dc3206c7aa4bbaee`. A read-only local checkout is at `C:\Users\didib\git\yolox` for reading source (do not modify it).
- `trainer/src/yolox_trainer/core/` and `grpc_api/` must not import torch, cv2 or yolox (host-testable). Only `engine/` may.
- Generated gRPC stubs are gitignored and produced by `scripts/gen_proto.sh` (host) and the Dockerfiles. Pin identical `grpcio`, `grpcio-tools`, `protobuf` versions in trainer and gateway.
- Data dirs at repo root: `datasets/`, `models/`, `pretrained/` — contents gitignored (keep `.gitkeep`).
- Same failure 3 times → stop and report BLOCKED instead of retrying.

## Shared interfaces (all tasks must use these names)

`trainer/src/yolox_trainer/core/`:
- `errors.py`: `class NotFound(Exception)`, `class InvalidArgument(ValueError)`, `class FailedPrecondition(Exception)`.
- `hyperparams.py`: frozen dataclass `Hyperparameters` (fields + defaults exactly as spec Appendix D, same names as proto); `resolve(overrides: Mapping[str, object]) -> Hyperparameters` merges non-None overrides onto defaults and validates (raises `InvalidArgument` with a field-specific message); `DEFAULTS = Hyperparameters()`.
- `ids.py`: `ID_PATTERN = r"^[a-z0-9][a-z0-9-]{0,63}$"`; `validate_id(value: str) -> str` (raises `InvalidArgument`); `new_model_id(dataset_name: str, now: datetime, rand_hex: str | None = None) -> str` → `YYYYMMDD-HHMMSS-<dataset with _ replaced by ->-<6 hex>`.
- `datasets.py`: frozen dataclass `DatasetInfo(name: str, root: Path, class_names: list[str], train_images: int, val_images: int)`; `class DatasetCatalog(root: Path)` with `list() -> list[DatasetInfo]` (sorted by name), `get(name) -> DatasetInfo` (raises `NotFound`/`InvalidArgument`), `annotation_path(name, split) -> Path`, `SPLITS = ("train", "val")`.
- `events.py` (picklable, torch-free; used by worker child too): `class JobState(enum.Enum)` STARTING/RUNNING/STOPPING/SUCCEEDED/FAILED/CANCELLED; dataclasses `JobStarted(dataset_name, hyperparameters, max_epoch, iters_per_epoch)`, `IterProgress(epoch, iter, global_iter, total_iters, lr, losses: dict[str,float], input_size, eta_seconds)`, `EpochEval(epoch, ap50, ap50_95, best_ap50_95)`, `JobFinished(state: JobState, model_id: str, error: str)`, `Event(job_id, seq, time_unix_ms, payload)`; worker messages `WorkerStarted(max_epoch, iters_per_epoch)`, `WorkerDone(outcome: Literal["succeeded","cancelled","failed"], weights_path: str | None, best_epoch: int, best_ap50: float, best_ap50_95: float, epochs_completed: int, error: str)` (worker also sends `IterProgress` and `EpochEval` directly); `event_to_dict(event) -> dict` / `event_from_dict(d) -> Event` for JSONL.
- `event_log.py`: `class EventLog(job_id: str, clock=time.time)` with `append(payload) -> Event`, `close()`, `closed: bool`, `last_seq: int`, `snapshot() -> list[Event]`, `read_after(after_seq: int, timeout: float) -> tuple[list[Event], bool]` (returns immediately if newer events exist; otherwise waits up to timeout; bool = log closed and no events after the returned ones). seq starts at 1.
- `registry.py`: frozen dataclass `ModelRecord(id, display_name, status: Literal["completed","stopped"], created_unix_ms, finished_unix_ms, base_model, dataset_name, class_names, hyperparameters: Hyperparameters, epochs_trained, best_epoch, best_ap50, best_ap50_95, eval_history: list[EpochEval])`; `class ModelRegistry(root: Path)` with `work_dir(model_id) -> Path` (`root/.work/<id>`), `register(record, weights_src: Path, events: list[Event], log_src: Path | None) -> ModelRecord`, `list() -> list[ModelRecord]` (newest first), `get(id)`, `delete(id)`, `weights_path(id) -> Path`, `cleanup() -> None`. Files: `<id>/model.json` (with `schema_version: 1`), `weights.pth`, `events.jsonl`, `train_log.txt`.
- `supervisor.py`: frozen dataclass `JobSpec(job_id, dataset_name, data_dir: str, class_names: list[str], hyperparameters: Hyperparameters, display_name: str, work_dir: str, pretrained_path: str)`; `class TrainingRunner(Protocol)`: `start(spec, on_message: Callable[[object], None], on_exit: Callable[[int | None], None]) -> None`, `request_stop() -> None`; frozen dataclass `JobSummary(job_id, state, dataset_name, hyperparameters, last_seq, started_unix_ms)`; `class TrainingSupervisor(runner_factory, registry, catalog, pretrained_path: Path, clock=time.time)` with `start(dataset_name, overrides: Mapping[str, object], display_name: str) -> tuple[str, Hyperparameters]`, `stop(job_id)`, `current() -> JobSummary | None`, `event_log(job_id) -> EventLog`, `shutdown(timeout: float)`.
- `coco_metrics.py`: frozen dataclasses `CocoStats` (12 floats named as proto), `ClassAp(class_name, ap, ap50)`, `EvaluationMetrics(stats, per_class, num_images)`; `compute(gt: dict, detections: list[dict]) -> EvaluationMetrics` (detections in COCO results format `{image_id, category_id, bbox:[x,y,w,h], score}`).
- `inference.py`: `class InferenceService(Protocol)`: `evaluate(model_id, dataset_name, split) -> EvaluationMetrics`, `predict(model_id, image: bytes, score_threshold: float) -> PredictResult`, `forget(model_id) -> None`; dataclasses `Detection(x1,y1,x2,y2,score,class_id,class_name)`, `PredictResult(image_width, image_height, detections)`.

Trainer config (env, `config.py`): `TRAINER_DATASETS_DIR` (default `/data/datasets`), `TRAINER_MODELS_DIR` (`/data/models`), `TRAINER_PRETRAINED` (`/data/pretrained/yolox_tiny.pth`), `TRAINER_PORT` (`50051`), `TRAINER_DEVICE` (`gpu`|`cpu`, default `gpu`).

Gateway REST API (JSON snake_case, prefix `/api`):
- `GET /health`; `GET /datasets`; `GET /hyperparameters/defaults`
- `POST /trainings` body `{dataset_name, display_name?, hyperparameters?: partial}` → 201 `{job_id, hyperparameters}`
- `GET /trainings/current` → `{job: JobSummary | null}`; `POST /trainings/{job_id}/stop` → 202
- `GET /trainings/{job_id}/events` SSE; resume from `Last-Event-ID` header or `?after_seq=`; SSE `id`=seq, `event` ∈ `started|iter|eval|finished`, `data`=JSON payload plus `seq`, `time_unix_ms`
- `GET /models`; `GET /models/{id}`; `DELETE /models/{id}` → 204
- `POST /models/{id}/evaluate` body `{dataset_name?, split: "val"|"train" = "val"}` (dataset defaults to the model's)
- `POST /models/{id}/predict` multipart `file`, form `score_threshold` (default 0.3) → `{image_width, image_height, detections}`
- gRPC→HTTP: NOT_FOUND→404, FAILED_PRECONDITION→409, INVALID_ARGUMENT→422, UNAVAILABLE→503, others→502.

---

## Task 1: Scaffold + Balloon dataset tooling

Files: `.gitignore`, `.gitattributes`, `datasets/.gitkeep`, `models/.gitkeep`, `pretrained/.gitkeep`, `scripts/pyproject.toml`, `scripts/src/dataset_tools/{__init__.py,via_to_coco.py,balloon.py,download_weights.py}`, `scripts/tests/{mechanical,semantic}/`.

Requirements:
- `.gitignore`: Python caches/venvs, `node_modules/`, `dist/`, generated `*_pb2*.py`/`*_pb2*.pyi`, `datasets/*`, `models/*`, `pretrained/*` (but not `.gitkeep`), `.superpowers/`. `.gitattributes`: `*.sh text eol=lf`, `Dockerfile text eol=lf`.
- `scripts` uv project (Python 3.12), deps: `pillow`; dev: `pytest`, `pycocotools`. Console scripts: `prepare-balloon`, `download-weights`.
- `via_to_coco.py` pure functions: `polygon_to_bbox(xs, ys) -> [x, y, w, h]` (min/max, float); `via_regions(entry) -> list[dict]` accepting VIA `regions` as dict or list; `build_coco(images: list[tuple[file_name, width, height, regions]], class_name: str) -> dict` producing COCO with `images` (ids from 1), `annotations` (ids from 1, `category_id` 1, `bbox`, `area=w*h`, `iscrowd=0`), `categories=[{"id":1,"name":class_name}]`.
- `balloon.py` CLI `prepare-balloon --out datasets/balloon [--train 40 --val 10 --seed 0] [--zip PATH]`: downloads `https://github.com/matterport/Mask_RCNN/releases/download/v2.1/balloon_dataset.zip` (urllib; skip if `--zip` given or cached in the out dir's parent `.cache`), pools the zip's train+val VIA images, picks a seeded disjoint subset, copies images into `<out>/train/` and `<out>/val/`, reads sizes with Pillow, writes `<out>/annotations/train.json` and `val.json` (class `balloon`). Layout per spec Appendix C.
- `download_weights.py` CLI `download-weights --out pretrained/yolox_tiny.pth`: downloads `https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_tiny.pth` if absent.
- Tests: mechanical — bbox from polygon; regions dict vs list; seeded subset deterministic, sizes 40/10, disjoint. Semantic — a built COCO dict (tiny synthetic input) loads in `pycocotools.coco.COCO`, `getAnnIds(iscrowd=False)` works, and COCOeval of GT-as-detections gives AP 1.0.
- After tests pass, actually run both CLIs from repo root and confirm `datasets/balloon/{train,val}` have 40/10 images and `pretrained/yolox_tiny.pth` exists (~40 MB). These data files are not committed.

## Task 2: Trainer container de-risk (torch + YOLOX + GPU)

Files: `trainer/pyproject.toml`, `trainer/uv.lock`, `trainer/Dockerfile`, `trainer/.dockerignore` or root `.dockerignore`, `trainer/src/yolox_trainer/__init__.py`, `trainer/tests/conftest.py`, `trainer/tests/engine/test_pretrained_load.py`, `docker-compose.yml` (trainer service only for now).

Requirements:
- `trainer` uv project, `requires-python = ">=3.10,<3.11"`. Base deps (host-installable): `grpcio`, `grpcio-health-checking`, `protobuf`, `numpy<2`, `pycocotools`. Optional extra `engine`: `torch==2.8.*`, `torchvision==0.23.*`, `yolox`, `opencv-python-headless`, `loguru`, `tabulate`, `psutil`, `tqdm`, `thop`, `tensorboard`, `ninja`, `setuptools`, `wheel`. Dev group: `pytest`, `grpcio-tools`.
- torch/torchvision from explicit index `https://download.pytorch.org/whl/cu128` (`[[tool.uv.index]] name="pytorch-cu128" explicit=true` + `[tool.uv.sources]`). `yolox = { git = "https://github.com/Megvii-BaseDetection/YOLOX", rev = "6ddff4824372906469a7fae2dc3206c7aa4bbaee" }`, `[tool.uv] no-build-isolation-package = ["yolox"]`, `[[tool.uv.dependency-metadata]] name="yolox" version="0.3.0" requires-dist=[]`.
- Locking on Windows host must work (`uv lock`); if the git build needs torch at lock time and fails, document and use the fallback: install yolox in the Dockerfile with `uv pip install --no-deps --no-build-isolation "git+https://github.com/Megvii-BaseDetection/YOLOX@6ddff48..."` and run with `UV_NO_SYNC=1`.
- Dockerfile (build context = repo root): builder `ghcr.io/astral-sh/uv:python3.10-bookworm`, BuildKit uv cache mount, sync torch first (`--no-install-package yolox`), then yolox; copy `proto/` (may not exist yet — tolerate) and `trainer/`; runtime `python:3.10-slim-bookworm` with the venv copied at the same path, `libgl`-free (headless opencv), `ENV PATH=/app/.venv/bin:$PATH`, include dev group (pytest) so tests run in the container. Default CMD can be a placeholder (`python -c "import yolox"`) until Task 8 sets the server.
- `docker-compose.yml` `trainer` service: build context `.`, dockerfile `trainer/Dockerfile`, GPU via `deploy.resources.reservations.devices: [{driver: nvidia, count: 1, capabilities: [gpu]}]`, `shm_size: 2gb`, volumes `./datasets:/data/datasets:ro`, `./models:/data/models`, `./pretrained:/data/pretrained:ro`, `./trainer/tests:/app/trainer/tests:ro` is NOT needed if tests are copied into the image.
- `tests/conftest.py`: register markers `engine` (requires torch+yolox; auto-skip when import fails) and `gpu` (requires `torch.cuda.is_available()`; auto-skip otherwise); `needs_weights` skip when `TRAINER_PRETRAINED` file is missing.
- Test (engine, needs_weights, runs on CPU): build `yolox.exp.default.yolox_tiny.Exp` with `num_classes=1`, `get_model()`, `load_ckpt(model, torch.load(path, map_location="cpu", weights_only=False)["model"])`; assert every state_dict key except those starting with `head.cls_preds.` equals the checkpoint tensor, and `head.cls_preds.*` shapes have out-dim 1.
- Verification commands (record output in report): `docker compose build trainer`; `docker compose run --rm trainer python -c "import torch, yolox; from yolox.tools.demo import Predictor; from yolox.layers import COCOeval_opt; print(torch.__version__, torch.cuda.is_available())"` → CUDA True (if `COCOeval_opt` import fails, report it; YOLOX falls back to pycocotools); `docker compose run --rm trainer pytest -m engine -q`.

## Task 3: Proto + stub generation + gateway project skeleton

Files: `proto/yoloxft/v1/trainer.proto` (exactly spec Appendix B), `scripts/gen_proto.sh`, `gateway/pyproject.toml`, `gateway/uv.lock`, `gateway/src/gateway/__init__.py`, trainer/gateway import tests, Dockerfile stub step in `trainer/Dockerfile`.

Requirements:
- `gen_proto.sh` (bash, LF) generates for both projects: `uv run --project <p> python -m grpc_tools.protoc -I proto --python_out=<p>/src --grpc_python_out=<p>/src --pyi_out=<p>/src proto/yoloxft/v1/trainer.proto`, creating `__init__.py` files under `<p>/src/yoloxft/` and `<p>/src/yoloxft/v1/` as needed (these `__init__.py` may be committed or generated — be consistent with `.gitignore`).
- `gateway` uv project (Python 3.12): deps `fastapi`, `uvicorn[standard]`, `sse-starlette`, `python-multipart`, `grpcio`, `protobuf`, `pydantic-settings`; dev `pytest`, `httpx`, `grpcio-tools`, `pytest-asyncio`. grpcio/grpcio-tools/protobuf versions identical to trainer's.
- `trainer/Dockerfile` runs the same protoc command during build.
- Mechanical test in each project: `from yoloxft.v1 import trainer_pb2, trainer_pb2_grpc` works and `TrainerServiceServicer` has all 11 RPC methods.

## Task 4: core — errors, hyperparams, ids, dataset catalog

Files: `trainer/src/yolox_trainer/core/{__init__.py,errors.py,hyperparams.py,ids.py,datasets.py}`, tests.

Requirements: as in "Shared interfaces". Validation table = spec Appendix D (also: `warmup_epochs < max_epoch`, `no_aug_epochs < max_epoch`). Catalog: a subdirectory is a dataset iff `annotations/train.json` and `annotations/val.json` exist and the name matches `^[a-z0-9][a-z0-9_-]{0,63}$`; class names = categories sorted by id; image counts = len(images) in each json. `get()` rejects invalid names with `InvalidArgument` before touching the filesystem.
Tests: mechanical — defaults equal Appendix D; each invalid field rejected; overrides with None ignored; `validate_id` rejects `../x`, uppercase, empty, 65 chars; `new_model_id` format. Semantic — catalog on `tmp_path` lists only complete datasets with correct classes/counts.

## Task 5: core — events + EventLog

Files: `core/events.py`, `core/event_log.py`, tests.
Requirements: as in "Shared interfaces". `event_to_dict`/`event_from_dict` round-trip every payload type (JobState as name string, Hyperparameters as dict).
Tests: mechanical — dict round-trip per payload type; pickle round-trip of worker messages. Semantic — `read_after` replays only seq > after_seq; a reader blocked in `read_after` wakes on `append` (thread); after `close()` a reader at the end returns `([], True)` immediately; concurrent appends from 4 threads give unique contiguous seqs.

## Task 6: core — model registry

Files: `core/registry.py`, tests.
Requirements: as in "Shared interfaces". `register` writes into `root/.staging-<id>/` then `os.replace` to `root/<id>`, copies weights (and log if given), writes `events.jsonl` (one `event_to_dict` per line) and `model.json`; afterwards removes `work_dir(id)` if present. `cleanup()` removes `root/.work/*` and `root/.staging-*`. `get/delete/weights_path` validate ids (`InvalidArgument`) and raise `NotFound`. `list` skips dirs without valid `model.json`.
Tests: semantic — register then get/list returns equal record; list newest-first; delete removes directory; a leftover `.staging-x` is invisible to `list` and removed by `cleanup`; traversal id rejected.

## Task 7: core — training supervisor

Files: `core/supervisor.py`, `tests/fakes.py` (FakeTrainingRunner), tests.
Requirements:
- `start`: raises `FailedPrecondition` if a job is STARTING/RUNNING/STOPPING or the pretrained file is missing; resolves hyperparams (`InvalidArgument`); `catalog.get` (`NotFound`); id from `new_model_id`; creates a fresh `EventLog` (replaces the previous job's); builds `JobSpec` (work_dir = `registry.work_dir(id)`); calls `runner_factory().start(spec, on_message, on_exit)`; state STARTING.
- Message mapping: `WorkerStarted` → state RUNNING + `JobStarted` event; `IterProgress`/`EpochEval` → appended as-is (also collect eval history); `WorkerDone` → if `weights_path` set: register `ModelRecord` (status `completed` for succeeded, `stopped` for cancelled; log `work_dir/<id>/train_log.txt` if present) then `JobFinished(SUCCEEDED|CANCELLED, model_id)`; if failed → `JobFinished(FAILED, "", error)`; cancelled without weights → `JobFinished(CANCELLED, "")`. Registration failure → FAILED with the error. Close the log after `JobFinished`.
- `on_exit(code)` without prior `WorkerDone` → `JobFinished(FAILED, "", "training process exited with code N")`.
- `stop(job_id)`: NotFound if not current job; if active → state STOPPING and `runner.request_stop()`; no-op if finished.
- `shutdown(timeout)` stops an active job and waits up to timeout for it to finish.
- All state guarded by a lock; callbacks may arrive on another thread.
Tests (semantic, fake runner driven from a thread): second start while running → FailedPrecondition; happy path emits started/iter/eval/finished and registers a model whose id is in `JobFinished`; stop → runner got stop request, fake replies cancelled with weights → status `stopped`; exit without done → FAILED; missing pretrained → FailedPrecondition.

## Task 8: core — COCO metrics

Files: `core/coco_metrics.py`, tests.
Requirements: `compute(gt, detections)`: build `COCO()` from the dict in memory (no temp file needed: set `dataset` and `createIndex()`), `loadRes` (empty detections → return all-zero stats and zero per-class AP without calling loadRes), run `COCOeval(..., "bbox")` evaluate/accumulate/summarize (suppress stdout), map `stats[0..11]`; per-class AP from `eval.eval["precision"]`: AP = mean of `precision[:, :, k, 0, -1]` over entries > -1, AP50 = mean of `precision[0, :, k, 0, -1]` over entries > -1 (0.0 if none); `num_images` = len(gt images).
Tests (semantic): perfect detections → ap = ap50 = 1.0; no detections → 0.0; two classes where only class A is predicted perfectly → per_class A=1.0, B=0.0.

## Task 9: gRPC servicer + server bootstrap

Files: `grpc_api/{__init__.py,convert.py,servicer.py}`, `server.py`, `config.py`, `__main__.py`, `tests/fakes.py` (FakeInferenceService), tests; `trainer/Dockerfile` CMD `python -m yolox_trainer`; compose healthcheck.
Requirements:
- `convert.py`: dataclass ↔ proto for Hyperparameters (proto optional fields → overrides dict of set fields only), DatasetInfo, ModelRecord→Model, Event→TrainingEvent, EvaluationMetrics→EvaluationResult, PredictResult→PredictResponse, JobSummary.
- `servicer.py` `TrainerServicer(supervisor, registry, catalog, inference: InferenceService)`; maps `NotFound`/`InvalidArgument`/`FailedPrecondition` to gRPC codes via `context.abort`. `StreamTrainingEvents` loops `log.read_after(after_seq, 1.0)` while `context.is_active()`, yields converted events, returns when closed and drained. `DeleteModel` also calls `inference.forget`. `EvaluateModel` defaults dataset to the model's dataset and split to `val`; rejects split outside `SPLITS`; rejects class-name mismatch with `FailedPrecondition`. `Predict` default `score_threshold` 0.3; empty image → InvalidArgument.
- `server.py` `build_server(servicer, port, max_workers=16) -> grpc.Server` with max send/receive 32 MB and health service (`SERVING`). `serve(config)` runs `registry.cleanup()`, builds engine objects (imports `yolox_trainer.engine` lazily — engine package may not exist yet; until Task 10–12 land, wire a placeholder that raises `FailedPrecondition("engine unavailable")`), handles SIGTERM/SIGINT → `supervisor.shutdown(30)` then `server.stop(5)`.
- Compose trainer healthcheck using `grpc_health_probe` is not available; use `python -c` with `grpc_health.v1` client.
Tests (semantic, real in-process `grpc.server` on a free port with fake runner + fake inference + tmp registry/catalog): ListDatasets; StartTraining busy → FAILED_PRECONDITION; GetModel unknown → NOT_FOUND; StreamTrainingEvents receives scripted events in order and ends after finished, and `after_seq` replays from the middle; Predict with a 5 MB payload succeeds.

## Task 10: engine — FinetuneExp + detection mapping

Files: `engine/__init__.py`, `engine/exp.py`, `engine/detections.py`, tests under `tests/engine/`.
Requirements:
- `FinetuneExp(yolox.exp.default.yolox_tiny.Exp)` constructed via `build_exp(data_dir: str, class_names: list[str], hp: Hyperparameters, output_dir: str, pretrained_path: str | None, num_workers: int = 2) -> FinetuneExp`. Settings per spec (input/test size, `random_size=(s//32-3, s//32+3)`, `max_epoch`, `warmup_epochs`, `no_aug_epochs`, `basic_lr_per_img = learning_rate / batch_size`, `mosaic_prob`, `enable_mixup`, mixup_prob 1.0 only when enabled, `eval_interval=1`, `print_interval=1`, `save_history_ckpt=False`, `data_num_workers`, `seed`, `output_dir`, `exp_name` irrelevant (trainer uses args.experiment_name)).
- Override `get_dataset(cache=False, cache_type="ram")` and `get_eval_dataset(**kwargs)` so images come from `data_dir/train` and `data_dir/<split>` with `json_file` `train.json` / `<split>.json` (`self.eval_split`, default `val`, settable). Keep upstream signatures (`get_eval_loader`/`get_evaluator` call `get_eval_dataset` with kwargs).
- `get_model()` loads `pretrained_path` once via `load_ckpt` when set (`weights_only=False` only for this official file).
- Override `eval(model, evaluator, is_distributed, half=False, return_outputs=False)`: call super, store `self.last_eval = (ap50_95, ap50)`, return super's value unchanged.
- `detections.py` pure: `to_detections(output: np.ndarray | None, ratio: float, width: int, height: int, class_names: list[str], min_score: float) -> list[Detection]` — rows `x1,y1,x2,y2,obj,cls_conf,cls_id`; boxes / ratio, clipped to image, score = obj*cls_conf, filter `score >= min_score`, sorted by score desc.
Tests: mechanical (no torch needed for `to_detections`, numpy only — put under `tests/mechanical`): coordinate mapping, clipping, filtering. Engine (marker `engine`): synthetic 4-image COCO fixture (random images written with cv2 into tmp) loads through `exp.get_dataset()` and yields boxes within the image after `MosaicDetection`; `get_eval_dataset` uses the requested split.

## Task 11: engine — training worker, process runner, wiring

Files: `engine/observable_trainer.py`, `engine/worker.py`, `engine/process_runner.py`, wire `ProcessTrainingRunner` into `server.serve`, tests `tests/engine/test_training_gpu.py`.
Requirements:
- `class TrainingStopped(Exception)`. `ObservableTrainer(Trainer)` per spec: `before_train` → emit `WorkerStarted(max_epoch, iters_per_epoch=self.max_iter)`; `after_iter` → if `should_stop()` raise `TrainingStopped`; read losses (`float(v.latest)`) and lr before `super().after_iter()`; emit `IterProgress` (epoch 1-based, iter 1-based, global_iter = `self.progress_in_iter + 1`, total_iters = `max_iter*max_epoch`, input_size = `self.input_size[0]`, ETA from wall-clock average); `after_epoch` → super, then emit `EpochEval(epoch, ap50, ap50_95, best_ap50_95=self.best_ap)` if `exp.last_eval` updated this epoch; track best epoch/ap50; check stop.
- `worker.run_job(spec: JobSpec, out_q, stop_event)`: `configure_module()`, seed, `cudnn.benchmark = True`; `exp = build_exp(...)` with `output_dir = dirname(spec.work_dir)` and `args.experiment_name = spec.job_id` (so YOLOX writes to `spec.work_dir`); args Namespace fields: `experiment_name, batch_size, fp16, resume=False, ckpt=None, start_epoch=None, cache=None, occupy=False, logger="tensorboard", exp_file=None, devices=1, dist_backend="nccl", dist_url=None, num_machines=1, machine_rank=0`; run `ObservableTrainer(...).train()`. On finish/stop: pick `best_ckpt.pth` else `latest_ckpt.pth` in work_dir; if present, save `{"model": ckpt["model"]}` to `work_dir/weights.pth` and pass that path; send `WorkerDone`. `TrainingStopped` → cancelled; other exceptions → failed with `traceback.format_exc()` (still export weights if a checkpoint exists? No: failed carries no weights).
- `ProcessTrainingRunner` per spec (spawn context, pump thread, `request_stop` with join(30)/terminate/join(5)/kill watchdog on a thread, `on_exit(exitcode)` after the queue is drained).
- `server.serve` uses `ProcessTrainingRunner` when engine import succeeds.
Tests (markers `engine`, `gpu`, `needs_weights`; dataset = `datasets/balloon` mounted at `TRAINER_DATASETS_DIR`, skip if absent): (1) `ProcessTrainingRunner` + 2-epoch run (input 320, batch 4, no_aug 1, warmup 0) delivers WorkerStarted → ≥1 IterProgress → 2 EpochEval → WorkerDone(succeeded) with a weights file loadable via `torch.load(..., weights_only=True)`; (2) setting stop after the first IterProgress yields WorkerDone(cancelled) within 30 s of the request. Run in container: `docker compose run --rm trainer pytest -m gpu -q`.

## Task 12: engine — evaluator, predictor, model cache

Files: `engine/inference_service.py` (implements `InferenceService`), wire into `server.serve`, tests.
Requirements:
- `ModelCache(registry, device, capacity=2)`: LRU of `(exp, model)`; builds `build_exp(data_dir="", class_names=record.class_names, hp=record.hyperparameters, output_dir=tmp, pretrained_path=None, num_workers=0)`, `model = exp.get_model()`, loads `weights.pth` with `weights_only=True`, `.eval()`, moves to cuda when device=gpu. `forget(id)`. A `threading.Lock` serializes GPU use.
- `predict`: decode bytes with `cv2.imdecode` (BGR; failure → `InvalidArgument`); `yolox.tools.demo.Predictor(model, exp, class_names, device, fp16=False)` with `confthre = 0.01`; `outputs, info = predictor.inference(img)`; `to_detections(outputs[0] as numpy, info["ratio"], w, h, class_names, score_threshold)`.
- `evaluate`: exp with `data_dir = dataset root`, `eval_split = split`, `num_workers=0`; `evaluator = exp.get_evaluator(batch_size=8, is_distributed=False)`; `_, image_wise = evaluator.evaluate(model, False, half=False, return_outputs=True)`; convert `image_wise` (`{image_id: {"bboxes": [xyxy], "scores": [...], "categories": [cat_id]}}`) to COCO results (xywh) and call `core.coco_metrics.compute(gt_json, results)`.
- Model weights at inference must match training pipeline (same Exp class, same test_size).
Tests (`engine`, `gpu`, `needs_weights`, dataset present): train 2 epochs via the runner (reuse a session fixture from Task 11 if convenient), register, then `evaluate(model_id, "balloon", "val").stats.ap` equals the job's best `ap50_95` within 1e-3; `predict` on a val image returns detections within image bounds and sorted by score.

## Task 13: gateway — client, datasets/models/trainings REST

Files: `gateway/src/gateway/{main.py,settings.py,schemas.py,trainer_client.py,errors.py,routes/{__init__.py,datasets.py,models.py,trainings.py}}`, tests.
Requirements:
- `settings.py`: `TRAINER_ADDR` (default `trainer:50051`).
- `schemas.py`: pydantic models mirroring proto messages (snake_case).
- `trainer_client.py`: `class TrainerClient(Protocol)` with async methods for each RPC returning schema objects (stream method = async iterator of `(event_type, seq, payload_dict)`); `GrpcTrainerClient(addr)` using `grpc.aio` with 32 MB message limits; raises `TrainerError(code: grpc.StatusCode, message)`.
- `errors.py`: exception handler mapping per Global REST section.
- Routes for everything in the REST API except SSE events, evaluate and predict (Task 14). `main.py` `create_app(client_factory)` with `get_client` dependency (override in tests).
Tests: mechanical — error mapping table; route validation (422 on bad body). Semantic — `GrpcTrainerClient` against an in-process `grpc.aio` server implementing a fake `TrainerServiceServicer` (from generated base) converts ListDatasets/StartTraining/GetModel correctly and maps NOT_FOUND to `TrainerError`.

## Task 14: gateway — SSE events, evaluate, predict, Dockerfile

Files: `routes/trainings.py` (SSE), `routes/models.py` (evaluate/predict), `gateway/Dockerfile`, `scripts/export_openapi` entry (`gateway` console script `export-openapi` printing `app.openapi()` JSON), compose `gateway` service, tests.
Requirements:
- SSE via `sse-starlette` `EventSourceResponse`: `id`=seq, `event` = `started|iter|eval|finished`, `data` = JSON (payload + `seq`, `time_unix_ms`); `after_seq` from `Last-Event-ID` header, else `?after_seq`, else 0; ping every 15 s; ends after `finished`; cancels the gRPC stream when the client disconnects.
- Evaluate: body `{dataset_name?: str, split: Literal["val","train"] = "val"}`. Predict: multipart `file` (≤ 20 MB, else 413), form `score_threshold: float = 0.3` in [0,1].
- Dockerfile (context repo root): uv python3.12 image, generate stubs, run `uvicorn gateway.main:app --host 0.0.0.0 --port 8000`. Compose `gateway` depends on trainer (healthy), env `TRAINER_ADDR=trainer:50051`, no published port (dev override may publish 8000).
Tests: semantic — SSE response with a fake client: frames contain correct `id/event/data`, `Last-Event-ID: 5` is passed as `after_seq=5`, stream ends after finished. Mechanical — predict forwards bytes and threshold to client; 413 on oversize.

## Task 15: frontend scaffold + nginx + full compose

Files: `frontend/` (Vite React TS: `package.json`, `vite.config.ts`, `tsconfig*.json`, `index.html`, `src/main.tsx`, `src/App.tsx`, `src/api/{client.ts,schema.d.ts,sse.ts}`, `src/features/*/` placeholders), `frontend/Dockerfile`, `frontend/nginx.conf`, compose `frontend` service.
Requirements:
- Deps: react, react-dom, react-router-dom, @tanstack/react-query, recharts; dev: vite, @vitejs/plugin-react, typescript, vitest, @testing-library/react, @testing-library/jest-dom, jsdom, openapi-typescript. Scripts: `dev`, `build`, `test` (vitest run), `gen:api` (`openapi-typescript src/api/openapi.json -o src/api/schema.d.ts`, with `openapi.json` produced by `uv run --project gateway export-openapi`).
- `client.ts`: typed fetch helpers for all REST endpoints using `schema.d.ts` types; throws an `ApiError(status, detail)`.
- `sse.ts`: `subscribeTrainingEvents(jobId, afterSeq, onEvent, onError) -> () => void` using `EventSource` (browser sends `Last-Event-ID` on auto-reconnect), closes on `finished`.
- App shell with nav: Training / Models / Inference; routes `/training`, `/models`, `/models/:id`, `/inference`.
- Vite dev proxy `/api` → `http://localhost:8000`. nginx: serve SPA with `try_files $uri /index.html`; `location /api/ { proxy_pass http://gateway:8000; proxy_http_version 1.1; proxy_buffering off; proxy_cache off; proxy_read_timeout 3600s; client_max_body_size 20m; }`. Compose `frontend` publishes `8080:80`.
- Verification: `npm run build` succeeds; `docker compose up -d --build` then `curl -s localhost:8080/api/datasets` lists balloon.

## Task 16: Training page

Files: `frontend/src/features/training/{TrainingPage.tsx,TrainingForm.tsx,trainingReducer.ts,charts.tsx}`, tests.
Requirements:
- Form: dataset select (from `/datasets`), display name, hyperparameter inputs prefilled from `/hyperparameters/defaults`; submit → POST; 409 shows "a training job is already running".
- On load, `GET /trainings/current`; if a job exists, subscribe from seq 0 (replay rebuilds charts after reload).
- `trainingReducer(state, event)`: ignores events with seq ≤ lastSeq; `started` → meta; `iter` → loss series (total_loss, iou_loss, conf_loss, cls_loss, l1_loss keyed by global_iter) + lr series + progress/ETA; `eval` → AP50/AP50-95 series by epoch; `finished` → final state, model_id link to `/models/:id`, error text.
- Charts (Recharts): loss vs iter, lr vs iter, AP vs epoch. Progress bar (global_iter/total_iters), epoch counter, ETA, state badge, Stop button (POST stop) while active.
Tests (vitest): reducer — builds series from a scripted event sequence, ignores duplicate/older seq, sets final state on finished. Form — submit sends `{dataset_name, hyperparameters}` with edited values (mock fetch).

## Task 17: Models + Evaluation pages

Files: `frontend/src/features/models/{ModelsPage.tsx,ModelDetailPage.tsx}`, `frontend/src/features/evaluation/EvaluationPanel.tsx`, tests only if logic beyond rendering appears.
Requirements:
- Models list table: display name, id, dataset, status, epochs, best AP50, best AP50-95, created; links to detail; delete with an inline confirm step (no `window.confirm`), invalidates the query.
- Detail: metadata, hyperparameters table, AP-vs-epoch chart from `eval_history`, link "Run inference" (`/inference?model=<id>`), Evaluation panel: split select (val/train), run → table of 12 COCO stats (show "—" for -1) and per-class AP table; note that val has few images.

## Task 18: Inference page

Files: `frontend/src/features/inference/{InferencePage.tsx,boxes.ts}`, tests.
Requirements:
- Model select (preselect from `?model=`), image file input, Run → POST predict with `score_threshold=0.05` (fetch once), then client-side threshold slider (default 0.3) filters boxes without refetching.
- Draw image on `<canvas>` scaled to fit max width; boxes + `class score` labels.
- `boxes.ts` pure `toCanvasBoxes(detections, imageW, imageH, canvasW, canvasH, threshold)` → scaled rects filtered by threshold.
Tests (vitest): `toCanvasBoxes` scaling and threshold filtering.

## Task 19: E2E smoke + README

Files: `scripts/e2e_smoke.py` (+ `httpx` in scripts dev deps, console script `e2e-smoke`), `README.md`, `CLAUDE.md` (add commands section only).
Requirements:
- `e2e-smoke --base http://localhost:8080 --epochs 5`: lists datasets → starts training on balloon → consumes SSE until finished (asserts started, ≥1 iter, `epochs` evals, finished SUCCEEDED) → GET model → evaluate val (ap50 ≥ 0 and stats present) → predict on a val image → starts a second job and stops it after the first iter → asserts CANCELLED within 60 s → deletes created models. Exit non-zero on failure.
- README: overview/architecture diagram (text), prerequisites (Docker Desktop + NVIDIA, uv, Node), setup (`prepare-balloon`, `download-weights`, `scripts/gen_proto.sh`), `docker compose up -d --build`, open `http://localhost:8080`, how to add your own COCO dataset (Appendix C layout), test commands per component.
- Run the full stack and `e2e-smoke`; then a Playwright pass (MCP browser): open the UI, start a short training, confirm charts update, open the model, evaluate, run inference; save screenshots to the SDD workspace and report.
