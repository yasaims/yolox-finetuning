# YOLOX-Tiny Finetuning Platform — Design & Implementation Plan

## Context

The repo `yolox-finetuning` is empty (README, CLAUDE.md: TDD, commit on `develop` after each task). Goal: a local platform to finetune the official **YOLOX-Tiny** on a **small dataset (~50 images)**, with a **containerized gRPC training engine on NVIDIA GPU** and a **browser UI** to watch training live, manage finetuned models, check accuracy and run inference.

Decisions (confirmed with user):
- Browser ↔ engine: **FastAPI BFF** (REST + SSE) → gRPC trainer.
- Base model: **YOLOX-Tiny only** (`yolox_tiny.pth`, release 0.1.1rc0; URL verified).
- Dataset: **Balloon** (matterport Mask_RCNN v2.1 release, VIA polygons, 1 class; URL verified) → seeded subset **train 40 / val 10**, converted to COCO bbox.
- No dataset upload: UI selects from server-side `datasets/`.
- One training job at a time (single GPU).

Environment: RTX 3080 12GB, Docker Desktop (WSL2, `nvidia` runtime present), uv, Node 22. Upstream YOLOX source is checked out at `C:\Users\didib\git\yolox` (used as reference; container installs from pinned git sha).

## Architecture

```
browser ──HTTP──> frontend (nginx :8080, SPA + /api proxy, buffering off)
                     └──> gateway (FastAPI :8000, REST + SSE)
                              └──gRPC──> trainer (Python 3.10, torch cu128, YOLOX, GPU)
                                            ├─ spawn child process per training job
                                            └─ volumes: datasets(ro) models pretrained(ro)
```

### Trainer (gRPC engine)
- **Training runs in a `spawn` subprocess**, not a thread: YOLOX `Trainer.__init__` hijacks loguru/stdout process-wide, cancellation must be forceful, crashes/OOM stay isolated, GPU memory is freed on exit.
- `engine/observable_trainer.py` — `ObservableTrainer(yolox.core.Trainer)`:
  - `before_train`: emit `Started(max_epoch, iters_per_epoch)`.
  - `after_iter`: check stop event → raise `TrainingStopped`; read losses from `meter.get_filtered_meter("loss")` (as `float`) and lr **before** `super().after_iter()` (it clears meters); emit `Iter`.
  - `after_epoch`: `super()` evaluates/saves; emit `Eval(ap50, ap50_95)` captured by `FinetuneExp.eval()` override (trainer discards the return value of `exp.eval`, `yolox/core/trainer.py:355`).
- `engine/worker.py` `run_job(spec, queue, stop_event)`: builds Exp + args Namespace, calls `trainer.train()` directly (not `tools/train.py`, whose `@logger.catch` swallows errors). Exports `weights.pth={"model": state_dict}` from `best_ckpt.pth`, **falling back to `latest_ckpt.pth`** (best is never written if AP stays 0). Sends `WorkerDone(outcome, ...)`.
- `engine/process_runner.py` `ProcessTrainingRunner`: queue pump thread; stop = set event → join(30) → terminate → kill.
- `core/supervisor.py` `TrainingSupervisor` (torch-free): state machine STARTING→RUNNING→STOPPING→SUCCEEDED/FAILED/CANCELLED; converts worker messages to public events; sole writer to registry; depends on a `TrainingRunner` Protocol (fake in tests).
- `core/event_log.py` `EventLog`: append-only list + `threading.Condition`; `read_after(seq, timeout)`. Gives replay for free (SSE `Last-Event-ID` → `after_seq`); kept for the last job and persisted as `events.jsonl` in the model dir.
- `engine/exp.py` `FinetuneExp(yolox.exp.default.yolox_tiny.Exp)`: `num_classes` from dataset, `data_dir=datasets/<name>`, override `get_dataset`/`get_eval_dataset` to use `train/` & `<split>/` image dirs; defaults: input 416, `random_size=(s//32-3, s//32+3)`, epochs 50, batch 8, lr 1.25e-3 (effective), warmup 3, no_aug 10, mosaic 1.0, mixup off, eval_interval 1, print_interval 1, fp16 on. Pretrained loaded in `get_model()` via `load_ckpt` (shape-mismatched `head.cls_preds.*` skipped automatically; explicit `weights_only=False` only for the official file).
- `engine/predictor.py`: reuse `yolox.tools.demo.Predictor` (ValTransform + postprocess); own pure fn `to_detections(...)` (undo ratio, score=obj·cls, clip). `ModelCache` LRU(2) + GPU lock.
- `engine/evaluator.py`: `exp.get_evaluator(...)`, `evaluate(return_outputs=True)`, `data_num_workers=0` in server process. `core/coco_metrics.py` (pycocotools only) computes 12 COCO stats + per-class AP from predictions.
- `core/registry.py`: `models/<id>/{model.json, weights.pth, events.jsonl, train_log.txt}`; id `YYYYMMDD-HHMMSS-<dataset>-<6hex>` (= job id), validated by `^[a-z0-9][a-z0-9-]{0,63}$`; atomic register via staging dir + `os.replace`; startup cleans `.work/` and `.staging-*`. Stopped jobs with ≥1 checkpoint are registered as `status=stopped`.

### Proto `proto/yoloxft/v1/trainer.proto`
`TrainerService`: `ListDatasets`, `GetDefaultHyperparameters`, `StartTraining` (FAILED_PRECONDITION if busy), `GetCurrentJob`, `StreamTrainingEvents(job_id, after_seq) → stream TrainingEvent{seq, oneof started|iter|eval|finished}`, `StopTraining`, `ListModels`, `GetModel`, `DeleteModel`, `EvaluateModel(model_id, dataset, split) → EvaluationResult{CocoStats, per_class}`, `Predict(model_id, image bytes, score_threshold) → detections`. `Hyperparameters` uses `optional` fields (unset = server default). Max message size 32 MB both ends. `grpcio-health-checking` for compose healthcheck.

### Gateway (FastAPI, Python 3.12)
- `TrainerClient` Protocol + `GrpcTrainerClient` (grpc.aio). Routes: `/api/datasets`, `/api/trainings` (POST start, GET current, POST stop, GET `/{id}/events` SSE via `sse-starlette` with `id: seq`, `event: <type>`), `/api/models` (list/get/delete, POST evaluate, POST predict multipart).
- gRPC→HTTP: NOT_FOUND→404, FAILED_PRECONDITION→409, INVALID_ARGUMENT→422, UNAVAILABLE→503.

### Frontend (React + Vite + TS)
- react-router, TanStack Query, types generated from gateway OpenAPI (`openapi-typescript`), Recharts.
- Pages: **Training** (form w/ server defaults, live loss/lr/AP charts, progress+ETA, Stop; reconnect replays via `Last-Event-ID`; close EventSource on `finished`), **Models** (table, detail with eval history curve, delete), **Evaluation** (COCO stats + per-class AP), **Inference** (upload image, canvas boxes, score slider).
- Built SPA served by nginx; `/api/` proxied with `proxy_buffering off`, long read timeout, `client_max_body_size 20m`.

### Repo layout
```
proto/yoloxft/v1/trainer.proto
trainer/   pyproject.toml uv.lock Dockerfile src/yolox_trainer/{core,engine,grpc_api}/ tests/{mechanical,semantic}/
gateway/   pyproject.toml uv.lock Dockerfile src/gateway/{routes,...} tests/
frontend/  package.json vite.config.ts Dockerfile nginx.conf src/{api,features/{training,models,evaluation,inference}}
scripts/   pyproject.toml src/dataset_tools/{via_to_coco.py,balloon.py,download_weights.py} gen_proto.sh e2e_smoke.py tests/
datasets/ models/ pretrained/   (contents gitignored)
docker-compose.yml .gitattributes .gitignore
```
- Three separate uv projects (trainer 3.10, gateway/scripts 3.12). Trainer core deps (grpc, protobuf, pycocotools, numpy<2) install on host for fast TDD; torch/YOLOX in an `engine` extra used in the container.
- Stubs via `grpc_tools.protoc` (`scripts/gen_proto.sh`, same command in Dockerfiles; generated files gitignored). Same pinned grpcio/protobuf versions in trainer and gateway.

### Trainer container
- Builder `ghcr.io/astral-sh/uv:python3.10-bookworm` → runtime `python:3.10-slim-bookworm`; torch 2.8 cu128 from explicit pytorch index (no CUDA base image needed).
- YOLOX from pinned git sha with `no-build-isolation-package = ["yolox"]` and `dependency-metadata` override to drop onnx/onnxsim/opencv-python; add `opencv-python-headless`, loguru, tabulate, thop, tensorboard, etc. explicitly. Two-stage `uv sync` (torch first, then yolox).
- Fallback if the uv git build fails: `uv pip install --no-deps --no-build-isolation git+...@sha` + `UV_NO_SYNC=1`.
- Compose: GPU `deploy.resources.reservations.devices`, **`shm_size: 2gb`**, volumes `datasets:ro`, `models`, `pretrained:ro`. Only frontend publishes a port (8080).

## Tests (TDD; only behavior-guaranteeing tests)

| Component | Mechanical | Semantic |
|---|---|---|
| scripts (host) | polygon→bbox; VIA regions dict/list; seeded disjoint 40/10 split | output loads in pycocotools; GT-vs-GT COCOeval AP=1.0 |
| trainer core (host) | hyperparam validation; id validation (traversal); proto↔dataclass | EventLog replay/wake/close; Supervisor w/ fake runner (busy, stop→CANCELLED, done→registered, crash→FAILED); registry atomicity; coco_metrics perfect/empty/per-class; servicer over in-process grpc server |
| trainer engine (container) | pretrained load skips only `cls_preds`; `to_detections` coordinate mapping | **GPU**: 2-epoch run emits started→iter→eval→done & weights load; stop ends < ~10s CANCELLED; evaluate reproduces training best AP |
| gateway (host) | gRPC→HTTP mapping; multipart predict | SSE framing & `Last-Event-ID`; GrpcTrainerClient vs in-process fake aio server |
| frontend (Vitest) | `toCanvasBoxes` scaling/threshold | `trainingReducer` series building & duplicate-seq dedupe |

Not covered by automated tests: actual accuracy level reached (checked manually in E2E), UI visuals (Playwright smoke only).

## Task order (each = TDD + commit on `develop`)

0. Scaffold: `develop` branch, gitignore/gitattributes, data dirs, save this design to `docs/superpowers/specs/2026-10-05-yolox-finetune-platform-design.md`.
1. scripts: VIA→COCO converter. 2. scripts: `prepare-balloon` + `download-weights` CLIs; run them.
3. **Trainer image de-risk first**: pyproject/Dockerfile/compose trainer; smoke: CUDA available, `import yolox`, pretrained-load test.
4. Proto + stub generation. 5. core: hyperparams, dataset catalog. 6. core: events/EventLog. 7. core: registry. 8. core: supervisor. 9. core: coco_metrics.
10. gRPC servicer + server bootstrap (health, startup cleanup, SIGTERM) with fakes.
11. engine: FinetuneExp. 12. engine: ObservableTrainer/worker/ProcessTrainingRunner wired to supervisor (GPU tests). 13. engine: predictor/evaluator/ModelCache wired.
14–16. gateway: datasets/models routes → trainings + SSE → evaluate/predict, client contract test, Dockerfile.
17. frontend scaffold + nginx + full compose. 18. Training page. 19. Models page. 20. Evaluation page. 21. Inference page.
22. `e2e_smoke.py`, README usage, Playwright smoke.

Execution: superpowers:subagent-driven-development (one subagent per task, review between tasks).

## Verification (end-to-end)
1. `uv run --project scripts prepare-balloon --out datasets/balloon` → 40/10 images + `annotations/{train,val}.json`; `download-weights` → `pretrained/yolox_tiny.pth`.
2. `docker compose up -d --build`; `docker compose exec trainer python -c "import torch,yolox;print(torch.cuda.is_available())"` → True; run trainer GPU tests in container; host `uv run pytest` in trainer/gateway/scripts; `npm test` in frontend.
3. Browser `http://localhost:8080`: start 50-epoch balloon run → loss/AP charts update live; reload mid-run → charts rebuilt by replay; expect AP50 ≳ 0.5.
4. Start another run → Stop → CANCELLED within ~10s, GPU memory freed (`nvidia-smi`).
5. Models: evaluate on balloon/val → AP50-95 matches model's best; Inference: upload val image → boxes align, slider filters; delete model → gone from list and `models/`.
6. `scripts/e2e_smoke.py` automates 3–5 with 5 epochs.

## Risks
- uv build of unmaintained YOLOX (no build isolation, C++ ext) → de-risked in task 3, fallback defined.
- Docker `/dev/shm` 64MB → `shm_size: 2gb`.
- torch ≥2.6 `weights_only` default → load official weights ourselves; own weights are tensor-only.
- Never fork from gRPC server process (spawn for training, `num_workers=0` for eval).
- 10 val images → noisy AP; UI allows evaluating on train split too.
- Windows: CRLF in `.sh` (`.gitattributes eol=lf`).

## Appendix A — YOLOX facts the design relies on (verified against upstream source)

- `Trainer.__init__` calls `setup_logger` → global `logger.remove()` and redirects `sys.stdout/stderr` (`yolox/utils/logger.py`). Device is hard-coded `cuda:{local_rank}`.
- `tools/train.py:main` is wrapped in `@logger.catch` (swallows exceptions) → call `Trainer(exp, args).train()` directly.
- `evaluate_and_save_model`: `(ap50_95, ap50, summary), predictions = self.exp.eval(...)`; `update_best_ckpt = ap50_95 > self.best_ap` (best_ap starts at 0 → `best_ckpt.pth` may never exist; fall back to `latest_ckpt.pth`).
- `after_iter` logs then calls `self.meter.clear_meters()` every `print_interval`; meter values may be tensors → `float()`.
- `COCODataset` reads images from `data_dir/<name>/<file_name>`, annotations from `data_dir/annotations/<json>`; upstream `get_eval_dataset` hard-codes `name="val2017"`, `get_dataset` uses `train2017` → override both.
- `yolox.utils.load_ckpt(model, state_dict)` skips shape-mismatched keys (`head.cls_preds.*` for 80→N classes).
- `COCOEvaluator.evaluate(model, return_outputs=True)` returns `((ap50_95, ap50, summary), image_wise_outputs)`; per-class AP only as text → compute structured metrics ourselves with pycocotools (`precision[:, :, k, 0, -1]` for AP, `precision[0, :, k, 0, -1]` for AP50, `-1` entries excluded).
- `yolox.tools.demo.Predictor(model, exp, cls_names, device, fp16)` provides `inference(img_ndarray)` = ValTransform + `postprocess`; returns `(outputs, img_info)` with `img_info["ratio"]`. Output rows: `x1,y1,x2,y2,obj,cls_conf,cls_id` in letterboxed coordinates.
- pycocotools requires `iscrowd` and `area` in annotations.
- `setup.py` builds a C++ ext and imports torch at build time; `requirements.txt` pulls onnx, onnx-simplifier==0.4.10, opencv-python (GUI) → drop via uv dependency-metadata override.

## Appendix B — Proto sketch (`proto/yoloxft/v1/trainer.proto`)

```proto
syntax = "proto3";
package yoloxft.v1;

service TrainerService {
  rpc ListDatasets(ListDatasetsRequest) returns (ListDatasetsResponse);
  rpc GetDefaultHyperparameters(GetDefaultHyperparametersRequest) returns (Hyperparameters);
  rpc StartTraining(StartTrainingRequest) returns (StartTrainingResponse);
  rpc GetCurrentJob(GetCurrentJobRequest) returns (GetCurrentJobResponse);
  rpc StreamTrainingEvents(StreamTrainingEventsRequest) returns (stream TrainingEvent);
  rpc StopTraining(StopTrainingRequest) returns (StopTrainingResponse);
  rpc ListModels(ListModelsRequest) returns (ListModelsResponse);
  rpc GetModel(GetModelRequest) returns (Model);
  rpc DeleteModel(DeleteModelRequest) returns (DeleteModelResponse);
  rpc EvaluateModel(EvaluateModelRequest) returns (EvaluationResult);
  rpc Predict(PredictRequest) returns (PredictResponse);
}

message ListDatasetsRequest {}
message Dataset { string name = 1; repeated string class_names = 2; int32 train_images = 3; int32 val_images = 4; }
message ListDatasetsResponse { repeated Dataset datasets = 1; }

message GetDefaultHyperparametersRequest {}
message Hyperparameters {
  optional int32 max_epoch = 1; optional int32 batch_size = 2; optional double learning_rate = 3;
  optional int32 input_size = 4; optional int32 warmup_epochs = 5; optional int32 no_aug_epochs = 6;
  optional double mosaic_prob = 7; optional bool enable_mixup = 8; optional bool fp16 = 9; optional int32 seed = 10;
}

message StartTrainingRequest { string dataset_name = 1; Hyperparameters hyperparameters = 2; string display_name = 3; }
message StartTrainingResponse { string job_id = 1; Hyperparameters resolved = 2; }

enum JobState {
  JOB_STATE_UNSPECIFIED = 0; JOB_STATE_STARTING = 1; JOB_STATE_RUNNING = 2; JOB_STATE_STOPPING = 3;
  JOB_STATE_SUCCEEDED = 4; JOB_STATE_FAILED = 5; JOB_STATE_CANCELLED = 6;
}

message GetCurrentJobRequest {}
message JobSummary { string job_id = 1; JobState state = 2; string dataset_name = 3; Hyperparameters hyperparameters = 4; int64 last_seq = 5; int64 started_unix_ms = 6; }
message GetCurrentJobResponse { optional JobSummary job = 1; }

message StreamTrainingEventsRequest { string job_id = 1; int64 after_seq = 2; }
message TrainingEvent {
  string job_id = 1; int64 seq = 2; int64 time_unix_ms = 3;
  oneof payload { JobStarted started = 10; IterProgress iter = 11; EpochEval eval = 12; JobFinished finished = 13; }
}
message JobStarted { string dataset_name = 1; Hyperparameters hyperparameters = 2; int32 max_epoch = 3; int32 iters_per_epoch = 4; }
message IterProgress { int32 epoch = 1; int32 iter = 2; int64 global_iter = 3; int64 total_iters = 4; double lr = 5; map<string, double> losses = 6; int32 input_size = 7; double eta_seconds = 8; }
message EpochEval { int32 epoch = 1; double ap50 = 2; double ap50_95 = 3; double best_ap50_95 = 4; }
message JobFinished { JobState state = 1; string model_id = 2; string error = 3; }

message StopTrainingRequest { string job_id = 1; }
message StopTrainingResponse {}

message ListModelsRequest {}
message Model {
  string id = 1; string display_name = 2; string status = 3; string dataset_name = 4; repeated string class_names = 5;
  Hyperparameters hyperparameters = 6; int32 epochs_trained = 7; int32 best_epoch = 8; double best_ap50 = 9;
  double best_ap50_95 = 10; int64 created_unix_ms = 11; repeated EpochEval eval_history = 12;
}
message ListModelsResponse { repeated Model models = 1; }
message GetModelRequest { string model_id = 1; }
message DeleteModelRequest { string model_id = 1; }
message DeleteModelResponse {}

message EvaluateModelRequest { string model_id = 1; string dataset_name = 2; string split = 3; }
message CocoStats { double ap = 1; double ap50 = 2; double ap75 = 3; double ap_small = 4; double ap_medium = 5; double ap_large = 6; double ar1 = 7; double ar10 = 8; double ar100 = 9; double ar_small = 10; double ar_medium = 11; double ar_large = 12; }
message ClassAp { string class_name = 1; double ap = 2; double ap50 = 3; }
message EvaluationResult { CocoStats stats = 1; repeated ClassAp per_class = 2; int32 num_images = 3; }

message PredictRequest { string model_id = 1; bytes image = 2; optional float score_threshold = 3; }
message Detection { float x1 = 1; float y1 = 2; float x2 = 3; float y2 = 4; float score = 5; int32 class_id = 6; string class_name = 7; }
message PredictResponse { int32 image_width = 1; int32 image_height = 2; repeated Detection detections = 3; }
```

Status codes: NOT_FOUND (unknown model/dataset/job), INVALID_ARGUMENT (bad hyperparams/id/split/image), FAILED_PRECONDITION (job running, pretrained missing, dataset classes ≠ model classes). `CocoStats` fields are `-1` when undefined (COCO convention).

## Appendix C — Dataset layout

```
datasets/<name>/
  train/<images>          val/<images>
  annotations/train.json  annotations/val.json   # COCO; categories id from 1; iscrowd=0; area=w*h
```
A directory is a dataset iff `annotations/train.json` and `annotations/val.json` exist. Class names come from `categories` sorted by id. Dataset name pattern `^[a-z0-9][a-z0-9_-]{0,63}$`.

## Appendix D — Hyperparameter defaults and validation

| field | default | valid |
|---|---|---|
| max_epoch | 50 | 1..300 |
| batch_size | 8 | 1..64 |
| learning_rate | 0.00125 (effective, = basic_lr_per_img × batch) | (0, 0.1] |
| input_size | 416 | 224..1024, multiple of 32 |
| warmup_epochs | 3 | 0..max_epoch-1 |
| no_aug_epochs | 10 | 0..max_epoch-1 |
| mosaic_prob | 1.0 | [0,1] |
| enable_mixup | false | |
| fp16 | true | |
| seed | 42 | ≥0 |
