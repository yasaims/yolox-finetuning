# YOLOX Finetune platform

Simple and handy local platform for finetuning YOLOX.

## Features

- Train run from browser
- Live AP and loss while training
- Review and eval models in registory
- Quick inference with your tuned models

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
