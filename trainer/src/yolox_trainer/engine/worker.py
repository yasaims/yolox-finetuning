"""Entry point of the training child process.

Heavy imports happen inside `run_job` so the parent can reference it without loading torch.
"""
from __future__ import annotations

import argparse
import multiprocessing.queues
import multiprocessing.synchronize
import random
import traceback
from pathlib import Path
from typing import Literal

from ..core.events import WorkerDone
from ..core.supervisor import JobSpec

WEIGHTS_FILE = "weights.pth"
CHECKPOINT_PREFERENCE = ("best_ckpt.pth", "latest_ckpt.pth")


def run_job(
    spec: JobSpec,
    out_q: multiprocessing.queues.Queue[object],
    stop_event: multiprocessing.synchronize.Event,
) -> None:
    """Trains `spec` and sends WorkerStarted, IterProgress, EpochEval and finally WorkerDone to `out_q`."""
    trainer = None
    outcome: Literal["succeeded", "cancelled", "failed"]
    weights_path: str | None = None
    error = ""
    try:
        import numpy as np
        import torch
        from torch.backends import cudnn
        from yolox.utils import configure_module

        from .exp import build_exp
        from .observable_trainer import ObservableTrainer, TrainingStopped

        configure_module()
        hp = spec.hyperparameters
        random.seed(hp.seed)
        np.random.seed(hp.seed)
        torch.manual_seed(hp.seed)
        cudnn.benchmark = True

        work_dir = Path(spec.work_dir)
        # YOLOX writes checkpoints and train_log.txt to output_dir/experiment_name == work_dir.
        exp = build_exp(spec.data_dir, spec.class_names, hp, str(work_dir.parent), spec.pretrained_path)
        args = _trainer_args(experiment_name=work_dir.name, batch_size=hp.batch_size, fp16=hp.fp16)
        trainer = ObservableTrainer(exp, args, out_q.put, stop_event.is_set)
        try:
            trainer.train()
            outcome = "succeeded"
        except TrainingStopped:
            outcome = "cancelled"
        weights_path = _export_weights(work_dir)
    except Exception:
        outcome, weights_path, error = "failed", None, traceback.format_exc()

    out_q.put(
        WorkerDone(
            outcome=outcome,
            weights_path=weights_path,
            best_epoch=trainer.best_epoch if trainer else 0,
            best_ap50=trainer.best_ap50 if trainer else 0.0,
            best_ap50_95=trainer.best_ap50_95 if trainer else 0.0,
            epochs_completed=trainer.epochs_completed if trainer else 0,
            error=error,
        )
    )


def _trainer_args(experiment_name: str, batch_size: int, fp16: bool) -> argparse.Namespace:
    """The `tools/train.py` arguments that `yolox.core.Trainer` reads, for one local GPU."""
    return argparse.Namespace(
        experiment_name=experiment_name,
        batch_size=batch_size,
        fp16=fp16,
        resume=False,
        ckpt=None,
        start_epoch=None,
        cache=None,
        occupy=False,
        logger="tensorboard",
        exp_file=None,
        devices=1,
        dist_backend="nccl",
        dist_url=None,
        num_machines=1,
        machine_rank=0,
    )


def _export_weights(work_dir: Path) -> str | None:
    """Saves the model state of the best (else latest) checkpoint as a tensor-only file."""
    import torch

    for name in CHECKPOINT_PREFERENCE:
        ckpt_path = work_dir / name
        if ckpt_path.is_file():
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            weights = work_dir / WEIGHTS_FILE
            torch.save({"model": ckpt["model"]}, weights)
            return str(weights)
    return None
