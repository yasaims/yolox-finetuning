from __future__ import annotations

import argparse
import time
from collections.abc import Callable

from yolox.core import Trainer

from ..core.events import EpochEval, IterProgress, WorkerStarted
from .exp import FinetuneExp


class TrainingStopped(Exception):
    """Raised inside the training loop when a stop was requested."""


class ObservableTrainer(Trainer):
    """YOLOX trainer that reports progress through `emit` and aborts when `should_stop()` is true."""

    exp: FinetuneExp

    def __init__(
        self,
        exp: FinetuneExp,
        args: argparse.Namespace,
        emit: Callable[[object], None],
        should_stop: Callable[[], bool],
    ) -> None:
        super().__init__(exp, args)
        self._emit = emit
        self._should_stop = should_stop
        self._train_started_at = 0.0
        self.epochs_completed = 0
        self.best_epoch = 0
        self.best_ap50 = 0.0

    @property
    def best_ap50_95(self) -> float:
        return float(self.best_ap)

    def before_train(self) -> None:
        super().before_train()
        self._emit(WorkerStarted(max_epoch=self.max_epoch, iters_per_epoch=self.max_iter))
        self._train_started_at = time.monotonic()

    def after_iter(self) -> None:
        if self._should_stop():
            raise TrainingStopped()
        # super().after_iter() clears the meters and may switch to a new random input size.
        losses = {name: float(meter.latest) for name, meter in self.meter.get_filtered_meter("loss").items()}
        lr = float(self.meter["lr"].latest)
        input_size = int(self.input_size[0])
        super().after_iter()

        done_iters = self.progress_in_iter + 1
        total_iters = self.max_iter * self.max_epoch
        elapsed = time.monotonic() - self._train_started_at
        self._emit(
            IterProgress(
                epoch=self.epoch + 1,
                iter=self.iter + 1,
                global_iter=done_iters,
                total_iters=total_iters,
                lr=lr,
                losses=losses,
                input_size=input_size,
                eta_seconds=elapsed / done_iters * (total_iters - done_iters),
            )
        )

    def after_epoch(self) -> None:
        self.exp.last_eval = None
        previous_best = self.best_ap
        super().after_epoch()
        self.epochs_completed = self.epoch + 1

        if self.exp.last_eval is not None:
            ap50_95, ap50 = self.exp.last_eval
            # Same condition YOLOX uses to overwrite best_ckpt.pth.
            if ap50_95 > previous_best:
                self.best_epoch = self.epoch + 1
                self.best_ap50 = ap50
            self._emit(
                EpochEval(epoch=self.epoch + 1, ap50=ap50, ap50_95=ap50_95, best_ap50_95=self.best_ap50_95)
            )

        if self.epochs_completed < self.max_epoch and self._should_stop():
            raise TrainingStopped()
