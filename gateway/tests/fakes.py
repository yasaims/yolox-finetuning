from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any

import grpc

from gateway.schemas import (
    ClassAp,
    CocoStats,
    Dataset,
    EvaluationResult,
    Hyperparameters,
    JobSummary,
    Model,
    PredictResponse,
    StartTrainingResult,
)
from gateway.trainer_client import TrainerError

HP = Hyperparameters(
    max_epoch=50,
    batch_size=8,
    learning_rate=0.00125,
    input_size=416,
    warmup_epochs=3,
    no_aug_epochs=10,
    mosaic_prob=1.0,
    enable_mixup=False,
    fp16=True,
    seed=0,
)

MODEL = Model(
    id="m1",
    display_name="m1",
    status="completed",
    dataset_name="balloon",
    class_names=["balloon"],
    hyperparameters=HP,
    epochs_trained=1,
    best_epoch=1,
    best_ap50=0.6,
    best_ap50_95=0.4,
    created_unix_ms=1,
    eval_history=[],
)

STATS = CocoStats(
    ap=0.4, ap50=0.6, ap75=0.4, ap_small=0, ap_medium=0, ap_large=0,
    ar1=0, ar10=0, ar100=0, ar_small=0, ar_medium=0, ar_large=0,
)  # fmt: skip


class FakeTrainerClient:
    """Records calls; `error` makes every call raise a TrainerError with that code."""

    def __init__(self, error: grpc.StatusCode | None = None) -> None:
        self.error = error
        self.calls: list[tuple[str, tuple[Any, ...]]] = []
        self.events: list[tuple[str, int, dict[str, Any]]] = []
        self.current: JobSummary | None = None

    def _record(self, name: str, *args: Any) -> None:
        self.calls.append((name, args))
        if self.error is not None:
            raise TrainerError(self.error, "boom")

    async def list_datasets(self) -> list[Dataset]:
        self._record("list_datasets")
        return [Dataset(name="balloon", class_names=["balloon"], train_images=40, val_images=10)]

    async def get_default_hyperparameters(self) -> Hyperparameters:
        self._record("get_default_hyperparameters")
        return HP

    async def start_training(
        self, dataset_name: str, hyperparameters: dict[str, Any], display_name: str
    ) -> StartTrainingResult:
        self._record("start_training", dataset_name, hyperparameters, display_name)
        return StartTrainingResult(job_id="j1", hyperparameters=HP)

    async def get_current_job(self) -> JobSummary | None:
        self._record("get_current_job")
        return self.current

    async def stream_events(self, job_id: str, after_seq: int) -> AsyncGenerator[tuple[str, int, dict[str, Any]], None]:
        self._record("stream_events", job_id, after_seq)
        for item in self.events:
            yield item

    async def stop_training(self, job_id: str) -> None:
        self._record("stop_training", job_id)

    async def list_models(self) -> list[Model]:
        self._record("list_models")
        return [MODEL]

    async def get_model(self, model_id: str) -> Model:
        self._record("get_model", model_id)
        return MODEL

    async def delete_model(self, model_id: str) -> None:
        self._record("delete_model", model_id)

    async def evaluate_model(self, model_id: str, dataset_name: str | None, split: str) -> EvaluationResult:
        self._record("evaluate_model", model_id, dataset_name, split)
        return EvaluationResult(
            stats=STATS, per_class=[ClassAp(class_name="balloon", ap=0.4, ap50=0.6)], num_images=10
        )

    async def predict(self, model_id: str, image: bytes, score_threshold: float) -> PredictResponse:
        self._record("predict", model_id, image, score_threshold)
        return PredictResponse(image_width=2, image_height=1, detections=[])

    async def close(self) -> None:
        pass
