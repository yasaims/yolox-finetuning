from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .coco_metrics import EvaluationMetrics


@dataclass(frozen=True)
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    class_id: int
    class_name: str


@dataclass(frozen=True)
class PredictResult:
    image_width: int
    image_height: int
    detections: list[Detection]


class InferenceService(Protocol):
    def evaluate(self, model_id: str, dataset_name: str, split: str) -> EvaluationMetrics: ...

    def predict(self, model_id: str, image: bytes, score_threshold: float) -> PredictResult: ...

    def forget(self, model_id: str) -> None:
        """Drops any cached state for a deleted model."""
        ...
