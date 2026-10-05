from __future__ import annotations

import json
import tempfile
import threading
from collections import OrderedDict
from typing import Any, Literal

import cv2
import numpy as np
import torch
from yolox.tools.demo import Predictor

from ..core.coco_metrics import EvaluationMetrics, compute
from ..core.datasets import DatasetCatalog
from ..core.errors import FailedPrecondition, InvalidArgument
from ..core.inference import PredictResult
from ..core.registry import ModelRegistry
from .detections import to_detections
from .exp import FinetuneExp, build_exp

Device = Literal["gpu", "cpu"]

PREDICT_CONF_THRESHOLD = 0.01
EVAL_BATCH_SIZE = 8


class ModelCache:
    """LRU cache of eval-mode models built from registered weights, keyed by model id."""

    def __init__(self, registry: ModelRegistry, device: Device, capacity: int = 2) -> None:
        self._registry = registry
        self._device = device
        self._capacity = capacity
        self._entries: OrderedDict[str, tuple[FinetuneExp, torch.nn.Module]] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, model_id: str) -> tuple[FinetuneExp, torch.nn.Module]:
        """Returns `(exp, model)`; raises NotFound for unknown models."""
        with self._lock:
            entry = self._entries.get(model_id)
            if entry is None:
                entry = self._load(model_id)
                self._entries[model_id] = entry
                while len(self._entries) > self._capacity:
                    self._entries.popitem(last=False)
            else:
                self._entries.move_to_end(model_id)
            return entry

    def forget(self, model_id: str) -> None:
        with self._lock:
            self._entries.pop(model_id, None)

    def _load(self, model_id: str) -> tuple[FinetuneExp, torch.nn.Module]:
        record = self._registry.get(model_id)
        weights = self._registry.weights_path(model_id)
        exp = build_exp(
            data_dir="",
            class_names=record.class_names,
            hp=record.hyperparameters,
            output_dir=tempfile.gettempdir(),
            pretrained_path=None,
            num_workers=0,
        )
        model = exp.get_model()
        model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True)["model"])
        model.eval()
        if self._device == "gpu":
            model.cuda()
        return exp, model


class EngineInferenceService:
    """Evaluation and prediction with registered models; one inference runs at a time."""

    def __init__(
        self, registry: ModelRegistry, catalog: DatasetCatalog, device: Device, cache_capacity: int = 2
    ) -> None:
        self._registry = registry
        self._catalog = catalog
        self._device = device
        self._cache = ModelCache(registry, device, cache_capacity)
        self._gpu_lock = threading.Lock()

    def evaluate(self, model_id: str, dataset_name: str, split: str) -> EvaluationMetrics:
        if self._device != "gpu":
            # COCOEvaluator moves batches to CUDA unconditionally.
            raise FailedPrecondition("evaluation requires a GPU")
        record = self._registry.get(model_id)
        dataset = self._catalog.get(dataset_name)
        gt = json.loads(self._catalog.annotation_path(dataset_name, split).read_text(encoding="utf-8"))
        exp = build_exp(
            data_dir=str(dataset.root),
            class_names=record.class_names,
            hp=record.hyperparameters,
            output_dir=tempfile.gettempdir(),
            pretrained_path=None,
            num_workers=0,
        )
        exp.eval_split = split
        with self._gpu_lock:
            _, model = self._cache.get(model_id)
            evaluator = exp.get_evaluator(batch_size=EVAL_BATCH_SIZE, is_distributed=False)
            _, image_wise = evaluator.evaluate(model, False, half=False, return_outputs=True)
        return compute(gt, _to_coco_results(image_wise))

    def predict(self, model_id: str, image: bytes, score_threshold: float) -> PredictResult:
        img = _decode_bgr(image)
        height, width = img.shape[:2]
        with self._gpu_lock:
            exp, model = self._cache.get(model_id)
            predictor = Predictor(model, exp, exp.class_names, device=self._device, fp16=False)
            predictor.confthre = PREDICT_CONF_THRESHOLD
            outputs, info = predictor.inference(img)
        output = None if outputs[0] is None else outputs[0].cpu().numpy()
        detections = to_detections(output, info["ratio"], width, height, exp.class_names, score_threshold)
        return PredictResult(image_width=width, image_height=height, detections=detections)

    def forget(self, model_id: str) -> None:
        self._cache.forget(model_id)


def _decode_bgr(image: bytes) -> np.ndarray:
    try:
        img = cv2.imdecode(np.frombuffer(image, dtype=np.uint8), cv2.IMREAD_COLOR)
    except cv2.error:
        img = None
    if img is None:
        raise InvalidArgument("image could not be decoded")
    return img


def _to_coco_results(image_wise: dict[int, dict[str, list[Any]]]) -> list[dict[str, Any]]:
    """COCOEvaluator per-image outputs (xyxy boxes, dataset category ids) to COCO results (xywh)."""
    return [
        {
            "image_id": int(image_id),
            "category_id": int(category_id),
            "bbox": [x1, y1, x2 - x1, y2 - y1],
            "score": float(score),
        }
        for image_id, outputs in image_wise.items()
        for (x1, y1, x2, y2), score, category_id in zip(
            outputs["bboxes"], outputs["scores"], outputs["categories"]
        )
    ]
