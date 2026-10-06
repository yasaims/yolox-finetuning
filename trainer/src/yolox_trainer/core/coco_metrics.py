from __future__ import annotations

import contextlib
import io
from dataclasses import dataclass
from typing import Any

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


@dataclass(frozen=True)
class CocoStats:
    ap: float
    ap50: float
    ap75: float
    ap_small: float
    ap_medium: float
    ap_large: float
    ar1: float
    ar10: float
    ar100: float
    ar_small: float
    ar_medium: float
    ar_large: float


@dataclass(frozen=True)
class ClassAp:
    class_name: str
    ap: float
    ap50: float


@dataclass(frozen=True)
class EvaluationMetrics:
    stats: CocoStats
    per_class: list[ClassAp]
    num_images: int


def _mean_valid(values: np.ndarray) -> float:
    valid = values[values > -1]
    return float(valid.mean()) if valid.size else 0.0


def compute(gt: dict[str, Any], detections: list[dict[str, Any]]) -> EvaluationMetrics:
    """COCO bbox evaluation of `detections` (COCO results format) against `gt`."""
    categories = gt["categories"]
    num_images = len(gt["images"])
    if not detections:
        zeros = CocoStats(*([0.0] * 12))
        return EvaluationMetrics(zeros, [ClassAp(c["name"], 0.0, 0.0) for c in categories], num_images)

    coco_gt = COCO()
    coco_gt.dataset = gt
    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt.createIndex()
        coco_dt = coco_gt.loadRes(detections)
        coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

    stats = CocoStats(*(float(s) for s in coco_eval.stats[:12]))
    precision = coco_eval.eval["precision"]  # [iou, recall, class, area, max_det]
    names = {c["id"]: c["name"] for c in categories}
    per_class = [
        ClassAp(
            names[cat_id],
            _mean_valid(precision[:, :, k, 0, -1]),
            _mean_valid(precision[0, :, k, 0, -1]),
        )
        for k, cat_id in enumerate(coco_eval.params.catIds)
    ]
    return EvaluationMetrics(stats, per_class, num_images)
