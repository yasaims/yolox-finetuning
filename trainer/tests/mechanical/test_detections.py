import numpy as np
import pytest

from yolox_trainer.core.inference import Detection
from yolox_trainer.engine.detections import to_detections

CLASSES = ["cat", "dog"]


def _rows(*rows: tuple[float, ...]) -> np.ndarray:
    return np.array(rows, dtype=np.float32)


def test_none_output_yields_no_detections() -> None:
    assert to_detections(None, 0.5, 100, 100, CLASSES, 0.0) == []


def test_boxes_are_divided_by_ratio_and_scored() -> None:
    output = _rows((10, 20, 30, 40, 0.8, 0.5, 1))

    [det] = to_detections(output, 0.5, 100, 100, CLASSES, 0.0)

    assert det == Detection(
        x1=20.0, y1=40.0, x2=60.0, y2=80.0, score=pytest.approx(0.4), class_id=1, class_name="dog"
    )
    assert all(type(v) is float for v in (det.x1, det.y1, det.x2, det.y2, det.score))
    assert type(det.class_id) is int


def test_boxes_are_clipped_to_image() -> None:
    output = _rows((-5, -10, 120, 90, 1.0, 1.0, 0))

    [det] = to_detections(output, 1.0, 100, 80, CLASSES, 0.0)

    assert (det.x1, det.y1, det.x2, det.y2) == (0.0, 0.0, 100.0, 80.0)


def test_filters_by_min_score_inclusive_and_sorts_descending() -> None:
    output = _rows(
        (0, 0, 1, 1, 0.5, 0.5, 0),  # 0.25, below threshold
        (0, 0, 2, 2, 0.6, 0.5, 0),  # 0.30, exactly at threshold
        (0, 0, 3, 3, 0.9, 1.0, 1),  # 0.90
    )

    dets = to_detections(output, 1.0, 10, 10, CLASSES, 0.3)

    assert [d.x2 for d in dets] == [3.0, 2.0]
    assert [d.class_name for d in dets] == ["dog", "cat"]
