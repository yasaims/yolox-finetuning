import pytest

from yolox_trainer.core.coco_metrics import compute

GT = {
    "images": [{"id": 1, "width": 100, "height": 100}, {"id": 2, "width": 100, "height": 100}],
    "categories": [{"id": 1, "name": "a"}, {"id": 2, "name": "b"}],
    "annotations": [
        {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 10, 40, 40], "area": 1600, "iscrowd": 0},
        {"id": 2, "image_id": 2, "category_id": 2, "bbox": [20, 20, 50, 50], "area": 2500, "iscrowd": 0},
    ],
}


def det(image_id: int, category_id: int, bbox: list[float]) -> dict:
    return {"image_id": image_id, "category_id": category_id, "bbox": bbox, "score": 0.9}


def test_perfect_detections_score_one() -> None:
    result = compute(GT, [det(1, 1, [10, 10, 40, 40]), det(2, 2, [20, 20, 50, 50])])

    assert result.stats.ap == pytest.approx(1.0)
    assert result.stats.ap50 == pytest.approx(1.0)
    assert [c.class_name for c in result.per_class] == ["a", "b"]
    assert [(c.ap, c.ap50) for c in result.per_class] == [pytest.approx((1.0, 1.0))] * 2
    assert result.num_images == 2


def test_no_detections_score_zero() -> None:
    result = compute(GT, [])

    assert result.stats.ap == 0.0
    assert result.stats.ar100 == 0.0
    assert [(c.class_name, c.ap, c.ap50) for c in result.per_class] == [("a", 0.0, 0.0), ("b", 0.0, 0.0)]
    assert result.num_images == 2


def test_per_class_ap_distinguishes_classes() -> None:
    result = compute(GT, [det(1, 1, [10, 10, 40, 40])])

    by_name = {c.class_name: c for c in result.per_class}
    assert by_name["a"].ap == pytest.approx(1.0)
    assert by_name["b"].ap == 0.0
