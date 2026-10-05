import json
from pathlib import Path

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from dataset_tools.via_to_coco import build_coco


def test_built_coco_loads_and_scores_perfectly_against_itself(tmp_path: Path) -> None:
    region = lambda xs, ys: {"shape_attributes": {"all_points_x": xs, "all_points_y": ys}}
    coco_dict = build_coco(
        [
            ("a.jpg", 200, 100, [region([10, 60, 60, 10], [10, 10, 50, 50])]),
            ("b.jpg", 200, 100, [region([100, 180, 140], [20, 30, 90]), region([5, 50], [60, 90])]),
        ],
        "balloon",
    )
    path = tmp_path / "gt.json"
    path.write_text(json.dumps(coco_dict))

    gt = COCO(str(path))
    assert len(gt.getAnnIds(iscrowd=False)) == 3
    assert [c["name"] for c in gt.loadCats(gt.getCatIds())] == ["balloon"]

    detections = [
        {"image_id": a["image_id"], "category_id": a["category_id"], "bbox": a["bbox"], "score": 1.0}
        for a in coco_dict["annotations"]
    ]
    evaluator = COCOeval(gt, gt.loadRes(detections), "bbox")
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    assert evaluator.stats[0] == 1.0
