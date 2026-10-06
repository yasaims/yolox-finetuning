from __future__ import annotations

import numpy as np

from ..core.inference import Detection


def to_detections(
    output: np.ndarray | None,
    ratio: float,
    width: int,
    height: int,
    class_names: list[str],
    min_score: float,
) -> list[Detection]:
    """Map YOLOX postprocess rows (x1,y1,x2,y2,obj,cls_conf,cls_id) in letterboxed
    coordinates to detections in original image coordinates."""
    if output is None or len(output) == 0:
        return []
    rows = np.asarray(output, dtype=np.float64)
    boxes = rows[:, :4] / ratio
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, width)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, height)
    scores = rows[:, 4] * rows[:, 5]
    class_ids = rows[:, 6].astype(int)

    detections = [
        Detection(
            x1=float(box[0]),
            y1=float(box[1]),
            x2=float(box[2]),
            y2=float(box[3]),
            score=float(score),
            class_id=int(cls),
            class_name=class_names[cls],
        )
        for box, score, cls in zip(boxes, scores, class_ids)
        if score >= min_score
    ]
    detections.sort(key=lambda d: d.score, reverse=True)
    return detections
