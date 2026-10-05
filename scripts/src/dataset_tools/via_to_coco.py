from typing import Any, Sequence

Region = dict[str, Any]
ImageEntry = tuple[str, int, int, list[Region]]


def polygon_to_bbox(xs: Sequence[float], ys: Sequence[float]) -> list[float]:
    """Axis-aligned bounding box [x, y, w, h] of a polygon."""
    x0, y0 = float(min(xs)), float(min(ys))
    return [x0, y0, float(max(xs)) - x0, float(max(ys)) - y0]


def via_regions(entry: dict[str, Any]) -> list[Region]:
    """VIA stores `regions` as a dict keyed by index (v1) or a list (v2)."""
    regions = entry.get("regions", [])
    return list(regions.values()) if isinstance(regions, dict) else list(regions)


def build_coco(images: Sequence[ImageEntry], class_name: str) -> dict[str, Any]:
    """Build a single-class COCO dict from (file_name, width, height, VIA regions)."""
    coco_images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    for image_id, (file_name, width, height, regions) in enumerate(images, start=1):
        coco_images.append(
            {"id": image_id, "file_name": file_name, "width": width, "height": height}
        )
        for region in regions:
            shape = region["shape_attributes"]
            bbox = polygon_to_bbox(shape["all_points_x"], shape["all_points_y"])
            annotations.append(
                {
                    "id": len(annotations) + 1,
                    "image_id": image_id,
                    "category_id": 1,
                    "bbox": bbox,
                    "area": bbox[2] * bbox[3],
                    "iscrowd": 0,
                }
            )
    return {
        "images": coco_images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": class_name}],
    }
