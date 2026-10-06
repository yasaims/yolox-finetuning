from dataset_tools.via_to_coco import polygon_to_bbox, via_regions


def test_polygon_to_bbox_uses_min_max() -> None:
    assert polygon_to_bbox([10, 30, 20], [5, 15, 40]) == [10.0, 5.0, 20.0, 35.0]


def test_via_regions_accepts_dict_and_list() -> None:
    region = {"shape_attributes": {"all_points_x": [1], "all_points_y": [2]}}
    assert via_regions({"regions": {"0": region}}) == [region]
    assert via_regions({"regions": [region]}) == [region]
