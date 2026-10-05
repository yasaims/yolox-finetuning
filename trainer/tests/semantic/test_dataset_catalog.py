import json
from pathlib import Path

import pytest

from yolox_trainer.core.datasets import DatasetCatalog
from yolox_trainer.core.errors import InvalidArgument, NotFound


def make_dataset(root: Path, name: str, train: int = 2, val: int = 1, splits: tuple[str, ...] = ("train", "val")) -> None:
    ann = root / name / "annotations"
    ann.mkdir(parents=True)
    categories = [{"id": 2, "name": "dog"}, {"id": 1, "name": "cat"}]
    for split, n in (("train", train), ("val", val)):
        if split in splits:
            data = {"images": [{"id": i} for i in range(n)], "annotations": [], "categories": categories}
            (ann / f"{split}.json").write_text(json.dumps(data))


def test_list_only_complete_valid_datasets(tmp_path: Path) -> None:
    make_dataset(tmp_path, "zeta", train=3, val=2)
    make_dataset(tmp_path, "alpha")
    make_dataset(tmp_path, "no-val", splits=("train",))
    make_dataset(tmp_path, "Upper")
    make_dataset(tmp_path, ".cache")
    (tmp_path / "stray.txt").write_text("x")
    (tmp_path / "empty").mkdir()

    infos = DatasetCatalog(tmp_path).list()

    assert [i.name for i in infos] == ["alpha", "zeta"]
    zeta = infos[1]
    assert zeta.class_names == ["cat", "dog"]
    assert (zeta.train_images, zeta.val_images) == (3, 2)
    assert zeta.root == tmp_path / "zeta"


def test_list_missing_root_is_empty(tmp_path: Path) -> None:
    assert DatasetCatalog(tmp_path / "absent").list() == []


def test_get_and_annotation_path(tmp_path: Path) -> None:
    make_dataset(tmp_path, "ds")
    catalog = DatasetCatalog(tmp_path)
    assert catalog.get("ds").name == "ds"
    assert catalog.annotation_path("ds", "val") == tmp_path / "ds" / "annotations" / "val.json"


def test_get_errors(tmp_path: Path) -> None:
    make_dataset(tmp_path, "incomplete", splits=("train",))
    catalog = DatasetCatalog(tmp_path)
    with pytest.raises(NotFound):
        catalog.get("missing")
    with pytest.raises(NotFound):
        catalog.get("incomplete")
    with pytest.raises(InvalidArgument):
        catalog.get("../etc")
    with pytest.raises(InvalidArgument):
        catalog.annotation_path("ds", "test")
