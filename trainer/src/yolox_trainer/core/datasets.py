from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from .errors import InvalidArgument, NotFound

SPLITS = ("train", "val")
_NAME_RE = re.compile(r"[a-z0-9][a-z0-9_-]{0,63}")


@dataclass(frozen=True)
class DatasetInfo:
    name: str
    root: Path
    class_names: list[str]
    train_images: int
    val_images: int


def _validate_name(name: str) -> None:
    if not isinstance(name, str) or not _NAME_RE.fullmatch(name):
        raise InvalidArgument(f"invalid dataset name: {name!r}")


def _load(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


class DatasetCatalog:
    def __init__(self, root: Path) -> None:
        self._root = root

    def annotation_path(self, name: str, split: str) -> Path:
        _validate_name(name)
        if split not in SPLITS:
            raise InvalidArgument(f"invalid split: {split!r}")
        return self._root / name / "annotations" / f"{split}.json"

    def get(self, name: str) -> DatasetInfo:
        _validate_name(name)
        paths = {split: self.annotation_path(name, split) for split in SPLITS}
        if not all(p.is_file() for p in paths.values()):
            raise NotFound(f"dataset not found: {name}")
        train, val = _load(paths["train"]), _load(paths["val"])
        categories = sorted(train.get("categories", []), key=lambda c: c["id"])
        return DatasetInfo(
            name=name,
            root=self._root / name,
            class_names=[c["name"] for c in categories],
            train_images=len(train.get("images", [])),
            val_images=len(val.get("images", [])),
        )

    def list(self) -> list[DatasetInfo]:
        if not self._root.is_dir():
            return []
        infos = []
        for entry in sorted(self._root.iterdir()):
            if not entry.is_dir() or not _NAME_RE.fullmatch(entry.name):
                continue
            try:
                infos.append(self.get(entry.name))
            except NotFound:
                continue
        return infos
