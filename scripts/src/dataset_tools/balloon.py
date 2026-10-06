import argparse
import json
import random
import shutil
import urllib.request
import zipfile
from pathlib import Path
from typing import Sequence

from PIL import Image

from dataset_tools.via_to_coco import ImageEntry, build_coco, via_regions

BALLOON_URL = (
    "https://github.com/matterport/Mask_RCNN/releases/download/v2.1/balloon_dataset.zip"
)
CLASS_NAME = "balloon"


def select_subset(
    names: Sequence[str], n_train: int, n_val: int, seed: int
) -> tuple[list[str], list[str]]:
    """Seeded, order-independent, disjoint train/val selection."""
    if n_train + n_val > len(names):
        raise ValueError(f"requested {n_train + n_val} images, only {len(names)} available")
    picked = random.Random(seed).sample(sorted(names), n_train + n_val)
    return picked[:n_train], picked[n_train:]


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    urllib.request.urlretrieve(url, tmp)
    tmp.replace(dest)


def _pool(extracted: Path) -> dict[str, tuple[Path, dict]]:
    """Map image file name -> (image path, VIA entry) over the zip's train and val parts."""
    pool: dict[str, tuple[Path, dict]] = {}
    for part in ("train", "val"):
        part_dir = next(extracted.rglob(part), None)
        if part_dir is None or not (part_dir / "via_region_data.json").exists():
            raise FileNotFoundError(f"'{part}/via_region_data.json' not found in zip")
        via = json.loads((part_dir / "via_region_data.json").read_text(encoding="utf-8"))
        for entry in via.values():
            image_path = part_dir / entry["filename"]
            if image_path.exists() and via_regions(entry):
                pool[entry["filename"]] = (image_path, entry)
    return pool


def prepare(
    out: Path, n_train: int, n_val: int, seed: int, zip_path: Path | None = None
) -> None:
    if zip_path is None:
        zip_path = out.parent / ".cache" / "balloon_dataset.zip"
        if not zip_path.exists():
            print(f"downloading {BALLOON_URL}")
            _download(BALLOON_URL, zip_path)

    for sub in ("train", "val", "annotations"):
        shutil.rmtree(out / sub, ignore_errors=True)
        (out / sub).mkdir(parents=True)

    extracted = out.parent / ".cache" / "balloon_extracted"
    shutil.rmtree(extracted, ignore_errors=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(extracted)
    try:
        pool = _pool(extracted)
        train, val = select_subset(list(pool), n_train, n_val, seed)
        for split, names in (("train", train), ("val", val)):
            entries: list[ImageEntry] = []
            for name in sorted(names):
                image_path, entry = pool[name]
                shutil.copy2(image_path, out / split / name)
                with Image.open(image_path) as img:
                    width, height = img.size
                entries.append((name, width, height, via_regions(entry)))
            coco = build_coco(entries, CLASS_NAME)
            (out / "annotations" / f"{split}.json").write_text(json.dumps(coco))
            print(f"{split}: {len(entries)} images, {len(coco['annotations'])} annotations")
    finally:
        shutil.rmtree(extracted, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser(prog="prepare-balloon")
    parser.add_argument("--out", type=Path, default=Path("datasets/balloon"))
    parser.add_argument("--train", type=int, default=40)
    parser.add_argument("--val", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--zip", type=Path, default=None, help="use a local balloon_dataset.zip")
    args = parser.parse_args()
    prepare(args.out, args.train, args.val, args.seed, args.zip)


if __name__ == "__main__":
    main()
