import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from yolox_trainer.core.hyperparams import DEFAULTS

pytestmark = pytest.mark.engine

CLASSES = ["cat", "dog"]
SPLIT_SIZES = {"train": 4, "val": 2}


def _write_split(root: Path, split: str, count: int, rng: np.random.Generator) -> None:
    import cv2

    (root / split).mkdir(parents=True)
    images, annotations = [], []
    for i in range(count):
        width, height = 320 + 32 * i, 240
        name = f"{split}_{i}.jpg"
        cv2.imwrite(str(root / split / name), rng.integers(0, 255, (height, width, 3), dtype=np.uint8))
        images.append({"id": i + 1, "file_name": name, "width": width, "height": height})
        for j, (x, y, w, h) in enumerate([(10, 20, 80, 60), (width - 90, height - 70, 80, 60)]):
            annotations.append(
                {
                    "id": len(annotations) + 1,
                    "image_id": i + 1,
                    "category_id": j + 1,
                    "bbox": [x, y, w, h],
                    "area": w * h,
                    "iscrowd": 0,
                }
            )
    categories = [{"id": k + 1, "name": n} for k, n in enumerate(CLASSES)]
    payload = {"images": images, "annotations": annotations, "categories": categories}
    (root / "annotations" / f"{split}.json").write_text(json.dumps(payload))


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    root = tmp_path / "toy"
    (root / "annotations").mkdir(parents=True)
    rng = np.random.default_rng(0)
    for split, count in SPLIT_SIZES.items():
        _write_split(root, split, count, rng)
    return root


def _exp(data_dir: Path, tmp_path: Path, pretrained: str | None = None, **hp_overrides: object):
    from yolox_trainer.engine.exp import build_exp

    hp = replace(DEFAULTS, **hp_overrides)
    return build_exp(str(data_dir), CLASSES, hp, str(tmp_path / "out"), pretrained, num_workers=0)


def test_build_exp_maps_hyperparameters(data_dir: Path, tmp_path: Path) -> None:
    exp = _exp(
        data_dir, tmp_path, max_epoch=20, batch_size=4, learning_rate=0.002, input_size=320,
        warmup_epochs=2, no_aug_epochs=5, mosaic_prob=0.5, enable_mixup=True, seed=7,
    )

    assert exp.num_classes == 2
    assert exp.input_size == exp.test_size == (320, 320)
    assert exp.random_size == (7, 13)
    assert (exp.max_epoch, exp.warmup_epochs, exp.no_aug_epochs) == (20, 2, 5)
    assert exp.basic_lr_per_img * 4 == pytest.approx(0.002)
    assert (exp.mosaic_prob, exp.enable_mixup, exp.mixup_prob) == (0.5, True, 1.0)
    assert (exp.eval_interval, exp.print_interval, exp.save_history_ckpt) == (1, 1, False)
    assert (exp.data_num_workers, exp.seed, exp.output_dir) == (0, 7, str(tmp_path / "out"))
    assert _exp(data_dir, tmp_path, enable_mixup=False).mixup_prob == 0.0


def test_train_loader_yields_boxes_inside_mosaic_image(data_dir: Path, tmp_path: Path) -> None:
    exp = _exp(data_dir, tmp_path, input_size=320)

    loader = exp.get_data_loader(batch_size=2, is_distributed=False)
    images, targets, _, _ = next(iter(loader))

    assert len(exp.dataset) == SPLIT_SIZES["train"]
    assert tuple(images.shape) == (2, 3, 320, 320)
    labels = targets.numpy()
    boxes = labels[labels[..., 3] > 0]  # (cls, cx, cy, w, h); zero rows are padding
    assert len(boxes) > 0
    assert set(np.unique(boxes[:, 0])) <= {0.0, 1.0}
    cx, cy, w, h = boxes[:, 1], boxes[:, 2], boxes[:, 3], boxes[:, 4]
    eps = 1e-3
    assert np.all(cx - w / 2 >= -eps) and np.all(cx + w / 2 <= 320 + eps)
    assert np.all(cy - h / 2 >= -eps) and np.all(cy + h / 2 <= 320 + eps)


@pytest.mark.parametrize("split", ["val", "train"])
def test_eval_dataset_uses_requested_split(data_dir: Path, tmp_path: Path, split: str) -> None:
    exp = _exp(data_dir, tmp_path)
    if split != "val":
        exp.eval_split = split

    dataset = exp.get_eval_dataset(legacy=False)
    image, _, _, _ = dataset[0]

    assert len(dataset) == SPLIT_SIZES[split]
    assert dataset.name == split
    assert dataset.json_file == f"{split}.json"
    assert image.shape == (3, 416, 416)


@pytest.mark.needs_weights
def test_get_model_loads_pretrained_once(data_dir: Path, tmp_path: Path, pretrained_weights: Path) -> None:
    import torch

    exp = _exp(data_dir, tmp_path, pretrained=str(pretrained_weights))
    ckpt = torch.load(pretrained_weights, map_location="cpu", weights_only=False)["model"]

    model = exp.get_model()
    again = exp.get_model()

    assert again is model
    for key, tensor in model.state_dict().items():
        if key.startswith("head.cls_preds."):
            assert tensor.shape[0] == len(CLASSES), key
        else:
            assert torch.equal(tensor, ckpt[key]), key


class _StubEvaluator:
    def evaluate(self, model, distributed, half, return_outputs=False):
        stats = (0.25, 0.5, "summary")
        return (stats, {"outputs": 1}) if return_outputs else stats


@pytest.mark.parametrize("return_outputs", [False, True])
def test_eval_records_last_ap_and_passes_result_through(
    data_dir: Path, tmp_path: Path, return_outputs: bool
) -> None:
    exp = _exp(data_dir, tmp_path)
    evaluator = _StubEvaluator()

    result = exp.eval(None, evaluator, False, return_outputs=return_outputs)

    assert result == evaluator.evaluate(None, False, False, return_outputs)
    assert exp.last_eval == (0.25, 0.5)
