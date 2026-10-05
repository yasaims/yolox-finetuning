from __future__ import annotations

import torch
from yolox.data import COCODataset, TrainTransform, ValTransform
from yolox.exp.default.yolox_tiny import Exp as YoloxTinyExp
from yolox.utils import load_ckpt

from ..core.hyperparams import Hyperparameters


class FinetuneExp(YoloxTinyExp):
    """YOLOX-Tiny experiment over a dataset laid out as `annotations/<split>.json`
    with images under `<split>/`."""

    def __init__(self) -> None:
        super().__init__()
        self.train_ann = "train.json"
        self.eval_split = "val"
        self.class_names: list[str] = []
        self.pretrained_path: str | None = None
        self.last_eval: tuple[float, float] | None = None
        self._pretrained_loaded = False

    @property
    def eval_split(self) -> str:
        return self._eval_split

    @eval_split.setter
    def eval_split(self, split: str) -> None:
        self._eval_split = split
        self.val_ann = f"{split}.json"

    def get_model(self) -> torch.nn.Module:
        # Upstream get_model re-initializes head biases, which would clobber loaded weights.
        if self._pretrained_loaded:
            return self.model
        model = super().get_model()
        if self.pretrained_path:
            # The official checkpoint is a trusted pickle that predates weights_only loading.
            ckpt = torch.load(self.pretrained_path, map_location="cpu", weights_only=False)
            load_ckpt(model, ckpt["model"])
            self._pretrained_loaded = True
        return model

    def get_dataset(self, cache: bool = False, cache_type: str = "ram") -> COCODataset:
        return COCODataset(
            data_dir=self.data_dir,
            json_file=self.train_ann,
            name="train",
            img_size=self.input_size,
            preproc=TrainTransform(max_labels=50, flip_prob=self.flip_prob, hsv_prob=self.hsv_prob),
            cache=cache,
            cache_type=cache_type,
        )

    def get_eval_dataset(self, **kwargs: object) -> COCODataset:
        return COCODataset(
            data_dir=self.data_dir,
            json_file=self.val_ann,
            name=self.eval_split,
            img_size=self.test_size,
            preproc=ValTransform(legacy=bool(kwargs.get("legacy", False))),
        )

    def eval(self, model, evaluator, is_distributed, half=False, return_outputs=False):  # type: ignore[no-untyped-def]
        result = super().eval(model, evaluator, is_distributed, half, return_outputs=return_outputs)
        stats = result[0] if return_outputs else result
        self.last_eval = (float(stats[0]), float(stats[1]))
        return result


def build_exp(
    data_dir: str,
    class_names: list[str],
    hp: Hyperparameters,
    output_dir: str,
    pretrained_path: str | None,
    num_workers: int = 2,
) -> FinetuneExp:
    exp = FinetuneExp()
    exp.data_dir = data_dir
    exp.class_names = list(class_names)
    exp.num_classes = len(class_names)
    exp.pretrained_path = pretrained_path
    exp.output_dir = output_dir

    size = hp.input_size
    exp.input_size = (size, size)
    exp.test_size = (size, size)
    exp.random_size = (size // 32 - 3, size // 32 + 3)

    exp.max_epoch = hp.max_epoch
    exp.warmup_epochs = hp.warmup_epochs
    exp.no_aug_epochs = hp.no_aug_epochs
    exp.basic_lr_per_img = hp.learning_rate / hp.batch_size
    exp.mosaic_prob = hp.mosaic_prob
    exp.enable_mixup = hp.enable_mixup
    exp.mixup_prob = 1.0 if hp.enable_mixup else 0.0
    exp.seed = hp.seed

    exp.eval_interval = 1
    exp.print_interval = 1
    exp.save_history_ckpt = False
    exp.data_num_workers = num_workers
    return exp
