from pathlib import Path

import pytest

pytestmark = [pytest.mark.engine, pytest.mark.needs_weights]


def test_pretrained_load_skips_only_cls_preds(pretrained_weights: Path) -> None:
    import torch
    from yolox.exp.default.yolox_tiny import Exp
    from yolox.utils import load_ckpt

    exp = Exp()
    exp.num_classes = 1
    model = exp.get_model()
    ckpt = torch.load(pretrained_weights, map_location="cpu", weights_only=False)["model"]

    load_ckpt(model, ckpt)

    state = model.state_dict()
    assert any(key.startswith("head.cls_preds.") for key in state)
    for key, tensor in state.items():
        if key.startswith("head.cls_preds."):
            assert tensor.shape[0] == 1, key
        else:
            assert torch.equal(tensor, ckpt[key]), key
