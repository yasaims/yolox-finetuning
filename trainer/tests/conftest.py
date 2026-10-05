import os
from functools import cache
from pathlib import Path

import pytest

DEFAULT_PRETRAINED = "/data/pretrained/yolox_tiny.pth"


def pretrained_path() -> Path:
    return Path(os.environ.get("TRAINER_PRETRAINED", DEFAULT_PRETRAINED))


@cache
def _engine_available() -> bool:
    try:
        import torch  # noqa: F401
        import yolox  # noqa: F401
    except ImportError:
        return False
    return True


@cache
def _gpu_available() -> bool:
    if not _engine_available():
        return False
    import torch

    return torch.cuda.is_available()


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "engine: requires torch and yolox")
    config.addinivalue_line("markers", "gpu: requires a CUDA device")
    config.addinivalue_line("markers", "needs_weights: requires the TRAINER_PRETRAINED checkpoint")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    for item in items:
        if "engine" in item.keywords and not _engine_available():
            item.add_marker(pytest.mark.skip(reason="torch/yolox not installed"))
        if "gpu" in item.keywords and not _gpu_available():
            item.add_marker(pytest.mark.skip(reason="CUDA not available"))
        if "needs_weights" in item.keywords and not pretrained_path().is_file():
            item.add_marker(pytest.mark.skip(reason=f"pretrained weights not found: {pretrained_path()}"))


@pytest.fixture
def pretrained_weights() -> Path:
    return pretrained_path()
