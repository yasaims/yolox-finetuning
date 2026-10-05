"""Torch/YOLOX-backed training and inference.

Importing this package stays torch-free; `build_runtime` and the submodules that need
torch or YOLOX import them on use.
"""
from __future__ import annotations

from collections.abc import Callable

from ..config import TrainerConfig
from ..core.datasets import DatasetCatalog
from ..core.inference import InferenceService
from ..core.registry import ModelRegistry
from ..core.supervisor import TrainingRunner


def build_runtime(
    config: TrainerConfig, registry: ModelRegistry, catalog: DatasetCatalog
) -> tuple[Callable[[], TrainingRunner], InferenceService]:
    """Returns `(runner_factory, inference_service)`; raises ImportError without torch/YOLOX."""
    import torch  # noqa: F401
    import yolox  # noqa: F401

    from ..server import _UnavailableInference
    from .process_runner import ProcessTrainingRunner

    return ProcessTrainingRunner, _UnavailableInference()
