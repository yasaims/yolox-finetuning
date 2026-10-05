"""Torch/YOLOX-backed training and inference.

Importing this package and its `detections`, `worker` and `process_runner` submodules does
not load torch or YOLOX; `exp`, `observable_trainer` and `inference_service` require them.
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
    from .inference_service import EngineInferenceService
    from .process_runner import ProcessTrainingRunner

    return ProcessTrainingRunner, EngineInferenceService(registry, catalog, config.device)
