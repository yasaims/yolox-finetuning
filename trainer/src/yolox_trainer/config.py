from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass(frozen=True)
class TrainerConfig:
    datasets_dir: Path = Path("/data/datasets")
    models_dir: Path = Path("/data/models")
    pretrained: Path = Path("/data/pretrained/yolox_tiny.pth")
    port: int = 50051
    device: Literal["gpu", "cpu"] = "gpu"

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> TrainerConfig:
        env = os.environ if env is None else env
        defaults = cls()
        device = env.get("TRAINER_DEVICE", defaults.device)
        if device not in ("gpu", "cpu"):
            raise ValueError(f"TRAINER_DEVICE must be 'gpu' or 'cpu', got {device!r}")
        return cls(
            datasets_dir=Path(env.get("TRAINER_DATASETS_DIR", str(defaults.datasets_dir))),
            models_dir=Path(env.get("TRAINER_MODELS_DIR", str(defaults.models_dir))),
            pretrained=Path(env.get("TRAINER_PRETRAINED", str(defaults.pretrained))),
            port=int(env.get("TRAINER_PORT", str(defaults.port))),
            device=device,  # type: ignore[arg-type]
        )
