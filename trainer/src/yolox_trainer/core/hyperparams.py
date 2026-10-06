from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from dataclasses import dataclass

from .errors import InvalidArgument


@dataclass(frozen=True)
class Hyperparameters:
    max_epoch: int = 50
    batch_size: int = 8
    learning_rate: float = 0.00125
    input_size: int = 416
    warmup_epochs: int = 3
    no_aug_epochs: int = 10
    mosaic_prob: float = 1.0
    enable_mixup: bool = False
    fp16: bool = True
    seed: int = 42


DEFAULTS = Hyperparameters()

_FIELD_TYPES = {f.name: f.type for f in dataclasses.fields(Hyperparameters)}


def _check_type(name: str, value: object) -> None:
    expected = _FIELD_TYPES[name]
    if expected == "bool":
        ok = isinstance(value, bool)
    elif expected == "int":
        ok = isinstance(value, int) and not isinstance(value, bool)
    else:
        ok = isinstance(value, (int, float)) and not isinstance(value, bool)
    if not ok:
        raise InvalidArgument(f"{name} must be of type {expected}, got {type(value).__name__}")


def _check_range(name: str, value: float, low: float, high: float) -> None:
    if not low <= value <= high:
        raise InvalidArgument(f"{name} must be in [{low}, {high}], got {value}")


def _validate(hp: Hyperparameters) -> None:
    _check_range("max_epoch", hp.max_epoch, 1, 300)
    _check_range("batch_size", hp.batch_size, 1, 64)
    if not 0 < hp.learning_rate <= 0.1:
        raise InvalidArgument(f"learning_rate must be in (0, 0.1], got {hp.learning_rate}")
    _check_range("input_size", hp.input_size, 224, 1024)
    if hp.input_size % 32:
        raise InvalidArgument(f"input_size must be a multiple of 32, got {hp.input_size}")
    _check_range("warmup_epochs", hp.warmup_epochs, 0, hp.max_epoch - 1)
    _check_range("no_aug_epochs", hp.no_aug_epochs, 0, hp.max_epoch - 1)
    _check_range("mosaic_prob", hp.mosaic_prob, 0, 1)
    if hp.seed < 0:
        raise InvalidArgument(f"seed must be >= 0, got {hp.seed}")


def resolve(overrides: Mapping[str, object]) -> Hyperparameters:
    """Apply non-None overrides onto the defaults and validate the result."""
    values: dict[str, object] = {}
    for name, value in overrides.items():
        if name not in _FIELD_TYPES:
            raise InvalidArgument(f"unknown hyperparameter: {name}")
        if value is None:
            continue
        _check_type(name, value)
        values[name] = value
    hp = dataclasses.replace(DEFAULTS, **values)  # type: ignore[arg-type]
    _validate(hp)
    return hp
