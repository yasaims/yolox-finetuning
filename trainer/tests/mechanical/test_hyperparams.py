import dataclasses

import pytest

from yolox_trainer.core.errors import InvalidArgument
from yolox_trainer.core.hyperparams import DEFAULTS, Hyperparameters, resolve


def test_defaults_match_spec() -> None:
    assert dataclasses.asdict(DEFAULTS) == {
        "max_epoch": 50,
        "batch_size": 8,
        "learning_rate": 0.00125,
        "input_size": 416,
        "warmup_epochs": 3,
        "no_aug_epochs": 10,
        "mosaic_prob": 1.0,
        "enable_mixup": False,
        "fp16": True,
        "seed": 42,
    }


def test_resolve_empty_returns_defaults() -> None:
    assert resolve({}) == DEFAULTS


def test_resolve_ignores_none_and_applies_overrides() -> None:
    hp = resolve({"max_epoch": None, "batch_size": 16, "fp16": False})
    assert hp == dataclasses.replace(DEFAULTS, batch_size=16, fp16=False)


def test_hyperparameters_is_frozen() -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        DEFAULTS.max_epoch = 1  # type: ignore[misc]


@pytest.mark.parametrize(
    "field, value",
    [
        ("max_epoch", 0),
        ("max_epoch", 301),
        ("batch_size", 0),
        ("batch_size", 65),
        ("learning_rate", 0.0),
        ("learning_rate", 0.11),
        ("input_size", 192),
        ("input_size", 1056),
        ("input_size", 420),
        ("warmup_epochs", -1),
        ("warmup_epochs", 50),
        ("no_aug_epochs", -1),
        ("no_aug_epochs", 50),
        ("mosaic_prob", -0.1),
        ("mosaic_prob", 1.1),
        ("seed", -1),
        ("max_epoch", True),
        ("max_epoch", 1.5),
        ("enable_mixup", 1),
        ("fp16", "yes"),
        ("learning_rate", "0.1"),
    ],
)
def test_resolve_rejects_invalid_field(field: str, value: object) -> None:
    with pytest.raises(InvalidArgument, match=field):
        resolve({field: value})


def test_resolve_rejects_unknown_field() -> None:
    with pytest.raises(InvalidArgument, match="bogus"):
        resolve({"bogus": 1})


def test_epoch_limits_are_checked_against_max_epoch() -> None:
    with pytest.raises(InvalidArgument, match="warmup_epochs"):
        resolve({"max_epoch": 3})  # default warmup_epochs=3 is not < 3
    with pytest.raises(InvalidArgument, match="no_aug_epochs"):
        resolve({"max_epoch": 5, "warmup_epochs": 0})  # default no_aug_epochs=10
    assert isinstance(resolve({"max_epoch": 4, "warmup_epochs": 3, "no_aug_epochs": 3}), Hyperparameters)


def test_boundaries_accepted() -> None:
    resolve({"max_epoch": 1, "warmup_epochs": 0, "no_aug_epochs": 0, "batch_size": 1,
             "learning_rate": 0.1, "input_size": 224, "mosaic_prob": 0, "seed": 0})
    resolve({"max_epoch": 300, "batch_size": 64, "input_size": 1024, "mosaic_prob": 1})
