from datetime import datetime

import pytest

from yolox_trainer.core.errors import InvalidArgument
from yolox_trainer.core.ids import new_model_id, validate_id


@pytest.mark.parametrize("value", ["a", "0abc", "20260101-120000-balloon-abc123", "a" * 64])
def test_validate_id_accepts(value: str) -> None:
    assert validate_id(value) == value


@pytest.mark.parametrize("value", ["../x", "Abc", "", "a" * 65, "-a", "a_b", "a/b", "a\n"])
def test_validate_id_rejects(value: str) -> None:
    with pytest.raises(InvalidArgument):
        validate_id(value)


def test_new_model_id_format() -> None:
    now = datetime(2026, 10, 5, 9, 8, 7)
    assert new_model_id("my_set", now, "abc123") == "20261005-090807-my-set-abc123"


def test_new_model_id_random_suffix_is_valid_id() -> None:
    model_id = new_model_id("balloon", datetime(2026, 1, 2, 3, 4, 5))
    assert validate_id(model_id) == model_id
    assert len(model_id.rsplit("-", 1)[1]) == 6


@pytest.mark.parametrize("dataset", ["a" * 64, "a" * 40 + "_" + "b" * 23, "a" * 40 + "-_" + "b" * 22])
def test_new_model_id_from_longest_dataset_names_is_valid(dataset: str) -> None:
    model_id = new_model_id(dataset, datetime(2026, 1, 2, 3, 4, 5), "abc123")
    assert validate_id(model_id) == model_id
    assert "--" not in model_id
