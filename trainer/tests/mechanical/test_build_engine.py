import importlib.util
from pathlib import Path

import pytest

from yolox_trainer import engine, server
from yolox_trainer.config import TrainerConfig
from yolox_trainer.core.datasets import DatasetCatalog
from yolox_trainer.core.errors import FailedPrecondition
from yolox_trainer.core.registry import ModelRegistry

real_find_spec = importlib.util.find_spec


def _build(tmp_path: Path):
    return server._build_engine(TrainerConfig(), ModelRegistry(tmp_path / "m"), DatasetCatalog(tmp_path / "d"))


def test_missing_torch_disables_training_and_inference(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        importlib.util, "find_spec", lambda name, *a: None if name == "torch" else real_find_spec(name, *a)
    )

    _, inference = _build(tmp_path)

    with pytest.raises(FailedPrecondition):
        inference.predict("m", b"img", 0.3)


def test_import_errors_from_engine_code_propagate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name, *a: object() if name in ("torch", "yolox") else real_find_spec(name, *a),
    )

    def broken_runtime(*args: object) -> None:
        raise ImportError("broken engine module", name="yolox_trainer.engine.inference_service")

    monkeypatch.setattr(engine, "build_runtime", broken_runtime)

    with pytest.raises(ImportError, match="broken engine module"):
        _build(tmp_path)
