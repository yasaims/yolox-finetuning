from pathlib import Path

import pytest

from yolox_trainer.core.errors import InvalidArgument, NotFound
from yolox_trainer.core.events import EpochEval, Event, JobFinished, JobState
from yolox_trainer.core.hyperparams import Hyperparameters
from yolox_trainer.core.registry import ModelRecord, ModelRegistry


def make_record(model_id: str, created: int = 1000) -> ModelRecord:
    return ModelRecord(
        id=model_id,
        display_name="demo",
        status="completed",
        created_unix_ms=created,
        finished_unix_ms=created + 500,
        base_model="yolox_tiny",
        dataset_name="ds",
        class_names=["a", "b"],
        hyperparameters=Hyperparameters(max_epoch=3, warmup_epochs=1, no_aug_epochs=1),
        epochs_trained=3,
        best_epoch=2,
        best_ap50=0.8,
        best_ap50_95=0.5,
        eval_history=[EpochEval(1, 0.5, 0.3, 0.3), EpochEval(2, 0.8, 0.5, 0.5)],
    )


def register(registry: ModelRegistry, tmp_path: Path, model_id: str, created: int = 1000) -> ModelRecord:
    weights = tmp_path / "w.pth"
    weights.write_bytes(b"weights")
    events = [Event(model_id, 1, 1, JobFinished(JobState.SUCCEEDED, model_id, ""))]
    return registry.register(make_record(model_id, created), weights, events, None)


def test_register_then_get_and_list_return_the_record(tmp_path: Path) -> None:
    registry = ModelRegistry(tmp_path / "models")
    record = register(registry, tmp_path, "m1")

    assert registry.get("m1") == record
    assert registry.list() == [record]
    assert registry.weights_path("m1").read_bytes() == b"weights"
    assert (tmp_path / "models" / "m1" / "events.jsonl").read_text().count("\n") == 1


def test_register_copies_log_and_removes_work_dir(tmp_path: Path) -> None:
    registry = ModelRegistry(tmp_path / "models")
    work = registry.work_dir("m1")
    work.mkdir(parents=True)
    (work / "scratch").write_text("x")
    log = tmp_path / "log.txt"
    log.write_text("log")
    weights = tmp_path / "w.pth"
    weights.write_bytes(b"w")

    registry.register(make_record("m1"), weights, [], log)

    assert (tmp_path / "models" / "m1" / "train_log.txt").read_text() == "log"
    assert not work.exists()


def test_list_is_newest_first(tmp_path: Path) -> None:
    registry = ModelRegistry(tmp_path / "models")
    register(registry, tmp_path, "old", created=1000)
    register(registry, tmp_path, "new", created=2000)

    assert [r.id for r in registry.list()] == ["new", "old"]


def test_delete_removes_model(tmp_path: Path) -> None:
    registry = ModelRegistry(tmp_path / "models")
    register(registry, tmp_path, "m1")

    registry.delete("m1")

    assert not (tmp_path / "models" / "m1").exists()
    with pytest.raises(NotFound):
        registry.get("m1")
    with pytest.raises(NotFound):
        registry.delete("m1")


def test_staging_leftovers_are_invisible_and_cleaned(tmp_path: Path) -> None:
    root = tmp_path / "models"
    registry = ModelRegistry(root)
    register(registry, tmp_path, "m1")
    (root / ".staging-x").mkdir()
    (root / ".staging-x" / "model.json").write_text("{}")
    (root / ".work" / "j1").mkdir(parents=True)
    (root / "broken").mkdir()

    assert [r.id for r in registry.list()] == ["m1"]

    registry.cleanup()

    assert not (root / ".staging-x").exists()
    assert not (root / ".work" / "j1").exists()
    assert registry.get("m1").id == "m1"


def test_traversal_ids_are_rejected(tmp_path: Path) -> None:
    registry = ModelRegistry(tmp_path / "models")

    for call in (registry.get, registry.delete, registry.weights_path):
        with pytest.raises(InvalidArgument):
            call("../x")
