import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from training_run import TrainedJob

from yolox_trainer.core.datasets import DatasetCatalog, DatasetInfo
from yolox_trainer.core.errors import InvalidArgument, NotFound
from yolox_trainer.core.events import WorkerDone
from yolox_trainer.core.hyperparams import DEFAULTS, Hyperparameters
from yolox_trainer.core.registry import ModelRecord, ModelRegistry

pytestmark = pytest.mark.engine

CLASSES = ["cat", "dog"]
SMALL_HP = replace(DEFAULTS, input_size=320)


def _record(model_id: str, class_names: list[str], hp: Hyperparameters) -> ModelRecord:
    return ModelRecord(
        id=model_id,
        display_name=model_id,
        status="completed",
        created_unix_ms=1000,
        finished_unix_ms=2000,
        base_model="yolox_tiny",
        dataset_name="ds",
        class_names=list(class_names),
        hyperparameters=hp,
        epochs_trained=1,
        best_epoch=1,
        best_ap50=0.0,
        best_ap50_95=0.0,
        eval_history=[],
    )


def _register_random_model(registry: ModelRegistry, tmp_path: Path, model_id: str) -> dict[str, Any]:
    import torch

    from yolox_trainer.engine.exp import build_exp

    state = build_exp("", CLASSES, SMALL_HP, str(tmp_path), None).get_model().state_dict()
    weights = tmp_path / f"{model_id}.pth"
    torch.save({"model": state}, weights)
    registry.register(_record(model_id, CLASSES, SMALL_HP), weights, [], None)
    return state


def _jpeg(width: int, height: int) -> bytes:
    import cv2

    ok, encoded = cv2.imencode(".jpg", np.zeros((height, width, 3), dtype=np.uint8))
    assert ok
    return encoded.tobytes()


def _service(registry: ModelRegistry, catalog: DatasetCatalog, device: str):
    from yolox_trainer.engine.inference_service import EngineInferenceService

    return EngineInferenceService(registry, catalog, device)


def test_model_cache_loads_registered_weights_and_evicts_least_recently_used(tmp_path: Path) -> None:
    import torch

    from yolox_trainer.engine.inference_service import ModelCache

    registry = ModelRegistry(tmp_path / "models")
    saved = {mid: _register_random_model(registry, tmp_path, mid) for mid in ("m-a", "m-b", "m-c")}
    cache = ModelCache(registry, "cpu", capacity=2)

    exp, a = cache.get("m-a")
    assert not a.training
    assert exp.test_size == (320, 320) and exp.class_names == CLASSES
    for key, tensor in a.state_dict().items():
        assert torch.equal(tensor, saved["m-a"][key]), key

    _, b = cache.get("m-b")
    assert cache.get("m-a")[1] is a
    cache.get("m-c")  # evicts m-b, the least recently used
    assert cache.get("m-a")[1] is a
    assert cache.get("m-b")[1] is not b

    cache.forget("m-a")
    assert cache.get("m-a")[1] is not a


def test_predict_unknown_model_raises_not_found(tmp_path: Path) -> None:
    service = _service(ModelRegistry(tmp_path / "models"), DatasetCatalog(tmp_path / "datasets"), "cpu")

    with pytest.raises(NotFound):
        service.predict("missing", _jpeg(64, 48), 0.3)


def test_predict_rejects_undecodable_image(tmp_path: Path) -> None:
    registry = ModelRegistry(tmp_path / "models")
    _register_random_model(registry, tmp_path, "m-a")
    service = _service(registry, DatasetCatalog(tmp_path / "datasets"), "cpu")

    with pytest.raises(InvalidArgument):
        service.predict("m-a", b"not an image", 0.3)


GPU_MODEL_ID = "gpu-model"


@pytest.fixture(scope="module")
def trained_registry(
    trained_job: TrainedJob, balloon: DatasetInfo, tmp_path_factory: pytest.TempPathFactory
) -> tuple[ModelRegistry, WorkerDone]:
    done = trained_job.recorder.messages[-1]
    assert isinstance(done, WorkerDone) and done.weights_path is not None
    registry = ModelRegistry(tmp_path_factory.mktemp("models"))
    record = _record(GPU_MODEL_ID, balloon.class_names, trained_job.spec.hyperparameters)
    registry.register(record, Path(done.weights_path), [], None)
    return registry, done


@pytest.mark.gpu
@pytest.mark.needs_weights
def test_evaluate_reproduces_training_best_ap(
    trained_registry: tuple[ModelRegistry, WorkerDone], balloon: DatasetInfo
) -> None:
    registry, done = trained_registry
    service = _service(registry, DatasetCatalog(balloon.root.parent), "gpu")

    metrics = service.evaluate(GPU_MODEL_ID, balloon.name, "val")

    assert done.best_ap50_95 > 0
    assert metrics.stats.ap == pytest.approx(done.best_ap50_95, abs=1e-3)
    assert metrics.num_images == balloon.val_images
    assert [c.class_name for c in metrics.per_class] == balloon.class_names


@pytest.mark.gpu
@pytest.mark.needs_weights
def test_predict_returns_in_bounds_detections_sorted_by_score(
    trained_registry: tuple[ModelRegistry, WorkerDone], balloon: DatasetInfo
) -> None:
    import cv2

    registry, _ = trained_registry
    service = _service(registry, DatasetCatalog(balloon.root.parent), "gpu")
    gt = json.loads((balloon.root / "annotations" / "val.json").read_text(encoding="utf-8"))
    image_path = balloon.root / "val" / gt["images"][0]["file_name"]
    height, width = cv2.imread(str(image_path)).shape[:2]
    threshold = 0.05

    result = service.predict(GPU_MODEL_ID, image_path.read_bytes(), threshold)

    assert (result.image_width, result.image_height) == (width, height)
    assert result.detections
    scores = [d.score for d in result.detections]
    assert scores == sorted(scores, reverse=True)
    for d in result.detections:
        assert 0 <= d.x1 <= d.x2 <= width and 0 <= d.y1 <= d.y2 <= height
        assert d.score >= threshold
        assert d.class_name == balloon.class_names[d.class_id]
