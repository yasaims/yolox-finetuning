import os
from pathlib import Path

import pytest
from training_run import RUN_TIMEOUT, Recorder, TrainedJob, make_spec

from yolox_trainer.core.datasets import DatasetCatalog, DatasetInfo
from yolox_trainer.core.errors import NotFound

DATASET = "balloon"


@pytest.fixture(scope="session")
def balloon() -> DatasetInfo:
    root = Path(os.environ.get("TRAINER_DATASETS_DIR", "/data/datasets"))
    try:
        return DatasetCatalog(root).get(DATASET)
    except NotFound:
        pytest.skip(f"dataset {DATASET!r} not found under {root}")


@pytest.fixture(scope="session")
def trained_job(
    tmp_path_factory: pytest.TempPathFactory, balloon: DatasetInfo, pretrained_weights: Path
) -> TrainedJob:
    """One 2-epoch training run through ProcessTrainingRunner, shared by the session."""
    from yolox_trainer.engine.process_runner import ProcessTrainingRunner

    spec = make_spec(tmp_path_factory.mktemp("trained"), balloon, pretrained_weights)
    recorder = Recorder()
    ProcessTrainingRunner().start(spec, recorder.on_message, recorder.on_exit)
    assert recorder.exited.wait(RUN_TIMEOUT), "training did not finish in time"
    return TrainedJob(spec, recorder)
