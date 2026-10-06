import json
import socket
from collections.abc import Iterator
from pathlib import Path

import grpc
import pytest
from fakes import FakeInferenceService, FakeTrainingRunner

from yolox_trainer.core.datasets import DatasetCatalog
from yolox_trainer.core.events import EpochEval, IterProgress, WorkerDone, WorkerStarted
from yolox_trainer.core.hyperparams import DEFAULTS
from yolox_trainer.core.registry import ModelRecord, ModelRegistry
from yolox_trainer.core.supervisor import TrainingSupervisor
from yolox_trainer.grpc_api.servicer import TrainerServicer
from yolox_trainer.server import build_server
from yoloxft.v1 import trainer_pb2 as pb
from yoloxft.v1 import trainer_pb2_grpc as pb_grpc

SCRIPT = [
    WorkerStarted(1, 2),
    IterProgress(1, 1, 1, 2, 0.001, {"total_loss": 3.0}, 416, 10.0),
    EpochEval(1, 0.6, 0.4, 0.4),
    WorkerDone("succeeded", "weights.pth", 1, 0.6, 0.4, 1, ""),
]
WORK_FILES = {"weights.pth": "weights"}
MAX_MESSAGE = 32 << 20


class Env:
    def __init__(self, tmp_path: Path) -> None:
        for name, classes in (("ds", ["cat"]), ("other", ["dog"])):
            ann = tmp_path / "datasets" / name / "annotations"
            ann.mkdir(parents=True)
            cats = [{"id": i + 1, "name": c} for i, c in enumerate(classes)]
            for split in ("train", "val"):
                (ann / f"{split}.json").write_text(json.dumps({"images": [{"id": 1}], "categories": cats}))
        self.tmp_path = tmp_path
        pretrained = tmp_path / "yolox_tiny.pth"
        pretrained.write_bytes(b"pretrained")
        self.registry = ModelRegistry(tmp_path / "models")
        self.catalog = DatasetCatalog(tmp_path / "datasets")
        self.inference = FakeInferenceService()
        self.runner = FakeTrainingRunner(SCRIPT, work_files=WORK_FILES)
        self.supervisor = TrainingSupervisor(lambda: self.runner, self.registry, self.catalog, pretrained)

    def add_model(self, model_id: str = "m1", dataset: str = "ds") -> None:
        weights = self.tmp_path / "w.pth"
        weights.write_bytes(b"w")
        record = ModelRecord(
            model_id, model_id, "completed", 1, 2, "yolox_tiny", dataset, ["cat"], DEFAULTS, 1, 1, 0.6, 0.4, []
        )
        self.registry.register(record, weights, [], None)


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def env(tmp_path: Path) -> Env:
    return Env(tmp_path)


@pytest.fixture
def stub(env: Env) -> Iterator[pb_grpc.TrainerServiceStub]:
    port = free_port()
    server = build_server(TrainerServicer(env.supervisor, env.registry, env.catalog, env.inference), port)
    server.start()
    options = [("grpc.max_send_message_length", MAX_MESSAGE), ("grpc.max_receive_message_length", MAX_MESSAGE)]
    with grpc.insecure_channel(f"127.0.0.1:{port}", options=options) as channel:
        yield pb_grpc.TrainerServiceStub(channel)
    env.supervisor.shutdown(2)
    server.stop(0).wait()


def code_of(excinfo: pytest.ExceptionInfo[grpc.RpcError]) -> grpc.StatusCode:
    return excinfo.value.code()  # type: ignore[no-any-return,attr-defined]


def test_list_datasets(stub: pb_grpc.TrainerServiceStub) -> None:
    response = stub.ListDatasets(pb.ListDatasetsRequest())
    assert [(d.name, list(d.class_names), d.train_images, d.val_images) for d in response.datasets] == [
        ("ds", ["cat"], 1, 1),
        ("other", ["dog"], 1, 1),
    ]


def test_start_training_resolves_only_set_overrides(stub: pb_grpc.TrainerServiceStub) -> None:
    request = pb.StartTrainingRequest(dataset_name="ds", hyperparameters=pb.Hyperparameters(max_epoch=20))
    resolved = stub.StartTraining(request).resolved
    assert resolved.max_epoch == 20
    assert resolved.batch_size == DEFAULTS.batch_size
    assert resolved.fp16 is True


def test_start_training_while_busy_is_failed_precondition(env: Env, stub: pb_grpc.TrainerServiceStub) -> None:
    env.runner.script = []
    env.runner.wait_for_stop = True
    stub.StartTraining(pb.StartTrainingRequest(dataset_name="ds"))
    with pytest.raises(grpc.RpcError) as busy:
        stub.StartTraining(pb.StartTrainingRequest(dataset_name="ds"))
    assert code_of(busy) == grpc.StatusCode.FAILED_PRECONDITION


def test_get_unknown_model_is_not_found(stub: pb_grpc.TrainerServiceStub) -> None:
    with pytest.raises(grpc.RpcError) as missing:
        stub.GetModel(pb.GetModelRequest(model_id="nope"))
    assert code_of(missing) == grpc.StatusCode.NOT_FOUND


def test_stream_delivers_events_in_order_and_replays_from_after_seq(stub: pb_grpc.TrainerServiceStub) -> None:
    job_id = stub.StartTraining(pb.StartTrainingRequest(dataset_name="ds")).job_id

    events = list(stub.StreamTrainingEvents(pb.StreamTrainingEventsRequest(job_id=job_id)))
    assert [e.seq for e in events] == [1, 2, 3, 4]
    assert [e.WhichOneof("payload") for e in events] == ["started", "iter", "eval", "finished"]
    assert events[-1].finished.state == pb.JOB_STATE_SUCCEEDED
    assert events[-1].finished.model_id == job_id

    replay = list(stub.StreamTrainingEvents(pb.StreamTrainingEventsRequest(job_id=job_id, after_seq=2)))
    assert [e.seq for e in replay] == [3, 4]


def test_evaluate_defaults_to_model_dataset_and_val_split(env: Env, stub: pb_grpc.TrainerServiceStub) -> None:
    env.add_model()
    result = stub.EvaluateModel(pb.EvaluateModelRequest(model_id="m1"))
    assert env.inference.evaluate_calls == [("m1", "ds", "val")]
    assert result.num_images == 3


def test_evaluate_rejects_bad_split_and_class_mismatch(env: Env, stub: pb_grpc.TrainerServiceStub) -> None:
    env.add_model()
    with pytest.raises(grpc.RpcError) as bad_split:
        stub.EvaluateModel(pb.EvaluateModelRequest(model_id="m1", split="test"))
    assert code_of(bad_split) == grpc.StatusCode.INVALID_ARGUMENT
    with pytest.raises(grpc.RpcError) as mismatch:
        stub.EvaluateModel(pb.EvaluateModelRequest(model_id="m1", dataset_name="other"))
    assert code_of(mismatch) == grpc.StatusCode.FAILED_PRECONDITION
    assert env.inference.evaluate_calls == []


def test_delete_model_forgets_cached_inference_state(env: Env, stub: pb_grpc.TrainerServiceStub) -> None:
    env.add_model()
    stub.DeleteModel(pb.DeleteModelRequest(model_id="m1"))
    assert env.inference.forgotten == ["m1"]
    assert env.registry.list() == []


def test_predict_accepts_5mb_image_and_defaults_threshold(env: Env, stub: pb_grpc.TrainerServiceStub) -> None:
    response = stub.Predict(pb.PredictRequest(model_id="m1", image=b"x" * (5 << 20)))
    assert env.inference.predict_calls == [("m1", 5 << 20, pytest.approx(0.3))]
    assert (response.image_width, response.image_height) == (640, 480)
    assert response.detections[0].class_name == "cat"


def test_predict_rejects_empty_image(stub: pb_grpc.TrainerServiceStub) -> None:
    with pytest.raises(grpc.RpcError) as empty:
        stub.Predict(pb.PredictRequest(model_id="m1"))
    assert code_of(empty) == grpc.StatusCode.INVALID_ARGUMENT
