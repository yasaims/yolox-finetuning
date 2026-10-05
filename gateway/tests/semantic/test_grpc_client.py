from collections.abc import AsyncIterator
from typing import Any

import grpc
import pytest

from gateway.trainer_client import GrpcTrainerClient, TrainerError
from yoloxft.v1 import trainer_pb2 as pb
from yoloxft.v1 import trainer_pb2_grpc as pb_grpc


class FakeServicer(pb_grpc.TrainerServiceServicer):
    def __init__(self) -> None:
        self.start_request: Any = None
        self.predict_request: Any = None

    async def ListDatasets(self, request: Any, context: Any) -> Any:
        return pb.ListDatasetsResponse(
            datasets=[pb.Dataset(name="balloon", class_names=["balloon"], train_images=40, val_images=10)]
        )

    async def StartTraining(self, request: Any, context: Any) -> Any:
        self.start_request = request
        resolved = pb.Hyperparameters(max_epoch=request.hyperparameters.max_epoch, fp16=True, seed=0)
        return pb.StartTrainingResponse(job_id="j1", resolved=resolved)

    async def GetCurrentJob(self, request: Any, context: Any) -> Any:
        return pb.GetCurrentJobResponse()

    async def GetModel(self, request: Any, context: Any) -> Any:
        if request.model_id != "m1":
            await context.abort(grpc.StatusCode.NOT_FOUND, f"model not found: {request.model_id}")
        return pb.Model(
            id="m1",
            display_name="First",
            status="completed",
            dataset_name="balloon",
            class_names=["balloon"],
            hyperparameters=pb.Hyperparameters(max_epoch=5),
            best_ap50=0.5,
            created_unix_ms=7,
            eval_history=[pb.EpochEval(epoch=1, ap50=0.5, ap50_95=0.3, best_ap50_95=0.3)],
        )

    async def Predict(self, request: Any, context: Any) -> Any:
        self.predict_request = request
        return pb.PredictResponse(
            image_width=4,
            image_height=3,
            detections=[pb.Detection(x1=1, y1=2, x2=3, y2=4, score=0.5, class_id=0, class_name="balloon")],
        )

    async def StreamTrainingEvents(self, request: Any, context: Any) -> AsyncIterator[Any]:
        progress = pb.IterProgress(
            epoch=1,
            iter=2,
            global_iter=2,
            total_iters=9,
            lr=0.1,
            losses={"total_loss": float("inf")},
            input_size=416,
            eta_seconds=3.0,
        )
        yield pb.TrainingEvent(job_id="j1", seq=6, time_unix_ms=10, iter=progress)
        finished = pb.JobFinished(state=pb.JOB_STATE_SUCCEEDED, model_id="m1")
        yield pb.TrainingEvent(job_id="j1", seq=7, time_unix_ms=11, finished=finished)


@pytest.fixture
async def served() -> AsyncIterator[tuple[GrpcTrainerClient, FakeServicer]]:
    servicer = FakeServicer()
    server = grpc.aio.server()
    pb_grpc.add_TrainerServiceServicer_to_server(servicer, server)
    port = server.add_insecure_port("127.0.0.1:0")
    await server.start()
    client = GrpcTrainerClient(f"127.0.0.1:{port}")
    yield client, servicer
    await client.close()
    await server.stop(None)


async def test_converts_datasets_and_current_job(served: tuple[GrpcTrainerClient, FakeServicer]) -> None:
    client, _ = served
    [dataset] = await client.list_datasets()
    assert (dataset.name, dataset.class_names, dataset.train_images, dataset.val_images) == (
        "balloon",
        ["balloon"],
        40,
        10,
    )
    assert await client.get_current_job() is None


async def test_start_training_sets_only_given_fields(served: tuple[GrpcTrainerClient, FakeServicer]) -> None:
    client, servicer = served
    result = await client.start_training("balloon", {"max_epoch": 3, "fp16": False}, "My run")
    request = servicer.start_request
    assert request.dataset_name == "balloon"
    assert request.display_name == "My run"
    assert request.hyperparameters.HasField("max_epoch")
    assert request.hyperparameters.HasField("fp16")
    assert not request.hyperparameters.HasField("batch_size")
    assert result.job_id == "j1"
    assert result.hyperparameters.max_epoch == 3
    assert result.hyperparameters.batch_size is None


async def test_get_model_converts_and_maps_not_found(served: tuple[GrpcTrainerClient, FakeServicer]) -> None:
    client, _ = served
    model = await client.get_model("m1")
    assert model.display_name == "First"
    assert model.hyperparameters.max_epoch == 5
    assert model.eval_history[0].ap50_95 == pytest.approx(0.3)
    with pytest.raises(TrainerError) as excinfo:
        await client.get_model("nope")
    assert excinfo.value.code == grpc.StatusCode.NOT_FOUND
    assert "nope" in excinfo.value.message


async def test_predict_sends_image_and_threshold(served: tuple[GrpcTrainerClient, FakeServicer]) -> None:
    client, servicer = served
    result = await client.predict("m1", b"img", 0.25)
    assert servicer.predict_request.image == b"img"
    assert servicer.predict_request.score_threshold == pytest.approx(0.25)
    assert (result.image_width, result.image_height) == (4, 3)
    assert result.detections[0].class_name == "balloon"


async def test_stream_events_yields_typed_payload_dicts(served: tuple[GrpcTrainerClient, FakeServicer]) -> None:
    client, _ = served
    events = [e async for e in client.stream_events("j1", 5)]
    assert [(kind, seq) for kind, seq, _ in events] == [("iter", 6), ("finished", 7)]
    iter_payload, finished_payload = events[0][2], events[1][2]
    assert iter_payload["seq"] == 6
    assert iter_payload["time_unix_ms"] == 10
    assert iter_payload["losses"]["total_loss"] == float("inf")
    assert finished_payload["state"] == "SUCCEEDED"
    assert finished_payload["model_id"] == "m1"
