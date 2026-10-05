from __future__ import annotations

from collections.abc import AsyncGenerator, Awaitable, Callable, Mapping
from typing import Any, Protocol, TypeVar

import grpc

from yoloxft.v1 import trainer_pb2 as pb
from yoloxft.v1 import trainer_pb2_grpc as pb_grpc

from .schemas import (
    ClassAp,
    CocoStats,
    Dataset,
    Detection,
    EpochEval,
    EvaluationResult,
    Hyperparameters,
    JobSummary,
    Model,
    PredictResponse,
    StartTrainingResult,
)

MAX_MESSAGE_BYTES = 32 << 20
_STATE_PREFIX = "JOB_STATE_"

T = TypeVar("T")
EventTuple = tuple[str, int, dict[str, Any]]


class TrainerError(Exception):
    def __init__(self, code: grpc.StatusCode, message: str) -> None:
        super().__init__(f"{code.name}: {message}")
        self.code = code
        self.message = message


class TrainerClient(Protocol):
    async def list_datasets(self) -> list[Dataset]: ...

    async def get_default_hyperparameters(self) -> Hyperparameters: ...

    async def start_training(
        self, dataset_name: str, hyperparameters: Mapping[str, Any], display_name: str
    ) -> StartTrainingResult: ...

    async def get_current_job(self) -> JobSummary | None: ...

    def stream_events(self, job_id: str, after_seq: int) -> AsyncGenerator[EventTuple, None]:
        """Yields (event_type, seq, payload); payload includes `seq` and `time_unix_ms`."""
        ...

    async def stop_training(self, job_id: str) -> None: ...

    async def list_models(self) -> list[Model]: ...

    async def get_model(self, model_id: str) -> Model: ...

    async def delete_model(self, model_id: str) -> None: ...

    async def evaluate_model(self, model_id: str, dataset_name: str | None, split: str) -> EvaluationResult: ...

    async def predict(self, model_id: str, image: bytes, score_threshold: float) -> PredictResponse: ...

    async def close(self) -> None: ...


def _state_name(state: int) -> str:
    return pb.JobState.Name(state).removeprefix(_STATE_PREFIX)


def _hyperparameters(message: Any) -> Hyperparameters:
    return Hyperparameters(
        **{name: getattr(message, name) for name in Hyperparameters.model_fields if message.HasField(name)}
    )


def _epoch_eval(message: Any) -> EpochEval:
    return EpochEval(
        epoch=message.epoch, ap50=message.ap50, ap50_95=message.ap50_95, best_ap50_95=message.best_ap50_95
    )


def _job_summary(message: Any) -> JobSummary:
    return JobSummary(
        job_id=message.job_id,
        state=_state_name(message.state),
        dataset_name=message.dataset_name,
        hyperparameters=_hyperparameters(message.hyperparameters),
        last_seq=message.last_seq,
        started_unix_ms=message.started_unix_ms,
    )


def _model(message: Any) -> Model:
    return Model(
        id=message.id,
        display_name=message.display_name,
        status=message.status,
        dataset_name=message.dataset_name,
        class_names=list(message.class_names),
        hyperparameters=_hyperparameters(message.hyperparameters),
        epochs_trained=message.epochs_trained,
        best_epoch=message.best_epoch,
        best_ap50=message.best_ap50,
        best_ap50_95=message.best_ap50_95,
        created_unix_ms=message.created_unix_ms,
        eval_history=[_epoch_eval(e) for e in message.eval_history],
    )


def _event_payload(message: Any) -> EventTuple:
    kind = message.WhichOneof("payload")
    body = getattr(message, kind)
    if kind == "started":
        data: dict[str, Any] = {
            "dataset_name": body.dataset_name,
            "hyperparameters": _hyperparameters(body.hyperparameters).model_dump(),
            "max_epoch": body.max_epoch,
            "iters_per_epoch": body.iters_per_epoch,
        }
    elif kind == "iter":
        data = {
            "epoch": body.epoch,
            "iter": body.iter,
            "global_iter": body.global_iter,
            "total_iters": body.total_iters,
            "lr": body.lr,
            "losses": dict(body.losses),
            "input_size": body.input_size,
            "eta_seconds": body.eta_seconds,
        }
    elif kind == "eval":
        data = _epoch_eval(body).model_dump()
    else:
        data = {"state": _state_name(body.state), "model_id": body.model_id, "error": body.error}
    data["seq"] = message.seq
    data["time_unix_ms"] = message.time_unix_ms
    return kind, message.seq, data


class GrpcTrainerClient:
    def __init__(self, addr: str) -> None:
        options = [
            ("grpc.max_send_message_length", MAX_MESSAGE_BYTES),
            ("grpc.max_receive_message_length", MAX_MESSAGE_BYTES),
        ]
        self.addr = addr
        self._channel = grpc.aio.insecure_channel(addr, options=options)
        self._stub = pb_grpc.TrainerServiceStub(self._channel)

    async def close(self) -> None:
        await self._channel.close()

    @staticmethod
    async def _unary(call: Callable[[], Awaitable[T]]) -> T:
        try:
            return await call()
        except grpc.aio.AioRpcError as e:
            raise TrainerError(e.code(), e.details() or "") from e

    async def list_datasets(self) -> list[Dataset]:
        reply = await self._unary(lambda: self._stub.ListDatasets(pb.ListDatasetsRequest()))
        return [
            Dataset(
                name=d.name,
                class_names=list(d.class_names),
                train_images=d.train_images,
                val_images=d.val_images,
            )
            for d in reply.datasets
        ]

    async def get_default_hyperparameters(self) -> Hyperparameters:
        reply = await self._unary(lambda: self._stub.GetDefaultHyperparameters(pb.GetDefaultHyperparametersRequest()))
        return _hyperparameters(reply)

    async def start_training(
        self, dataset_name: str, hyperparameters: Mapping[str, Any], display_name: str
    ) -> StartTrainingResult:
        request = pb.StartTrainingRequest(
            dataset_name=dataset_name,
            display_name=display_name,
            hyperparameters=pb.Hyperparameters(**hyperparameters),
        )
        reply = await self._unary(lambda: self._stub.StartTraining(request))
        return StartTrainingResult(job_id=reply.job_id, hyperparameters=_hyperparameters(reply.resolved))

    async def get_current_job(self) -> JobSummary | None:
        reply = await self._unary(lambda: self._stub.GetCurrentJob(pb.GetCurrentJobRequest()))
        return _job_summary(reply.job) if reply.HasField("job") else None

    async def stream_events(self, job_id: str, after_seq: int) -> AsyncGenerator[EventTuple, None]:
        call = self._stub.StreamTrainingEvents(pb.StreamTrainingEventsRequest(job_id=job_id, after_seq=after_seq))
        try:
            async for message in call:
                yield _event_payload(message)
        except grpc.aio.AioRpcError as e:
            raise TrainerError(e.code(), e.details() or "") from e
        finally:
            call.cancel()

    async def stop_training(self, job_id: str) -> None:
        await self._unary(lambda: self._stub.StopTraining(pb.StopTrainingRequest(job_id=job_id)))

    async def list_models(self) -> list[Model]:
        reply = await self._unary(lambda: self._stub.ListModels(pb.ListModelsRequest()))
        return [_model(m) for m in reply.models]

    async def get_model(self, model_id: str) -> Model:
        return _model(await self._unary(lambda: self._stub.GetModel(pb.GetModelRequest(model_id=model_id))))

    async def delete_model(self, model_id: str) -> None:
        await self._unary(lambda: self._stub.DeleteModel(pb.DeleteModelRequest(model_id=model_id)))

    async def evaluate_model(self, model_id: str, dataset_name: str | None, split: str) -> EvaluationResult:
        request = pb.EvaluateModelRequest(model_id=model_id, dataset_name=dataset_name or "", split=split)
        reply = await self._unary(lambda: self._stub.EvaluateModel(request))
        return EvaluationResult(
            stats=CocoStats(**{name: getattr(reply.stats, name) for name in CocoStats.model_fields}),
            per_class=[ClassAp(class_name=c.class_name, ap=c.ap, ap50=c.ap50) for c in reply.per_class],
            num_images=reply.num_images,
        )

    async def predict(self, model_id: str, image: bytes, score_threshold: float) -> PredictResponse:
        request = pb.PredictRequest(model_id=model_id, image=image, score_threshold=score_threshold)
        reply = await self._unary(lambda: self._stub.Predict(request))
        return PredictResponse(
            image_width=reply.image_width,
            image_height=reply.image_height,
            detections=[
                Detection(
                    x1=d.x1, y1=d.y1, x2=d.x2, y2=d.y2, score=d.score, class_id=d.class_id, class_name=d.class_name
                )
                for d in reply.detections
            ],
        )
