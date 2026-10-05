from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import grpc

from yoloxft.v1 import trainer_pb2 as pb
from yoloxft.v1 import trainer_pb2_grpc as pb_grpc

from ..core.datasets import SPLITS, DatasetCatalog
from ..core.errors import FailedPrecondition, InvalidArgument, NotFound
from ..core.hyperparams import DEFAULTS
from ..core.inference import InferenceService
from ..core.registry import ModelRegistry
from ..core.supervisor import TrainingSupervisor
from . import convert

DEFAULT_SCORE_THRESHOLD = 0.3
DEFAULT_SPLIT = "val"
STREAM_POLL_SECONDS = 1.0

_CODES: tuple[tuple[type[Exception], grpc.StatusCode], ...] = (
    (NotFound, grpc.StatusCode.NOT_FOUND),
    (InvalidArgument, grpc.StatusCode.INVALID_ARGUMENT),
    (FailedPrecondition, grpc.StatusCode.FAILED_PRECONDITION),
)


@contextmanager
def _abort_on_domain_errors(context: grpc.ServicerContext) -> Iterator[None]:
    try:
        yield
    except (NotFound, InvalidArgument, FailedPrecondition) as e:
        code = next(code for exc, code in _CODES if isinstance(e, exc))
        context.abort(code, str(e))


class TrainerServicer(pb_grpc.TrainerServiceServicer):
    def __init__(
        self,
        supervisor: TrainingSupervisor,
        registry: ModelRegistry,
        catalog: DatasetCatalog,
        inference: InferenceService,
    ) -> None:
        self._supervisor = supervisor
        self._registry = registry
        self._catalog = catalog
        self._inference = inference

    def ListDatasets(self, request: Any, context: grpc.ServicerContext) -> Any:
        with _abort_on_domain_errors(context):
            return pb.ListDatasetsResponse(datasets=[convert.dataset_to_proto(d) for d in self._catalog.list()])

    def GetDefaultHyperparameters(self, request: Any, context: grpc.ServicerContext) -> Any:
        return convert.hyperparameters_to_proto(DEFAULTS)

    def StartTraining(self, request: Any, context: grpc.ServicerContext) -> Any:
        with _abort_on_domain_errors(context):
            overrides = convert.hyperparameter_overrides(request.hyperparameters)
            job_id, resolved = self._supervisor.start(request.dataset_name, overrides, request.display_name)
            return pb.StartTrainingResponse(job_id=job_id, resolved=convert.hyperparameters_to_proto(resolved))

    def GetCurrentJob(self, request: Any, context: grpc.ServicerContext) -> Any:
        summary = self._supervisor.current()
        if summary is None:
            return pb.GetCurrentJobResponse()
        return pb.GetCurrentJobResponse(job=convert.job_summary_to_proto(summary))

    def StreamTrainingEvents(self, request: Any, context: grpc.ServicerContext) -> Iterator[Any]:
        with _abort_on_domain_errors(context):
            log = self._supervisor.event_log(request.job_id)
        last_seq = max(request.after_seq, 0)
        while context.is_active():
            events, drained = log.read_after(last_seq, STREAM_POLL_SECONDS)
            for event in events:
                yield convert.event_to_proto(event)
                last_seq = event.seq
            if drained:
                return

    def StopTraining(self, request: Any, context: grpc.ServicerContext) -> Any:
        with _abort_on_domain_errors(context):
            self._supervisor.stop(request.job_id)
            return pb.StopTrainingResponse()

    def ListModels(self, request: Any, context: grpc.ServicerContext) -> Any:
        with _abort_on_domain_errors(context):
            return pb.ListModelsResponse(models=[convert.model_to_proto(m) for m in self._registry.list()])

    def GetModel(self, request: Any, context: grpc.ServicerContext) -> Any:
        with _abort_on_domain_errors(context):
            return convert.model_to_proto(self._registry.get(request.model_id))

    def DeleteModel(self, request: Any, context: grpc.ServicerContext) -> Any:
        with _abort_on_domain_errors(context):
            self._registry.delete(request.model_id)
            self._inference.forget(request.model_id)
            return pb.DeleteModelResponse()

    def EvaluateModel(self, request: Any, context: grpc.ServicerContext) -> Any:
        with _abort_on_domain_errors(context):
            model = self._registry.get(request.model_id)
            split = request.split or DEFAULT_SPLIT
            if split not in SPLITS:
                raise InvalidArgument(f"invalid split: {split!r}")
            dataset = self._catalog.get(request.dataset_name or model.dataset_name)
            if dataset.class_names != model.class_names:
                raise FailedPrecondition(
                    f"class names of dataset {dataset.name} {dataset.class_names} "
                    f"do not match model {model.id} {model.class_names}"
                )
            metrics = self._inference.evaluate(model.id, dataset.name, split)
            return convert.evaluation_to_proto(metrics)

    def Predict(self, request: Any, context: grpc.ServicerContext) -> Any:
        with _abort_on_domain_errors(context):
            if not request.image:
                raise InvalidArgument("image must not be empty")
            threshold = (
                request.score_threshold if request.HasField("score_threshold") else DEFAULT_SCORE_THRESHOLD
            )
            result = self._inference.predict(request.model_id, request.image, threshold)
            return convert.predict_result_to_proto(result)
