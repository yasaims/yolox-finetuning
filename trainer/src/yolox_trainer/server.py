from __future__ import annotations

import importlib.util
import logging
import signal
import threading
from collections.abc import Callable
from concurrent import futures
from types import FrameType

import grpc
from grpc_health.v1 import health, health_pb2, health_pb2_grpc

from yoloxft.v1 import trainer_pb2, trainer_pb2_grpc

from .config import TrainerConfig
from .core.coco_metrics import EvaluationMetrics
from .core.datasets import DatasetCatalog
from .core.errors import FailedPrecondition
from .core.inference import InferenceService, PredictResult
from .core.registry import ModelRegistry
from .core.supervisor import JobSpec, TrainingRunner, TrainingSupervisor
from .grpc_api.servicer import TrainerServicer

log = logging.getLogger(__name__)

MAX_MESSAGE_BYTES = 32 * 1024 * 1024
TRAINING_SHUTDOWN_SECONDS = 30
SERVER_STOP_GRACE_SECONDS = 5
ENGINE_PACKAGES = ("torch", "yolox")


def build_server(servicer: TrainerServicer, port: int, max_workers: int = 16) -> grpc.Server:
    """Creates a gRPC server bound to `port` (not started) with the trainer and health services."""
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=max_workers),
        options=[
            ("grpc.max_send_message_length", MAX_MESSAGE_BYTES),
            ("grpc.max_receive_message_length", MAX_MESSAGE_BYTES),
            ("grpc.so_reuseport", 0),
        ],
    )
    trainer_pb2_grpc.add_TrainerServiceServicer_to_server(servicer, server)
    health_servicer = health.HealthServicer()
    health_pb2_grpc.add_HealthServicer_to_server(health_servicer, server)
    service_name = trainer_pb2.DESCRIPTOR.services_by_name["TrainerService"].full_name
    for name in ("", service_name):
        health_servicer.set(name, health_pb2.HealthCheckResponse.SERVING)
    server.add_insecure_port(f"[::]:{port}")
    return server


class _UnavailableRunner:
    def start(self, spec: JobSpec, on_message: Callable[[object], None], on_exit: Callable[[int | None], None]) -> None:
        raise FailedPrecondition("engine unavailable")

    def request_stop(self) -> None:
        pass


class _UnavailableInference:
    def evaluate(self, model_id: str, dataset_name: str, split: str) -> EvaluationMetrics:
        raise FailedPrecondition("engine unavailable")

    def predict(self, model_id: str, image: bytes, score_threshold: float) -> PredictResult:
        raise FailedPrecondition("engine unavailable")

    def forget(self, model_id: str) -> None:
        pass


def _engine_installed() -> bool:
    return all(importlib.util.find_spec(name) is not None for name in ENGINE_PACKAGES)


def _build_engine(
    config: TrainerConfig, registry: ModelRegistry, catalog: DatasetCatalog
) -> tuple[Callable[[], TrainingRunner], InferenceService]:
    """Builds the torch-based engine; without torch/YOLOX installed, training and inference
    requests fail with FAILED_PRECONDITION."""
    if not _engine_installed():
        log.warning("torch/yolox not installed; training and inference are disabled")
        return _UnavailableRunner, _UnavailableInference()
    from .engine import build_runtime

    return build_runtime(config, registry, catalog)


def serve(config: TrainerConfig) -> None:
    registry = ModelRegistry(config.models_dir)
    registry.cleanup()
    catalog = DatasetCatalog(config.datasets_dir)
    runner_factory, inference = _build_engine(config, registry, catalog)
    supervisor = TrainingSupervisor(runner_factory, registry, catalog, config.pretrained)
    server = build_server(TrainerServicer(supervisor, registry, catalog, inference), config.port)

    stopping = threading.Event()

    def request_shutdown(signum: int, frame: FrameType | None) -> None:
        stopping.set()

    signal.signal(signal.SIGTERM, request_shutdown)
    signal.signal(signal.SIGINT, request_shutdown)

    server.start()
    log.info("trainer listening on port %d", config.port)
    stopping.wait()

    log.info("shutting down")
    supervisor.shutdown(TRAINING_SHUTDOWN_SECONDS)
    server.stop(SERVER_STOP_GRACE_SECONDS).wait()
