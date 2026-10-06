from __future__ import annotations

import dataclasses
from typing import Any

from yoloxft.v1 import trainer_pb2 as pb

from ..core.coco_metrics import EvaluationMetrics
from ..core.datasets import DatasetInfo
from ..core.events import EpochEval, Event, IterProgress, JobFinished, JobStarted, JobState
from ..core.hyperparams import Hyperparameters
from ..core.inference import PredictResult
from ..core.registry import ModelRecord
from ..core.supervisor import JobSummary

_HP_FIELDS = [f.name for f in dataclasses.fields(Hyperparameters)]


def hyperparameter_overrides(message: Any) -> dict[str, object]:
    """Only the fields explicitly set on the proto message."""
    return {name: getattr(message, name) for name in _HP_FIELDS if message.HasField(name)}


def hyperparameters_to_proto(hp: Hyperparameters) -> Any:
    return pb.Hyperparameters(**dataclasses.asdict(hp))


def job_state_to_proto(state: JobState) -> Any:
    return pb.JobState.Value(f"JOB_STATE_{state.name}")


def dataset_to_proto(info: DatasetInfo) -> Any:
    return pb.Dataset(
        name=info.name,
        class_names=info.class_names,
        train_images=info.train_images,
        val_images=info.val_images,
    )


def _eval_to_proto(e: EpochEval) -> Any:
    return pb.EpochEval(epoch=e.epoch, ap50=e.ap50, ap50_95=e.ap50_95, best_ap50_95=e.best_ap50_95)


def model_to_proto(record: ModelRecord) -> Any:
    return pb.Model(
        id=record.id,
        display_name=record.display_name,
        status=record.status,
        dataset_name=record.dataset_name,
        class_names=record.class_names,
        hyperparameters=hyperparameters_to_proto(record.hyperparameters),
        epochs_trained=record.epochs_trained,
        best_epoch=record.best_epoch,
        best_ap50=record.best_ap50,
        best_ap50_95=record.best_ap50_95,
        created_unix_ms=record.created_unix_ms,
        eval_history=[_eval_to_proto(e) for e in record.eval_history],
    )


def job_summary_to_proto(summary: JobSummary) -> Any:
    return pb.JobSummary(
        job_id=summary.job_id,
        state=job_state_to_proto(summary.state),
        dataset_name=summary.dataset_name,
        hyperparameters=hyperparameters_to_proto(summary.hyperparameters),
        last_seq=summary.last_seq,
        started_unix_ms=summary.started_unix_ms,
    )


def event_to_proto(event: Event) -> Any:
    message = pb.TrainingEvent(job_id=event.job_id, seq=event.seq, time_unix_ms=event.time_unix_ms)
    p = event.payload
    if isinstance(p, JobStarted):
        message.started.CopyFrom(
            pb.JobStarted(
                dataset_name=p.dataset_name,
                hyperparameters=hyperparameters_to_proto(p.hyperparameters),
                max_epoch=p.max_epoch,
                iters_per_epoch=p.iters_per_epoch,
            )
        )
    elif isinstance(p, IterProgress):
        message.iter.CopyFrom(
            pb.IterProgress(
                epoch=p.epoch,
                iter=p.iter,
                global_iter=p.global_iter,
                total_iters=p.total_iters,
                lr=p.lr,
                losses=p.losses,
                input_size=p.input_size,
                eta_seconds=p.eta_seconds,
            )
        )
    elif isinstance(p, EpochEval):
        message.eval.CopyFrom(_eval_to_proto(p))
    elif isinstance(p, JobFinished):
        message.finished.CopyFrom(
            pb.JobFinished(state=job_state_to_proto(p.state), model_id=p.model_id, error=p.error)
        )
    else:
        raise TypeError(f"unsupported event payload: {type(p).__name__}")
    return message


def evaluation_to_proto(metrics: EvaluationMetrics) -> Any:
    return pb.EvaluationResult(
        stats=pb.CocoStats(**dataclasses.asdict(metrics.stats)),
        per_class=[pb.ClassAp(class_name=c.class_name, ap=c.ap, ap50=c.ap50) for c in metrics.per_class],
        num_images=metrics.num_images,
    )


def predict_result_to_proto(result: PredictResult) -> Any:
    return pb.PredictResponse(
        image_width=result.image_width,
        image_height=result.image_height,
        detections=[pb.Detection(**dataclasses.asdict(d)) for d in result.detections],
    )
