from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Schema(BaseModel):
    """Non-finite floats serialize as null so every response parses with JSON.parse."""

    model_config = ConfigDict(ser_json_inf_nan="null")


class Hyperparameters(Schema):
    model_config = ConfigDict(extra="forbid")

    max_epoch: int | None = None
    batch_size: int | None = None
    learning_rate: float | None = None
    input_size: int | None = None
    warmup_epochs: int | None = None
    no_aug_epochs: int | None = None
    mosaic_prob: float | None = None
    enable_mixup: bool | None = None
    fp16: bool | None = None
    seed: int | None = None


class Dataset(Schema):
    name: str
    class_names: list[str]
    train_images: int
    val_images: int


class StartTrainingRequest(Schema):
    dataset_name: str = Field(min_length=1)
    display_name: str = ""
    hyperparameters: Hyperparameters = Field(default_factory=Hyperparameters)


class StartTrainingResult(Schema):
    job_id: str
    hyperparameters: Hyperparameters


class JobSummary(Schema):
    job_id: str
    state: str
    dataset_name: str
    hyperparameters: Hyperparameters
    last_seq: int
    started_unix_ms: int


class CurrentJobResponse(Schema):
    job: JobSummary | None


class EpochEval(Schema):
    epoch: int
    ap50: float
    ap50_95: float
    best_ap50_95: float


class Model(Schema):
    id: str
    display_name: str
    status: str
    dataset_name: str
    class_names: list[str]
    hyperparameters: Hyperparameters
    epochs_trained: int
    best_epoch: int
    best_ap50: float
    best_ap50_95: float
    created_unix_ms: int
    eval_history: list[EpochEval]


class EvaluateRequest(Schema):
    dataset_name: str | None = None
    split: Literal["val", "train"] = "val"


class CocoStats(Schema):
    ap: float
    ap50: float
    ap75: float
    ap_small: float
    ap_medium: float
    ap_large: float
    ar1: float
    ar10: float
    ar100: float
    ar_small: float
    ar_medium: float
    ar_large: float


class ClassAp(Schema):
    class_name: str
    ap: float
    ap50: float


class EvaluationResult(Schema):
    stats: CocoStats
    per_class: list[ClassAp]
    num_images: int


class Detection(Schema):
    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    class_id: int
    class_name: str


class PredictResponse(Schema):
    image_width: int
    image_height: int
    detections: list[Detection]
