from __future__ import annotations

import dataclasses
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from .errors import NotFound
from .events import EpochEval, Event, event_to_dict
from .hyperparams import Hyperparameters
from .ids import validate_id

SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ModelRecord:
    id: str
    display_name: str
    status: Literal["completed", "stopped"]
    created_unix_ms: int
    finished_unix_ms: int
    base_model: str
    dataset_name: str
    class_names: list[str]
    hyperparameters: Hyperparameters
    epochs_trained: int
    best_epoch: int
    best_ap50: float
    best_ap50_95: float
    eval_history: list[EpochEval]


def _record_to_json(record: ModelRecord) -> dict[str, Any]:
    data = dataclasses.asdict(record)
    data["schema_version"] = SCHEMA_VERSION
    return data


def _record_from_json(data: dict[str, Any]) -> ModelRecord:
    data = {k: v for k, v in data.items() if k != "schema_version"}
    data["hyperparameters"] = Hyperparameters(**data["hyperparameters"])
    data["eval_history"] = [EpochEval(**e) for e in data["eval_history"]]
    return ModelRecord(**data)


class ModelRegistry:
    def __init__(self, root: Path) -> None:
        self._root = Path(root)

    def work_dir(self, model_id: str) -> Path:
        return self._root / ".work" / validate_id(model_id)

    def register(
        self,
        record: ModelRecord,
        weights_src: Path,
        events: list[Event],
        log_src: Path | None,
    ) -> ModelRecord:
        model_id = validate_id(record.id)
        self._root.mkdir(parents=True, exist_ok=True)
        staging = self._root / f".staging-{model_id}"
        shutil.rmtree(staging, ignore_errors=True)
        staging.mkdir()
        try:
            shutil.copyfile(weights_src, staging / "weights.pth")
            if log_src is not None:
                shutil.copyfile(log_src, staging / "train_log.txt")
            with open(staging / "events.jsonl", "w", encoding="utf-8") as f:
                for event in events:
                    f.write(json.dumps(event_to_dict(event)) + "\n")
            (staging / "model.json").write_text(
                json.dumps(_record_to_json(record), indent=2), encoding="utf-8"
            )
            os.replace(staging, self._root / model_id)
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        shutil.rmtree(self.work_dir(model_id), ignore_errors=True)
        return record

    def _read(self, model_dir: Path) -> ModelRecord | None:
        try:
            data = json.loads((model_dir / "model.json").read_text(encoding="utf-8"))
            return _record_from_json(data)
        except (OSError, ValueError, KeyError, TypeError):
            return None

    def list(self) -> list[ModelRecord]:
        if not self._root.is_dir():
            return []
        records = [
            record
            for entry in self._root.iterdir()
            if not entry.name.startswith(".") and entry.is_dir()
            for record in [self._read(entry)]
            if record is not None
        ]
        return sorted(records, key=lambda r: r.created_unix_ms, reverse=True)

    def get(self, model_id: str) -> ModelRecord:
        record = self._read(self._root / validate_id(model_id))
        if record is None:
            raise NotFound(f"model not found: {model_id}")
        return record

    def delete(self, model_id: str) -> None:
        model_dir = self._root / validate_id(model_id)
        if not model_dir.is_dir():
            raise NotFound(f"model not found: {model_id}")
        shutil.rmtree(model_dir)

    def weights_path(self, model_id: str) -> Path:
        path = self._root / validate_id(model_id) / "weights.pth"
        if not path.is_file():
            raise NotFound(f"model not found: {model_id}")
        return path

    def cleanup(self) -> None:
        shutil.rmtree(self._root / ".work", ignore_errors=True)
        if not self._root.is_dir():
            return
        for entry in self._root.glob(".staging-*"):
            shutil.rmtree(entry, ignore_errors=True)
