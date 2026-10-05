"""End-to-end smoke test against a running stack, through the gateway REST API."""

import argparse
import json
import sys
import time
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

STOP_DEADLINE_SECONDS = 60
BUSY_RETRY_SECONDS = 15


class SmokeFailure(Exception):
    pass


@dataclass(frozen=True)
class SseFrame:
    event: str
    id: str
    data: Any


def parse_sse(lines: Iterable[str]) -> Iterator[SseFrame]:
    """Parses SSE text lines into frames; comments and data-less frames are skipped."""
    event, frame_id, data = "message", "", []
    for line in lines:
        if line == "":
            if data:
                yield SseFrame(event, frame_id, json.loads("\n".join(data)))
            event, frame_id, data = "message", "", []
        elif line.startswith(":"):
            continue
        else:
            field, _, value = line.partition(":")
            value = value.removeprefix(" ")
            if field == "event":
                event = value
            elif field == "id":
                frame_id = value
            elif field == "data":
                data.append(value)


def check(condition: bool, message: str) -> None:
    if not condition:
        raise SmokeFailure(message)


def step(message: str) -> None:
    print(f"==> {message}", flush=True)


def start_training(client: httpx.Client, dataset: str, epochs: int, name: str) -> str:
    body = {
        "dataset_name": dataset,
        "display_name": name,
        "hyperparameters": {
            "max_epoch": epochs,
            "input_size": 320,
            "warmup_epochs": 1,
            "no_aug_epochs": 1,
        },
    }
    deadline = time.monotonic() + BUSY_RETRY_SECONDS
    while True:
        response = client.post("/api/trainings", json=body)
        if response.status_code == 409 and time.monotonic() < deadline:
            time.sleep(1)
            continue
        check(response.status_code == 201, f"start training: {response.status_code} {response.text}")
        return response.json()["job_id"]


@contextmanager
def open_events(client: httpx.Client, job_id: str, read_timeout: float) -> Iterator[Iterator[SseFrame]]:
    timeout = httpx.Timeout(read_timeout, connect=10)
    with client.stream("GET", f"/api/trainings/{job_id}/events", timeout=timeout) as response:
        check(response.status_code == 200, f"events: {response.status_code}")
        yield parse_sse(response.iter_lines())


def check_training(client: httpx.Client, dataset: str, epochs: int, timeout: float) -> str:
    """Runs a full training job and returns the registered model id."""
    step(f"training {dataset} for {epochs} epochs")
    job_id = start_training(client, dataset, epochs, "e2e-smoke")
    kinds: list[str] = []
    finished: dict[str, Any] | None = None
    deadline = time.monotonic() + timeout
    with open_events(client, job_id, read_timeout=60) as frames:
        for frame in frames:
            kinds.append(frame.event)
            if frame.event == "finished":
                finished = frame.data
                break
            check(time.monotonic() < deadline, f"training did not finish within {timeout:.0f} s")
    if finished is None:
        raise SmokeFailure("event stream ended without a finished event")
    check(kinds[0] == "started", f"first event was {kinds[0]!r}, expected 'started'")
    check("iter" in kinds, "no iter event received")
    check(kinds.count("eval") == epochs, f"expected {epochs} eval events, got {kinds.count('eval')}")
    check(finished["state"] == "SUCCEEDED", f"finished state {finished['state']}: {finished.get('error')}")
    check(bool(finished["model_id"]), "finished event has no model_id")
    return finished["model_id"]


def check_model(client: httpx.Client, model_id: str, dataset: str, epochs: int) -> None:
    step(f"model {model_id}")
    response = client.get(f"/api/models/{model_id}")
    check(response.status_code == 200, f"get model: {response.status_code}")
    model = response.json()
    check(model["status"] == "completed", f"model status {model['status']}")
    check(model["dataset_name"] == dataset, f"model dataset {model['dataset_name']}")
    check(len(model["eval_history"]) == epochs, f"eval_history has {len(model['eval_history'])} entries")
    check(any(m["id"] == model_id for m in client.get("/api/models").json()), "model missing from list")


def check_evaluate(client: httpx.Client, model_id: str) -> None:
    step("evaluate on val")
    response = client.post(f"/api/models/{model_id}/evaluate", json={"split": "val"})
    check(response.status_code == 200, f"evaluate: {response.status_code} {response.text}")
    result = response.json()
    ap50 = result["stats"]["ap50"]
    check(ap50 is not None and ap50 >= 0, "ap50 missing or negative")
    check(result["num_images"] > 0, "evaluation covered no images")
    print(f"    ap50={ap50:.3f} images={result['num_images']}")


def first_val_image(datasets_dir: Path, dataset: str) -> Path:
    root = datasets_dir / dataset
    images = json.loads((root / "annotations" / "val.json").read_text(encoding="utf-8"))["images"]
    check(bool(images), "val split has no images")
    return root / "val" / images[0]["file_name"]


def check_predict(client: httpx.Client, model_id: str, image: Path) -> None:
    step(f"predict on {image.name}")
    response = client.post(
        f"/api/models/{model_id}/predict",
        files={"file": (image.name, image.read_bytes(), "image/jpeg")},
        data={"score_threshold": "0.1"},
    )
    check(response.status_code == 200, f"predict: {response.status_code} {response.text}")
    result = response.json()
    check(result["image_width"] > 0 and result["image_height"] > 0, "invalid image size in response")
    print(f"    {len(result['detections'])} detections")


def check_cancel(client: httpx.Client, dataset: str, epochs: int) -> str:
    """Stops a running job after its first iter event. Returns the model id it registered, if any."""
    step("cancel a running job")
    job_id = start_training(client, dataset, epochs, "e2e-smoke-cancel")
    stop_deadline: float | None = None
    with open_events(client, job_id, read_timeout=STOP_DEADLINE_SECONDS) as frames:
        for frame in frames:
            if frame.event == "iter" and stop_deadline is None:
                response = client.post(f"/api/trainings/{job_id}/stop")
                check(response.status_code == 202, f"stop: {response.status_code}")
                stop_deadline = time.monotonic() + STOP_DEADLINE_SECONDS
            if frame.event == "finished":
                check(stop_deadline is not None, "job finished before the stop request")
                check(frame.data["state"] == "CANCELLED", f"cancelled job ended as {frame.data['state']}")
                return frame.data.get("model_id", "")
            if stop_deadline is not None:
                check(time.monotonic() < stop_deadline, f"no finished event within {STOP_DEADLINE_SECONDS} s of stop")
    raise SmokeFailure("event stream ended without a finished event")


def run(
    client: httpx.Client,
    dataset: str,
    epochs: int,
    datasets_dir: Path,
    timeout: float,
    created: list[str],
) -> None:
    step("datasets")
    response = client.get("/api/datasets")
    check(response.status_code == 200, f"list datasets: {response.status_code}")
    check(any(d["name"] == dataset for d in response.json()), f"dataset {dataset!r} not listed")

    model_id = check_training(client, dataset, epochs, timeout)
    created.append(model_id)
    check_model(client, model_id, dataset, epochs)
    check_evaluate(client, model_id)
    check_predict(client, model_id, first_val_image(datasets_dir, dataset))

    cancelled_model = check_cancel(client, dataset, epochs)
    if cancelled_model:
        created.append(cancelled_model)


def cleanup(client: httpx.Client, model_ids: list[str]) -> None:
    for model_id in model_ids:
        response = client.delete(f"/api/models/{model_id}")
        print(f"    delete {model_id}: {response.status_code}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="e2e-smoke")
    parser.add_argument("--base", default="http://localhost:8080")
    parser.add_argument("--dataset", default="balloon")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--datasets-dir", type=Path, default=Path("datasets"))
    parser.add_argument("--timeout", type=float, default=1800, help="seconds allowed for the training run")
    args = parser.parse_args()

    created: list[str] = []
    with httpx.Client(base_url=args.base, timeout=120) as client:
        try:
            run(client, args.dataset, args.epochs, args.datasets_dir, args.timeout, created)
        except (SmokeFailure, httpx.HTTPError, OSError) as error:
            print(f"FAILED: {error!r}", file=sys.stderr)
            sys.exit(1)
        finally:
            step("cleanup")
            cleanup(client, created)
    print("OK")


if __name__ == "__main__":
    main()
