from conftest import ClientFactory
from fakes import HP, FakeTrainerClient

from gateway.schemas import JobSummary


async def test_health(http: ClientFactory) -> None:
    response = await http(FakeTrainerClient()).get("/api/health")
    assert response.status_code == 200


async def test_list_datasets_and_defaults(http: ClientFactory) -> None:
    client = http(FakeTrainerClient())
    datasets = await client.get("/api/datasets")
    assert datasets.json() == [{"name": "balloon", "class_names": ["balloon"], "train_images": 40, "val_images": 10}]
    defaults = await client.get("/api/hyperparameters/defaults")
    assert defaults.json() == HP.model_dump()


async def test_start_training_forwards_only_given_hyperparameters(http: ClientFactory) -> None:
    fake = FakeTrainerClient()
    response = await http(fake).post(
        "/api/trainings",
        json={"dataset_name": "balloon", "hyperparameters": {"max_epoch": 3, "fp16": False, "seed": None}},
    )
    assert response.status_code == 201
    assert response.json() == {"job_id": "j1", "hyperparameters": HP.model_dump()}
    assert fake.calls == [("start_training", ("balloon", {"max_epoch": 3, "fp16": False}, ""))]


async def test_start_training_rejects_invalid_body(http: ClientFactory) -> None:
    client = http(FakeTrainerClient())
    assert (await client.post("/api/trainings", json={})).status_code == 422
    unknown = await client.post("/api/trainings", json={"dataset_name": "d", "hyperparameters": {"lr": 1}})
    assert unknown.status_code == 422
    wrong_type = await client.post("/api/trainings", json={"dataset_name": "d", "hyperparameters": {"max_epoch": "x"}})
    assert wrong_type.status_code == 422


async def test_current_job_is_null_when_idle(http: ClientFactory) -> None:
    fake = FakeTrainerClient()
    client = http(fake)
    assert (await client.get("/api/trainings/current")).json() == {"job": None}
    fake.current = JobSummary(
        job_id="j1", state="RUNNING", dataset_name="balloon", hyperparameters=HP, last_seq=3, started_unix_ms=5
    )
    job = (await client.get("/api/trainings/current")).json()["job"]
    assert job["job_id"] == "j1"
    assert job["state"] == "RUNNING"


async def test_stop_returns_202_and_delete_model_204(http: ClientFactory) -> None:
    fake = FakeTrainerClient()
    client = http(fake)
    assert (await client.post("/api/trainings/j1/stop")).status_code == 202
    assert (await client.delete("/api/models/m1")).status_code == 204
    assert [c[0] for c in fake.calls] == ["stop_training", "delete_model"]


async def test_models_list_and_get(http: ClientFactory) -> None:
    client = http(FakeTrainerClient())
    assert [m["id"] for m in (await client.get("/api/models")).json()] == ["m1"]
    assert (await client.get("/api/models/m1")).json()["class_names"] == ["balloon"]


async def test_non_finite_floats_serialize_as_null(http: ClientFactory) -> None:
    fake = FakeTrainerClient()
    fake.current = JobSummary(
        job_id="j1",
        state="RUNNING",
        dataset_name="balloon",
        hyperparameters=HP.model_copy(update={"learning_rate": float("inf")}),
        last_seq=1,
        started_unix_ms=1,
    )
    response = await http(fake).get("/api/trainings/current")
    assert response.json()["job"]["hyperparameters"]["learning_rate"] is None
