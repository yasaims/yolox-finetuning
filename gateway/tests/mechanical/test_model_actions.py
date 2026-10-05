from conftest import ClientFactory
from fakes import FakeTrainerClient

MAX_IMAGE_BYTES = 20 << 20


async def test_evaluate_defaults_to_val_split_and_model_dataset(http: ClientFactory) -> None:
    fake = FakeTrainerClient()
    response = await http(fake).post("/api/models/m1/evaluate", json={})
    assert response.status_code == 200
    assert response.json()["num_images"] == 10
    assert response.json()["per_class"] == [{"class_name": "balloon", "ap": 0.4, "ap50": 0.6}]
    assert fake.calls == [("evaluate_model", ("m1", None, "val"))]


async def test_evaluate_forwards_dataset_and_split(http: ClientFactory) -> None:
    fake = FakeTrainerClient()
    await http(fake).post("/api/models/m1/evaluate", json={"dataset_name": "other", "split": "train"})
    assert fake.calls == [("evaluate_model", ("m1", "other", "train"))]


async def test_evaluate_rejects_unknown_split(http: ClientFactory) -> None:
    response = await http(FakeTrainerClient()).post("/api/models/m1/evaluate", json={"split": "test"})
    assert response.status_code == 422


async def test_predict_forwards_bytes_and_threshold(http: ClientFactory) -> None:
    fake = FakeTrainerClient()
    response = await http(fake).post(
        "/api/models/m1/predict", files={"file": ("a.jpg", b"jpeg-bytes")}, data={"score_threshold": "0.55"}
    )
    assert response.status_code == 200
    assert response.json() == {"image_width": 2, "image_height": 1, "detections": []}
    assert fake.calls == [("predict", ("m1", b"jpeg-bytes", 0.55))]


async def test_predict_default_threshold(http: ClientFactory) -> None:
    fake = FakeTrainerClient()
    await http(fake).post("/api/models/m1/predict", files={"file": ("a.jpg", b"x")})
    assert fake.calls == [("predict", ("m1", b"x", 0.3))]


async def test_predict_rejects_out_of_range_threshold_and_missing_file(http: ClientFactory) -> None:
    client = http(FakeTrainerClient())
    too_high = await client.post(
        "/api/models/m1/predict", files={"file": ("a.jpg", b"x")}, data={"score_threshold": "1.5"}
    )
    assert too_high.status_code == 422
    assert (await client.post("/api/models/m1/predict", data={"score_threshold": "0.5"})).status_code == 422


async def test_predict_rejects_oversize_image(http: ClientFactory) -> None:
    fake = FakeTrainerClient()
    response = await http(fake).post(
        "/api/models/m1/predict", files={"file": ("big.jpg", b"0" * (MAX_IMAGE_BYTES + 1))}
    )
    assert response.status_code == 413
    assert fake.calls == []
