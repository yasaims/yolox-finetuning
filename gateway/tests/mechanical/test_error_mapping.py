import grpc
import pytest
from conftest import ClientFactory
from fakes import FakeTrainerClient


@pytest.mark.parametrize(
    ("code", "status"),
    [
        (grpc.StatusCode.NOT_FOUND, 404),
        (grpc.StatusCode.FAILED_PRECONDITION, 409),
        (grpc.StatusCode.INVALID_ARGUMENT, 422),
        (grpc.StatusCode.UNAVAILABLE, 503),
        (grpc.StatusCode.INTERNAL, 502),
        (grpc.StatusCode.DEADLINE_EXCEEDED, 502),
    ],
)
async def test_trainer_error_maps_to_http_status(http: ClientFactory, code: grpc.StatusCode, status: int) -> None:
    response = await http(FakeTrainerClient(error=code)).get("/api/datasets")
    assert response.status_code == status
    assert response.json() == {"detail": "boom"}
