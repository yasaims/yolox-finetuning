from collections.abc import AsyncIterator, Callable

import httpx
import pytest
from fakes import FakeTrainerClient

from gateway.main import create_app

ClientFactory = Callable[[FakeTrainerClient], httpx.AsyncClient]


@pytest.fixture
async def http() -> AsyncIterator[ClientFactory]:
    opened: list[httpx.AsyncClient] = []

    def make(fake: FakeTrainerClient) -> httpx.AsyncClient:
        app = create_app(lambda: fake)
        client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")
        opened.append(client)
        return client

    yield make
    for client in opened:
        await client.aclose()
