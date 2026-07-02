import asyncio

from httpx import ASGITransport, AsyncClient

from tracemind.api.app import app


def test_health_endpoint() -> None:
    async def request_health():
        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            return await client.get("/health")

    response = asyncio.run(request_health())

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
