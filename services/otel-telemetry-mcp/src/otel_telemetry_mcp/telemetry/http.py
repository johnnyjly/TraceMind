"""Shared HTTP and JSON error translation for telemetry requests."""

from typing import Any

import httpx

from otel_telemetry_mcp.telemetry.errors import TelemetryClientError


async def request_json(
    client: httpx.AsyncClient,
    method: str,
    url: str,
    *,
    source: str,
    params: dict[str, str | int | list[str]] | None = None,
    json: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Perform one HTTP request and require a JSON object response.

    Translate HTTP failures, invalid JSON, and non-object payloads to
    TelemetryClientError with backend context. Client settings own
    transport timeouts; this helper adds no retries.
    """
    try:
        response = await client.request(method, url, params=params, json=json)
        response.raise_for_status()
    except httpx.HTTPError as exc:
        raise TelemetryClientError(f"{source} request failed: {exc}") from exc

    try:
        payload = response.json()
    except ValueError as exc:
        raise TelemetryClientError(f"{source} returned invalid JSON") from exc

    if not isinstance(payload, dict):
        raise TelemetryClientError(
            f"{source} returned an unexpected JSON payload"
        )

    return payload
