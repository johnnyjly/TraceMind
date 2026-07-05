from datetime import UTC, datetime
from typing import Any

import httpx

from tracemind.models import Evidence, EvidenceSource
from tracemind.telemetry.errors import TelemetryClientError
from tracemind.telemetry.http import request_json


class PrometheusClient:
    def __init__(self, base_url: str, http_client: httpx.AsyncClient) -> None:
        self._base_url = base_url.rstrip("/")
        self._http_client = http_client

    async def query(
        self,
        expression: str,
        *,
        at: datetime | None = None,
        limit: int = 50,
    ) -> list[Evidence]:
        params: dict[str, str | int] = {
            "query": expression,
            "limit": limit,
        }
        if at is not None:
            params["time"] = at.isoformat()

        payload = await request_json(
            self._http_client,
            "GET",
            f"{self._base_url}/api/v1/query",
            source="Prometheus",
            params=params,
        )

        if payload.get("status") != "success":
            error = payload.get("error", "unknown query error")
            raise TelemetryClientError(f"Prometheus query failed: {error}")

        data = payload.get("data")
        if not isinstance(data, dict) or data.get("resultType") != "vector":
            raise TelemetryClientError(
                "Prometheus returned an unsupported result type; expected vector"
            )

        results = data.get("result")
        if not isinstance(results, list):
            raise TelemetryClientError("Prometheus response is missing vector results")

        return [self._to_evidence(item, expression) for item in results]

    @staticmethod
    def _to_evidence(item: Any, expression: str) -> Evidence:
        if not isinstance(item, dict):
            raise TelemetryClientError("Prometheus returned an invalid vector sample")

        labels = item.get("metric")
        sample = item.get("value")
        if not isinstance(labels, dict) or not isinstance(sample, list) or len(sample) != 2:
            raise TelemetryClientError("Prometheus returned an invalid vector sample")

        timestamp, value = sample
        try:
            observed_at = datetime.fromtimestamp(float(timestamp), tz=UTC)
        except (TypeError, ValueError, OSError) as exc:
            raise TelemetryClientError(
                "Prometheus returned an invalid sample timestamp"
            ) from exc

        metric_name = str(labels.get("__name__", expression))
        service = labels.get("service_name") or labels.get("job")

        return Evidence(
            source=EvidenceSource.METRIC,
            summary=f"Metric {metric_name}={value}",
            observed_at=observed_at,
            service=str(service) if service is not None else None,
            query=expression,
            attributes={
                "labels": labels,
                "value": str(value),
            },
        )

