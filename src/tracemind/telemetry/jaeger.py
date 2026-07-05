from datetime import UTC, datetime
from typing import Any

import httpx

from tracemind.models import Evidence, EvidenceSource
from tracemind.telemetry.errors import TelemetryClientError
from tracemind.telemetry.http import request_json


class JaegerClient:
    def __init__(self, base_url: str, http_client: httpx.AsyncClient) -> None:
        self._base_url = base_url.rstrip("/")
        self._http_client = http_client

    async def search_traces(
        self,
        service: str,
        *,
        start: datetime,
        end: datetime,
        limit: int = 20,
    ) -> list[Evidence]:
        payload = await request_json(
            self._http_client,
            "GET",
            f"{self._base_url}/api/traces",
            source="Jaeger",
            params={
                "service": service,
                "start": int(start.timestamp() * 1_000_000),
                "end": int(end.timestamp() * 1_000_000),
                "limit": limit,
            },
        )

        traces = payload.get("data")
        if not isinstance(traces, list):
            raise TelemetryClientError("Jaeger response is missing trace results")

        return [self._to_evidence(trace, service) for trace in traces]

    @staticmethod
    def _to_evidence(trace: Any, service: str) -> Evidence:
        if not isinstance(trace, dict):
            raise TelemetryClientError("Jaeger returned an invalid trace")

        trace_id = trace.get("traceID")
        spans = trace.get("spans")
        if not isinstance(trace_id, str) or not isinstance(spans, list) or not spans:
            raise TelemetryClientError("Jaeger returned an incomplete trace")

        try:
            starts = [int(span["startTime"]) for span in spans]
            ends = [
                int(span["startTime"]) + int(span["duration"])
                for span in spans
            ]
        except (KeyError, TypeError, ValueError) as exc:
            raise TelemetryClientError("Jaeger returned an invalid span timestamp") from exc

        error_span_count = sum(_span_has_error(span) for span in spans)
        duration_ms = (max(ends) - min(starts)) / 1_000
        observed_at = datetime.fromtimestamp(min(starts) / 1_000_000, tz=UTC)

        return Evidence(
            source=EvidenceSource.TRACE,
            summary=(
                f"Trace {trace_id}: {len(spans)} spans, "
                f"{error_span_count} error spans, {duration_ms:.2f} ms"
            ),
            observed_at=observed_at,
            service=service,
            query=f"service={service}",
            attributes={
                "trace_id": trace_id,
                "span_count": len(spans),
                "error_span_count": error_span_count,
                "duration_ms": duration_ms,
            },
        )


def _span_has_error(span: Any) -> bool:
    if not isinstance(span, dict):
        return False

    tags = span.get("tags", [])
    if not isinstance(tags, list):
        return False

    for tag in tags:
        if not isinstance(tag, dict):
            continue
        key = tag.get("key")
        value = tag.get("value")
        if key == "error" and value is True:
            return True
        if key == "otel.status_code" and str(value).upper() == "ERROR":
            return True
    return False

