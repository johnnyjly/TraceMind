"""Jaeger trace search and error-span evidence normalization."""

from datetime import UTC, datetime
from typing import Any
from urllib.parse import quote

import httpx

from tracemind.models import Evidence, EvidenceSource
from tracemind.telemetry.errors import TelemetryClientError
from tracemind.telemetry.http import request_json


class JaegerClient:
    """Normalize Jaeger trace-search and trace-detail API responses."""

    def __init__(self, base_url: str, http_client: httpx.AsyncClient) -> None:
        """Configure the endpoint using an HTTP client owned by the caller."""
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
        """Query service traces within a window and normalize each result.

        Send start/end as epoch microseconds and request a backend
        result limit. Return trace summaries; this method does not
        locally truncate the response.
        """
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
            raise TelemetryClientError(
                "Jaeger response is missing trace results"
            )

        return [self._to_evidence(trace, service) for trace in traces]

    async def get_trace_detail(self, trace_id: str) -> list[Evidence]:
        """Fetch one trace and return evidence for spans carrying error
        tags.
        """
        payload = await request_json(
            self._http_client,
            "GET",
            f"{self._base_url}/api/traces/{quote(trace_id, safe='')}",
            source="Jaeger",
        )

        traces = payload.get("data")
        if not isinstance(traces, list) or len(traces) != 1:
            raise TelemetryClientError(
                "Jaeger response is missing the requested trace"
            )

        trace = traces[0]
        if not isinstance(trace, dict):
            raise TelemetryClientError("Jaeger returned an invalid trace")

        spans = trace.get("spans")
        processes = trace.get("processes")
        if not isinstance(spans, list) or not isinstance(processes, dict):
            raise TelemetryClientError("Jaeger trace detail is incomplete")

        evidence: list[Evidence] = []
        for span in spans:
            if _span_has_error(span):
                evidence.append(
                    self._error_span_to_evidence(span, processes, trace_id)
                )
        return evidence

    @staticmethod
    def _to_evidence(trace: Any, service: str) -> Evidence:
        """Summarize trace duration, span counts, and earliest observed
        time.
        """
        if not isinstance(trace, dict):
            raise TelemetryClientError("Jaeger returned an invalid trace")

        trace_id = trace.get("traceID")
        spans = trace.get("spans")
        if (
            not isinstance(trace_id, str)
            or not isinstance(spans, list)
            or not spans
        ):
            raise TelemetryClientError("Jaeger returned an incomplete trace")

        try:
            starts = [int(span["startTime"]) for span in spans]
            ends = [
                int(span["startTime"]) + int(span["duration"])
                for span in spans
            ]
        except (KeyError, TypeError, ValueError) as exc:
            raise TelemetryClientError(
                "Jaeger returned an invalid span timestamp"
            ) from exc

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

    @staticmethod
    def _error_span_to_evidence(
        span: Any,
        processes: dict[str, Any],
        trace_id: str,
    ) -> Evidence:
        """Resolve a span service and normalize timing, operation, and
        error tags.
        """
        if not isinstance(span, dict):
            raise TelemetryClientError("Jaeger returned an invalid span")

        process = processes.get(span.get("processID"))
        if not isinstance(process, dict) or not isinstance(
            process.get("serviceName"), str
        ):
            raise TelemetryClientError("Jaeger span is missing its service")

        try:
            observed_at = datetime.fromtimestamp(
                int(span["startTime"]) / 1_000_000,
                tz=UTC,
            )
            duration_ms = int(span["duration"]) / 1_000
        except (KeyError, TypeError, ValueError, OSError) as exc:
            raise TelemetryClientError(
                "Jaeger returned an invalid span timestamp"
            ) from exc

        operation = str(span.get("operationName", "unknown operation"))
        tags = _tags_by_key(span)
        service = process["serviceName"]

        attributes: dict[str, Any] = {
            "trace_id": trace_id,
            "span_id": str(span.get("spanID", "")),
            "operation": operation,
            "duration_ms": duration_ms,
        }
        for key in ("error.type", "otel.status_description", "status.message"):
            if key in tags:
                attributes[key.replace(".", "_")] = tags[key]

        return Evidence(
            source=EvidenceSource.TRACE,
            summary=f"Error span in {service}: {operation}",
            observed_at=observed_at,
            service=service,
            query=f"trace_id={trace_id}",
            attributes=attributes,
        )


def _span_has_error(span: Any) -> bool:
    """Recognize error=True or an ERROR OpenTelemetry status among span
    tags.
    """
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


def _tags_by_key(span: dict[str, Any]) -> dict[str, Any]:
    """Index valid tag entries by string key, keeping the last repeated
    key.
    """
    tags = span.get("tags", [])
    if not isinstance(tags, list):
        return {}
    return {
        str(tag["key"]): tag.get("value")
        for tag in tags
        if isinstance(tag, dict) and "key" in tag
    }
