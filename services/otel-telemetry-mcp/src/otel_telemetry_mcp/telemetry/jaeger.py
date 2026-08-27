"""Jaeger trace search plus bounded error and latency span evidence."""

from datetime import UTC, datetime
from typing import Any
from urllib.parse import quote

import httpx

from otel_telemetry_mcp.observations import Evidence, EvidenceSource
from otel_telemetry_mcp.telemetry.errors import TelemetryClientError
from otel_telemetry_mcp.telemetry.http import request_json


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
        result limit. Summarize only spans starting in the inclusive window;
        omit traces with no such spans even if the backend returned them.
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

        evidence = [self._to_evidence(trace, service, start=start, end=end)
                    for trace in traces]
        return [item for item in evidence if item is not None]

    async def get_trace_detail(self, trace_id: str) -> list[Evidence]:
        """Return error spans and the twenty longest non-error spans.

        Successful slow calls still need service/operation timing evidence.
        Keep every existing error observation and bound the added context.
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

        errors, successful = [], []
        for span in spans:
            item = self._span_to_evidence(span, processes, trace_id)
            (errors if _span_has_error(span) else successful).append(item)
        successful.sort(key=lambda item: item.attributes["duration_ms"],
                        reverse=True)
        return errors + successful[:20]

    @staticmethod
    def _to_evidence(
        trace: Any, service: str, *, start: datetime, end: datetime,
    ) -> Evidence | None:
        """Summarize selected spans without assigning an old trace start.

        A trace can include historical spans alongside current spans. The
        selected span extent is not a whole-request latency measurement.
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
            received = len(spans)
            spans = [span for span in spans if (
                start.timestamp() * 1_000_000 <= int(span["startTime"])
                <= end.timestamp() * 1_000_000
            )]
            if not spans:
                return None
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
                f"Trace {trace_id}: {len(spans)} spans starting in the "
                f"investigation window, {error_span_count} error spans, "
                f"{duration_ms:.2f} ms selected span extent"
            ),
            observed_at=observed_at,
            service=service,
            query=f"service={service}",
            attributes={
                "trace_id": trace_id,
                "span_count": len(spans),
                "error_span_count": error_span_count,
                "duration_ms": duration_ms,
                "summary_scope": "spans_started_in_window",
                "excluded_span_count": received - len(spans),
            },
        )

    @staticmethod
    def _span_to_evidence(
        span: Any,
        processes: dict[str, Any],
        trace_id: str,
    ) -> Evidence:
        """Resolve service, timing, parent relation and diagnostic tags.
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
            "is_error": _span_has_error(span),
        }
        for reference in span.get("references", []):
            if (reference.get("refType") == "CHILD_OF" and
                    reference.get("traceID") == trace_id):
                attributes["parent_span_id"] = str(reference["spanID"])
                break
        for key in ("error.type", "otel.status_description", "status.message"):
            if key in tags:
                attributes[key.replace(".", "_")] = tags[key]

        return Evidence(
            source=EvidenceSource.TRACE,
            summary=(f"{'Error span' if _span_has_error(span) else 'Span'} "
                     f"in {service}: {operation}"),
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
