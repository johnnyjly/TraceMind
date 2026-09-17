"""Jaeger trace search plus bounded error and latency span evidence."""

from datetime import UTC, datetime
from hashlib import sha256
import json
from typing import Any
from urllib.parse import quote

import httpx

from otel_telemetry_mcp.observations import Evidence, EvidenceSource
from otel_telemetry_mcp.telemetry.errors import TelemetryClientError
from otel_telemetry_mcp.telemetry.http import request_json
from otel_telemetry_mcp.telemetry.attributes import diagnostic_attributes


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
        min_duration_ms: float | None = None,
        errors_only: bool = False,
        operation: str | None = None,
    ) -> list[Evidence]:
        """Query service traces within a window and normalize each result.

        Send start/end as epoch microseconds and request a backend
        result limit. Summarize only spans starting in the inclusive window;
        omit traces with no such spans even if the backend returned them.
        """
        params = {
            "service": service,
            "start": int(start.timestamp() * 1_000_000),
            "end": int(end.timestamp() * 1_000_000),
            "limit": limit,
        }
        if min_duration_ms is not None:
            params["minDuration"] = f"{min_duration_ms:.9f}ms"
        if errors_only:
            params["tags"] = json.dumps({"error": "true"})
        if operation:
            params["operation"] = operation
        payload = await request_json(
            self._http_client,
            "GET",
            f"{self._base_url}/api/traces",
            source="Jaeger",
            params=params,
        )

        traces = payload.get("data")
        if not isinstance(traces, list):
            raise TelemetryClientError(
                "Jaeger response is missing trace results"
            )

        evidence = [
            self._to_evidence(trace, service, start=start, end=end)
            for trace in traces
        ]
        return [item for item in evidence if item is not None]

    async def service_hints(self, service: str) -> dict:
        """Return bounded global names, not time-window observations."""
        payload = await request_json(
            self._http_client,
            "GET",
            f"{self._base_url}/api/services",
            source="Jaeger",
        )
        services = payload.get("data")
        if not isinstance(services, list) or any(
            not isinstance(name, str) for name in services
        ):
            raise TelemetryClientError("Jaeger returned an invalid catalogue")
        names = sorted(set(services))
        available = [name for name in names if len(name) <= 240][:50]
        return {
            "kind": "trace_search_hints",
            "requested_service": service,
            "service_known": service in names,
            "available_services": available,
            "omitted_services": len(names) - len(available),
            "catalogue_scope": "global_not_incident_window",
            "guidance": (
                "No matching traces were returned. "
                "This does not prove health. "
                "Catalogue names are lookup hints, not diagnostic evidence."
            ),
        }

    async def get_trace_detail(
        self,
        trace_id: str,
        *,
        service: str | None = None,
        operation: str | None = None,
        offset: int = 0,
        limit: int = 25,
    ) -> list[Evidence] | dict:
        """Page all spans, prioritizing errors then duration, after filters."""
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

        items = [self._span_to_evidence(s, processes, trace_id) for s in spans]
        # Jaeger warnings may change between reads without changing any span.
        # Hash the exposed observations before filtering, paging or UUIDs.
        fingerprint = sha256(
            json.dumps(
                [e.model_dump(mode="json", exclude={"id"})
                 for e in sorted(
                     items, key=lambda e: e.attributes["span_id"])],
                sort_keys=True,
            ).encode()
        ).hexdigest()
        items = [
            e
            for e in items
            if (service is None or e.service == service)
            and (operation is None or e.attributes["operation"] == operation)
        ]
        items.sort(
            key=lambda e: (
                -int(e.attributes["is_error"]),
                -e.attributes["duration_ms"],
                e.attributes["span_id"],
            )
        )
        page = items[offset: offset + limit]
        coverage = {
            "trace_span_count": len(spans),
            "matched_span_count": len(items),
            "page_offset": offset,
            "page_size": len(page),
            "next_offset": offset + limit
            if offset + limit < len(items)
            else None,
            "trace_fingerprint": fingerprint,
            "selection": {"service": service, "operation": operation},
        }
        for item in page:
            item.attributes["coverage"] = coverage
        return page if page else {"kind": "trace_detail_coverage", **coverage}

    @staticmethod
    def _to_evidence(
        trace: Any,
        service: str,
        *,
        start: datetime,
        end: datetime,
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
            spans = [
                span
                for span in spans
                if (
                    start.timestamp() * 1_000_000
                    <= int(span["startTime"])
                    <= end.timestamp() * 1_000_000
                )
            ]
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
        processes = trace.get("processes", {})
        services = sorted(
            {
                process["serviceName"]
                for span in spans
                if isinstance(processes, dict)
                and isinstance(
                    process := processes.get(span.get("processID")), dict
                )
                and isinstance(process.get("serviceName"), str)
                and process["serviceName"]
            }
        )

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
                "timestamp_source": "earliest_selected_span_startTime",
                "excluded_span_count": received - len(spans),
                "services": services[:20],
                "services_truncated": len(services) > 20,
            },
        )

    @staticmethod
    def _span_to_evidence(
        span: Any,
        processes: dict[str, Any],
        trace_id: str,
    ) -> Evidence:
        """Resolve service, timing, parent relation and diagnostic tags."""
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
            "timestamp_source": "span_startTime",
            "is_error": _span_has_error(span),
        }
        for reference in span.get("references", []):
            if (
                reference.get("refType") == "CHILD_OF"
                and reference.get("traceID") == trace_id
            ):
                attributes["parent_span_id"] = str(reference["spanID"])
                break
        for key in ("error.type", "otel.status_description", "status.message"):
            if key in tags:
                attributes[key.replace(".", "_")] = tags[key]
        attributes["diagnostic"] = diagnostic_attributes(tags)

        return Evidence(
            source=EvidenceSource.TRACE,
            summary=(
                f"{'Error span' if _span_has_error(span) else 'Span'} "
                f"in {service}: {operation}"
            ),
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
