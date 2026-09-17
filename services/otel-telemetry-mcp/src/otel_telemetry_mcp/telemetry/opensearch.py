"""OpenSearch incident-window log queries and evidence normalization."""

import json
from datetime import UTC, datetime
from typing import Any
from urllib.parse import quote

import httpx

from otel_telemetry_mcp.observations import Evidence, EvidenceSource
from otel_telemetry_mcp.telemetry.errors import TelemetryClientError
from otel_telemetry_mcp.telemetry.http import request_json
from otel_telemetry_mcp.telemetry.attributes import diagnostic_attributes


EPOCH = datetime(1970, 1, 1, tzinfo=UTC)


class OpenSearchClient:
    """Query SS4O log documents and normalize them as evidence."""

    def __init__(
        self,
        base_url: str,
        logs_index: str,
        http_client: httpx.AsyncClient,
    ) -> None:
        """Configure an encoded index pattern and a caller-owned HTTP
        client.
        """
        self._base_url = base_url.rstrip("/")
        self._logs_index = quote(logs_index, safe="*,-_.")
        self._http_client = http_client

    async def search_logs(
        self,
        service: str,
        *,
        start: datetime,
        end: datetime,
        text: str | None = None,
        limit: int = 100,
        trace_id: str | None = None,
    ) -> list[Evidence]:
        """Search by event time, or observation time when event time is unset.

        Fetch bounded, disjoint groups and merge by the selected time.
        Taking each group's newest N preserves the combined newest N
        without backend scripts. Valid old events are never admitted just
        because their observation time is recent.
        """
        bounds = {"gte": start.isoformat(), "lte": end.isoformat()}
        unset_event = {
            "bool": {
                "minimum_should_match": 1,
                "should": [
                    {
                        "bool": {
                            "must_not": [{"exists": {"field": "@timestamp"}}]
                        }
                    },
                    {"term": {"@timestamp": EPOCH.isoformat()}},
                ],
            }
        }
        groups = (
            (
                "@timestamp",
                [
                    {"range": {"@timestamp": bounds}},
                    {"range": {"@timestamp": {"gt": EPOCH.isoformat()}}},
                ],
            ),
            (
                "observedTimestamp",
                [unset_event, {"range": {"observedTimestamp": bounds}}],
            ),
        )
        evidence = []
        for field, time_filters in groups:
            filters = [
                {"term": {"resource.service.name.keyword": service}},
                *time_filters,
            ]
            if trace_id:
                filters.append({"term": {"traceId": trace_id}})
            query: dict[str, Any] = {"bool": {"filter": filters}}
            if text:
                query["bool"]["must"] = [{"match": {"body": text}}]
            payload = await request_json(
                self._http_client,
                "POST",
                f"{self._base_url}/{self._logs_index}/_search",
                source="OpenSearch",
                json={
                    "size": limit,
                    "sort": [
                        {field: {"order": "desc", "unmapped_type": "date"}}
                    ],
                    "query": query,
                },
            )
            hits = payload.get("hits")
            if not isinstance(hits, dict) or not isinstance(
                hits.get("hits"), list
            ):
                raise TelemetryClientError(
                    "OpenSearch response is missing log hits"
                )
            evidence.extend(
                self._to_evidence(hit, service, text) for hit in hits["hits"]
            )
        evidence = [
            item for item in evidence if start <= item.observed_at <= end
        ]
        return sorted(
            evidence, key=lambda item: item.observed_at, reverse=True
        )[:limit]

    @staticmethod
    def _to_evidence(hit: Any, service: str, text: str | None) -> Evidence:
        """Normalize a log body, timestamp, severity, and trace/span
        references.
        """
        if not isinstance(hit, dict) or not isinstance(
            hit.get("_source"), dict
        ):
            raise TelemetryClientError(
                "OpenSearch returned an invalid log hit"
            )

        source = hit["_source"]
        timestamp = source.get("@timestamp")
        body = source.get("body")
        if body is None:
            raise TelemetryClientError("OpenSearch log is missing body")

        try:
            observed_at = (
                datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                if timestamp is not None
                else EPOCH
            )
            time_attributes = {}
            if observed_at == EPOCH:
                observed_at = datetime.fromisoformat(
                    source["observedTimestamp"].replace("Z", "+00:00")
                )
                time_attributes = {
                    "timestamp_source": "observedTimestamp",
                    "original_event_timestamp": timestamp,
                    "original_observed_timestamp": source["observedTimestamp"],
                }
            if observed_at.tzinfo is None or observed_at <= EPOCH:
                raise ValueError("timestamp must be a positive aware time")
        except (ValueError, AttributeError, KeyError, TypeError) as exc:
            raise TelemetryClientError(
                "OpenSearch log has an invalid timestamp"
            ) from exc

        summary = (
            body if isinstance(body, str) else json.dumps(body, sort_keys=True)
        )
        body_attributes = {}
        if isinstance(body, str) and not body.strip():
            summary = "[Empty log body]"
            body_attributes = {"body_empty": True, "original_body": body}
        severity = source.get("severity")
        if not isinstance(severity, dict):
            severity = {}
        attributes = {
            **time_attributes,
            **body_attributes,
            "document_id": str(hit.get("_id", "")),
            "severity_text": severity.get("text"),
            "severity_number": severity.get("number"),
            "trace_id": source.get("traceId"),
            "span_id": source.get("spanId"),
            "diagnostic": diagnostic_attributes(
                source.get("attributes") or {}
            ),
        }

        return Evidence(
            source=EvidenceSource.LOG,
            summary=summary,
            observed_at=observed_at,
            service=service,
            query=f"service={service}" + (f" text={text}" if text else ""),
            attributes={
                key: value
                for key, value in attributes.items()
                if value is not None
            },
        )
