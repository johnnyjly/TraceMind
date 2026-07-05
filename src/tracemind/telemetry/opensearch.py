import json
from datetime import datetime
from typing import Any
from urllib.parse import quote

import httpx

from tracemind.models import Evidence, EvidenceSource
from tracemind.telemetry.errors import TelemetryClientError
from tracemind.telemetry.http import request_json


class OpenSearchClient:
    def __init__(
        self,
        base_url: str,
        logs_index: str,
        http_client: httpx.AsyncClient,
    ) -> None:
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
    ) -> list[Evidence]:
        filters: list[dict[str, Any]] = [
            {"term": {"resource.attributes.service.name": service}},
            {
                "range": {
                    "@timestamp": {
                        "gte": start.isoformat(),
                        "lte": end.isoformat(),
                    }
                }
            },
        ]
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
                "sort": [{"@timestamp": {"order": "desc"}}],
                "query": query,
            },
        )

        hits = payload.get("hits")
        if not isinstance(hits, dict) or not isinstance(hits.get("hits"), list):
            raise TelemetryClientError("OpenSearch response is missing log hits")

        return [self._to_evidence(hit, service, text) for hit in hits["hits"]]

    @staticmethod
    def _to_evidence(hit: Any, service: str, text: str | None) -> Evidence:
        if not isinstance(hit, dict) or not isinstance(hit.get("_source"), dict):
            raise TelemetryClientError("OpenSearch returned an invalid log hit")

        source = hit["_source"]
        timestamp = source.get("@timestamp")
        body = source.get("body")
        if not isinstance(timestamp, str) or body is None:
            raise TelemetryClientError("OpenSearch log is missing timestamp or body")

        try:
            observed_at = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        except ValueError as exc:
            raise TelemetryClientError("OpenSearch log has an invalid timestamp") from exc

        summary = body if isinstance(body, str) else json.dumps(body, sort_keys=True)
        attributes = {
            "document_id": str(hit.get("_id", "")),
            "severity_text": source.get("severity_text"),
            "severity_number": source.get("severity_number"),
            "trace_id": source.get("traceId"),
            "span_id": source.get("spanId"),
        }

        return Evidence(
            source=EvidenceSource.LOG,
            summary=summary,
            observed_at=observed_at,
            service=service,
            query=f"service={service}" + (f" text={text}" if text else ""),
            attributes={key: value for key, value in attributes.items() if value is not None},
        )

