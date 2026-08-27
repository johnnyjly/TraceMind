"""Prometheus metric discovery, instant queries and evidence normalization."""

import json
from datetime import UTC, datetime
from typing import Any

import httpx

from otel_telemetry_mcp.observations import Evidence, EvidenceSource
from otel_telemetry_mcp.telemetry.errors import TelemetryClientError
from otel_telemetry_mcp.telemetry.http import request_json


class PrometheusClient:
    """Discover metric metadata and normalize instant-vector observations."""

    def __init__(self, base_url: str, http_client: httpx.AsyncClient) -> None:
        """Configure the endpoint using an HTTP client owned by the caller."""
        self._base_url = base_url.rstrip("/")
        self._http_client = http_client

    async def discover_metrics(
        self, *, start: datetime, end: datetime,
        service: str | None = None, metric: str | None = None, limit: int = 20,
    ) -> dict[str, Any]:
        """Discover bounded series metadata, never sampled fault evidence.

        Match conventional service labels or a bounded global catalogue.
        Prometheus metadata timestamps are approximate and do not establish
        sample availability at the instant used by query().
        """
        if not 1 <= limit <= 50:
            raise ValueError("metric discovery limit must be between 1 and 50")
        name_filter = (f"__name__={json.dumps(metric)}" if metric is not None
                       else '__name__=~".+"')
        matchers = ([f"{{{name_filter},{label}={json.dumps(service)}}}"
                     for label in ("service_name", "service", "job")]
                    if service is not None else [f"{{{name_filter}}}"])

        async def metadata(path: str) -> list:
            """Read bounded metadata under the caller's timeout."""
            payload = await request_json(
                self._http_client, "GET", f"{self._base_url}/api/v1/{path}",
                source="Prometheus",
                params={"match[]": matchers, "start": start.isoformat(),
                        "end": end.isoformat(), "limit": limit},
            )
            if payload.get("status") != "success":
                raise TelemetryClientError(
                    f"Prometheus discovery failed: "
                    f"{payload.get('error', 'unknown metadata error')}"
                )
            data = payload.get("data")
            if not isinstance(data, list):
                raise TelemetryClientError(
                    "Prometheus returned invalid metadata"
                )
            return data

        # Discover distinct names separately: a series limit can otherwise
        # return only buckets of a single histogram and hide other metrics.
        names = (await metadata("label/__name__/values")
                 if metric is None else [])
        if not all(isinstance(name, str) for name in names):
            raise TelemetryClientError(
                "Prometheus returned invalid metric names"
            )
        series = await metadata("series")
        metrics = {name: {
            "metric": name, "label_keys": [], "example_labels": [],
            "labels_truncated": False,
        } for name in names[:limit]}
        for labels in series[:limit]:
            if (not isinstance(labels, dict)
                    or not isinstance(labels.get("__name__"), str)
                    or not all(isinstance(k, str) and isinstance(v, str)
                               for k, v in labels.items())):
                raise TelemetryClientError(
                    "Prometheus returned invalid series labels"
                )
            name = labels["__name__"]
            if metric is None and name not in metrics:
                continue
            entry = metrics.setdefault(name, {
                "metric": name, "label_keys": [], "example_labels": [],
                "labels_truncated": False,
            })
            keys = sorted(set(entry["label_keys"]) | (
                set(labels) - {"__name__"}
            ))
            entry["label_keys"] = keys[:20]
            example = {key: labels[key] for key in sorted(labels)
                       if key != "__name__" and len(key) <= 120
                       and len(labels[key]) <= 160}
            entry["labels_truncated"] |= (
                len(keys) > 20 or len(example) != len(labels) - 1
                or len(example) > 12
            )
            example = dict(list(example.items())[:12])
            if (example not in entry["example_labels"]
                    and len(entry["example_labels"]) < 2):
                entry["example_labels"].append(example)
        result = {
            "kind": "metric_metadata", "matchers": matchers,
            "investigation_start": start.isoformat(),
            "investigation_end": end.isoformat(),
            "series_limit": limit, "returned_series": min(len(series), limit),
            "possibly_truncated": len(series) >= limit or len(names) >= limit,
            "metrics": [metrics[name] for name in sorted(metrics)],
            "omitted_metrics": 0,
            "guidance": (
                "Metadata is not evidence and may describe series without "
                "samples in this window. Use actual metric names and label "
                "examples to construct a focused query_metrics expression. "
                "If a listed metric has no label examples, pass its exact "
                "name as metric to discover_metrics to inspect its labels. "
                "This catalogue is bounded, not exhaustive. If no service "
                "series match, labels may differ; a bounded discovery without "
                "service can reveal actual labels. Missing metadata does not "
                "prove the service is healthy or has no metrics."
            ),
        }
        # Bound model-facing metadata even if a backend ignores its limit or
        # returns unusually large label sets. Full metadata is not retained.
        while (len(json.dumps(result, ensure_ascii=False).encode()) > 8192
               and result["metrics"]):
            result["metrics"].pop()
            result["omitted_metrics"] += 1
            result["possibly_truncated"] = True
        return result

    async def query(
        self,
        expression: str,
        *,
        at: datetime | None = None,
        limit: int = 50,
    ) -> list[Evidence]:
        """Run instant PromQL at the supplied time or the backend default
        time.

        Request the backend result limit and require a successful vector
        response. Normalize samples without local truncation. HTTP and
        malformed response errors are reported as TelemetryClientError.
        """
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
                "Prometheus returned an unsupported result type; "
                "expected vector"
            )

        results = data.get("result")
        if not isinstance(results, list):
            raise TelemetryClientError(
                "Prometheus response is missing vector results"
            )

        return [self._to_evidence(item, expression) for item in results]

    @staticmethod
    def _to_evidence(item: Any, expression: str) -> Evidence:
        """Normalize one vector sample with its labels, value, and UTC
        timestamp.
        """
        if not isinstance(item, dict):
            raise TelemetryClientError(
                "Prometheus returned an invalid vector sample"
            )

        labels = item.get("metric")
        sample = item.get("value")
        if (
            not isinstance(labels, dict)
            or not isinstance(sample, list)
            or len(sample) != 2
        ):
            raise TelemetryClientError(
                "Prometheus returned an invalid vector sample"
            )

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
