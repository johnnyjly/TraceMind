"""Read-only telemetry tool bindings and run-local evidence collection."""

import hashlib
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Annotated

from langchain_core.tools import BaseTool, tool
from pydantic import Field

from tracemind.agent.context import json_text
from tracemind.agent.harness import RunHarness
from tracemind.models import Evidence, EvidenceSource
from tracemind.models.run import StopReason, ToolName
from tracemind.telemetry import (
    JaegerClient,
    OpenSearchClient,
    PrometheusClient,
)


@dataclass(frozen=True)
class ToolSpec:
    """Describe a tool preview limit and its capability label."""

    name: ToolName
    max_results: int
    capability: str = "telemetry.read"


TOOL_REGISTRY = {
    spec.name: spec
    for spec in (
        ToolSpec("query_metrics", 50),
        ToolSpec("discover_metrics", 1, capability="telemetry.metadata.read"),
        ToolSpec("search_traces", 20),
        ToolSpec("get_trace_detail", 100),
        ToolSpec("search_logs", 100),
        ToolSpec("read_evidence", 20, capability="evidence.read"),
    )
}


class EvidenceCapacityError(Exception):
    """Signal that retaining an observation would exceed the byte limit."""


class EvidenceCollector:
    """Store run-local observations with exact deduplication and stable IDs.

    The byte limit covers serialized evidence, not total Python memory or
    backend response buffers. A persistent run can snapshot additions through
    the on_change callback.
    """

    def __init__(
        self, *, max_bytes: int = 8_388_608,
        initial: Sequence[Evidence] = (),
        on_change: Callable[[], None] | None = None,
    ) -> None:
        """Initialize an evidence index with a serialized-byte capacity."""
        self._evidence: dict[str, Evidence] = {}
        self._fingerprints: dict[str, str] = {}
        self._documents: dict[str, str] = {}
        self._max_bytes = max_bytes
        self._on_change = None
        self.bytes_used = 0
        self.add(initial)
        self._on_change = on_change

    def add(self, items: Sequence[Evidence]) -> list[dict[str, object]]:
        """Store observations and return previews with canonical IDs.

        Exact content matches excluding the generated ID reuse the first ID.
        New observations are copied. On EvidenceCapacityError, earlier items
        in the batch remain stored; insertion is not transactional.
        """
        ids = []
        for item in items:
            data = item.model_dump(mode="json", exclude={"id"})
            fingerprint = hashlib.sha256(json_text(data).encode()).hexdigest()
            evidence_id = self._fingerprints.get(fingerprint)
            if evidence_id is None:
                evidence_id = str(item.id)
                document = json_text(item.model_dump(mode="json"))
                size = len(document.encode())
                if self.bytes_used + size > self._max_bytes:
                    raise EvidenceCapacityError
                # Own the stored observation; external mutation must not change
                # the evidence behind a previously issued citation.
                self._evidence[evidence_id] = item.model_copy(deep=True)
                self._documents[evidence_id] = document
                self._fingerprints[fingerprint] = evidence_id
                self.bytes_used += size
                if self._on_change is not None:
                    self._on_change()
            if evidence_id not in ids:
                ids.append(evidence_id)
        return [self.preview(key) for key in ids]

    def preview(self, evidence_id: str) -> dict[str, object]:
        """Return complete evidence or a marked preview for a known ID."""
        item = self._evidence[evidence_id]
        document = self._documents[evidence_id]
        if len(document.encode()) <= 1200:
            return item.model_dump(mode="json")
        return {
            "id": evidence_id,
            "source": item.source.value,
            "service": item.service[:120] if item.service else None,
            "observed_at": item.observed_at.isoformat(),
            "summary": item.summary[:240],
            "truncated": True,
            "detail_chars": len(document),
        }

    def read(
        self, evidence_id: str | None, offset: int, *,
        service: str | None = None, source: EvidenceSource | None = None,
    ) -> list[dict]:
        """Page previews by entry offset or JSON by character offset.

        Callers validate nonnegative offsets. Unknown IDs and out-of-range
        detail offsets return error records; next_offset=None ends paging.
        Filters apply only to the catalogue, whose offsets count matches.
        """
        if evidence_id is None:
            ids = [key for key, item in self._evidence.items()
                   if (service is None or item.service == service)
                   and (source is None or item.source == source)]
            end = offset + TOOL_REGISTRY["read_evidence"].max_results
            result = [self.preview(key) for key in ids[offset:end]]
            result.append({
                "total_evidence": len(ids),
                "next_offset": end if end < len(ids) else None,
            })
            return result
        document = self._documents.get(evidence_id)
        if document is None:
            return [{"error": "unknown_evidence_id"}]
        if offset >= len(document):
            return [{"error": "offset_out_of_range"}]
        end = min(offset + 2048, len(document))
        return [{
            "id": evidence_id, "offset": offset,
            "content": document[offset:end],
            "next_offset": end if end < len(document) else None,
            "total_chars": len(document),
        }]

    @property
    def evidence(self) -> list[Evidence]:
        """Return collected evidence in a new list.

        The Evidence objects are shared with the collector; callers should
        treat them as read-only.
        """
        return list(self._evidence.values())


def context_index(items: list[dict]) -> dict:
    """Build application-owned compact references for context reduction."""
    # Artifact is application-owned metadata, not parsed model/backend text.
    index = []
    for item in items:
        if "id" in item:
            entry = {key: item[key] for key in
                     ("id", "source", "observed_at", "offset", "next_offset")
                     if key in item}
            if item.get("service") is not None:
                entry["service"] = str(item["service"])[:120]
            entry["summary"] = str(item.get("summary", "Stored detail"))[:80]
            entry["truncated"] = True
            index.append(entry)
        else:
            index.append(item)
    return {"context_index": index}


def create_telemetry_tools(
    prometheus: PrometheusClient,
    jaeger: JaegerClient,
    opensearch: OpenSearchClient,
    *,
    start: datetime,
    end: datetime,
    collector: EvidenceCollector,
    harness: RunHarness,
) -> list[BaseTool]:
    """Bind read-only telemetry tools to a fixed investigation window.

    Filter visibility by the harness allowlist. Execution rechecks
    authorization and reserves budgets. Store observations before selecting
    bounded previews, retaining omitted details for paginated retrieval.
    Final citation validation checks observation times.
    """
    async def execute(
        name: ToolName,
        arguments: dict[str, object],
        call: Callable[[], Awaitable[Sequence[Evidence]]],
    ) -> tuple[list[dict], dict]:
        """Execute a query and return previews with a compact index."""
        spec = TOOL_REGISTRY[name]

        async def collect() -> list[dict]:
            """Store observations, select previews, and trace retention."""
            observations = await call()
            harness.check_active()
            before = len(collector.evidence)
            try:
                previews = collector.add(observations)
            except EvidenceCapacityError:
                harness.emit("observations.limit",
                             parent_span_id=harness.state.id,
                             data={"stored_bytes": collector.bytes_used})
                harness.stop(StopReason.CONTEXT_LIMIT)
            finally:
                harness.state.evidence_ids = [e.id for e in collector.evidence]
            result = previews[:spec.max_results]
            omitted = len(previews) - len(result)
            if omitted:
                result.append({
                    "omitted_count": omitted,
                    "retrieval": "read_evidence without evidence_id lists all",
                })
            harness.emit("observations.stored",
                         parent_span_id=harness.state.id,
                         data={"received": len(observations),
                               "added": len(collector.evidence) - before,
                               "duplicates": (
                                   len(observations)
                                   - (len(collector.evidence) - before)
                               ),
                               "omitted_previews": omitted,
                               "stored_bytes": collector.bytes_used})
            return result

        result = await harness.tool_call(name, arguments, collect)
        return result, context_index(result)

    @tool(response_format="content_and_artifact")
    async def query_metrics(
        expression: Annotated[str, Field(description=(
            "PromQL expression returning an instant vector, using confirmed "
            "metric names and label values. Not a bare range vector or "
            "scalar. "
            "Query time is fixed to the investigation end by the application."
        ))],
    ) -> tuple[list[dict], dict]:
        """Query an instant vector at the investigation end time.

        Use when sample values are needed to check a hypothesis. Reuse known
        names and labels; discover_metrics is optional if they are confirmed.
        Skip this tool when existing evidence already answers the question.
        Returns normalized observations with evidence IDs, or empty-query
        guidance without evidence IDs. An empty result does not establish
        health: check names, labels and sample availability before retrying.

        Conditional example: ONLY if discovery returned request_count_total
        and the label job="orders", query_metrics with expression
        'request_count_total{job="orders"}' can inspect those samples.
        If names or labels are unknown, discover them instead of guessing.
        """
        result, artifact = await execute(
            "query_metrics", {"expression": expression},
            lambda: prometheus.query(expression, at=end),
        )
        if not result:
            result = [{
                "kind": "empty_metric_query", "query_time": end.isoformat(),
                "guidance": (
                    "No matching samples for this expression at query_time. "
                    "This is not evidence of health or absence of a fault. "
                    "Check metric names, actual label values, and instant "
                    "sample availability. Use discover_metrics if authorized "
                    "instead of repeatedly guessing query variants."
                ),
            }]
            artifact = context_index(result)
        return result, artifact

    @tool(response_format="content_and_artifact")
    async def discover_metrics(
        service: Annotated[str | None, Field(
            max_length=120, description=(
                "Exact service value matched against service_name, service "
                "or job labels, not a substring or regex. Omit or use null "
                "for a bounded global catalogue when these labels are unknown."
            ),
        )] = None,
        metric: Annotated[str | None, Field(
            max_length=240, description=(
                "One exact metric name returned by discovery, not PromQL or "
                "a regex. Set it to inspect that metric's label examples; "
                "omit or use null to discover distinct metric names first."
            ),
        )] = None,
        limit: Annotated[int, Field(
            ge=1, le=50, description=(
                "Maximum distinct names and maximum series inspected per "
                "metadata request (1-50, default 20), not an evidence count "
                "or time range. Results may be incomplete even at this limit."
            ),
        )] = 20,
    ) -> tuple[list[dict], dict]:
        """Discover metric names and label examples before writing PromQL.

        Use when relevant names or labels are unknown, including after an
        empty query with suspected mismatched names/labels. Reuse previously
        discovered metadata instead of rediscovering it for every query.
        Skip discovery when names/labels are known or evidence is sufficient.
        The application fixes the investigation window; do not supply dates.

        Returns metrics containing metric, label_keys and example_labels.
        Empty example_labels means no examples were included, not that the
        metric has no labels; pass its exact returned name as metric to inspect
        it. possibly_truncated, omitted_metrics and labels_truncated mark
        incomplete coverage. Empty discovery may mean service labels differ;
        bounded global discovery can reveal them. Absence never proves health.
        Metadata has no evidence IDs or sample values and cannot be cited.

        Examples: If the incident names orders and metric names are unknown,
        discover_metrics(service="orders") finds candidates to query.
        If retained evidence already supports a user's explanation request,
        answer directly without discovery or another metric query.
        """
        async def discover() -> list[dict]:
            """Return metadata without storing observations or evidence IDs."""
            return [await prometheus.discover_metrics(
                service=service, metric=metric,
                start=start, end=end, limit=limit,
            )]

        result = await harness.tool_call(
            "discover_metrics", {"service": service, "metric": metric,
                                 "limit": limit}, discover,
        )
        return result, context_index(result)

    @tool(response_format="content_and_artifact")
    async def search_traces(service: str) -> tuple[list[dict], dict]:
        """Find recent Jaeger traces for one service in the incident window."""
        return await execute(
            "search_traces", {"service": service},
            lambda: jaeger.search_traces(service, start=start, end=end),
        )

    @tool(response_format="content_and_artifact")
    async def get_trace_detail(trace_id: str) -> tuple[list[dict], dict]:
        (
            "Inspect one Jaeger trace and return its normalized error-span "
            "evidence."
        )
        return await execute(
            "get_trace_detail", {"trace_id": trace_id},
            lambda: jaeger.get_trace_detail(trace_id),
        )

    @tool(response_format="content_and_artifact")
    async def search_logs(
        service: str,
        text: str | None = None,
    ) -> tuple[list[dict], dict]:
        """Find OpenSearch logs for one service, optionally matching text."""
        return await execute(
            "search_logs", {"service": service, "text": text},
            lambda: opensearch.search_logs(
                service, start=start, end=end, text=text
            ),
        )

    @tool(response_format="content_and_artifact")
    async def read_evidence(
        evidence_id: str | None = None,
        offset: Annotated[int, Field(ge=0)] = 0,
        service: str | None = None,
        source: EvidenceSource | None = None,
    ) -> tuple[list[dict], dict]:
        """List stored previews, or page evidence JSON by ID.

        Prefer a known evidence_id for missing details. Otherwise filter the
        catalogue by service and/or source. Follow next_offset only when more
        entries are needed to answer the question.

        Without an ID, offset counts evidence entries. With an ID it counts
        characters in the stored JSON and ignores catalogue filters.
        Only this run's evidence is visible.
        """
        async def read() -> list[dict]:
            """Read a page from this run's evidence collector."""
            return collector.read(evidence_id, offset,
                                  service=service, source=source)

        result = await harness.tool_call(
            "read_evidence",
            {"evidence_id": evidence_id, "offset": offset,
             "service": service, "source": source}, read
        )
        artifact = context_index(result)
        artifact["detail_page"] = evidence_id is not None
        return result, artifact

    return [
        item for item in (
            query_metrics, discover_metrics, search_traces,
            get_trace_detail, search_logs,
            read_evidence,
        )
        if item.name in harness.state.allowed_tools
    ]
