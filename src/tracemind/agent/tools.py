from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from datetime import datetime

from langchain_core.tools import BaseTool, tool

from tracemind.agent.harness import RunHarness
from tracemind.models import Evidence
from tracemind.models.run import ToolName
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
        ToolSpec("search_traces", 20),
        ToolSpec("get_trace_detail", 100),
        ToolSpec("search_logs", 100),
    )
}


class EvidenceCollector:
    """Collect run-local evidence by ID for final citation validation."""

    def __init__(self) -> None:
        self._evidence: dict[str, Evidence] = {}

    def add(self, items: Sequence[Evidence]) -> list[dict[str, object]]:
        """Retain observations by ID and return JSON-compatible tool results.

        An existing ID is replaced by the supplied object. Content is not
        deduplicated and observations are not copied or persisted.
        """
        for item in items:
            self._evidence[str(item.id)] = item
        return [item.model_dump(mode="json") for item in items]

    @property
    def evidence(self) -> list[Evidence]:
        """Return collected evidence in a new list.

        The Evidence objects are shared with the collector; callers should
        treat them as read-only.
        """
        return list(self._evidence.values())


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
    authorization, reserves budgets, and caps result counts before adding
    evidence to the collector. Trace detail queries use their trace ID;
    final citation validation checks times.
    """
    async def execute(
        name: ToolName,
        arguments: dict[str, object],
        call: Callable[[], Awaitable[Sequence[Evidence]]],
    ) -> list[dict[str, object]]:
        spec = TOOL_REGISTRY[name]
        evidence = await harness.tool_call(
            name, arguments, call, max_results=spec.max_results
        )
        result = collector.add(evidence)
        harness.state.evidence_ids = [item.id for item in collector.evidence]
        return result

    @tool
    async def query_metrics(expression: str) -> list[dict[str, object]]:
        (
            "Run an instant PromQL query and return normalized metric "
            "evidence."
        )
        return await execute(
            "query_metrics", {"expression": expression},
            lambda: prometheus.query(expression, at=end),
        )

    @tool
    async def search_traces(service: str) -> list[dict[str, object]]:
        """Find recent Jaeger traces for one service in the incident window."""
        return await execute(
            "search_traces", {"service": service},
            lambda: jaeger.search_traces(service, start=start, end=end),
        )

    @tool
    async def get_trace_detail(trace_id: str) -> list[dict[str, object]]:
        (
            "Inspect one Jaeger trace and return its normalized error-span "
            "evidence."
        )
        return await execute(
            "get_trace_detail", {"trace_id": trace_id},
            lambda: jaeger.get_trace_detail(trace_id),
        )

    @tool
    async def search_logs(
        service: str,
        text: str | None = None,
    ) -> list[dict[str, object]]:
        """Find OpenSearch logs for one service, optionally matching text."""
        return await execute(
            "search_logs", {"service": service, "text": text},
            lambda: opensearch.search_logs(
                service, start=start, end=end, text=text
            ),
        )

    return [
        item for item in (
            query_metrics, search_traces, get_trace_detail, search_logs
        )
        if item.name in harness.state.allowed_tools
    ]
