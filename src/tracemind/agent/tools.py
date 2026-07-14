from collections.abc import Sequence
from datetime import datetime

from langchain_core.tools import BaseTool, tool

from tracemind.models import Evidence
from tracemind.telemetry import JaegerClient, OpenSearchClient, PrometheusClient


class EvidenceCollector:
    def __init__(self) -> None:
        self._evidence: dict[str, Evidence] = {}
        self._tool_calls: dict[str, int] = {}

    def record_call(self, tool_name: str) -> None:
        self._tool_calls[tool_name] = self._tool_calls.get(tool_name, 0) + 1

    def add(self, items: Sequence[Evidence]) -> list[dict[str, object]]:
        for item in items:
            self._evidence[str(item.id)] = item
        return [item.model_dump(mode="json") for item in items]

    @property
    def evidence(self) -> list[Evidence]:
        return list(self._evidence.values())

    @property
    def tool_calls(self) -> dict[str, int]:
        return dict(self._tool_calls)


def create_telemetry_tools(
    prometheus: PrometheusClient,
    jaeger: JaegerClient,
    opensearch: OpenSearchClient,
    *,
    start: datetime,
    end: datetime,
    collector: EvidenceCollector,
) -> list[BaseTool]:
    @tool
    async def query_metrics(expression: str) -> list[dict[str, object]]:
        """Run an instant PromQL query and return normalized metric evidence."""
        collector.record_call("query_metrics")
        return collector.add(await prometheus.query(expression, at=end))

    @tool
    async def search_traces(service: str) -> list[dict[str, object]]:
        """Find recent Jaeger traces for one service in the incident window."""
        collector.record_call("search_traces")
        return collector.add(
            await jaeger.search_traces(service, start=start, end=end)
        )

    @tool
    async def get_trace_detail(trace_id: str) -> list[dict[str, object]]:
        """Inspect one Jaeger trace and return its normalized error-span evidence."""
        collector.record_call("get_trace_detail")
        return collector.add(await jaeger.get_trace_detail(trace_id))

    @tool
    async def search_logs(
        service: str,
        text: str | None = None,
    ) -> list[dict[str, object]]:
        """Find OpenSearch logs for one service, optionally matching text."""
        collector.record_call("search_logs")
        return collector.add(
            await opensearch.search_logs(
                service,
                start=start,
                end=end,
                text=text,
            )
        )

    return [query_metrics, search_traces, get_trace_detail, search_logs]
