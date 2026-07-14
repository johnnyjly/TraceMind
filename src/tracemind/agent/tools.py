from collections.abc import Sequence
from datetime import datetime

from langchain_core.tools import BaseTool, tool

from tracemind.models import Evidence
from tracemind.telemetry import JaegerClient, OpenSearchClient, PrometheusClient


class EvidenceCollector:
    def __init__(self) -> None:
        self._evidence: dict[str, Evidence] = {}

    def add(self, items: Sequence[Evidence]) -> list[dict[str, object]]:
        for item in items:
            self._evidence[str(item.id)] = item
        return [item.model_dump(mode="json") for item in items]

    @property
    def evidence(self) -> list[Evidence]:
        return list(self._evidence.values())


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
        return collector.add(await prometheus.query(expression, at=end))

    @tool
    async def search_traces(service: str) -> list[dict[str, object]]:
        """Find recent Jaeger traces for one service in the incident window."""
        return collector.add(
            await jaeger.search_traces(service, start=start, end=end)
        )

    @tool
    async def search_logs(
        service: str,
        text: str | None = None,
    ) -> list[dict[str, object]]:
        """Find OpenSearch logs for one service, optionally matching text."""
        return collector.add(
            await opensearch.search_logs(
                service,
                start=start,
                end=end,
                text=text,
            )
        )

    return [query_metrics, search_traces, search_logs]
