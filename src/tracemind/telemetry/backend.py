"""Deprecated direct/stdio contract retained for compatibility and replay."""

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

from tracemind.models import Evidence
from tracemind.telemetry import (
    JaegerClient, OpenSearchClient, PrometheusClient,
)


class TelemetryBackend(Protocol):
    """Supply observations or metadata without owning execution policy."""

    transport: str

    async def query_metrics(
        self, expression: str, *, at: datetime,
    ) -> list[Evidence]:
        """Return normalized instant-vector samples."""
        ...

    async def discover_metrics(
        self, *, service: str | None, metric: str | None,
        start: datetime, end: datetime, limit: int,
    ) -> dict:
        """Return metric metadata, never diagnostic observations."""
        ...

    async def search_traces(
        self, service: str, *, start: datetime, end: datetime,
    ) -> list[Evidence]:
        """Return trace summaries for one service and investigation window."""
        ...

    async def get_trace_detail(self, trace_id: str) -> list[Evidence]:
        """Return error-span observations from one known trace."""
        ...

    async def search_logs(
        self, service: str, *, start: datetime, end: datetime,
        text: str | None = None,
    ) -> list[Evidence]:
        """Return service logs with an optional body-text filter."""
        ...


@dataclass
class DirectTelemetryBackend:
    """Delegate to existing clients, preserving their limits and decoders."""

    prometheus: PrometheusClient
    jaeger: JaegerClient
    opensearch: OpenSearchClient
    transport: str = "direct"

    async def query_metrics(
        self, expression: str, *, at: datetime,
    ) -> list[Evidence]:
        """Read instant samples using the existing Prometheus client."""
        return await self.prometheus.query(expression, at=at)

    async def discover_metrics(
        self, *, service: str | None, metric: str | None,
        start: datetime, end: datetime, limit: int,
    ) -> dict:
        """Read bounded names and labels without creating Evidence."""
        return await self.prometheus.discover_metrics(
            service=service, metric=metric, start=start, end=end, limit=limit,
        )

    async def search_traces(
        self, service: str, *, start: datetime, end: datetime,
    ) -> list[Evidence]:
        """Read service traces with the client's existing result limit."""
        return await self.jaeger.search_traces(service, start=start, end=end)

    async def get_trace_detail(self, trace_id: str) -> list[Evidence]:
        """Read normalized error spans from a known trace."""
        return await self.jaeger.get_trace_detail(trace_id)

    async def search_logs(
        self, service: str, *, start: datetime, end: datetime,
        text: str | None = None,
    ) -> list[Evidence]:
        """Read service logs with the client's existing result limit."""
        return await self.opensearch.search_logs(
            service, start=start, end=end, text=text,
        )
