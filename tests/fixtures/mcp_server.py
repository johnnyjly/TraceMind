"""Real SDK subprocess server with deterministic telemetry."""

import asyncio
import os
import sys
from datetime import timedelta
from uuid import UUID

import httpx
from mcp.server.stdio import stdio_server
from mcp.types import CallToolResult

from tracemind.config import Settings
from tracemind.mcp.server import create_server
from tracemind.models import Evidence, EvidenceSource
from tracemind.telemetry import (
    JaegerClient, OpenSearchClient, PrometheusClient,
)
from tracemind.telemetry.backend import DirectTelemetryBackend
from tracemind.telemetry.errors import TelemetryClientError


class FixtureBackend:
    """Provide five deterministic operations and injected external failures."""

    attempts = 0

    async def query_metrics(self, expression, *, at):
        """Return a fixed sample or an explicitly selected failure mode."""
        if expression == "crash":
            os._exit(0)
        if expression == "slow":
            await asyncio.sleep(2)
        if expression == "retry":
            self.attempts += 1
            if self.attempts == 1:
                request = httpx.Request("GET", "http://fixture/metrics")
                error = httpx.HTTPStatusError(
                    "remote-secret-must-not-leak", request=request,
                    response=httpx.Response(503, request=request,
                                            headers={"Retry-After": "0"}),
                )
                raise TelemetryClientError(
                    "private backend payload") from error
        return [Evidence(
            id=UUID(int=1), source=EvidenceSource.METRIC,
            summary=("x" * 2048 if expression == "oversize"
                     else "Metric payment_errors_total=42"), observed_at=at,
            service="payment", query=expression,
            attributes={"value": "42", "pid": os.getpid()},
        )]

    async def discover_metrics(self, **arguments):
        """Return metadata with no observation IDs."""
        return {"metrics": [{"metric": "payment_errors_total",
                             "label_keys": ["service_name"],
                             "example_labels": [{"service_name": "payment"}]}]}

    async def search_traces(self, service, *, start, end):
        """Return a fixed trace summary with a usable trace identifier."""
        return [Evidence(
            id=UUID(int=2), source=EvidenceSource.TRACE,
            summary="payment span failed", observed_at=end, service=service,
            attributes={"trace_id": "0123456789abcdef"},
        )]

    async def get_trace_detail(self, trace_id):
        """Return a fixed error span for the requested trace."""
        from datetime import UTC, datetime

        return [Evidence(
            id=UUID(int=3), source=EvidenceSource.TRACE,
            summary="charge failed", service="payment",
            observed_at=datetime(2024, 9, 1, 1, tzinfo=UTC),
            attributes={"trace_id": trace_id},
        )]

    async def search_logs(self, service, *, start, end, text=None):
        """Include an old observation to exercise window filtering."""
        return [Evidence(
            id=UUID(int=4), source=EvidenceSource.LOG,
            summary=text or "payment error", service=service, observed_at=end,
        ), Evidence(
            id=UUID(int=5), source=EvidenceSource.LOG,
            summary="second observation", service=service,
            observed_at=(end if text == "capacity"
                         else start - timedelta(seconds=1)),
        )]


async def main():
    """Run the actual stdio protocol, optionally returning malformed output."""
    if len(sys.argv) > 1 and sys.argv[1] == "startup":
        await asyncio.sleep(5)
    if len(sys.argv) > 1 and sys.argv[1] == "http":
        def response(request):
            """Serve a real Prometheus response to the production decoder."""
            assert request.url.params["query"] == "up"
            assert request.url.params["time"] == "2024-09-01T01:00:00+00:00"
            return httpx.Response(200, json={
                "status": "success", "data": {
                    "resultType": "vector", "result": [{
                        "metric": {"__name__": "payment_errors_total",
                                   "service_name": "payment"},
                        "value": [1725152400, "42"],
                    }],
                },
            })

        async with httpx.AsyncClient(
            transport=httpx.MockTransport(response),
        ) as client:
            backend = DirectTelemetryBackend(
                PrometheusClient("http://fixture", client),
                JaegerClient("http://fixture", client),
                OpenSearchClient("http://fixture", "logs", client),
            )
            server = create_server(Settings(_env_file=None), backend)
            async with stdio_server() as (read, write):
                await server.run(read, write,
                                 server.create_initialization_options())
        return
    server = create_server(Settings(_env_file=None), FixtureBackend())
    if len(sys.argv) > 1 and sys.argv[1] == "malformed":
        @server.call_tool()
        async def malformed(name, arguments):
            return CallToolResult(content=[], structuredContent={"bogus": 1})

    async with stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
