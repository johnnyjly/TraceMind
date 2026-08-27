"""Ensure successful slow dependencies remain visible to MCP clients."""

import asyncio
from pathlib import Path
import sys

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]
                       / "services/otel-telemetry-mcp/src"))

from otel_telemetry_mcp.telemetry.jaeger import JaegerClient  # noqa: E402


def test_details_include_slow_successful_dependency_and_preserve_errors():
    trace_id = "a" * 32
    spans = [{
        "spanID": f"{i:016x}", "processID": "shipping",
        "operationName": "POST /ship-order", "startTime": 1725148800000000,
        "duration": (i + 1) * 1000, "tags": [],
        "references": [{"refType": "CHILD_OF", "traceID": trace_id,
                        "spanID": "b" * 16}],
    } for i in range(25)]
    failed = {**spans[0], "spanID": "c" * 16,
              "tags": [{"key": "error", "value": True}]}
    requests = []

    def handler(request):
        """Return both an error and many successful spans in one trace."""
        requests.append(request)
        return httpx.Response(200, json={"data": [{
            "spans": [failed, *spans], "processes": {
                "shipping": {"serviceName": "shipping"}},
        }]})

    async def detail():
        """Exercise the independently packaged telemetry decoder."""
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler),
        ) as http:
            return await JaegerClient("http://jaeger", http)\
                .get_trace_detail(trace_id)

    items = asyncio.run(detail())
    assert len(requests) == 1
    assert len(items) == 21
    assert items[0].summary == "Error span in shipping: POST /ship-order"
    assert items[0].attributes["is_error"] is True
    assert [e.attributes["duration_ms"] for e in items[1:]] == list(
        range(25, 5, -1))
    assert all(not e.attributes["is_error"] for e in items[1:])
    assert items[1].service == "shipping"
    assert items[1].attributes["parent_span_id"] == "b" * 16
