"""Reject expired or changed captured traces before admitting model calls."""

from tracemind.mcp.client import HttpMcpClient
from tracemind.mcp.results import telemetry_reply
from tracemind.telemetry.errors import RemoteTelemetryError


async def verify_captured_traces(client, captured):
    """Compare whole-trace fingerprints through the same MCP as diagnosis."""
    expected = {}
    for item in captured["traces"]:
        attributes = item["attributes"]
        trace_id = attributes["trace_id"]
        coverage = attributes["coverage"]
        identity = (
            coverage["trace_fingerprint"], coverage["trace_span_count"],
        )
        if not identity[0] or identity[1] <= 0:
            raise ValueError("captured trace has no fingerprint or spans")
        if expected.setdefault(trace_id, identity) != identity:
            raise ValueError("captured trace changed while collecting pages")
    if not expected:
        raise ValueError("capture has no traces; recapture the window")
    await client.discover_tools()
    for trace_id, identity in sorted(expected.items()):
        try:
            reply = telemetry_reply(await client.call(
                "get_trace_detail", {"trace_id": trace_id, "limit": 1},
            ))
        except RemoteTelemetryError as exc:
            raise ValueError(
                f"captured trace {trace_id} unavailable ({exc.category}, "
                f"HTTP {exc.status_code}); recapture before evaluation"
            ) from exc
        if not reply.evidence:
            raise ValueError(f"captured trace {trace_id} returned no spans")
        attributes = reply.evidence[0].attributes
        coverage = attributes.get("coverage", {})
        if attributes.get("trace_id") != trace_id or (
            coverage.get("trace_fingerprint"), coverage.get("trace_span_count")
        ) != identity:
            raise ValueError(
                f"captured trace {trace_id} changed; recapture the window"
            )
    return {"checked_traces": len(expected), "fingerprints_match": True}


async def verify_live_window(url, captured):
    client = HttpMcpClient(url)
    try:
        return await verify_captured_traces(client, captured)
    finally:
        await client.aclose()
