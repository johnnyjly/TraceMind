"""Adapt explicitly declared telemetry envelopes or generic MCP references."""

from mcp.types import CallToolResult, Tool
from pydantic import ValidationError

from tracemind.mcp.contracts import TelemetryReply
from tracemind.telemetry.errors import RemoteTelemetryError


RESULT_CONTRACT_KEY = "tracemind/result-contract"
TELEMETRY_CONTRACT = "telemetry.v1"


def uses_telemetry_contract(definition: Tool) -> bool:
    """Select a domain adapter by contract, never by tool name or payload."""
    contract = (definition.meta or {}).get(RESULT_CONTRACT_KEY)
    if contract is not None and contract != TELEMETRY_CONTRACT:
        raise RemoteTelemetryError("tool_contract_mismatch")
    if contract is None:
        # Deployed telemetry servers predate the explicit version marker.
        # Their full declared output schema still identifies this contract.
        return definition.outputSchema == TelemetryReply.model_json_schema()
    return contract == TELEMETRY_CONTRACT


def telemetry_reply(result: CallToolResult) -> TelemetryReply:
    """Validate observations and retain the existing safe retry envelope."""
    try:
        reply = TelemetryReply.model_validate(result.structuredContent)
    except ValidationError as exc:
        raise RemoteTelemetryError("malformed_reply") from exc
    if reply.error is not None:
        raise RemoteTelemetryError(**reply.error.model_dump())
    if result.isError:
        raise RemoteTelemetryError("malformed_reply")
    return reply


def reference_content(result: CallToolResult) -> list[dict]:
    """Expose text/JSON without generating evidence IDs or leaking errors.

    Structured content is authoritative when present; the text block often
    duplicates it. Multimedia and resource content require separate adapters.
    """
    if result.isError:
        raise RemoteTelemetryError("invalid_request")
    if any(block.type != "text" for block in result.content):
        raise RemoteTelemetryError("unsupported_content")
    if result.structuredContent is not None:
        return [result.structuredContent]
    return [{"text": block.text} for block in result.content]
