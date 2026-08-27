"""Deprecated stdio compatibility server; HTTP service lives in services/."""

import asyncio
from contextlib import asynccontextmanager

import httpx
from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server
from mcp.types import CallToolResult, TextContent, Tool, ToolAnnotations
from pydantic import ValidationError

from tracemind.agent.recovery import classify_failure
from tracemind.config import Settings
from tracemind.mcp.contracts import (
    RemoteFailure, TOOL_ARGUMENTS, TelemetryReply,
)
from tracemind.telemetry import (
    JaegerClient, OpenSearchClient, PrometheusClient, TelemetryClientError,
)
from tracemind.telemetry.backend import (
    DirectTelemetryBackend, TelemetryBackend,
)


def create_server(
    settings: Settings, backend: TelemetryBackend | None = None,
) -> Server:
    """Create a fixed tool set; the process lifespan owns the HTTP client."""
    @asynccontextmanager
    async def lifespan(server):
        """Reuse one HTTP pool until the MCP process shuts down."""
        async with httpx.AsyncClient(
            timeout=settings.agent_tool_timeout_seconds,
        ) as client:
            yield backend or DirectTelemetryBackend(
                PrometheusClient(str(settings.prometheus_url), client),
                JaegerClient(str(settings.jaeger_url), client),
                OpenSearchClient(str(settings.opensearch_url),
                                 settings.opensearch_logs_index, client),
            )

    server = Server("tracemind-telemetry", lifespan=lifespan)

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        """Advertise only the five typed, read-only telemetry operations."""
        return [Tool(
            name=name, description=arguments.__doc__,
            inputSchema=arguments.model_json_schema(),
            outputSchema=TelemetryReply.model_json_schema(),
            annotations=ToolAnnotations(readOnlyHint=True),
        ) for name, arguments in TOOL_ARGUMENTS.items()]

    @server.call_tool(validate_input=False)
    async def call_tool(name: str, arguments: dict) -> CallToolResult:
        """Validate transport arguments and return normalized, bounded data."""
        try:
            if name not in TOOL_ARGUMENTS:
                raise ValueError("unknown tool")
            args = TOOL_ARGUMENTS[name].model_validate(arguments)
        except (ValidationError, ValueError):
            reply = TelemetryReply(error=RemoteFailure(
                category="invalid_request", status_code=400,
            ))
        else:
            source = server.request_context.lifespan_context
            try:
                result = await getattr(source, name)(**args.model_dump())
            except TelemetryClientError as exc:
                failure = classify_failure(exc)
                reply = TelemetryReply(error=RemoteFailure(
                    category=failure.category, retryable=failure.retryable,
                    status_code=failure.status_code,
                    retry_after=failure.retry_after,
                ))
            else:
                reply = (TelemetryReply(metadata=result)
                         if name == "discover_metrics"
                         else TelemetryReply(evidence=result))
        encoded = reply.model_dump_json()
        if len(encoded.encode()) > settings.telemetry_mcp_response_bytes:
            reply = TelemetryReply(error=RemoteFailure(
                category="response_limit",
            ))
        return CallToolResult(
            content=[TextContent(type="text", text="Telemetry reply")],
            structuredContent=reply.model_dump(mode="json", exclude_none=True),
            isError=reply.error is not None,
        )

    return server


async def serve() -> None:
    """Serve stdio using environment endpoints without loading .env or keys."""
    server = create_server(Settings(_env_file=None))
    async with stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(serve())
