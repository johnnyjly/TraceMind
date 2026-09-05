"""Serve read-only telemetry tools over standalone Streamable HTTP."""

from contextlib import asynccontextmanager

import httpx
import uvicorn
from mcp.server.lowlevel import Server
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from mcp.server.transport_security import TransportSecuritySettings
from mcp.types import CallToolResult, TextContent, Tool, ToolAnnotations
from pydantic import ValidationError
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route

from otel_telemetry_mcp.backend import DirectTelemetryBackend, TelemetryBackend
from otel_telemetry_mcp.config import Settings
from otel_telemetry_mcp.contracts import (
    RemoteFailure, TOOL_ARGUMENTS, TelemetryReply,
)
from otel_telemetry_mcp.descriptions import (
    DESCRIPTIONS, PARAMETER_DESCRIPTIONS,
)
from otel_telemetry_mcp.failures import classify_failure
from otel_telemetry_mcp.telemetry.errors import TelemetryClientError
from otel_telemetry_mcp.telemetry.jaeger import JaegerClient
from otel_telemetry_mcp.telemetry.opensearch import OpenSearchClient
from otel_telemetry_mcp.telemetry.prometheus import PrometheusClient


class McpEndpoint:
    """Adapt the SDK ASGI handler without Starlette request conversion."""

    def __init__(self, manager: StreamableHTTPSessionManager) -> None:
        """Retain the app-owned session manager."""
        self.manager = manager

    async def __call__(self, scope, receive, send) -> None:
        """Forward HTTP requests without altering MCP response semantics."""
        await self.manager.handle_request(scope, receive, send)


def create_server(settings: Settings, backend: TelemetryBackend | None = None):
    """Create protocol handlers; each session owns its backend HTTP pool."""
    @asynccontextmanager
    async def lifespan(server):
        """Keep telemetry connections separate from client MCP sessions."""
        async with httpx.AsyncClient(
            timeout=settings.timeout_seconds,
        ) as client:
            yield backend or DirectTelemetryBackend(
                PrometheusClient(str(settings.prometheus_url), client),
                JaegerClient(str(settings.jaeger_url), client),
                OpenSearchClient(str(settings.opensearch_url),
                                 settings.opensearch_logs_index, client),
            )

    server = Server("otel-telemetry", lifespan=lifespan)

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        """Supply the authoritative descriptions, schemas and result types."""
        def schema(name, arguments):
            """Attach guidance to the service-owned parameter schema."""
            result = arguments.model_json_schema()
            for field, description in PARAMETER_DESCRIPTIONS[name].items():
                result["properties"][field]["description"] = description
            return result

        return [Tool(
            name=name, description=DESCRIPTIONS[name],
            inputSchema=schema(name, arguments),
            outputSchema=TelemetryReply.model_json_schema(),
            _meta={
                "tracemind/result-contract": "telemetry.v1",
                "tracemind/preview-limit": {
                    "query_metrics": 50, "discover_metrics": 1,
                    "search_traces": 20, "get_trace_detail": 100,
                    "search_logs": 100,
                }[name],
            },
            annotations=ToolAnnotations(readOnlyHint=True),
        ) for name, arguments in TOOL_ARGUMENTS.items()]

    @server.call_tool(validate_input=False)
    async def call_tool(name: str, arguments: dict) -> CallToolResult:
        """Validate the wire contract and return observations or metadata."""
        try:
            if name not in TOOL_ARGUMENTS:
                raise ValueError("unknown tool")
            args = TOOL_ARGUMENTS[name].model_validate(arguments)
        except (ValidationError, ValueError):
            reply = TelemetryReply(error=RemoteFailure(
                category="invalid_request", status_code=400))
        else:
            try:
                result = await getattr(
                    server.request_context.lifespan_context, name,
                )(**args.model_dump())
            except TelemetryClientError as exc:
                reply = TelemetryReply(error=classify_failure(exc))
            else:
                reply = (TelemetryReply(metadata=result)
                         if isinstance(result, dict)
                         else TelemetryReply(evidence=result))
        if len(reply.model_dump_json().encode()) > settings.response_bytes:
            reply = TelemetryReply(error=RemoteFailure(
                category="response_limit"))
        response = CallToolResult(
            content=[TextContent(type="text", text=reply.model_dump_json(
                exclude_none=True))],
            structuredContent=reply.model_dump(mode="json", exclude_none=True),
            isError=reply.error is not None,
        )
        if len(response.model_dump_json().encode()) > settings.response_bytes:
            failure = TelemetryReply(error=RemoteFailure(
                category="response_limit"))
            return CallToolResult(
                content=[TextContent(type="text", text="response_limit")],
                structuredContent=failure.model_dump(
                    mode="json", exclude_none=True),
                isError=True,
            )
        return response

    return server


def create_app(settings: Settings | None = None,
               backend: TelemetryBackend | None = None) -> Starlette:
    """Expose /mcp and a readiness endpoint without any agent runtime."""
    settings = settings or Settings()
    manager = StreamableHTTPSessionManager(
        create_server(settings, backend), json_response=True,
        security_settings=TransportSecuritySettings(
            allowed_hosts=settings.allowed_hosts,
            allowed_origins=settings.allowed_origins,
        ),
    )

    @asynccontextmanager
    async def lifespan(app):
        """Drain active protocol sessions when the service stops."""
        async with manager.run():
            yield

    async def health(request):
        """Report process readiness without querying telemetry backends."""
        return JSONResponse({"status": "ok"})

    return Starlette(routes=[Route("/health", health),
                             Route("/mcp", endpoint=McpEndpoint(manager))],
                     lifespan=lifespan)


def main() -> None:
    """Start the independently configured MCP HTTP service."""
    settings = Settings()
    uvicorn.run(create_app(settings), host=settings.host, port=settings.port)


if __name__ == "__main__":
    main()
