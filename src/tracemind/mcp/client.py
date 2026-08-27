"""MCP sessions with an HTTP default and deprecated stdio compatibility."""

import asyncio
import logging
import sys
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import cast

import anyio
import httpx
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamable_http_client
from mcp.shared.exceptions import McpError
from mcp.types import Tool
from pydantic import ValidationError

from tracemind.config import Settings
from tracemind.mcp.contracts import TOOL_ARGUMENTS, TelemetryReply
from tracemind.models import Evidence
from tracemind.telemetry.errors import RemoteTelemetryError


logger = logging.getLogger(__name__)


def server_parameters(settings: Settings) -> StdioServerParameters:
    """Build the deprecated stdio launch configuration without model keys."""
    return StdioServerParameters(
        command=sys.executable, args=["-m", "tracemind.mcp.server"],
        env={
            "TRACEMIND_PROMETHEUS_URL": str(settings.prometheus_url),
            "TRACEMIND_JAEGER_URL": str(settings.jaeger_url),
            "TRACEMIND_OPENSEARCH_URL": str(settings.opensearch_url),
            "TRACEMIND_OPENSEARCH_LOGS_INDEX": settings.opensearch_logs_index,
            "TRACEMIND_AGENT_TOOL_TIMEOUT_SECONDS": str(
                settings.agent_tool_timeout_seconds),
            "TRACEMIND_TELEMETRY_MCP_RESPONSE_BYTES": str(
                settings.telemetry_mcp_response_bytes),
        },
    )


class _McpSession:
    """Share protocol validation and single-task session ownership."""

    transport = "mcp"

    def __init__(
        self, *, timeout: float = 20,
        max_bytes: int = 16_777_216,
    ) -> None:
        """Initialize session ownership without opening a transport."""
        self.timeout = timeout
        self.max_bytes = max_bytes
        self._ready = asyncio.Event()
        self._stop = asyncio.Event()
        self._task: asyncio.Task | None = None
        self._session: ClientSession | None = None
        self._error: Exception | None = None
        self._closed = False
        self._tool_names: tuple[str, ...] = ()

    async def _serve(self) -> None:
        """Own one transport in the concrete HTTP or legacy stdio client."""
        raise NotImplementedError

    async def _connect(self) -> ClientSession:
        """Start at most one session, including concurrent first tool calls."""
        if self._closed:
            raise RemoteTelemetryError("transport_closed")
        if self._task is None:
            self._task = asyncio.create_task(self._serve())
        await self._ready.wait()
        session = self._session
        if session is None:
            raise RemoteTelemetryError("transport_closed") from self._error
        return session

    async def available_tools(self) -> list[str]:
        """Check initialization and discovery without querying telemetry."""
        await self._connect()
        return list(self._tool_names)

    async def _call(self, name: str, arguments: dict) -> TelemetryReply:
        """Validate typed replies; reconnect and retries are not implicit."""
        session = await self._connect()
        try:
            result = await session.call_tool(name, arguments)
        except McpError as exc:
            code = exc.error.code
            category = ("invalid_request" if code in {-32600, -32601, -32602}
                        else "transient" if code == 408
                        else "transport_closed")
            raise RemoteTelemetryError(category, retryable=code == 408,
                                       status_code=408 if code == 408
                                       else None) from exc
        except (anyio.BrokenResourceError, anyio.ClosedResourceError,
                anyio.EndOfStream) as exc:
            raise RemoteTelemetryError("transport_closed") from exc
        except (RuntimeError, ValidationError) as exc:
            # The SDK uses RuntimeError for output-schema validation errors.
            raise RemoteTelemetryError("malformed_reply") from exc
        if len(result.model_dump_json().encode()) > self.max_bytes:
            raise RemoteTelemetryError("response_limit")
        try:
            reply = TelemetryReply.model_validate(result.structuredContent)
        except ValidationError as exc:
            raise RemoteTelemetryError("malformed_reply") from exc
        if reply.error is not None:
            raise RemoteTelemetryError(**reply.error.model_dump())
        if result.isError:
            raise RemoteTelemetryError("malformed_reply")
        if (name == "discover_metrics") != (reply.metadata is not None):
            raise RemoteTelemetryError("malformed_reply")
        return reply

    async def aclose(self) -> None:
        """Close the transport owner, including interrupted initialization."""
        self._closed = True
        self._stop.set()
        if self._task is not None:
            if not self._ready.is_set():
                self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)


class McpTelemetryBackend(_McpSession):
    """Deprecated stdio backend for explicit compatibility callers only."""

    def __init__(
        self, parameters: StdioServerParameters, *, timeout: float = 20,
        max_bytes: int = 16_777_216,
    ) -> None:
        """Retain the trusted legacy subprocess command without starting it."""
        super().__init__(timeout=timeout, max_bytes=max_bytes)
        self.parameters = parameters

    async def _serve(self) -> None:
        """Own initialization, transport pumps and shutdown in one task."""
        try:
            async with stdio_client(self.parameters) as (read, write):
                async with ClientSession(
                    read, write,
                    read_timeout_seconds=timedelta(seconds=self.timeout),
                ) as session:
                    await session.initialize()
                    tools = await session.list_tools()
                    if not set(TOOL_ARGUMENTS).issubset(
                        {tool.name for tool in tools.tools}
                    ):
                        raise RemoteTelemetryError("tool_contract_mismatch")
                    self._tool_names = tuple(tool.name for tool in tools.tools
                                             if tool.name in TOOL_ARGUMENTS)
                    self._session = session
                    self._ready.set()
                    await self._stop.wait()
        except Exception as exc:
            # SDK transport task groups may wrap failures. Retain the cause
            # for inspection, but never print remote payloads or error text.
            self._error = exc
            logger.warning("MCP session ended (%s)", type(exc).__name__)
        finally:
            self._session = None
            self._ready.set()

    async def query_metrics(
        self, expression: str, *, at: datetime,
    ) -> list[Evidence]:
        """Read normalized instant samples over MCP."""
        reply = await self._call("query_metrics", {
            "expression": expression, "at": at.isoformat(),
        })
        return cast(list[Evidence], reply.evidence)

    async def discover_metrics(
        self, *, service: str | None, metric: str | None,
        start: datetime, end: datetime, limit: int,
    ) -> dict:
        """Read metadata without converting it into Evidence."""
        reply = await self._call("discover_metrics", {
            "service": service, "metric": metric, "limit": limit,
            "start": start.isoformat(), "end": end.isoformat(),
        })
        return cast(dict, reply.metadata)

    async def search_traces(
        self, service: str, *, start: datetime, end: datetime,
    ) -> list[Evidence]:
        """Read one service's trace summaries over MCP."""
        reply = await self._call("search_traces", {
            "service": service,
            "start": start.isoformat(), "end": end.isoformat(),
        })
        return cast(list[Evidence], reply.evidence)

    async def get_trace_detail(self, trace_id: str) -> list[Evidence]:
        """Read a known trace's normalized error spans over MCP."""
        reply = await self._call("get_trace_detail", {"trace_id": trace_id})
        return cast(list[Evidence], reply.evidence)

    async def search_logs(
        self, service: str, *, start: datetime, end: datetime,
        text: str | None = None,
    ) -> list[Evidence]:
        """Read service logs with the application-selected window."""
        reply = await self._call("search_logs", {
            "service": service, "text": text,
            "start": start.isoformat(), "end": end.isoformat(),
        })
        return cast(list[Evidence], reply.evidence)


class HttpMcpClient(_McpSession):
    """Discover authoritative tool definitions over Streamable HTTP.

    The shared session owner handles SDK context lifetimes. No local query
    implementation or per-tool backend method is used by HTTP consumers.
    """

    def __init__(self, url: str, *, timeout: float = 20,
                 max_bytes: int = 16_777_216) -> None:
        """Capture connection settings; discovery starts the session lazily."""
        super().__init__(timeout=timeout, max_bytes=max_bytes)
        self.url = url
        self._definitions: list[Tool] = []

    async def _serve(self) -> None:
        """Own the HTTP pool, MCP handshake and bounded tool discovery."""
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                async with streamable_http_client(
                    self.url, http_client=client,
                ) as (read, write, _):
                    async with ClientSession(
                        read, write,
                        read_timeout_seconds=timedelta(seconds=self.timeout),
                    ) as session:
                        await session.initialize()
                        cursor = None
                        seen = set()
                        while True:
                            page = await session.list_tools(cursor=cursor)
                            size = len(page.model_dump_json().encode())
                            if size > self.max_bytes:
                                raise RemoteTelemetryError("response_limit")
                            self._definitions.extend(page.tools)
                            if len(self._definitions) > 100:
                                raise RemoteTelemetryError(
                                    "tool_contract_mismatch")
                            cursor = page.nextCursor
                            if cursor is None:
                                break
                            if cursor in seen:
                                raise RemoteTelemetryError(
                                    "tool_contract_mismatch")
                            seen.add(cursor)
                        names = [t.name for t in self._definitions]
                        if len(names) != len(set(names)):
                            raise RemoteTelemetryError(
                                "tool_contract_mismatch")
                        self._tool_names = tuple(names)
                        self._session = session
                        self._ready.set()
                        await self._stop.wait()
        except Exception as exc:
            self._error = exc
            logger.warning("MCP HTTP session ended (%s)", type(exc).__name__)
        finally:
            self._session = None
            self._ready.set()

    async def discover_tools(self) -> list[Tool]:
        """Return session tool definitions within a discovery deadline."""
        try:
            async with asyncio.timeout(self.timeout):
                await self._connect()
        except TimeoutError as exc:
            raise RemoteTelemetryError("transient", retryable=True) from exc
        return list(self._definitions)

    async def call(self, name: str, arguments: dict) -> TelemetryReply:
        """Execute an authorized tool without a query-specific wrapper."""
        return await self._call(name, arguments)


@asynccontextmanager
async def open_telemetry_backend(settings: Settings):
    """Scope an MCP session to one request, worker run or evaluation."""
    if settings.telemetry_transport == "direct":
        yield None
        return
    backend = HttpMcpClient(
        str(settings.telemetry_mcp_url),
        timeout=settings.agent_tool_timeout_seconds,
        max_bytes=settings.telemetry_mcp_response_bytes,
    )
    try:
        yield backend
    finally:
        await backend.aclose()
