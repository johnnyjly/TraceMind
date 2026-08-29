"""HTTP MCP client and the tool discovery/execution boundary."""

import asyncio
from contextlib import asynccontextmanager
from datetime import timedelta
import logging
from typing import Protocol

import anyio
import httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from mcp.shared.exceptions import McpError
from mcp.types import Tool
from pydantic import ValidationError

from tracemind.config import Settings
from tracemind.mcp.contracts import TelemetryReply
from tracemind.telemetry.errors import RemoteTelemetryError


logger = logging.getLogger(__name__)


class ToolClient(Protocol):
    """Tool source consumed by diagnosis, implemented by HTTP and replay."""

    transport: str

    async def discover_tools(self) -> list[Tool]: ...

    async def call(self, name: str, arguments: dict) -> TelemetryReply: ...


class HttpMcpClient:
    """Own one HTTP MCP session with validated replies and discovered tools."""

    transport = "mcp"

    def __init__(
        self, url: str, *, timeout: float = 20,
        max_bytes: int = 16_777_216,
    ) -> None:
        """Initialize session ownership without opening a transport."""
        self.url = url
        self._definitions: list[Tool] = []
        self.timeout = timeout
        self.max_bytes = max_bytes
        self._ready = asyncio.Event()
        self._stop = asyncio.Event()
        self._task: asyncio.Task | None = None
        self._session: ClientSession | None = None
        self._error: Exception | None = None
        self._closed = False

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

    async def call(self, name: str, arguments: dict) -> TelemetryReply:
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


@asynccontextmanager
async def open_telemetry_backend(settings: Settings):
    """Scope an MCP session to one request, worker run or evaluation."""
    backend = HttpMcpClient(
        str(settings.telemetry_mcp_url),
        timeout=settings.agent_tool_timeout_seconds,
        max_bytes=settings.telemetry_mcp_response_bytes,
    )
    try:
        yield backend
    finally:
        await backend.aclose()
