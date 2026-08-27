"""Check the configured HTTP MCP server without models or telemetry queries."""

import asyncio
import logging

from tracemind.config import get_settings
from tracemind.mcp.client import HttpMcpClient
from tracemind.telemetry.errors import RemoteTelemetryError


logger = logging.getLogger("tracemind.mcp.check")


async def check() -> list[str]:
    """Initialize Streamable HTTP and list tools, then close the session."""
    settings = get_settings()
    backend = HttpMcpClient(
        str(settings.telemetry_mcp_url),
        timeout=settings.agent_tool_timeout_seconds,
        max_bytes=settings.telemetry_mcp_response_bytes,
    )
    try:
        return [tool.name for tool in await backend.discover_tools()]
    finally:
        await backend.aclose()


def main() -> None:
    """Report tool names or a safe failure category without payload logging."""
    logging.basicConfig(level=logging.WARNING,
                        format="%(levelname)s %(name)s %(message)s")
    logger.setLevel(logging.INFO)
    try:
        tools = asyncio.run(check())
    except RemoteTelemetryError as exc:
        logger.error("MCP check failed: %s", exc.category)
        raise SystemExit(1) from None
    except KeyboardInterrupt:
        logger.info("MCP check interrupted")
        raise SystemExit(130) from None
    logger.info("MCP Streamable HTTP ready: %s", ", ".join(tools))
    logger.info("Session closed; model calls: 0; telemetry queries: 0")


if __name__ == "__main__":
    main()
