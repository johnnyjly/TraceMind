"""Reply validation retained after removal of the stdio compatibility suite."""

import asyncio

import anyio
from mcp.types import CallToolResult, TextContent
import pytest

from tracemind.mcp.client import HttpMcpClient
from tracemind.mcp.results import telemetry_reply
from tracemind.telemetry.errors import RemoteTelemetryError


@pytest.mark.parametrize('payload,is_error,tool,category', [
    (None, False, 'query_metrics', 'malformed_reply'),
    ({
        'evidence': [],
        'metadata': {}}, False, 'query_metrics', 'malformed_reply'),
    ({'evidence': 'not a list'}, False, 'query_metrics', 'malformed_reply'),
    ({'metadata': []}, False, 'discover_metrics', 'malformed_reply'),
    ({'evidence': []}, True, 'query_metrics', 'malformed_reply'),
    ({'error': {'category': 'invalid_request', 'status_code': 400}}, True,
     'query_metrics', 'invalid_request'),
])
def test_invalid_reply_cannot_become_observations(
    monkeypatch,
    payload,
    is_error,
    tool, category,
):
    class Session:
        async def call_tool(self, name, arguments):
            return CallToolResult(
                content=[],
                structuredContent=payload,
                isError=is_error)

    client = HttpMcpClient('http://unused')

    async def connect():
        return Session()
    monkeypatch.setattr(client, '_connect', connect)

    async def exercise():
        return telemetry_reply(await client.call(tool, {}))
    with pytest.raises(RemoteTelemetryError) as caught:
        asyncio.run(exercise())
    assert caught.value.category == category


def test_decoded_reply_limit_is_independent_of_evidence_retention(monkeypatch):
    class Session:
        async def call_tool(self, name, arguments):
            return CallToolResult(
                content=[TextContent(type='text', text='x' * 2048)],
                structuredContent={'evidence': []},
            )
    client = HttpMcpClient('http://unused', max_bytes=1024)

    async def connect():
        return Session()
    monkeypatch.setattr(client, '_connect', connect)
    with pytest.raises(RemoteTelemetryError) as caught:
        asyncio.run(client.call('query_metrics', {}))
    assert caught.value.category == 'response_limit'


@pytest.mark.parametrize(
    'error',
    [
        anyio.EndOfStream,
        anyio.BrokenResourceError,
        anyio.ClosedResourceError,
    ],
)
def test_disconnected_rpc_is_classified_without_backend_details(
    monkeypatch,
    error,
):
    class Session:
        async def call_tool(self, name, arguments):
            raise error('private transport details')
    client = HttpMcpClient('http://unused')

    async def connect():
        return Session()
    monkeypatch.setattr(client, '_connect', connect)
    with pytest.raises(RemoteTelemetryError) as caught:
        asyncio.run(client.call('query_metrics', {}))
    assert caught.value.category == 'transport_closed'
    assert 'private' not in str(caught.value)
