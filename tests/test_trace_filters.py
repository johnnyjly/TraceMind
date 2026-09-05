"""Trace filters reach Jaeger before sampling; hints never become evidence."""
import asyncio
from datetime import UTC, datetime, timedelta
import json

import httpx
import pytest
from pydantic import ValidationError
from otel_telemetry_mcp.backend import DirectTelemetryBackend
from otel_telemetry_mcp.contracts import LogSearch, TraceSearch
from otel_telemetry_mcp.telemetry.errors import TelemetryClientError
from otel_telemetry_mcp.telemetry.jaeger import JaegerClient

from tracemind.evaluation.replay import ReplayResponse, ReplayTelemetry

END = datetime(2026, 10, 11, tzinfo=UTC)
START = END - timedelta(minutes=1)


def test_filters_are_sent_before_backend_limit_and_empty_returns_hints():
    async def exercise():
        def handler(request):
            if request.url.path == '/api/traces':
                params = request.url.params
                assert params['service'] == 'productcatalog'
                assert params['minDuration'] == '1000.000000000ms'
                assert json.loads(params['tags']) == {'error': 'true'}
                assert params['limit'] == '20'
                return httpx.Response(200, json={'data': []})
            assert request.url.path == '/api/services'
            return httpx.Response(200, json={'data': ['product-catalog']})
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler),
        ) as client:
            backend = DirectTelemetryBackend(None, JaegerClient(
                'http://jaeger', client), None)
            result = await backend.search_traces(
                'productcatalog', start=START, end=END,
                min_duration_ms=1000, errors_only=True)
        assert result['service_known'] is False
        assert result['available_services'] == ['product-catalog']
        assert result['filters']['errors_only'] is True
        assert result['catalogue_scope'] == 'global_not_incident_window'
        assert 'id' not in result
    asyncio.run(exercise())


def test_catalogue_is_bounded_but_membership_uses_full_catalogue():
    async def exercise():
        names = [f'service-{i:03}' for i in range(70)] + ['x' * 241]
        async with httpx.AsyncClient(transport=httpx.MockTransport(
            lambda r: httpx.Response(200, json={'data': names}),
        )) as client:
            result = await JaegerClient('http://jaeger', client).service_hints(
                'service-069')
        assert result['service_known'] is True
        assert len(result['available_services']) == 50
        assert result['omitted_services'] == 21
    asyncio.run(exercise())


@pytest.mark.parametrize('data', [None, {}, [123]])
def test_invalid_catalogue_is_not_silently_empty(data):
    async def exercise():
        async with httpx.AsyncClient(transport=httpx.MockTransport(
            lambda r: httpx.Response(200, json={'data': data}),
        )) as client:
            with pytest.raises(TelemetryClientError):
                await JaegerClient('http://jaeger', client).service_hints('a')
    asyncio.run(exercise())


@pytest.mark.parametrize('duration', [-1, 0, float('nan'), float('inf')])
def test_invalid_duration_and_no_log_schema_leak(duration):
    with pytest.raises(ValidationError):
        TraceSearch(service='a', start=START, end=END,
                    min_duration_ms=duration)
    with pytest.raises(ValidationError):
        LogSearch(service='a', start=START, end=END, errors_only=True)


def test_replay_preserves_filter_types_and_metadata_envelope():
    async def exercise():
        args = {'service': 'a', 'min_duration_ms': 123.5, 'errors_only': True}
        response = ReplayResponse(method='search_traces', arguments=args,
                                  metadata={'kind': 'trace_search_hints'})
        assert response.arguments == args
        assert response.arguments['errors_only'] is True
        reply = await ReplayTelemetry([response]).call('search_traces', args)
        assert reply.structuredContent == {
            'metadata': {'kind': 'trace_search_hints'}}
    asyncio.run(exercise())
