"""Adapt existing synthetic observations to the production MCP tool binder."""
from otel_telemetry_mcp.contracts import TOOL_ARGUMENTS
from otel_telemetry_mcp.telemetry.errors import (
    TelemetryClientError as ServerError,
)
from pydantic import ValidationError

from tracemind.agent.runtime.tools import bind_mcp_tools
from tracemind.evaluation.replay import recorded_tools
from tracemind.mcp.contracts import TelemetryReply
from tracemind.models import Evidence
from tracemind.telemetry.errors import (
    RemoteTelemetryError,
    TelemetryClientError,
)


class FixtureToolClient:
    transport = 'replay'

    def __init__(self, prometheus, jaeger, opensearch):
        self.prometheus, self.jaeger, self.opensearch = (
            prometheus, jaeger, opensearch
        )

    async def discover_tools(self):
        return recorded_tools()

    async def call(self, name, arguments):
        try:
            args = TOOL_ARGUMENTS[name].model_validate(arguments).model_dump()
        except ValidationError as exc:
            raise RemoteTelemetryError(
                'invalid_request',
                status_code=400) from exc
        method = {
            'query_metrics': (self.prometheus, 'query'),
            'discover_metrics': (self.prometheus, 'discover_metrics'),
            'search_traces': (self.jaeger, 'search_traces'),
            'get_trace_detail': (self.jaeger, 'get_trace_detail'),
            'search_logs': (self.opensearch, 'search_logs'),
        }
        target, operation = method[name]
        try:
            positional = {
                'query_metrics': 'expression',
                'search_traces': 'service',
                'get_trace_detail': 'trace_id',
                'search_logs': 'service',
            }
            if name in positional:
                value = args.pop(positional[name])
                result = await getattr(target, operation)(value, **args)
            else:
                result = await getattr(target, operation)(**args)
        except ServerError as exc:
            raise TelemetryClientError(str(exc)) from exc.__cause__
        if name == 'discover_metrics':
            return TelemetryReply(metadata=result)
        return TelemetryReply(evidence=[
            Evidence.model_validate(i.model_dump()) for i in result
        ])


def bind_fixture_tools(prometheus, jaeger, opensearch, **kwargs):
    return bind_mcp_tools(FixtureToolClient(prometheus, jaeger, opensearch),
                          recorded_tools(), **kwargs)
