"""Keep source context bounded, attributable and explicitly sanitized."""

import asyncio
from datetime import UTC, datetime
import json

import httpx
from otel_telemetry_mcp.backend import DirectTelemetryBackend
from otel_telemetry_mcp.contracts import TraceDetail
from otel_telemetry_mcp.telemetry.jaeger import JaegerClient
from otel_telemetry_mcp.telemetry.opensearch import OpenSearchClient
from pydantic import ValidationError
import pytest


TRACE_ID = "a" * 32
SPAN_ID = "b" * 16
START_US = 1_725_148_800_000_000


def tags(values):
    return [{"key": key, "value": value} for key, value in values.items()]


def span_item(*, span_tags=None, events=None, resource=None):
    return JaegerClient._span_to_evidence({
        "spanID": SPAN_ID, "processID": "p", "operationName": "Charge",
        "startTime": START_US, "duration": 3_000_000,
        "tags": tags(span_tags or {}), "logs": events or [],
    }, {"p": {"serviceName": "payment", "tags": tags(resource or {})}},
        TRACE_ID)


def log_item(*, attributes=None, resource=None, body="charge failed",
             event_time="2024-09-01T00:00:01Z"):
    return OpenSearchClient._to_evidence({
        "_id": "log-1", "_source": {
            "body": body, "@timestamp": event_time,
            "observedTimestamp": "2024-09-01T00:00:03Z",
            "traceId": TRACE_ID, "spanId": SPAN_ID,
            "resource": resource or {}, "attributes": attributes or {},
        },
    }, "payment", None)


def test_jaeger_preserves_resource_alias_conflicts_and_exact_code_fields():
    item = span_item(resource={
        "service.version": "1.8.0", "service.namespace": "checkout",
        "service.instance.id": "payment-2",
        "deployment.environment": "production",
        "k8s.pod.name": "payment-2-abc", "k8s.namespace.name": "shop",
        "authorization": "private-resource-secret",
        "process.command_args": "private-command-secret",
    }, span_tags={
        "service.version": "incorrect-span-version",
        "deployment.environment.name": "staging",
        "host.name": "host-1", "code.file.path": "/app/charge.py",
        "code.filepath": "/old/charge.py", "code.line.number": 42,
        "code.lineno": 41, "code.function.name": "Charge.run",
        "code.function": "run", "code.namespace": "Charge",
    })
    attrs = item.attributes
    assert attrs["resource"]["service.version"] == "1.8.0"
    assert attrs["resource"]["deployment.environment.name"] == "production"
    assert attrs["resource"]["host.name"] == "host-1"
    meta = attrs["resource_metadata"]
    assert meta["provenance"]["deployment.environment.name"] == {
        "source": "jaeger.process.tags", "field": "deployment.environment",
    }
    assert [c["value"] for c in meta["conflicts"]["service.version"]] == [
        "1.8.0", "incorrect-span-version",
    ]
    assert attrs["code"]["file"] == "/app/charge.py"
    assert attrs["code"]["line"] == 42
    assert attrs["code"]["function"] == "Charge.run"
    assert attrs["code"]["namespace"] == "Charge"
    assert attrs["code"]["provenance"]["file"] == {
        "source": "jaeger.span.tags", "field": "code.file.path",
    }
    assert [c["value"] for c in attrs["code"]["conflicts"]["line"]] == [42, 41]
    assert "private-" not in item.model_dump_json()


def test_jaeger_exception_event_time_is_distinct_and_does_not_set_error():
    item = span_item(events=[{
        "timestamp": START_US + 1_500_000,
        "fields": tags({"event": "exception", "exception.type": "Timeout",
                        "exception.message": "dependency timed out",
                        "exception.stacktrace": "Timeout\n at Charge.run",
                        "code.filepath": "/app/charge.py", "code.lineno": 19}),
    }, {"timestamp": START_US + 2_000_000,
        "fields": tags({"event": "retry", "message": "recovered"})}])
    assert item.attributes["is_error"] is False
    assert item.observed_at == datetime(2024, 9, 1, tzinfo=UTC)
    event, = item.attributes["exceptions"]
    assert event["timestamp"] == "2024-09-01T00:00:01.500000+00:00"
    assert event["source"] == "jaeger.span.logs"
    assert event["event_index"] == 0
    assert event["code"]["line"] == 19
    assert item.attributes["exception_coverage"]["total_events"] == 1
    assert not item.attributes["exception_coverage"]["truncated"]


def test_invalid_exception_time_and_untimed_tags_do_not_borrow_span_time():
    item = span_item(span_tags={"exception.type": "Timeout"}, events=[{
        "timestamp": "bad-time", "fields": tags({"exception.type": "Failure"}),
    }])
    tag_event, log_event = item.attributes["exceptions"]
    assert "timestamp" not in tag_event
    assert "timestamp" not in log_event
    assert log_event["timestamp_invalid"]
    assert tag_event["source"] == "jaeger.span.tags"


def test_log_resource_nested_fields_and_modern_alias_precedence():
    item = log_item(resource={
        "service": {"name": "payment", "version": "1.8.0", "instance": {
            "id": "pod-7"}},
        "deployment.environment.name": "prod",
        "deployment.environment": "legacy-prod",
        "request.body": "private-payload",
    }, attributes={
        "service.version": "1.9.0", "code.filepath": "/app/charge.js",
        "code.lineno": 71, "code.function": "charge",
        "exception.type": "TypeError", "exception.message": "bad input",
    })
    assert item.attributes["resource"]["service.instance.id"] == "pod-7"
    assert item.attributes["resource"]["deployment.environment.name"] == "prod"
    assert item.attributes["resource_metadata"]["provenance"][
        "service.version"] == {
            "source": "opensearch.resource", "field": "service.version",
    }
    assert "service.version" in item.attributes[
        "resource_metadata"]["conflicts"]
    assert item.attributes["code"]["file"] == "/app/charge.js"
    event, = item.attributes["exceptions"]
    assert event["timestamp"] == "2024-09-01T00:00:01+00:00"
    assert event["source"] == "opensearch.attributes"
    assert event["code"]["line"] == 71
    assert "private-payload" not in item.model_dump_json()


@pytest.mark.parametrize("event_time", [None, "1970-01-01T00:00:00Z"])
def test_log_exception_does_not_invent_event_time_from_observation(event_time):
    item = log_item(event_time=event_time, attributes={
        "exception.type": "Timeout", "exception.message": "late observation",
    })
    assert item.observed_at == datetime(2024, 9, 1, 0, 0, 3, tzinfo=UTC)
    assert item.attributes["timestamp_source"] == "observedTimestamp"
    assert "timestamp" not in item.attributes["exceptions"][0]


@pytest.mark.parametrize("payload,secrets", [
    ("Authorization: Bearer auth-private-value", ["auth-private-value"]),
    ("Cookie: session=cookie-private; csrf=second-private",
     ["cookie-private", "second-private"]),
    ('password="password-private" api_key=api-private',
     ["password-private", "api-private"]),
    ("cookie=session=cookie-private; csrf=second-private",
     ["cookie-private", "second-private"]),
    ('{"token": "json-private", "client_secret": "client-private"}',
     ["json-private", "client-private"]),
    ("connect postgresql://alice:url-private@database:5432/shop",
     ["url-private", "alice"]),
    ("https://api/charge?access_token=query-private&region=east",
     ["query-private"]),
])
def test_credentials_are_redacted_in_all_duplicated_exception_outputs(
    payload, secrets,
):
    fields = {"exception.type": "Timeout", "exception.message": payload,
              "exception.stacktrace": payload,
              "otel.status_description": payload, "db.statement": payload}
    trace = span_item(span_tags=fields, events=[{
        "timestamp": START_US, "fields": tags(fields),
    }])
    log = log_item(body=payload, attributes=fields)
    for item in (trace, log):
        rendered = item.model_dump_json()
        assert all(secret not in rendered for secret in secrets)
        assert "[REDACTED]" in rendered
        assert item.attributes["exception_coverage"]["redacted"]
        assert item.attributes["diagnostic"]["db.statement.redacted"]
    assert log.attributes["body_redacted"]
    assert trace.attributes["otel_status_description_redacted"]


def test_structured_body_and_conflicts_do_not_export_sensitive_values():
    item = log_item(body={
        "headers": {"authorization": "body-private",
                    "cookie": "cookie-private"},
        "http.request.header.authorization": ["header-private"],
        "details": [{"password": "nested-private"}],
    }, resource={
        "service.version": "token=version-private",
        "deployment.environment.name": "token=env-private",
        "deployment.environment": "token=other-private",
    })
    assert "private" not in item.model_dump_json()
    assert item.attributes["body_redacted"]
    assert item.attributes["resource_metadata"]["redacted_fields"] == [
        "deployment.environment.name", "service.version",
    ]
    # Redaction may make different values look equal; keep the conflict marker.
    assert "deployment.environment.name" in item.attributes[
        "resource_metadata"]["conflicts"]


def test_exception_limits_keep_utf8_head_tail_and_report_omissions():
    stack = "FirstFrame\n" + "出错\n\"" * 20_000 + "\nLastFrame"
    item = span_item(events=[{
        "timestamp": START_US + i,
        "fields": tags({"exception.type": "Failure",
                        "exception.stacktrace": stack,
                        "exception.message": "message " * 4000,
                        "code.file.path": "/" + "源" * 4000}),
    } for i in range(7)])
    events = item.attributes["exceptions"]
    coverage = item.attributes["exception_coverage"]
    assert len(events) == 4
    assert coverage["total_events"] == 7
    assert coverage["omitted_events"] == 3
    size = len(json.dumps(events, ensure_ascii=False).encode())
    assert size == coverage["retained_bytes"] <= 32768
    assert coverage["content_limit_bytes"] == 32768
    assert coverage["truncated"]
    for event in events:
        assert event["stacktrace"].startswith("FirstFrame")
        assert event["stacktrace"].endswith("LastFrame")
        assert "truncated" in event["stacktrace"]
        assert event["truncated_fields"]["stacktrace"]["original_bytes"] == (
            len(stack.encode()))
        assert "�" not in event["stacktrace"]


def test_large_repeated_log_body_is_redacted_and_explicitly_bounded():
    body = "FirstFrame password=body-private\n" + "源" * 20_000 + "\nLastFrame"
    item = log_item(body=body, attributes={"exception.stacktrace": body})
    assert item.summary.startswith("FirstFrame password=[REDACTED]")
    assert item.summary.endswith("LastFrame")
    assert "body-private" not in item.model_dump_json()
    truncation = item.attributes["body_truncation"]
    assert truncation["source"] == "opensearch.body"
    assert truncation["retained_bytes"] == len(item.summary.encode()) <= 16384
    assert truncation["original_bytes"] > truncation["retained_bytes"]


def test_long_duplicated_status_fields_cannot_bypass_diagnostic_bounds():
    item = span_item(span_tags={
        "status.message": "password=status-private " + "x" * 20_000,
    })
    assert "status-private" not in item.model_dump_json()
    assert item.attributes["status_message_redacted"]
    assert item.attributes["status_message_truncated"]
    assert len(item.attributes["status_message"]) == 1600
    assert item.attributes["status_message"] == item.attributes[
        "diagnostic"]["status.message"]


def test_missing_and_malformed_optional_context_remains_absent():
    for item in (span_item(), log_item(), log_item(
            attributes=["malformed"], resource=["malformed"])):
        assert not {"resource", "resource_metadata", "code", "exceptions",
                    "exception_coverage"}.intersection(item.attributes)
    item = span_item(span_tags={
        "code.line.number": True, "code.lineno": "42",
        "code.function": [], "exception.type": {},
    })
    assert "code" not in item.attributes
    assert "type" not in item.attributes["exceptions"][0]


def test_large_exception_without_code_does_not_claim_code_was_omitted():
    item = span_item(span_tags={"exception.stacktrace": "\n" * 5000})
    event = item.attributes["exceptions"][0]
    assert "code" not in event
    assert "code_omitted" not in event
    assert "stacktrace" in event["truncated_fields"]


def test_blank_log_body_cannot_bypass_size_limit():
    item = log_item(body=" " * 100_000)
    assert item.summary == "[Empty log body]"
    assert item.attributes["body_empty"]
    assert len(item.attributes["original_body"].encode()) <= 16384
    assert item.attributes["body_truncation"]["original_bytes"] == 100_000


@pytest.mark.parametrize("selected_id", [
    "000000000000000a", "000000000000000A",
])
def test_span_filter_runs_before_pagination_and_backend_forwards_it(
    selected_id,
):
    spans = [{
        "spanID": f"{i:016x}", "processID": "p", "operationName": "Charge",
        "startTime": START_US, "duration": i * 1000, "tags": [],
    } for i in range(1, 31)]

    async def exercise():
        async with httpx.AsyncClient(transport=httpx.MockTransport(
            lambda _: httpx.Response(200, json={"data": [{
                "spans": spans, "processes": {"p": {"serviceName": "payment"}},
            }]}),
        )) as http:
            backend = DirectTelemetryBackend(
                None, JaegerClient("http://jaeger", http), None)
            selected = await backend.get_trace_detail(
                TRACE_ID, span_id=selected_id, limit=1)
            assert selected[0].attributes["span_id"] == selected_id.lower()
            coverage = selected[0].attributes["coverage"]
            assert coverage["trace_span_count"] == 30
            assert coverage["matched_span_count"] == 1
            assert coverage["next_offset"] is None
            assert coverage["selection"]["span_id"] == selected_id
            empty = await backend.get_trace_detail(
                TRACE_ID, span_id=selected_id, service="other")
            assert empty["matched_span_count"] == 0
            assert empty["selection"]["span_id"] == selected_id

    asyncio.run(exercise())


@pytest.mark.parametrize("invalid", [
    "", "a" * 15, "a" * 17, "g" * 16, "../trace", 123,
])
def test_span_id_schema_rejects_non_exact_hex_identifiers(invalid):
    with pytest.raises(ValidationError):
        TraceDetail(trace_id=TRACE_ID, span_id=invalid)


def test_span_id_is_optional_and_does_not_enable_environment_search():
    assert TraceDetail(trace_id=TRACE_ID).span_id is None
    assert TraceDetail(trace_id=TRACE_ID, span_id=SPAN_ID).span_id == SPAN_ID
    with pytest.raises(ValidationError):
        TraceDetail(trace_id=TRACE_ID, environment="production")


def test_resource_and_exception_changes_update_trace_fingerprint():
    reads = []

    def response(_):
        number = len(reads)
        reads.append(number)
        return httpx.Response(200, json={"data": [{
            "spans": [{
                "spanID": SPAN_ID, "processID": "p", "operationName": "Charge",
                "startTime": START_US, "duration": 1000,
                "tags": tags({"exception.message": str(number > 1)}),
            }],
            "processes": {"p": {"serviceName": "payment", "tags": tags({
                "service.version": "v1" if number == 0 else "v2",
            })}},
        }]})

    async def exercise():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(response),
        ) as http:
            client = JaegerClient("http://jaeger", http)
            return [(await client.get_trace_detail(TRACE_ID))[0].attributes[
                "coverage"]["trace_fingerprint"] for _ in range(3)]

    assert len(set(asyncio.run(exercise()))) == 3
