"""Verify MCP log time selection without changing event-time semantics."""

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
import sys

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]
                       / "services/otel-telemetry-mcp/src"))

from otel_telemetry_mcp.telemetry.errors import (  # noqa: E402
    TelemetryClientError,
)
from otel_telemetry_mcp.telemetry.opensearch import (  # noqa: E402
    OpenSearchClient,
)


def hit(name, event_time, observed_time=None):
    """Build a stored SS4O log with independently supplied timestamps."""
    source = {"body": name, "@timestamp": event_time}
    if event_time is None:
        source.pop("@timestamp")
    if observed_time is not None:
        source["observedTimestamp"] = observed_time
    return {"_id": name, "_source": source}


@pytest.mark.parametrize("event_time", [
    None, "1970-01-01T00:00:00Z", "1970-01-01T01:00:00+01:00",
])
def test_unset_event_time_uses_observation_with_provenance(event_time):
    observed = "2026-10-07T18:19:48.81224425Z"
    item = OpenSearchClient._to_evidence(
        hit("delay", event_time, observed), "shipping", None)
    assert item.observed_at == datetime.fromisoformat(observed)
    assert item.attributes["timestamp_source"] == "observedTimestamp"
    assert item.attributes["original_observed_timestamp"] == observed
    if event_time is not None:
        assert item.attributes["original_event_timestamp"] == event_time


def test_valid_event_time_is_not_overridden_by_new_observation():
    item = OpenSearchClient._to_evidence(hit(
        "historical event", "2024-09-01T00:00:00Z",
        "2026-10-07T18:19:48Z"), "shipping", None)
    assert item.observed_at == datetime(2024, 9, 1, tzinfo=UTC)
    assert "timestamp_source" not in item.attributes


@pytest.mark.parametrize("event_time, observed_time", [
    (None, None), (None, "not-a-date"),
    ("1970-01-01T00:00:00Z", "1970-01-01T00:00:00Z"),
    (None, "2026-10-07T18:19:48"),
    ("malformed", "2026-10-07T18:19:48Z"),
])
def test_no_valid_selected_time_is_an_explicit_error(
    event_time, observed_time,
):
    with pytest.raises(TelemetryClientError, match="invalid timestamp"):
        OpenSearchClient._to_evidence(
            hit("delay", event_time, observed_time), "shipping", None)


def test_query_merges_bounded_groups_and_preserves_filters():
    requests = []
    start = datetime(2024, 9, 1, tzinfo=UTC)
    end = datetime(2024, 9, 1, 1, tzinfo=UTC)

    def handler(request):
        """Return event and observation groups as separate bounded results."""
        body = json.loads(request.content)
        requests.append(body)
        assert body["size"] == 2
        filters = body["query"]["bool"]["filter"]
        assert filters[0] == {
            "term": {"resource.service.name.keyword": "shipping"}}
        assert body["query"]["bool"]["must"] == [
            {"match": {"body": "delay"}}]
        if "@timestamp" in body["sort"][0]:
            assert filters[1] == {"range": {"@timestamp": {
                "gte": start.isoformat(), "lte": end.isoformat()}}}
            return httpx.Response(200, json={"hits": {"hits": [
                hit("event-new", "2024-09-01T00:40:00Z"),
                hit("event-old", "2024-09-01T00:20:00Z"),
            ]}})
        assert filters[1]["bool"]["minimum_should_match"] == 1
        assert filters[1]["bool"]["should"] == [
            {"bool": {"must_not": [{"exists": {"field": "@timestamp"}}]}},
            {"term": {"@timestamp": "1970-01-01T00:00:00+00:00"}},
        ]
        assert filters[2] == {"range": {"observedTimestamp": {
            "gte": start.isoformat(), "lte": end.isoformat()}}}
        return httpx.Response(200, json={"hits": {"hits": [
            hit("observed-new", "1970-01-01T00:00:00Z",
                "2024-09-01T00:50:00Z"),
            hit("observed-old", None, "2024-09-01T00:10:00Z"),
        ]}})

    async def search():
        """Use the real client with a recorded backend transport."""
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler),
        ) as http:
            return await OpenSearchClient("http://backend", "logs", http)\
                .search_logs("shipping", start=start, end=end,
                             text="delay", limit=2)

    items = asyncio.run(search())
    assert len(requests) == 2
    assert [item.summary for item in items] == ["observed-new", "event-new"]


def test_backend_failure_on_observation_query_is_not_silently_empty():
    def handler(request):
        """Simulate the second backend query failing."""
        if "observedTimestamp" in json.loads(request.content)["sort"][0]:
            return httpx.Response(503)
        return httpx.Response(200, json={"hits": {"hits": []}})

    async def search():
        """Propagate the backend failure to the existing MCP error layer."""
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler),
        ) as http:
            await OpenSearchClient("http://backend", "logs", http)\
                .search_logs("shipping",
                             start=datetime(2024, 9, 1, tzinfo=UTC),
                             end=datetime(2024, 9, 2, tzinfo=UTC))

    with pytest.raises(TelemetryClientError, match="request failed"):
        asyncio.run(search())
