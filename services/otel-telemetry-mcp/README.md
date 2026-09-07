# OpenTelemetry telemetry MCP service

Independent read-only telemetry tools over Streamable HTTP at `/mcp`. The
package requires neither TraceMind nor LangChain. It queries Prometheus, Jaeger
and OpenSearch HTTP APIs; business microservices need no MCP instrumentation.
This service is unrelated to the Astronomy Shop's shopping/cart MCP service.

## Run independently

From this directory:

```bash
uv sync --frozen
uv run otel-telemetry-mcp
```

It listens on `127.0.0.1:8012` by default. Configure environment variables using
the names in `.env.example`. The service does not automatically load `.env`.
`GET /health` checks readiness without querying backend data. Other compatible
MCP clients can connect to `http://127.0.0.1:8012/mcp` and use tools/list/call.

## Docker with the existing Demo

From the TraceMind repository root, with the Demo already running:

```bash
docker compose -f deploy/compose.telemetry-mcp.yaml up -d --build
docker compose -f deploy/compose.telemetry-mcp.yaml ps
```

The Compose project is independent and joins the external `opentelemetry-demo`
network. Set `OTEL_DEMO_NETWORK` to override the network name. It queries
`prometheus:9090`, `jaeger:16686` and `opensearch:9200` inside that network.
The image build context is only this directory, so TraceMind code and credentials
are not included. Re-run the build command after changing service source.
Why: The image is built from the independent package, so a running image does not pick
up source edits.

The service is intended for localhost access. Host/Origin checks are enabled;
configure their allowlists for another deployment. Remote authentication and
tenant isolation are not provided by this local Compose setup.
Why: Host checks are a local access boundary and cannot substitute for remote
authentication.

## Tool contract

Every tool declares the domain envelope through
`_meta["tracemind/result-contract"] = "telemetry.v1"`, alongside its standard
MCP output schema. This lets consumers select an observation adapter without
matching tool names. `_meta["tracemind/preview-limit"]` suggests the number of
retained-observation previews to show; consumers enforce their own limits.
These fields describe results and do not grant execution permissions. Other
MCP clients may ignore them and use the standard content/output schema.

| Tool | Required arguments | Optional arguments |
|---|---|---|
| query_metrics | expression, at | — |
| discover_metrics | start, end | service, metric, limit |
| search_traces | service, start, end | min_duration_ms (> 0), errors_only (default false) |
| get_trace_detail | trace_id | — |
| search_logs | service, start, end | text |

All timestamps must include a timezone; windows require end > start. Context
fields are marked `x-investigation-context: start/end` in the advertised schema.
Clients may bind these fields from their application context, as TraceMind does,
or supply them explicitly. Input validation always occurs on the server.
Why: Timezone-aware ordered windows prevent ambiguous or inverted telemetry queries.

`search_logs` prefers the event timestamp. Missing or zero event times use
`observedTimestamp`, with provenance preserved in evidence attributes. A valid
old event is not admitted merely because its observation time is recent.
Both groups are queried with the same service/text/window restrictions and
merged into at most 100 observations, newest selected time first.

`get_trace_detail` retains error spans and adds the twenty longest successful
spans with service, operation, duration and available parent IDs. This exposes
slow dependencies without requiring an error. The result is bounded context,
not a complete span tree; parent and child durations must not be summed.
Why: Trace details contain overlapping and bounded spans, so summing them misstates
request latency.

Replies contain exactly one of:

- `evidence`: observations with UUID, source, summary, observed_at, service,
  query and attributes; clients decide how to retain and cite them;
- `metadata`: metric discovery or empty-trace service-catalogue hints, without
  evidence IDs; these guide queries and do not establish incident facts;
- `error`: safe category, retryable flag, status and optional Retry-After hint.

Both structuredContent and readable text content are returned. Backend failures
do not transfer exception text. The server does not retry calls or own agent
budgets, diagnosis state, checkpoints or memory. Streamable HTTP uses JSON
responses; it does not require an SSE response stream.
Why: The telemetry service owns observations while client applications own retries,
permissions and execution state.

## Validation

The repository's `tests/test_mcp_http.py` starts a real HTTP server with fake
telemetry and exercises discovery, execution and application controls. Docker
verification additionally confirms that TraceMind/LangChain are not installed.
The service lock retains the dependency versions used by the application.
