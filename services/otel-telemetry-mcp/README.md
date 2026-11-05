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
| get_trace_detail | trace_id | service, operation, span_id, offset, limit |
| search_logs | service, start, end | text, trace_id |

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

`get_trace_detail` applies exact service/operation/span filters before paging
all spans, errors first then duration. `span_id` is an optional 16-character
hexadecimal ID within the supplied trace. Pages default to 25 spans (maximum
100); `coverage` reports matched counts, `next_offset`, filters and a fingerprint
of the normalized trace. This exposes successful slow dependencies as well as
fast controls. Results are not a complete span tree unless coverage establishes
it; parent and child durations must not be summed.
Why: Trace details contain overlapping and bounded spans, so summing them misstates
request latency.

Trace spans and logs retain source-reported context in `attributes`:

- `resource` contains allowlisted OTel service name/namespace/version/instance,
  deployment environment, Kubernetes namespace/pod/cluster, container and host
  identifiers. `resource_metadata.provenance` identifies each input field;
  `conflicts` retains differing reported values. Resource scope takes precedence
  over span/log attributes, then modern names over legacy aliases within a scope.
- `code` contains reported `file`, `line`, `function` and legacy `namespace`, with
  provenance and conflict metadata. Modern `code.file.path`, `code.line.number`,
  `code.function.name` take precedence over `code.filepath`, `code.lineno` and
  `code.function`. String values are limited to 1 KiB; truncation is explicit.
- `exceptions` contains up to four records from Jaeger span tags/events or log
  attributes, in source order. Each record contains its source, reported event
  timestamp, type/message/stacktrace and code context when present. Span-tag
  exceptions have no independent event time. Log observation-time fallback does
  not become an exception event timestamp. Recording an exception does not
  change span error status, and stack frames are not inferred by parsing text.

Exception records share a 32 KiB serialized UTF-8 budget per evidence item.
Per-field limits (512-byte type, 1 KiB message, 4 KiB stack) retain head and tail;
escaping and metadata may require further reduction or omission of event code.
`exception_coverage` reports retained bytes, total/returned/omitted event counts,
redaction and truncation. Resource/code conflicts and provenance also remain
bounded; absent values are not invented. Log body previews are capped at 16 KiB,
with `body_truncation` identifying their source and retained size.

Explicit credential forms (authorization, cookies, password/token/API-key
assignments and credentials in connection URLs) are redacted before returning
exception text, log bodies and duplicated status/diagnostic fields. Structured
log-body credential keys are also redacted. Markers identify redacted fields;
this targeted sanitation is not a general PII detector. Arbitrary resource and
span attributes are not exported. Resource labels remain telemetry claims,
not verified mappings to a deployment or source revision. Environment/version
search selectors are not offered without verified backend index support.

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

### Address and success-state evidence

`diagnostic` preserves `upstream.host`, `upstream.cluster`, `server.address`,
`destination.address`, `source.address` and `event.name` independently when
present. `diagnostic_source` identifies `jaeger.span.tags` or
`opensearch.attributes`. Addresses are not inferred from log-body columns.
Interpret each field using the actual instrumentation/log format and span kind;
`server.address` alone is not proof of an upstream destination. In the Demo
Envoy `proxy.access` format it is the downstream local address, while
`upstream.host` names the upstream host. This is not a universal OTel mapping.

Original `otel.status_code`, `error` and RPC/HTTP status fields remain visible.
`is_error=false` means no recognized error marker; absent status remains
unknown. Explicit RPC/HTTP success reports that operation's status, not proof
of order persistence, payment settlement, final cart state or safe retries.
Business outcomes require linked business-state evidence; retry decisions also
require transaction-state and idempotency checks. Redaction/truncation markers
continue to apply to the retained fields.
