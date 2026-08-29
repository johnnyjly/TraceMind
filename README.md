# TraceMind

TraceMind is an evidence-driven incident diagnosis system for distributed
microservices. It will investigate synthetic incidents in the OpenTelemetry
Demo by querying metrics, traces, and logs, then return a structured diagnosis
with supporting evidence.

The single-agent baseline includes context management, execution controls,
recovery, persistence, tracing and optional feedback memory; see
[the single-agent design](DESIGN.md). Telemetry tools are discovered from an
independently deployed MCP service over Streamable HTTP. Multi-agent orchestration, repository
and Runbook tools, RAG integration, mutation approval and reinforcement learning remain
later phases.

## Current status

The current single-agent baseline provides:

- typed incident, evidence, and diagnosis models;
- environment-based application settings;
- a minimal FastAPI application and health endpoint;
- independent Docker MCP service querying Prometheus, Jaeger and OpenSearch;
- server-owned tool descriptions/schemas discovered over Streamable HTTP;
- normalization of backend responses into typed evidence;
- a bounded LangGraph single-agent loop with read-only telemetry tools;
- Gemini and OpenAI model providers selected through environment settings;
- validation that a completed diagnosis cites evidence actually returned by a tool;
- synchronous diagnosis and queued persistent run API endpoints;
- unit tests for the public model and API contracts.

## Requirements

- Python 3.12 or newer
- `uv`
- Docker Desktop with Docker Compose for the OpenTelemetry Demo integration

## Local setup

```bash
uv sync --dev
cp .env.example .env
uv run uvicorn tracemind.api.app:app --reload
```

The API health endpoint is available at `GET /health`.

By default TraceMind uses `gemini-3.7-flash`. Put a Google AI API key in the
local `.env` file before requesting a diagnosis:

```dotenv
TRACEMIND_LLM_PROVIDER=google
TRACEMIND_LLM_MODEL=gemini-3.7-flash
TRACEMIND_GOOGLE_API_KEY=your-key
```

To use the OpenAI fallback instead:

```dotenv
TRACEMIND_LLM_PROVIDER=openai
TRACEMIND_LLM_MODEL=gpt-5.6-luna
TRACEMIND_OPENAI_API_KEY=your-key
```

To use DeepSeek:

```dotenv
TRACEMIND_LLM_PROVIDER=deepseek
TRACEMIND_LLM_MODEL=deepseek-flash
TRACEMIND_DEEPSEEK_API_KEY=your-key
```

DeepSeek uses its OpenAI-compatible Chat Completions endpoint with thinking
disabled. Structured diagnoses use tool calls; SDK retries remain disabled.
The harness applies the same output, context, call, and time limits.

Never commit the populated `.env` file. Model configuration is loaded only
when the diagnosis endpoint is called, so the health endpoint remains usable
without an API key.

## RAG infrastructure (optional)

The first RAG increment provides an isolated retrieval runtime. It is not
registered as an Agent tool and does not change the single-agent baseline.
No documents are automatically imported from `knowledge/`.

```text
Query → LightRAG mix (graph + vector recall) ─┐
      → SQLite FTS5 BM25 recall ────────────┤
                         deduplicate / RRF → local BGE reranking
                                          → cited knowledge chunks
```

LightRAG uses Qdrant for vectors, NetworkX for relationships, and local JSON
stores for framework metadata. A separate SQLite database holds canonical
chunk provenance and two FTS5 indexes: stemmed English prose and literal
identifiers. LightRAG's extracted keywords drive entity/relationship vector
search; they are not a BM25 full-text index. Our FTS5 branch supplies that
independent lexical recall. `mix` is the primary mode; `naive` provides vector recall for
comparison. Both use the same BM25 fusion and reranker. Knowledge references
remain separate from current-incident telemetry evidence.

### Set up and check

```bash
uv sync --extra rag --dev
docker compose -f deploy/compose.rag.yaml up -d
uv run --extra rag python scripts/rag.py models
uv run --extra rag python scripts/rag.py init
uv run --extra rag python scripts/rag.py check
```

The `models` command downloads revision-pinned `BAAI/bge-small-en-v1.5`
(384 dimensions) and `BAAI/bge-reranker-base`, then checks local inference.
Subsequent runtime initialization uses cached weights only. CPU is the default;
Apple Silicon MPS can be selected through `TRACEMIND_RAG_DEVICE=mps`.
Qdrant is an independent Docker service on localhost:6333 with a named volume.
It does not require the OpenTelemetry Demo or receive model credentials.

Settings are listed in `.env.example` under `TRACEMIND_RAG_*`. Workspaces and
model caches live under ignored `artifacts/rag/`. Each workspace records model
revisions, dimensions and chunk settings in `runtime.json`; incompatible
configuration requires a different `TRACEMIND_RAG_SNAPSHOT` identifier.
The default `bootstrap` workspace contains zero knowledge chunks and is
explicitly marked unpublished.

```bash
uv run --extra rag python scripts/rag.py query --query "RPC connection errors" --mode mix
```

An empty workspace returns `[]` without model calls. Once populated, retrieval
returns structured references, not a generated diagnosis. LightRAG keyword
extraction can call the configured model; the bridge enforces per-operation call, output
and time limits and logs reported tokens. Embedding and reranking run locally,
with tokenizer checks that reject inputs exceeding model limits.

The CLI selects `TRACEMIND_RAG_LLM_PROVIDER` and `TRACEMIND_RAG_LLM_MODEL`,
using the selected provider's existing application credential. DeepSeek is the
configuration default, not an automatic fallback. The retrieval library never
loads application credentials or chooses a provider: callers inject an async
completion function into `RagRuntime`. Missing injection raises an explicit
error only when retrieval needs an LLM. Provider-specific request options are
owned by the CLI adapter. Initializing empty stores requires no model key.

### Remaining boundaries

Source verification, format selection, section-aware chunking, extraction,
immutable snapshot publication and retrieval evaluation are the next stage.
Agent integration will additionally supply English query planning, charge
internal model calls to the outer run budget, and keep knowledge citations
distinct from telemetry evidence IDs. The CLI is not a distributed retrieval
service: use one process per workspace; local graph/JSON stores do not support
concurrent cross-process writers. Qdrant does not make those stores distributed.

Interfaces follow the pinned [LightRAG core documentation](https://github.com/HKUDS/LightRAG/blob/main/docs/ProgramingWithCore.md)
and the [BGE embedding](https://huggingface.co/BAAI/bge-small-en-v1.5) and
[reranker](https://huggingface.co/BAAI/bge-reranker-base) model contracts.

## Telemetry MCP service

The default query path is `TRACEMIND_TELEMETRY_TRANSPORT=mcp`, using
`TRACEMIND_TELEMETRY_MCP_URL=http://127.0.0.1:8012/mcp`. The independent
[otel-telemetry-mcp service](services/otel-telemetry-mcp/README.md) owns the five
query implementations, descriptions and parameter schemas. Its package and
Docker image do not depend on TraceMind, LangChain or LangGraph. Any compatible
MCP client can discover and call it. This is separate from the Demo's shopping
MCP service, which exposes cart/product/checkout operations.

```text
Session initialization → MCP tools/list → allowlist filter → LangChain tools
Agent tool proposal → harness authorization/budget/timeout
                    → MCP tools/call over Streamable HTTP
                    → server HTTP clients → telemetry backends
                    → validated observations → run-local store → bounded preview
```

The server exposes `query_metrics`, `discover_metrics`, `search_traces`,
`get_trace_detail` and `search_logs`. The model-facing definitions come from
`tools/list`, not hand-written local query functions. A generic adapter binds
application-selected dates using the server schema's `x-investigation-context`
annotations; those fields are hidden from the model and cannot be overridden.
TraceMind still owns allowlists, retries, evidence deduplication, compression,
citation validation and progress. `read_evidence` remains a local agent tool.
Metric metadata never becomes evidence.

### Start or update the container

First start the Demo using the existing scenario script. Its Docker network
must exist. Then, from the TraceMind project directory:

```bash
docker compose -f deploy/compose.telemetry-mcp.yaml up -d --build
uv run python scripts/check_mcp.py
```

This is a separate Compose project attached to the existing `opentelemetry-demo`
network. Set `OTEL_DEMO_NETWORK` if the Demo uses a different network name.
Only the MCP container is built/recreated. The host port binds to loopback.
The service reads `TELEMETRY_MCP_` settings, owns its backend HTTP connections
and receives no model keys or TraceMind `.env` file. Rebuild after editing its
source; containerization does not prevent later changes.

### Try a diagnosis

HTTP MCP is the only live telemetry path. Remove an old
`TRACEMIND_TELEMETRY_TRANSPORT=direct` setting or change it to `mcp`; `direct`
is now rejected during configuration validation. Backend URLs belong to the
independent service’s `TELEMETRY_MCP_` configuration. Start the API and worker:

```bash
TRACEMIND_TELEMETRY_TRANSPORT=mcp uv run uvicorn tracemind.api.app:app --reload
TRACEMIND_TELEMETRY_TRANSPORT=mcp uv run python scripts/run_worker.py
```

Then submit the existing demo from another terminal:

```bash
uv run python scripts/demo_investigation.py --lookback-minutes 10
```

Tool discovery happens once per investigation session, before the model loop,
within the run deadline and the tool timeout. Discovery events record its
outcome and duration; it consumes no model or query attempt. Parallel queries
reuse the session. Local-only explanation and resumed finalization skip remote
discovery. Closing the request/worker/evaluation session leaves the independently
running server available. Discovery failures stop explicitly; no direct fallback
is performed. Query retries still consume the original shared budgets.

`TRACEMIND_TELEMETRY_MCP_RESPONSE_BYTES` caps decoded replies and discovery pages,
not peak parser memory. The server has its own response cap. Run/report metadata
retains `telemetry_transport=mcp`; old stored records still default to `direct`.
The HTTP server uses JSON responses supported by Streamable HTTP; TraceMind's
user-facing SSE progress is a separate API channel.

### Runtime boundaries and offline replay

The old direct clients, manually registered telemetry tools and stdio
server/client have been removed. Production diagnosis uses tools discovered
from the independent HTTP MCP service. Offline replay uses recorded tool
schemas and the same binder, with `telemetry_transport=replay`; it does not
contact a server or measure live-model accuracy. A contract test compares the
recorded definitions with live HTTP discovery. Old stored `direct` labels
remain readable, without re-enabling their execution path.

API, worker and live evaluation share service assembly. Queue endpoints do not
construct model clients or require model credentials; the worker validates
provider configuration when it executes the run. Synchronous diagnosis still
validates its own model configuration before execution.

Each API app initializes a run store once per database path, in the request
thread pool. SQLite transactions and worker storage callbacks execute outside
the event loop. Harness admission reserves budget in memory before yielding,
then awaits a serialized durable snapshot before calling a provider/tool.
Tool completion/rejection is committed before the graph advances. Cancellation
waits for an in-flight storage operation before terminal cleanup, so a late
progress write cannot overtake finalization. Lock waits can still delay the
request or run that needs the database, but do not block unrelated async work.
Run and memory repositories share explicit transactions; feedback lease checks,
revision checks and receipts remain atomic in the same database. There is no
schema migration in this refactor.

HTTP integration uses the real SDK/server with scripted models and fake
backends. Network authentication/multi-tenant access and distributed diagnosis
workers remain outside this local deployment.

Submit a bounded investigation window to `POST /api/v1/diagnoses`:

```json
{
  "incident": {
    "title": "Checkout HTTP 500 spike",
    "description": "Checkout failures began after a synthetic fault",
    "severity": "high",
    "affected_services": ["checkout"]
  },
  "investigation_start": "2024-01-01T14:00:00Z",
  "investigation_end": "2024-01-01T14:30:00Z"
}
```

The agent can query Prometheus, search Jaeger traces, and search OpenSearch
logs. These tools are read-only and enforce fixed result limits. A diagnosis
with status `complete` is rejected unless every cited evidence ID came from an
actual tool result.

Run the tests with:

```bash
uv run pytest
```

## Repeatable evaluation (S5, first increment)

Run the six committed synthetic replays without Docker, credentials, or model
API calls:

```bash
uv run python scripts/run_evaluation_suite.py
```

Fixtures in `evals/replays/` cover payment failure, misleading checkout
symptoms, missing telemetry, conflicting evidence, transient recovery, and
backend failure. Each executes a scripted model through the real LangGraph
and harness against fixed normalized telemetry. These are workflow regression
checks; their scores and latency do not measure a real model's accuracy or
network latency. Ground truth and scripted answers stay outside runtime
prompts. Unrecorded backend requests are rejected. A real model can be supplied
through `run_replay_evaluation(..., model=...)` for a separate `model_replay`
measurement against the same strict transcript.

The batch JSON defaults to `artifacts/evaluations/replay-suite.json`. It retains
individual reports and groups by execution mode, provider, and model. Regression cases include expected completions, inconclusive outcomes and
operational failures such as `tool_error`. Expected inconclusive
answers pass only cases that explicitly expect them. A complete answer that
fails the cause/evidence checks, or completes an inconclusive case, counts as
a false completion. This remains deterministic heuristic scoring.

Aggregate existing live reports without another model call:

```bash
uv run python scripts/run_evaluation_suite.py \
  --reports artifacts/evaluations/payment-failure.json \
  --output artifacts/evaluations/live-summary.json
```

Summaries include pass rate, false completion rate among complete diagnoses,
failures/cancellations, attempts/retries, successful recovery, mean/p95 latency,
reported tokens, unreported calls, and trace coverage. Optional `--pricing`
accepts a JSON mapping such as
`{"provider:model":{"input_per_million":1,"output_per_million":2}}` with
explicit USD rates. The estimate is null when any run lacks pricing, complete
usage, or per-model attribution after fallback; cached-token discounts are
not modeled. Raw telemetry and diagnoses remain in the report, so protect
the artifacts directory when using data beyond the synthetic Demo.

Service-reported diagnosis failures and cancellations produce reports with
partial state; errors outside that boundary still propagate. The live CLI exits 1
on failure, and the suite exits 1 when its selected gates fail. Existing report
JSON remains loadable. OpenTelemetry trace export and shared-service controls
remain subsequent S5 work.

## Single-agent execution controls (S1)

API and evaluation runs use one harness for tool authorization, budgets, and
termination. Defaults are 8 model calls, 12 admitted telemetry calls, at most
2 calls with identical tool arguments, and 120 seconds per run. Model and tool
timeouts are 60 and 20 seconds. The graph recursion limit remains a separate
guard and can stop a run earlier. Configure these values using the
`TRACEMIND_AGENT_*` settings in [.env.example](.env.example).

`TRACEMIND_AGENT_ALLOWED_TOOLS` is a JSON array of tool names. The harness filters
visible tools and checks authorization again before execution. This is a
tool-name policy; it does not provide service or tenant isolation. SDK retries
are disabled in S1 so hidden retries cannot multiply calls. Failed calls are
counted; calls rejected before backend execution are not admitted tool calls.

Model/context/recursion exhaustion produces an `inconclusive` diagnosis. Tool
budget and repetition limits close investigation and allow a final diagnosis
within remaining hard budgets. Operational
timeouts return HTTP 504; tool/model failures and invalid diagnoses return 502.
Error `detail` remains a string. Responses for completed or failed runs include
`X-TraceMind-Run-ID` and `X-TraceMind-Stop-Reason` headers. Investigation
timestamps require timezones. Final citations must be unique, observed, and
inside the investigation window; these checks do not prove causal correctness.

Structured events are written to `artifacts/runs/<run_id>.jsonl` by default.
They contain call relationships, durations, hashed arguments, evidence IDs,
usage when the provider reports it, and a terminal status. Unreported usage is
tracked separately from reported token totals. Raw prompts, queries, results,
and exception messages are not included in these event files. `--verbose`
also logs this safe event metadata to the terminal at DEBUG level.
Trace file open/write/close failures emit a server warning without replacing
the diagnosis result or its original error. Events remain available in memory;
the file can be missing or incomplete. Expected provider and telemetry errors
are classified separately from unexpected code errors (`internal_error`).

Evaluation reports include `run_state` with counters, events, and stop reason.
The synchronous diagnosis and evaluation paths keep run state in memory. S4
persistence applies to queued runs and investigation rounds described below. JSONL
events are diagnostic output, not resumable checkpoints.

## Persistent local runs (S4)

Run the API and worker in separate terminals with the same `.env`:

```bash
uv run uvicorn tracemind.api.app:app --reload
uv run python scripts/run_worker.py
```

Submit the same JSON body shown above to `POST /api/v1/runs`. It returns 202
and a run ID. `Idempotency-Key` is optional; repeating a request with the same
key and body returns its existing run, while using that key for another body
returns 409. Poll `GET /api/v1/runs/{id}` and read the diagnosis and evidence
at `GET /api/v1/runs/{id}/result` when available. `POST
/api/v1/runs/{id}/cancel` requests cancellation; `POST
/api/v1/runs/{id}/resume` requeues an interrupted run after its worker lease
expires. The worker also reclaims expired leases automatically. `--once`
processes at most one queued run.

The run ledger and evidence use `TRACEMIND_AGENT_RUN_DATABASE`; LangGraph
checkpoints use `TRACEMIND_AGENT_CHECKPOINT_DATABASE`. Defaults are separate
SQLite files under `artifacts/`. Back up both files together. The worker
restores call budgets, evidence, elapsed deadline, and graph state after a
restart. Calls interrupted between an external response and a checkpoint can
be retried, with previous attempts still consuming budget; ambiguous in-flight
calls are marked in the run events. Cancellation stops admission of new calls
and interrupts the active worker task. This local setup is designed for one
worker process, without shared-service authentication or admission controls.

S2 context controls and S3 recovery are described below.

## Incident collaboration and readable progress

The autonomous first diagnosis stays usable without human interaction. To
retain an incident for later questions, submit the same diagnosis request to
`POST /api/v1/investigations`. Its 202 response contains an investigation `id`
and the first `run`. A run owns its checkpoint, evidence, deadline and budgets;
an investigation associates successive rounds with the same incident.

After that run finishes, `POST /api/v1/investigations/{id}/messages` accepts:

```json
{"message":"Why is payment the root cause?","purpose":"explain"}
```

`explain` allows only retained-evidence reading, so the model cannot silently
perform fresh telemetry queries. Use `purpose: "investigate"` to verify an
alternative hypothesis or collect more observations. User suggestions and
previous conclusions are untrusted claims, not observed evidence. Every round
gets a new run ID and budget; the previous outcome is preserved. An active
round returns 409 for a new question; finish or cancel it first. Both creation
and follow-up accept `Idempotency-Key`.

An omitted window inherits the previous round's fixed bounds. For a recurrence,
provide both timezone-aware `investigation_start` and `investigation_end`.
Only prior observations inside that window and the new capacity are retained,
with cited observations selected first. Explain requests must keep the previous
window. Full evidence stays paginated through `read_evidence`; the prompt uses
up to 20 bounded observations, prioritizing citations and interleaving services
and sources. Small observations include normalized attributes; large ones mark
truncated or omitted fields. The pack is capped at 16 KB or one third of the
input allowance, with history trimmed first. The previous conclusion remains
a historical claim, alongside at most three recent questions.
Explanation follows the user's requested language, length and detail. It can
answer directly from the pack or read necessary details by ID; a short question
does not require a complete catalogue scan or new remediation suggestions.
Older rounds remain available through
`GET /api/v1/investigations/{id}?offset=0&limit=20` and individual run results.
This is incident-local history; it is not cross-incident long-term memory.

### Optional cross-incident memory (S6, first increment)

Memory is **disabled by default**. When enabled, the agent learns from ordinary
follow-up feedback; no "remember this" request or manual registration is needed.
It proposes reusable preferences/skills and human-reported confirmed or
corrected diagnoses alongside its answer. The workflow verifies the feedback
quote and provenance, then persists the proposal. Model confidence and a
completed run alone never establish human confirmation.

Examples of ordinary follow-up messages:

```json
{"message":"以后先给简短结论，再列证据。","purpose":"explain"}
```

```json
{"message":"已确认是路由配置，修正路由后恢复了。","purpose":"explain"}
```

Use `POST /api/v1/investigations/{id}/messages`, or the interactive
`scripts/demo_investigation.py` follow-up prompt. Existing diagnosis/model/time
budgets still apply. Learning uses the same structured model response and adds
no separate extraction call. The model chooses whether feedback is enduring;
"这次简短输出" should be classified as temporary and is not written to long-term
memory. A later correction updates the same stable key; disputed items can be
revoked through feedback. The inference is a model decision, not a keyword
rule, so mistaken classifications remain possible and reviewable.

Preferences/skills use a local-user profile independent of deployment version.
New rounds capture a bounded instruction snapshot, including explain rounds;
current requests take priority over saved habits. Skills cannot authorize tools,
change budgets or create evidence. Inferred diagnostic confirmations reference
the parent run and the current feedback run/quote, with normalized source
observations as provenance. Human feedback can correct an inconclusive source
with observations; it does not independently prove causality for a new run.
Automatic confirmations expire after `TRACEMIND_AGENT_MEMORY_AUTO_EXPIRY_DAYS`
(default 30). Explanation still performs no new historical-diagnosis retrieval.

`guidance.selected` and `memory.feedback_saved` expose sanitized progress and
counts; full quotes/proposals remain in SQLite. Writes check the active worker
lease, quoted current input and concurrent revisions. A transaction stores the
updates and a per-run receipt, so restart cannot duplicate learning or recreate
a record subsequently deleted by a human. Previous snapshots and receipt
contents remain available for audit. Failed quote/revision/record validation
does not invalidate an otherwise valid diagnosis; database failures propagate.

| Guidance route | Action |
| --- | --- |
| `GET /api/v1/guidance?offset=0&limit=20` | Review inferred preferences and skills with source feedback |
| `PUT /api/v1/guidance/{id}` | Correct `instruction` with `expected_revision` |
| `DELETE /api/v1/guidance/{id}?expected_revision=1` | Remove from future retrieval |

The API is for trusted local use; this profile is not a tenant boundary and
`confirmed_by` is an attestation, not an authenticated identity. Shared-service
authentication remains S5 work. Autonomous evaluations never load live user
guidance or learn from follow-up feedback; human-assisted learning needs a
separate fixed-feedback evaluation. Evaluate real-model classification quality separately.

The existing manual confirmation API remains available for review and import.

After reviewing a source run, write a `memory-confirmation.json` file:

```json
{
  "source_run_id": "<completed-run-UUID>",
  "environment": "otel-demo-local",
  "deployment_version": "demo-3.1.0-local-v1",
  "services": ["checkout", "payment"],
  "symptom": "Checkout requests returned HTTP 500",
  "resolution": "<human-verified cause and resolution>",
  "diagnostic_queries": ["<useful query or check>"],
  "remediation": ["<verified recovery action>"],
  "confirmed_by": "<reviewer>",
  "confirmation_note": "<how the cause and recovery were verified>",
  "expires_at": "<future timezone-aware timestamp>"
}
```

Submit it with:

```bash
curl -sS -X POST http://127.0.0.1:8000/api/v1/memories \
  -H 'Content-Type: application/json' --data-binary @memory-confirmation.json
```

The workflow checks the source run, citations, time window and services, and
stores provenance. Environment/version are human attestations for older runs
without captured memory scope; scoped runs must match their saved policy.
An incorrect model diagnosis can be corrected in the confirmed resolution.
Only future, explicitly enabled investigations retrieve these records:

```dotenv
TRACEMIND_AGENT_MEMORY_ENABLED=true
TRACEMIND_AGENT_MEMORY_ENVIRONMENT=otel-demo-local
TRACEMIND_AGENT_MEMORY_DEPLOYMENT_VERSION=demo-3.1.0-local-v1
TRACEMIND_AGENT_MEMORY_MAX_RECORDS=3
TRACEMIND_AGENT_MEMORY_CONTEXT_BYTES=4096
TRACEMIND_AGENT_MEMORY_AUTO_EXPIRY_DAYS=30
```

Restart the API and worker after configuring them. Retrieval requires exact
environment/version and at least one affected-service match, excludes the same
incident and revoked/expired records, and selects recent confirmations first.
A run captures its policy at creation and freezes selected content before its
first model call. Recovery reuses that snapshot even after library changes.
Explain rounds perform no new historical-diagnosis lookup; investigate rounds
can. Both can load saved preferences and skills.
The prompt labels bounded previews as untrusted historical hints. Historical
IDs cannot substitute for current telemetry citations or establish completion.
`memory.selected` records the snapshot hash, record IDs/revisions and context
omissions; SSE shows retrieval progress without exposing resolution content.

| Management route | Action |
| --- | --- |
| `GET /api/v1/memories?offset=0&limit=20` | Review current records, including expired/revoked entries |
| `GET /api/v1/memories/{id}` | Inspect confirmation and provenance |
| `PUT /api/v1/memories/{id}` | Replace and reconfirm all fields with `expected_revision` |
| `POST /api/v1/memories/{id}/revoke?expected_revision=1` | Exclude from future retrieval |
| `DELETE /api/v1/memories/{id}?expected_revision=2` | Remove from the active library |

Concurrent stale edits return 409. Reconfirmation increments the revision.
Deletion does not erase copies already saved in run snapshots, receipts,
checkpoints or evaluation files. SQLite schema v4 adds guidance and feedback
receipts to the v3 memory table, preserving existing run data and checkpoint
schema. No vector search or new dependency is
introduced. Memory does not reduce the existing evidence, context or call gates.

### Frozen memory evaluation

Evaluation **always defaults to memory-off**, regardless of application memory
enablement. It reads historical hints only from `--memory-snapshot`, never the
live database. Export a scoped corpus **before** collecting held-out incidents:

```bash
curl -sS --get http://127.0.0.1:8000/api/v1/memories/snapshot \
  --data-urlencode environment=otel-demo-local \
  --data-urlencode deployment_version=demo-3.1.0-local-v1 \
  --data-urlencode service=checkout --data-urlencode limit=10 \
  -o artifacts/evaluations/memory-corpus.json
```

The snapshot timestamp must precede the evaluation window. Same-incident,
post-freeze confirmations, mismatched scope and expired entries are excluded;
expiry is also checked at the held-out window's start. Ground Truth and replay
scripts are never inputs to retrieval. Dataset authors must keep training and
test incidents separate; timestamps/IDs cannot detect copied answer text.

Try the fully offline synthetic memory-on/off comparison:

```bash
uv run python scripts/run_evaluation_suite.py \
  --memory-snapshot evals/memory/synthetic_training.json --compare-memory \
  --output artifacts/evaluations/memory-comparison.json
```

Each replay gets independent telemetry/model state and identical budgets.
Reports include memory mode and full-corpus hash; aggregation separates modes
and corpora. This scripted comparison verifies workflow isolation, **not model
accuracy improvement**. The synthetic record is deliberately an earlier network
failure, not the held-out payment answer. For a real-model comparison, run
`scripts/run_evaluation.py` twice on controlled equivalent fault windows, once
without the flag and once with `--memory-snapshot <frozen-file>`, keeping model,
limits and deployment fixed. Evaluate real-model memory quality separately.

Subscribe to `GET /api/v1/runs/{run_id}/events` for readable SSE progress.
It replays saved events then tails updates, including tool/service/window,
observations added, retries, recovery and termination. It excludes raw prompts,
queries and log payloads. The detailed operational events remain in run state
and best-effort JSONL traces; workflow-only queued/early-terminal events are
authoritative in SQLite. Progress describes actual actions, not private model
reasoning or an inferred investigation plan.

Reconnect with `Last-Event-ID: <sequence>` or `?after=<sequence>` (the header
takes precedence). Delivery across reconnects is at least once; clients should
deduplicate by run ID and sequence. Heartbeat comments arrive while idle.
The `complete` event is sent after the terminal transaction commits and contains
the final status and an optional result URL; close the subscription on that
event. Closing a subscription does not cancel the run. Existing synchronous
diagnoses/evaluations and their standalone JSONL files are not live SSE runs.

With the API, worker and Demo running, the terminal demo makes real model calls:

```bash
uv run python scripts/demo_investigation.py
uv run python scripts/demo_investigation.py --investigation-id <id> \
  --purpose explain --message "为什么认为 payment 是根因？"
uv run python scripts/demo_investigation.py --investigation-id <id> \
  --purpose investigate --message "检查是否也存在网络故障"
```

Use `--lookback-minutes 10` on an investigate request to start a new current
window; omit it to reuse the previous window. `--run-id <id> --after <sequence>`
subscribes to an existing run without starting another. The demo automatically
reconnects on transport failures and prints the final diagnosis.

The run store migrates schema v1 to v2 transactionally on opening, adding an
investigations table and a nullable run association. Existing runs remain
independent and readable. Back up the run and checkpoint stores before upgrade;
no checkpoint schema or dependency changes are required. Local SQLite remains
intended for one worker. Automatic fault detection and alert triggering are
not implemented. Autonomous evaluation stays separate from human-assisted
evaluation; human follow-up tests verify workflow mechanics, not model accuracy.

## Bounded context and evidence retrieval (S2)

Each run stores complete normalized observations in memory. Exact duplicates
(excluding their generated UUID, including source, timestamp, service, query,
and all attributes) share the first evidence ID. Similar messages at different
times or with different error details remain separate observations. This is
conservative deduplication, not semantic grouping or a causal judgment.

Tools return previews: small observations remain complete; large observations
return their ID, source, time, bounded service/summary, and a truncation marker.
All observations returned by a backend are stored before selecting the limited
preview batch. Omitted previews are counted explicitly. Backend queries can
still limit which observations are retrieved in the first place.

The new read-only `read_evidence` tool can:

- list stored previews with no `evidence_id` (20 entries per page);
- filter that catalogue by exact `service` and/or `source` (`metric`, `trace`,
  or `log`); offsets and `total_evidence` then refer to matching entries;
- retrieve an observation's complete JSON in 2,048-character pages by ID;
- continue either operation using the returned `next_offset`.

Reads are isolated to the current run and consume the same tool-call,
repetition, and timeout budgets as telemetry queries. Explicit existing
`TRACEMIND_AGENT_ALLOWED_TOOLS` lists must add `read_evidence` to enable it.
Without that permission, compressed details are unavailable to the agent.

### Metric discovery and empty queries

`discover_metrics(service="payment")` provides a bounded catalogue of actual
metric names and label examples. It reads distinct names and series metadata
within the fixed investigation window, using `service_name`, `service`, or
`job` matches. Omitting the service permits a bounded global discovery when
these labels are unknown. `limit` defaults to 20 and cannot exceed 50; the
model-facing metadata is capped at 8 KB. Truncation is explicit.

Series samples can all belong to one histogram, so the names catalogue is
queried separately. For a listed name without label examples, use
`discover_metrics(service="payment", metric="rpc_calls_total")` to inspect
that metric's labels. An ordinary discovery attempt performs two HTTP reads;
a targeted attempt performs one. Both use one tool-attempt slot and the same
harness timeout, retry and permission controls as other tools.

Discovery metadata is not sampled fault evidence: it receives no evidence IDs,
is not retained in the collector, and does not advance the evidence progress
counter. Prometheus metadata may describe series without samples in the window.
Explicit allowlists must add `discover_metrics` to enable it; existing persisted
runs keep their original permissions. Explain rounds still expose only
`read_evidence`.

An empty `query_metrics` result includes its query time and guidance to check
names, labels and sample availability. It creates no evidence or progress.
Empty results do not prove service health or exclude a fault. Diagnoses must
separate observed errors from hypotheses about how those errors were generated,
and label unverified mechanisms and suggested checks accordingly.

Before every model call, middleware budgets the system message, conversation,
tool schemas, and diagnosis schema, plus an output reservation. The initial
estimator (`utf8-bytes-v1`) counts serialized UTF-8 bytes plus 1,024 units of
framing allowance. It deliberately overestimates ordinary text/JSON tokens;
it is not a provider tokenizer or a guarantee about billing or model limits.
Provider-reported usage remains separately recorded.

The example diagnostic profile in `.env.example` is:

| Setting suffix (`TRACEMIND_AGENT_`) | Example profile | Meaning |
| --- | --- | --- |
| `CONTEXT_TOKENS` | 131072 | Estimated input plus reserved output allowance |
| `OUTPUT_TOKENS` | 8192 | Output reservation and provider output cap |
| `OBSERVATION_BYTES` | 8388608 | Maximum stored serialized evidence bytes |
| `RECURSION_LIMIT` | 50 | Graph step limit; a model/tool cycle uses multiple steps |
| `MAX_MODEL_CALLS` | 15 | Model attempts, including retries and repairs |
| `MAX_TOOL_CALLS` | 25 | Tool attempts, including retries and evidence reads |
| `TIMEOUT_SECONDS` | 300 | Whole-run deadline in seconds |

The input allowance is 122,880 estimate units after reserving output. These
settings override the smaller code defaults when `.env` is loaded. Choose
allowances compatible with your provider's model capacity. They limit maximum
work; the agent can finish earlier when it has sufficient evidence.

`.env.example` lists all application settings, grouped by function. Keep local
keys and backend addresses in `.env`. After changing it, restart the API and
worker and create a new run: existing runs retain their saved `RunLimits`.
The evaluation runner also caps tool calls by the case's `max_tool_calls`.

When input exceeds its allowance, the oldest tool results are replaced with
compact evidence indexes first. If those still exceed the allowance and
`read_evidence` is authorized, old indexes become bounded previews with explicit
`context_omitted_count` records and a catalogue retrieval instruction. Preview
selection preserves result order and does not rank causal importance. All
evidence IDs remain stored and discoverable through `read_evidence`; every
assistant tool-call/result pair is retained. Without that tool, indexes retain
all references and the context gate can stop the run. System instructions, incident
information, and assistant messages are preserved. A newly requested detail
page reaches the model intact. Compression changes the model input view, not
the stored observations or graph message history; no summarization LLM is used.

If the remaining input cannot fit, or the evidence store reaches its byte
limit, the run returns `inconclusive` with `context_limit` and retains partial
evidence. The byte limit bounds serialized retained observations, not Python
heap usage or HTTP response buffering. This is not a persistent artifact store.

`context.selected` events record input estimates before/after reduction,
reduced-message counts, omitted-reference counts, input allowance, output
reservation, and admission outcome.
`observations.stored` events record received/new/duplicate/omitted-preview
counts and stored bytes. `observations.limit` distinguishes store exhaustion.
These events contain no raw observation payloads. Evaluation reports already
include these events through `run_state`; reports also contain full evidence.

Scripted real-loop tests cover large results, omitted counterevidence retrieval,
read budgets/permissions, and context exhaustion. Unit tests cover parallel
message pairing, lossless Unicode pagination, exact deduplication, isolation,
output settings for both providers, and schema-inclusive estimates. Use live evaluations to assess diagnostic quality.

## OpenTelemetry Demo

The official OpenTelemetry Demo is kept as a separate sibling checkout rather
than copied into this repository:

```text
repos/
├── tracemind/
└── opentelemetry-demo/
```

TraceMind connects to the Demo's Prometheus, Jaeger, and OpenSearch APIs via the
URLs in `.env`. The evaluation environment is pinned to OpenTelemetry Demo
`3.1.0` at commit `dedc0178918e260823323b8d95005a8cb924b007`. The Demo's
built-in agentic mode is not used.

On macOS, install and start [Docker Desktop](https://docs.docker.com/desktop/setup/install/mac-install/).
Open a new terminal after installation, then verify that its CLI and Compose
are available:

```bash
docker info
docker compose version
```

Clone the pinned Demo as a sibling of TraceMind:

```bash
git clone https://github.com/open-telemetry/opentelemetry-demo.git ../opentelemetry-demo
git -C ../opentelemetry-demo checkout dedc0178918e260823323b8d95005a8cb924b007
```

The first evaluation case injects `paymentFailure=100%`. Its generated flagd
configuration is written under ignored `artifacts/`; the upstream Demo checkout
is not modified.

```bash
uv run python scripts/demo_scenario.py prepare
uv run python scripts/demo_scenario.py start
uv run python scripts/demo_scenario.py status
```

After the load generator has produced checkout failures, run the live diagnosis
and deterministic ground-truth checks:

```bash
uv run python scripts/run_evaluation.py --lookback-minutes 10
```

Add `--verbose` to show structured run, model, tool, and context events in the
terminal. The event payload contains operational metadata, not raw model
messages or telemetry results.

The report is written to `artifacts/evaluations/payment-failure.json`. Stop and
remove the scenario containers and volumes with:

```bash
uv run python scripts/demo_scenario.py stop
```

### Additional live fault demos

Three cases share the same running Demo and HTTP MCP service:

| Case | Fault | Symptoms | Required evidence |
| --- | --- | --- | --- |
| `payment_failure` | `paymentFailure=100%` | Checkout fails during payment | Payment error trace |
| `cart_failure` | `cartFailure=100%` | Clearing the cart fails; items can remain after an order | Cart error trace |
| `shipping_slowdown` | `intlShippingSlowdown=5sec` | Non-US orders take longer; US orders are unaffected | Shipping delay log |

Switch the existing deployment to the cart scenario:

```bash
uv run python scripts/demo_scenario.py switch --scenario cart_failure
```

Add an item in the shop at `http://localhost:8080`, then clear the cart.
The fault routes `EmptyCart` to an invalid storage endpoint; it does not
break every cart operation. Checkout ignores the cart-clearing error, so a
successful order can leave items behind.

For the latency scenario:

```bash
uv run python scripts/demo_scenario.py switch --scenario shipping_slowdown
```

Place an order with a Canadian address, then compare with a US address.
Only non-US `/ship-order` requests receive the injected delay. The bundled
load generator also includes a Canadian customer, but explicit requests are
more reliable for a quick demonstration. Trace detail currently includes
error spans and the twenty longest successful spans; use service/operation
timing to locate the delay, then shipping logs to establish its mechanism.

The MCP log adapter prefers event time and uses `observedTimestamp` only
when event time is missing or zero (exported as `1970-01-01`). Such evidence
records `timestamp_source` and the original timestamps. Observation time
means when telemetry was observed, not a confirmed event occurrence time.
Valid historical events remain historical even if collected recently.

Each switch builds from the pristine Demo flags, so these three faults do
not accumulate. It saves the previous active configuration and recreates
only flagd. Existing business services, MCP, and telemetry data remain.
Allow fresh traffic and telemetry export after switching; wait at least
two minutes and use a one-minute lookback to exclude the previous fault.

Run either the interactive investigation (API and worker must be running)
or the scored evaluation. Replace the case path for the shipping scenario:

```bash
uv run python scripts/demo_investigation.py \
  --case evals/cases/cart_failure.json --lookback-minutes 1

TRACEMIND_AGENT_MAX_MODEL_CALLS=5 uv run python scripts/run_evaluation.py \
  --case evals/cases/cart_failure.json --lookback-minutes 1

uv run python scripts/demo_investigation.py \
  --case evals/cases/shipping_slowdown.json --lookback-minutes 1

TRACEMIND_AGENT_MAX_MODEL_CALLS=5 uv run python scripts/run_evaluation.py \
  --case evals/cases/shipping_slowdown.json --lookback-minutes 1
```

Default reports are named after the case, e.g.
`artifacts/evaluations/cart-failure.json` and `shipping-slowdown.json`.
Use `--output` to preserve multiple runs of the same case. Ground truth and
flag configuration are used by scenario setup and scoring, not sent to the
diagnosis model. Scoring checks source/service citations and keywords; it
does not validate every statement in the explanation.

```bash
# Return to the configuration before the most recent start/switch.
uv run python scripts/demo_scenario.py restore

# Explicitly select the payment scenario again.
uv run python scripts/demo_scenario.py switch --scenario payment_failure

# Generate a candidate configuration without activating it.
uv run python scripts/demo_scenario.py prepare --scenario shipping_slowdown
```

`restore` keeps one previous configuration, not a stack. `prepare` writes
`artifacts/otel-demo/<scenario>.flagd.json`; the active mount retains the
legacy `payment-failure.flagd.json` filename for existing deployments.
`status` and `stop` do not rewrite flags. `stop` still removes the Demo
containers and volumes; use `restore` to undo a fault without deleting data.

### Terminal logging

The evaluation CLI writes standard logging output to stderr. Evaluation
summaries use INFO. `--verbose` enables DEBUG for the harness event logger
without enabling third-party HTTP client debug output. Records include
timestamps and logger names; verbose event JSON carries the investigation
run ID and span relationships. Queries, prompts, telemetry payloads, and raw
exception messages are excluded from these events.

`RunHarness.emit()` records events in run state and JSONL regardless of terminal
log level. Applications embedding TraceMind can attach their own logging
handlers. Trace-file warnings use standard logging.

## Bounded recovery (S3)

The harness owns retries; provider SDK retries remain disabled. Transient
transport failures, timeouts, and HTTP 408/429/500/502/503/504 receive at most
`TRACEMIND_AGENT_RETRY_ATTEMPTS` retries (default 2), with capped exponential
backoff, jitter, and Retry-After handling. Every attempt consumes the existing
model/tool budget; backoff consumes the run deadline. Cancellation stops retries.
Authentication failures and permanent model request errors stop explicitly.

Exhausted transient tool failures and invalid queries return sanitized error
feedback, allowing the agent to correct arguments or investigate another source.
Failed calls produce no evidence. Unclassified telemetry failures stop the run.

Fallback is disabled by default. Set both `TRACEMIND_AGENT_FALLBACK_PROVIDER`
and `TRACEMIND_AGENT_FALLBACK_MODEL`, with that provider's API key, to enable
one model switch after transient primary failures exhaust retries. History,
evidence, and budgets are shared, and the fallback stays selected for that run.
A model switch emits `model.switched`; retry scheduling emits
`recovery.retry_scheduled`. Failed model attempts with unavailable token usage
increase the unreported-call counter, rather than reporting zero token cost.

Missing diagnosis structure (including text-only or empty responses), schema
errors, and citation errors allow one diagnosis repair by default. Native
structured output and tool-based output use the same bounded repair loop.
Repair feedback identifies the violated rule and up to 20 offending evidence
IDs or schema field errors. `diagnosis.rejected` records `validation_code`,
`validation_message`, and the relevant `validation_evidence_ids` or
`validation_fields` in the run ledger and JSONL. Its `candidate_preview` retains
at most 4096 UTF-8 bytes of rejected diagnosis output; `candidate_truncated`
marks incomplete excerpts. These events are persisted before checking the
repair allowance, so the final rejected candidate survives even if its model
node never commits a graph checkpoint. Full prompts and tool payloads are not
copied into these events, and raw provider exceptions are excluded.
Repair calls consume model/time/context budgets and reported token usage,
but do not advance the no-progress counter. Rejected exchanges enter graph
history when the model node completes, and mixed batches execute telemetry calls
with paired call IDs. Exhaustion returns an inconclusive diagnosis with
`invalid_diagnosis`. Three consecutive investigation rounds without new
usable observations trigger `no_progress`; this is an observation-based
stall heuristic, not a judgment of reasoning quality. Rereading a previously
returned preview does not advance it; reading a previously unseen detail page
can. Configure these limits through
`TRACEMIND_AGENT_DIAGNOSIS_REPAIRS` and `TRACEMIND_AGENT_NO_PROGRESS_ROUNDS`.

Jaeger trace summaries count only spans starting within the inclusive
investigation window. `summary_scope=spans_started_in_window` and
`excluded_span_count` describe the selection; `duration_ms` is the selected
span extent, not whole-request latency. Traces without selected spans are
omitted. Tool bindings exclude out-of-window observations before storage and
return exclusion guidance without evidence IDs. `observations.filtered` records
these exclusions; an empty result does not prove service health. Legacy stored
evidence still receives the same strict final citation validation.

`get_trace_detail` requires an exact 16- or 32-character hexadecimal ID copied
from returned `attributes.trace_id`, rather than an evidence UUID or span ID.
Invalid formats are rejected before a backend request; the model receives
argument-validation feedback to correct the call.

### Final diagnosis after tool investigation closes

`no_progress`, `tool_budget`, and `repeated_tool_call` close investigation when
there is remaining model-call budget and time. Middleware removes all telemetry
and `read_evidence` tools from subsequent model requests and instructs the model
to submit its diagnosis using evidence already supplied. With tool-based
structured output, only `DiagnosisDraft` remains exposed as the result-submission
mechanism. Native structured output needs no tool for submitting the result.

Already-admitted parallel queries can finish within existing timeouts. Calls
rejected by the closure return paired error messages without reaching a backend
or consuming additional tool attempts. A model ignoring closure receives only
the configured diagnosis repair allowance; tools stay closed during repairs.
Citation and investigation-window validation still apply. A valid diagnosis
can be complete or inconclusive with a useful explanation; there is no forced
claim of a confirmed cause.

Finalization adds no model attempts, timeout, context space, graph steps, retry,
or repair allowance. Model/context/graph exhaustion, cancellation, and the
deadline retain their existing stop behavior. If final output cannot be obtained
or validated within those limits, the existing fallback/failure path is used.
`policy.finalizing` records why investigation closed, while the terminal stop
reason describes the final outcome (`completed`, `insufficient_evidence`, or a
hard stop). `tool.skipped` identifies rejected tool dispatches. SSE displays
the transition as “工具调查已停止，正在根据已有证据整理结论”. The persisted phase
event keeps tools closed when a worker resumes; no database migration is needed.
