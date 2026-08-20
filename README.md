# TraceMind

TraceMind is an evidence-driven incident diagnosis system for distributed
microservices. It will investigate synthetic incidents in the OpenTelemetry
Demo by querying metrics, traces, and logs, then return a structured diagnosis
with supporting evidence.

The first implementation phase starts with a single-agent baseline. The next
priority is its context management, execution controls, recovery, persistence,
and tracing; see [the single-agent design](DESIGN.md). Multi-agent orchestration,
MCP, RAG, human approval, and reinforcement learning are later phases and are
not part of the current implementation.

## Current status

The current single-agent baseline provides:

- typed incident, evidence, and diagnosis models;
- environment-based application settings;
- a minimal FastAPI application and health endpoint;
- asynchronous Prometheus, Jaeger, and OpenSearch clients;
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
