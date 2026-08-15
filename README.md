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

Milestones 1 through 3 provide:

- typed incident, evidence, and diagnosis models;
- environment-based application settings;
- a minimal FastAPI application and health endpoint;
- asynchronous Prometheus, Jaeger, and OpenSearch clients;
- normalization of backend responses into typed evidence;
- a bounded LangGraph single-agent loop with read-only telemetry tools;
- Gemini and OpenAI model providers selected through environment settings;
- validation that a completed diagnosis cites evidence actually returned by a tool;
- a synchronous diagnosis API endpoint;
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

Budget/repetition/recursion stops produce an `inconclusive` diagnosis. Operational
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
The run state is currently in memory; JSONL events are not resumable checkpoints.
S2 context controls are described below. Automatic retries/fallback and
persistent recovery remain subsequent stages in [DESIGN.md](DESIGN.md).

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
- retrieve an observation's complete JSON in 2,048-character pages by ID;
- continue either operation using the returned `next_offset`.

Reads are isolated to the current run and consume the same tool-call,
repetition, and timeout budgets as telemetry queries. Explicit existing
`TRACEMIND_AGENT_ALLOWED_TOOLS` lists must add `read_evidence` to enable it.
Without that permission, compressed details are unavailable to the agent.

Before every model call, middleware budgets the system message, conversation,
tool schemas, and diagnosis schema, plus an output reservation. The initial
estimator (`utf8-bytes-v1`) counts serialized UTF-8 bytes plus 1,024 units of
framing allowance. It deliberately overestimates ordinary text/JSON tokens;
it is not a provider tokenizer or a guarantee about billing or model limits.
Provider-reported usage remains separately recorded.

Defaults in `.env.example` are:

| Setting suffix (`TRACEMIND_AGENT_`) | Default | Meaning |
| --- | --- | --- |
| `CONTEXT_TOKENS` | 32768 | Estimated input plus reserved output allowance |
| `OUTPUT_TOKENS` | 4096 | Output reservation and provider output cap |
| `OBSERVATION_BYTES` | 8388608 | Maximum stored serialized evidence bytes |

When input exceeds its allowance, the oldest tool results are replaced with
compact evidence indexes first. Every displayed evidence ID and every
assistant tool-call/result pair is retained. System instructions, incident
information, and assistant messages are preserved. A newly requested detail
page reaches the model intact. Compression changes the model input view, not
the stored observations or graph message history; no summarization LLM is used.

If the remaining input cannot fit, or the evidence store reaches its byte
limit, the run returns `inconclusive` with `context_limit` and retains partial
evidence. The byte limit bounds serialized retained observations, not Python
heap usage or HTTP response buffering. This is not a persistent artifact store.

`context.selected` events record input estimates before/after reduction,
reduced-message counts, output reservation, and admission outcome.
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
