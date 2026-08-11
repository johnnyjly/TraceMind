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
- Podman and `podman-compose` for the OpenTelemetry Demo integration

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
and exception messages are not included in these event files. The optional
`--verbose` console callback remains a separate development debugging facility.
Trace file open/write/close failures emit a server warning without replacing
the diagnosis result or its original error. Events remain available in memory;
the file can be missing or incomplete. Expected provider and telemetry errors
are classified separately from unexpected code errors (`internal_error`).

Evaluation reports include `run_state` with counters, events, and stop reason.
The run state is currently in memory; JSONL events are not resumable checkpoints.
Context compression, automatic retries/fallback, and persistent recovery are
subsequent stages in [DESIGN.md](DESIGN.md).

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

On Ubuntu or WSL, install the rootless container runtime and Compose provider:

```bash
sudo apt-get update
sudo apt-get install -y podman podman-compose
systemctl --user enable --now podman.socket
podman info
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

Add `--verbose` to print every model input, model response, tool call, and tool
result. Each event is limited to 12,000 characters by default; adjust the limit
with `--verbose-max-chars`.

The report is written to `artifacts/evaluations/payment-failure.json`. Stop and
remove the scenario containers and volumes with:

```bash
uv run python scripts/demo_scenario.py stop
```
