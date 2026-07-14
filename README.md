# TraceMind

TraceMind is an evidence-driven incident diagnosis system for distributed
microservices. It will investigate synthetic incidents in the OpenTelemetry
Demo by querying metrics, traces, and logs, then return a structured diagnosis
with supporting evidence.

The first implementation phase intentionally starts with a single-agent
baseline. Multi-agent orchestration, MCP, RAG, human approval, and reinforcement
learning are later phases and are not part of the current implementation.

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
- Docker Compose for the later OpenTelemetry Demo integration

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

## OpenTelemetry Demo

The official OpenTelemetry Demo is kept as a separate sibling checkout rather
than copied into this repository:

```text
repos/
├── tracemind/
└── opentelemetry-demo/
```

TraceMind connects to the Demo's Prometheus, Jaeger, and OpenSearch APIs via the
URLs in `.env`. A pinned Demo revision and reproducible injected-fault run are
the next milestone. The Demo's built-in agentic mode is not used.
