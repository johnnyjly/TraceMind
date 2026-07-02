# TraceMind

TraceMind is an evidence-driven incident diagnosis system for distributed
microservices. It will investigate synthetic incidents in the OpenTelemetry
Demo by querying metrics, traces, and logs, then return a structured diagnosis
with supporting evidence.

The first implementation phase intentionally starts with a single-agent
baseline. Multi-agent orchestration, MCP, RAG, human approval, and reinforcement
learning are later phases and are not part of the current implementation.

## Current status

Milestone 1 provides:

- typed incident, evidence, and diagnosis models;
- environment-based application settings;
- a minimal FastAPI application and health endpoint;
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

TraceMind will connect to the Demo's Prometheus, Jaeger, and OpenSearch APIs via
the URLs in `.env`. The tested upstream revision and startup procedure will be
recorded when telemetry integration is implemented. The Demo's built-in agentic
mode will not be used.
