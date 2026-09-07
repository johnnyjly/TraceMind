# TraceMind

TraceMind investigates microservice incidents using metrics, traces and logs,
then returns a structured diagnosis with supporting evidence. It runs against
reproducible faults in the OpenTelemetry Demo.

The current baseline is one bounded LangGraph agent with context management,
recovery, persistent queued runs and readable progress. Telemetry comes from
an independent read-only HTTP MCP service. Snapshot-based RAG and feedback
memory are optional and disabled by default.

## Requirements

- Python 3.12+, `uv`, and a model API key for diagnoses.
- Docker Desktop with Compose for the Demo and telemetry MCP service.
- A sibling OpenTelemetry Demo checkout pinned to commit
  `dedc0178918e260823323b8d95005a8cb924b007` (version 3.1.0).

Commands below run from this repository root. See
[Demo and evaluation](docs/evaluation.md) for cloning and scenario details.

## Quick start

### 1. Install and configure

```bash
uv sync --dev
cp .env.example .env
```

Set the provider, model and matching key in your local `.env`:

```dotenv
TRACEMIND_LLM_PROVIDER=google
TRACEMIND_LLM_MODEL=gemini-3.7-flash
TRACEMIND_GOOGLE_API_KEY=your-key
```

Other supported configurations:

| Provider | Example model | Key variable |
| --- | --- | --- |
| `openai` | `gpt-5.6-luna` | `TRACEMIND_OPENAI_API_KEY` |
| `deepseek` | `deepseek-flash` | `TRACEMIND_DEEPSEEK_API_KEY` |

Never commit a populated `.env`. [.env.example](.env.example) lists settings
and the local diagnostic budget profile. Restart API and worker after changes;
existing queued runs retain their saved limits and permissions.
Why: Keep credentials local and make configuration changes explicit without altering
admitted work.

### 2. Start the Demo and telemetry service

With the pinned Demo at `../opentelemetry-demo` and Docker running:

```bash
uv run python scripts/demo_scenario.py prepare
uv run python scripts/demo_scenario.py start
docker compose -f deploy/compose.telemetry-mcp.yaml up -d --build
uv run python scripts/check_mcp.py
```

The default scenario enables `paymentFailure=100%`. Allow the load generator
to produce checkout failures before starting a diagnosis. Generated flags live
under ignored `artifacts/`; the upstream Demo checkout is preserved.

The MCP endpoint defaults to `http://127.0.0.1:8012/mcp`. Its backend URLs belong
to the independent [telemetry service](services/otel-telemetry-mcp/README.md).

### 3. Start API and worker

If online knowledge is enabled and `TRACEMIND_RAG_SERVICE_URL` is set, first
start the independent retrieval process in its own terminal:
Why: The configured HTTP path needs a ready owner for its models and stores.

```bash
uv run --extra rag python scripts/run_rag_service.py
curl --fail http://127.0.0.1:8011/ready
```

`TRACEMIND_RAG_SNAPSHOT` must name an existing published snapshot and match
`TRACEMIND_AGENT_KNOWLEDGE_SNAPSHOT`. The process loads and warms the local
models before accepting queries. It binds localhost, uses two inference threads,
and retains its model/store lifetime independently of API and worker restarts.
Why: Matching published snapshots prevents references from silently pointing to another
corpus.
Qdrant remains in `deploy/compose.rag.yaml`. See
[service operation](docs/rag-online-integration.md#independent-query-service).

In separate terminals with the same `.env`:
Why: API admission and worker execution need consistent provider and policy settings.

```bash
uv run uvicorn tracemind.api.app:app --reload
```

```bash
uv run python scripts/run_worker.py
```

When online knowledge is enabled, add `--extra rag` to both startup commands.
The HTTP client also imports the pinned LightRAG keyword prompt, so API and worker
still need the optional dependencies when using the independent service.

`GET /health` works without a model key. Interactive API schemas are available
at `http://127.0.0.1:8000/docs`. Queued runs require the worker to execute.

### 4. Run a diagnosis

From another terminal:

```bash
uv run python scripts/demo_investigation.py --lookback-minutes 10
```

This makes real model calls, submits an investigation, streams progress and
prints the diagnosis. Keep its investigation ID for follow-up questions:

```bash
uv run python scripts/demo_investigation.py --investigation-id <id> \
  --purpose explain --message "Why is payment the root cause?"
```

Explanation reads retained evidence and knowledge. Use `--purpose investigate`
for fresh telemetry queries; each round receives its own run and budgets.

## API overview

| Endpoint | Purpose |
| --- | --- |
| `POST /api/v1/diagnoses` | Wait for a synchronous diagnosis |
| `POST /api/v1/runs` | Queue a persistent run; returns 202 |
| `GET /api/v1/runs/{id}` | Read run status |
| `GET /api/v1/runs/{id}/result` | Read diagnosis and evidence |
| `POST /api/v1/runs/{id}/cancel` or `/resume` | Cancel or requeue an interrupted run |
| `GET /api/v1/runs/{id}/events` | Subscribe to durable SSE progress |
| `POST /api/v1/investigations` | Create an incident and its first queued run |
| `POST /api/v1/investigations/{id}/messages` | Ask a follow-up question |

Creation accepts an incident and a fixed, timezone-aware investigation window.
Request examples and lifecycle details are in [Running TraceMind](docs/operations.md).

## Tests and evaluation

```bash
uv run pytest
uv run --extra rag pytest --require-rag
```

The base installation skips optional RAG modules. The full RAG acceptance command
requires the extra and fails early if its dependencies are missing.
Why: A passing base test run must not be mistaken for validation of optional retrieval.

The replay suite needs no Docker or model credentials. Scripted replay measures
workflow behavior; live diagnosis quality requires separate controlled cases.
Why: Fixed answers test execution controls without measuring model judgment.
Evaluation uses local Langfuse; deployment and dataset setup are described in
[Evaluation](docs/evaluation.md). For a scored live payment-failure run:

```bash
uv run python scripts/run_experiment.py --task agent --name payment-baseline \
  --case evals/cases/payment_failure.json --lookback-minutes 10 \
  --env-file .env --env-file deploy/.env.langfuse --judge --max-model-calls 20
```

Online completion verification is opt-in with `TRACEMIND_VERIFIER_ENABLED=true`.
It reviews a complete candidate once, before acceptance; rejection returns
inconclusive and review failure fails the run. It never automatically repairs
the diagnosis. See [verification](docs/evaluation.md#online-completion-verification).
Why: Keep review costs and failure outcomes explicit before enabling the saved
completion gate.

## Documentation

| Document | Contents |
| --- | --- |
| [Design](DESIGN.md) | Current architecture, contracts and limitations |
| [Running TraceMind](docs/operations.md) | API, budgets, persistence, progress and memory |
| [Demo and evaluation](docs/evaluation.md) | Fault scenarios, replay, live and memory evaluation |
| [RAG setup](docs/rag.md) | Models, ingestion, snapshots and retrieval evaluation |
| [Online RAG integration](docs/rag-online-integration.md) | Module ownership, callback and lifetime contracts |
| [Telemetry MCP service](services/otel-telemetry-mcp/README.md) | Independent deployment and tool protocol |
| [Changelog](CHANGELOG.md) | Feature changes |

TraceMind targets a local, single-worker deployment. Shared-service authentication,
distributed workers, automatic alert ingestion, mutation tools, multi-agent
orchestration and reinforcement learning remain outside the current baseline.
