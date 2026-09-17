# TraceMind

TraceMind investigates microservice incidents using metrics, traces and logs,
then produces a structured diagnosis with supporting evidence, hypotheses and
explicit gaps. It runs against reproducible faults in the OpenTelemetry Demo.

A bounded LangGraph agent is the default. An opt-in multi-agent workflow adds a
coordinator, up to two concurrent investigators and bounded verifier feedback.
Both modes share permissions, model/tool budgets, retained evidence and durable
run state. Telemetry comes from an independent read-only HTTP MCP service.

## What is implemented

- Fixed investigation windows, evidence references and source validation.
- Context limits, bounded retries, optional fallback and controlled finalization.
- Persistent queued runs, worker leases, checkpoints, cancellation and recovery.
- Durable SSE progress and incident follow-up questions.
- Completion verification, enabled by default, with explicit partial/failure outcomes.
- Optional snapshot-based RAG and feedback memory, disabled by default.
- Langfuse tracing and evaluation, scripted regression replays and live fault cases.

The supported deployment is local with one worker. Parallel investigation tasks
run inside that worker; distributed workers, shared-service authentication,
automatic alert ingestion and remediation tools are outside the current scope.
Recovery does not guarantee exactly-once external model or tool execution.

## Requirements

- Python 3.12+, `uv`, and credentials for the configured diagnosis and review models.
- Docker with Compose for the Demo, telemetry MCP and optional supporting services.
- A sibling OpenTelemetry Demo checkout pinned to
  `dedc0178918e260823323b8d95005a8cb924b007` (version 3.1.0).

Commands below run from the repository root. See
[Demo setup](docs/evaluation.md) for checkout and fault scenario instructions.

## Quick start

### 1. Install and configure

```bash
uv sync --locked --dev
cp .env.example .env
```

The example configuration uses Google for diagnosis and DeepSeek for verification:

```dotenv
TRACEMIND_LLM_PROVIDER=google
TRACEMIND_LLM_MODEL=gemini-3.7-flash
TRACEMIND_GOOGLE_API_KEY=your-google-key
TRACEMIND_VERIFIER_ENABLED=true
TRACEMIND_VERIFIER_PROVIDER=deepseek
TRACEMIND_VERIFIER_MODEL=deepseek-v4-pro
TRACEMIND_DEEPSEEK_API_KEY=your-deepseek-key
```

Diagnosis supports `google`, `openai` and `deepseek`; set the matching model and
`TRACEMIND_GOOGLE_API_KEY`, `TRACEMIND_OPENAI_API_KEY` or
`TRACEMIND_DEEPSEEK_API_KEY`. Review has its own provider/model settings. The
values above are repository configuration examples, not availability guarantees.
To explicitly disable review in single-agent mode, set
`TRACEMIND_VERIFIER_ENABLED=false`; multi-agent mode requires review.

[.env.example](.env.example) documents settings and diagnostic budgets. Keep populated
environment files local. Restart API and worker after configuration changes;
admitted runs retain their saved limits and policies.

### 2. Start the Demo and telemetry service

With the pinned Demo at `../opentelemetry-demo` and Docker running:

```bash
uv run python scripts/demo_scenario.py prepare
uv run python scripts/demo_scenario.py start
docker compose -f deploy/compose.telemetry-mcp.yaml up -d --build
uv run python scripts/check_mcp.py
```

The default scenario enables `paymentFailure=100%`. Allow the load generator to
produce checkout failures before investigating. Generated configuration lives
under ignored `artifacts/`. The telemetry MCP endpoint defaults to
`http://127.0.0.1:8012/mcp`; its backend configuration belongs to the independent
[telemetry service](services/otel-telemetry-mcp/README.md).

### 3. Start API and worker

Run these in separate terminals with the same environment:

```bash
uv run uvicorn tracemind.api.app:app --reload
```

```bash
uv run python scripts/run_worker.py
```

`GET /health` requires no model key. Interactive API schemas are available at
`http://127.0.0.1:8000/docs`. Queued runs require the worker to execute.

### 4. Investigate and follow up

```bash
uv run python scripts/demo_investigation.py --lookback-minutes 10
```

This submits an investigation, makes real model calls, streams progress and
prints a diagnosis. Keep the investigation ID for follow-up questions:

```bash
uv run python scripts/demo_investigation.py --investigation-id <id> \
  --purpose explain --message "Why is payment the root cause?"
```

Explanation uses retained evidence and knowledge. `--purpose investigate`
permits fresh telemetry queries; each round has its own run and budgets.

## Optional capabilities

**Multi-agent investigation:** set `TRACEMIND_AGENT_MODE=multi`. The `graph-v6`
workflow delegates at most two tasks per batch and performs bounded synthesis
and verification rounds. Partial task results and original observations survive
recoverable interruptions. See [multi-agent architecture](docs/multi-agent.md)
for admission requirements, context limits and recovery boundaries.

**Knowledge retrieval:** install the `rag` extra, prepare and publish a snapshot,
then configure knowledge permissions and snapshot settings. RAG combines graph,
vector and BM25 candidates, rank fusion and reranking. With
`TRACEMIND_RAG_SERVICE_URL` configured, start the independent service first:

```bash
uv run --extra rag python scripts/run_rag_service.py
curl --fail http://127.0.0.1:8011/ready
```

`TRACEMIND_RAG_SNAPSHOT` must match `TRACEMIND_AGENT_KNOWLEDGE_SNAPSHOT`.
API and worker also need `--extra rag` when knowledge is enabled. Qdrant is
configured in `deploy/compose.rag.yaml`. See [RAG setup](docs/rag.md) and
[service operation](docs/rag-online-integration.md#independent-query-service).

**Feedback memory:** single-agent mode can retain scoped, human-confirmed
incidents and durable guidance. It requires explicit environment/deployment
scope. Multi-agent mode currently rejects enabled feedback memory. See
[operations](docs/operations.md) for configuration and retention rules.

## API overview

| Endpoint | Purpose |
| --- | --- |
| `POST /api/v1/diagnoses` | Wait for a synchronous diagnosis |
| `POST /api/v1/runs` | Queue a persistent run; returns 202 |
| `GET /api/v1/runs/{id}` | Read run status |
| `GET /api/v1/runs/{id}/result` | Read diagnosis and evidence |
| `POST /api/v1/runs/{id}/cancel` or `/resume` | Cancel or requeue a run |
| `GET /api/v1/runs/{id}/events` | Subscribe to durable SSE progress |
| `POST /api/v1/investigations` | Create an incident and its first run |
| `POST /api/v1/investigations/{id}/messages` | Ask a follow-up question |

Creation requires a timezone-aware investigation window. Synchronous responses
expose `X-TraceMind-Run-ID` and `X-TraceMind-Stop-Reason`. Request examples and
lifecycle rules are in [operations](docs/operations.md).

## Tests and evaluation

```bash
uv run pytest
uv run --extra rag pytest --require-rag
```

Base installations skip optional RAG modules. The second command requires all
RAG dependencies. Scripted tests and replays check workflow behavior, not model
diagnosis quality. Live evaluations require separate controlled faults, model
credentials and explicit call budgets.

After configuring local Langfuse, a scored payment-failure experiment can run as:

```bash
uv run python scripts/run_experiment.py --task agent --name payment-baseline \
  --case evals/cases/payment_failure.json --lookback-minutes 10 \
  --env-file .env --env-file deploy/.env.langfuse --judge --max-model-calls 20
```

The completion verifier reviews numbered assertions against retained evidence.
In single-agent mode, rejection retains supported partial findings without an
automatic investigation revision; multi-agent mode can use bounded feedback to
revise or investigate again. Review failures cannot approve an unreviewed draft.
Unknown mechanisms do not erase evidence-backed service localization.

Evaluate multi-agent quality with controlled comparisons. Keep failed,
inconclusive and timeout attempts in evaluation records. See
[evaluation](docs/evaluation.md) for the evaluation method.

## Documentation

| Document | Contents |
| --- | --- |
| [Design](DESIGN.md) | Architecture, contracts and limitations |
| [Operations](docs/operations.md) | API, budgets, persistence, progress and memory |
| [Multi-agent diagnosis](docs/multi-agent.md) | Coordination, budgets and recovery |
| [Demo and evaluation](docs/evaluation.md) | Fault scenarios, live tests and scoring |
| [RAG setup](docs/rag.md) | Ingestion, snapshots and retrieval evaluation |
| [Online RAG integration](docs/rag-online-integration.md) | Service and lifetime contracts |
| [Telemetry MCP service](services/otel-telemetry-mcp/README.md) | Deployment and protocol |
| [Changelog](CHANGELOG.md) | Feature changes |

The Python distribution and import package are both `tracemind`; application
settings use `TRACEMIND_`. Existing local deployments must update their environment
keys, import paths and response-header consumers. Renamed Compose projects and
named volumes do not automatically migrate existing local service data.
