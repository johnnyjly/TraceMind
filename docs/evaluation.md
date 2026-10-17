# Evaluation with Langfuse

Langfuse stores datasets, experiment results, scores and comparisons. TraceMind
supplies task adapters and evaluators through its Python SDK. The agent harness
owns permissions, execution budgets, checkpoints and recovery. This guide
covers repeatable evaluation procedures; generated inputs and results belong
under ignored `artifacts/` or in your Langfuse instance.

## Local deployment

```bash
uv sync --locked --extra rag --dev
uv run python scripts/setup_langfuse.py
docker compose --env-file deploy/.env.langfuse -f deploy/compose.langfuse.yaml up -d
```

Open `http://127.0.0.1:3000`. Bootstrap creates `deploy/.env.langfuse` with mode
0600 and refuses to replace an existing file. The generated administrator is
`tracemind@localhost.invalid`; its password is `LANGFUSE_INIT_USER_PASSWORD`
in that file. Keep provider API keys in your ignored application `.env`.

The generated file contains application model settings as well as Langfuse
credentials. In commands that load `.env` followed by `deploy/.env.langfuse`,
the latter overrides duplicate settings. Check both files when changing the
models used by an experiment.

The Compose project runs Web, Worker, Postgres, ClickHouse, Redis and MinIO.
Named volumes retain data; Web port 3000 and MinIO port 9092 bind to loopback.
To stop these services while retaining their data:

```bash
docker compose --env-file deploy/.env.langfuse -f deploy/compose.langfuse.yaml stop
```

Python evaluators run in the TraceMind process. The Langfuse UI displays their
results; it does not execute the agent or retrieval pipeline.

## Models and call admission

Diagnosis, fallback, retrieval, online verification and offline Judge have
independent settings. See [.env.example](../.env.example) for the current
provider, timeout and output configuration. Configure matching credentials
for every enabled model role. Agreement between models from one family is
not independent ground truth; compare review decisions with human labels.

Every experiment requires `--max-model-calls`. The cap includes investigation,
fallback, retries, auxiliary retrieval, verification and Judge calls within
that CLI invocation. Admitted failed attempts consume capacity. Multiple
invocations have separate caps; account for them together when setting an
experiment-wide budget. Provider SDK retries are disabled so application
admission can count actual attempts.

Reasoning tokens consume model output limits. Per-model, review and overall
run deadlines must leave room for the enabled workflow. Judge and verifier
also have input byte limits; oversized evidence is rejected rather than
silently dropping observations or counterevidence.

Record usage when the provider reports it and retain unknown usage explicitly.
A missing measurement is not zero. Cost estimates depend on configured pricing
and attribution; token counts alone do not establish a billable cost.

## Experiment input

Choose exactly one input source: `--data FILE`, `--case FILE` or
`--dataset NAME`. Every invocation also needs `--task`, `--name` and
`--max-model-calls`.

| Task | Input and purpose |
| --- | --- |
| `agent` | Incident and fixed window; runs a diagnosis against telemetry |
| `saved` | Saved diagnosis and retained evidence; evaluates an existing answer |
| `retrieval` | Query and relevance labels for a published knowledge snapshot |
| `verification` | Candidate and retained evidence; checks the runtime verifier |

`--case` applies only to `agent`; `--judge` applies only to `agent` and `saved`.

`expected_output` and item metadata stay outside investigator and retrieval
inputs. Labels are for evaluators. An illustrative agent item is:

```json
[
  {
    "input": {
      "incident": {
        "title": "Checkout errors",
        "description": "Investigate failures in the specified window",
        "affected_services": ["checkout"]
      },
      "start": "2024-01-01T00:00:00Z",
      "end": "2024-01-01T00:10:00Z"
    },
    "expected_output": {
      "expected_status": "complete",
      "reference_answer": "Human-reviewed root cause and evidential limits"
    }
  }
]
```

Replace the illustrative timestamps and reference with your incident data.
Save the list to `artifacts/evaluations/held-out-items.json` before running:

```bash
uv run python scripts/run_experiment.py --task agent --name agent-baseline \
  --data artifacts/evaluations/held-out-items.json \
  --env-file .env --env-file deploy/.env.langfuse --judge --max-model-calls 20
```

The alternative `--case evals/cases/payment_failure.json --lookback-minutes 10`
creates a fixed UTC window ending at invocation time. It does not activate the
fault. Prepare the scenario separately and wait for fresh telemetry. The case's
tool-call cap still applies.

For datasets hosted in Langfuse, use `--dataset NAME` and optionally
`--dataset-version` with the dataset's actual ISO timestamp. File-backed
experiments publish scored observations and experiment summaries but do not
automatically create hosted dataset versions.

### Saved diagnoses

Saved items contain a diagnosis, retained evidence, investigation window and,
when available, run state. Re-evaluation cannot assess observations that were
never collected. A bounded real-Judge check using manufactured inputs is:

```bash
uv run python scripts/run_experiment.py --task saved --name synthetic-judge \
  --data evals/langfuse/synthetic-diagnoses.json \
  --env-file .env --env-file deploy/.env.langfuse --judge --max-model-calls 2
```

This dataset contains both supported and deliberately unsupported answers.
Inspect each verdict and rationale; an overall failure alone can also result
from infrastructure or output-format errors. Judge calls are real model calls.

## RAG retrieval

Build and verify a snapshot using [RAG setup](rag.md), then create relevance
labels against its document IDs. Retrieval returns knowledge chunks, so its
scores measure retrieval rather than generated-answer correctness.

An illustrative experiment input is:

```json
[
  {
    "input": {
      "query": "How should connection errors be investigated?",
      "mode": "mix"
    },
    "expected_output": {
      "relevance": {"your.corpus.document.id": 3}
    }
  }
]
```

Replace the placeholder document ID with one from your published corpus and
save the list to `artifacts/evaluations/retrieval-items.json`:

```bash
uv run --extra rag python scripts/run_experiment.py --task retrieval --name rag-baseline \
  --data artifacts/evaluations/retrieval-items.json \
  --env-file .env --env-file deploy/.env.langfuse --max-model-calls 20
```

Labels use document IDs and grades 0–3. Repeated chunks from the same document
count once. Available vector/graph, BM25, fused and final stages report
precision, recall, MRR and nDCG at the configured result limit. Empty relevance
labels measure abstention; missing labels omit quality scores and cannot satisfy
an explicit threshold. Use `--min-score NAME=MIN` for chosen acceptance gates.

Use development labels to select a reranker cutoff, then freeze it before
holdout evaluation. Labels, model revision and snapshot must agree. Store
corpus-specific suites under ignored `artifacts/`; changing a snapshot name
does not make labels valid for a different corpus. See [retrieval evaluation](rag.md#evaluate-retrieval)
for the suite format and `scripts/rag.py evaluate` workflow.

## Scores and acceptance

| Score or group | What it establishes |
| --- | --- |
| `execution_success` | A result was produced without an operational error |
| `citation_integrity` | Cited telemetry exists in retained, in-window evidence |
| `required_source_citation` | Labeled service/source provenance, not causal correctness |
| `trace_integrity`, `tool_budget` | Execution accounting and tool-limit compliance |
| `expected_status` | Match to an explicitly supplied status expectation |
| `verification_completed`, `verification_error` | Runtime review completion or failure |
| `grounded_findings`, `supported_finding_count` | Reviewed findings retained with supporting citations |
| `judge_completed`, `evidence_support`, `uncertainty_appropriate` | Judge execution and evidence-based assessment |
| `root_cause_correctness` | Match to a held-out reference, when supplied |
| `false_completion` | An answer claims completion without sufficient support or correctness |
| `model_calls`, `tool_calls`, `reported_tokens`, `unreported_model_calls` | Execution and usage accounting |
| `evaluation_coverage`, `acceptance` | Required scores and thresholds across all items |

Default gates require execution success and citation integrity for diagnoses.
Agent runs also require trace integrity and tool-budget compliance. Reviewed
complete answers need accepted runtime verification, including task completion
when that score is available. Explicit expected-status requirements stay strict;
without one, reviewed answers still need a grounded finding to pass the
usefulness gate. Judge runs require completed review, support and appropriate
uncertainty. A held-out correctness reference is required for correctness scoring.

`--min-score NAME=MIN` sets or overrides thresholds. Missing required scores,
failed evaluators or incomplete publication fail the experiment. Exit status 0
means both score acceptance and Langfuse publication verification succeeded.
Acceptance applies to the selected dataset and rubric; it is not general
production accuracy.

### Context-cache measurements

Model events retain reported cache hit/miss tokens and whether usage was
available. Cache hits are already part of input tokens. Aggregate the weighted
rate as `sum(hit) / sum(hit + miss)` over measured requests, and report coverage
alongside it. Missing usage cannot provide a measured hit rate.

Compare matching model, tool schemas, workload and context limits. Separate
initial investigation, later requests, review and retrieval calls. Compression,
finalization and tool changes can affect prefix reuse; inspect latency, total
usage and answer quality alongside cache statistics.

## Online completion verification

Runtime verification is enabled by default. Single-agent deployments can set
`TRACEMIND_VERIFIER_ENABLED=false`; multi-agent mode always requires verification.
The reviewer uses independent `TRACEMIND_VERIFIER_*` settings. Its code defaults
are a 180-second deadline, 131072 input bytes and 8192 output tokens; environment
settings can override them for newly admitted runs.

The investigator submits structured findings, hypotheses, limits and proposed
checks with run-local evidence references. Application code validates these
references and renders the diagnosis. The verifier reviews numbered statements
against retained observations, including counterevidence, and separately checks
whether the original task criteria were met. Knowledge can explain a mechanism
but cannot prove it occurred in the current incident. Empty queries alone do
not establish service health.

| Outcome | Behavior |
| --- | --- |
| Supported statements and satisfied completion criteria | The candidate can pass review |
| Insufficient support or unmet criteria | Preserve supported findings and identify gaps |
| Malformed review or invalid references | At most one format-repair attempt |
| Timeout, provider failure, exhausted repair or oversized input | Explicit review failure; never approve an unchecked draft |

Single-agent rejection does not automatically revise the investigation.
Collaborative mode can use bounded feedback to investigate or submit again.
The verifier itself does not retry a semantic rejection. Review and format
repair share the run's budgets and review deadline.

Run the verifier against manufactured cases with held-out expected decisions:

```bash
uv run python scripts/run_experiment.py --task verification \
  --name verifier-calibration --data evals/langfuse/verifier-calibration.json \
  --env-file .env --env-file deploy/.env.langfuse --max-model-calls 16
```

This task calls the verifier without an investigator or additional Judge.
A correct rejection passes the classification check. Calibration fixtures
exercise particular boundaries; they do not establish representative quality.

## Runtime and regression boundaries

Load Langfuse settings into API/worker processes when runtime tracing is desired.
Regular diagnosis works with tracing disabled; evaluation requires a configured,
authenticated Langfuse instance. Observability does not own execution state:
SQLite ledgers, graph checkpoints and SSE retain their runtime responsibilities.

Synchronous agent evaluation calls service assembly directly. It does not test
queued worker/checkpoint recovery merely by using database paths. Test recovery
separately through the persistence and worker tests. Interactive investigations
require API and worker processes, as described in [README](../README.md).

```bash
uv run pytest
uv run --extra rag pytest --require-rag
```

Scripted replays use fixed inputs and model responses to check execution,
isolation, failure handling and accounting without paid model calls. Sanitized
regression fixtures contain illustrative IDs/timings and preserve specific
claim boundaries. They are not local run transcripts or model-quality results.
The full RAG command requires the optional dependencies.

Frozen-memory comparison uses `--memory-snapshot FILE --compare-memory` with an
agent task. It runs independent off/on variants under one invocation's call cap.
Prepare the snapshot before held-out incident windows, keep it under ignored
`artifacts/`, and do not substitute live memory for a frozen evaluation corpus.

## OpenTelemetry Demo

Keep the upstream Demo in a sibling checkout pinned to version 3.1.0:

```bash
git clone https://github.com/open-telemetry/opentelemetry-demo.git ../opentelemetry-demo
git -C ../opentelemetry-demo checkout dedc0178918e260823323b8d95005a8cb924b007
uv run python scripts/demo_scenario.py prepare
uv run python scripts/demo_scenario.py start
uv run python scripts/demo_scenario.py status
docker compose -f deploy/compose.telemetry-mcp.yaml up -d --build
uv run python scripts/check_mcp.py
```

For an existing checkout, verify its revision instead of cloning again.
`--demo-path` selects another checkout. Generated flags stay in ignored
`artifacts/otel-demo/`; the upstream files are preserved. MCP joins the Demo
network; set `OTEL_DEMO_NETWORK` when using a different network name.
The Compose override supplies resource limits; validate capacity against your
Docker allocation and workload before collecting an evaluation window.

## Live fault cases

| Case | Fault | Expected observations |
| --- | --- | --- |
| `payment_failure` | `paymentFailure=100%` | Checkout failures and Payment Charge error spans |
| `cart_failure` | `cartFailure=100%` | EmptyCart errors; other cart operations can succeed |
| `shipping_slowdown` | `intlShippingSlowdown=5sec` | Slow non-US shipping and a delay log |
| `product_catalog_lock_contention` | `productCatalogLockContention=on` | Slow catalog reads and a table-lock log |
| `payment_unreachable` | `paymentUnreachable=on` | Checkout payment RPC destination-resolution errors |

After the load generator produces fresh fault telemetry:

```bash
uv run python scripts/run_experiment.py --task agent --name payment-live \
  --case evals/cases/payment_failure.json --lookback-minutes 10 \
  --env-file .env --env-file deploy/.env.langfuse --judge --max-model-calls 20
```

Switch the fault with `scripts/demo_scenario.py switch --scenario CASE`, then
select its matching `evals/cases/CASE.json`. A switch preserves old telemetry;
wait for new requests and choose a window containing the intended fault only.
For cart, explicitly exercise clearing items; for shipping, compare US and
non-US orders. Service localization alone does not establish the underlying
mechanism or justify remediation.

## Restore and stop

```bash
uv run python scripts/demo_scenario.py restore
```

`restore` returns to the previous flag configuration. `prepare` only generates
a candidate and `status` does not rewrite flags. When finished:

```bash
uv run python scripts/demo_scenario.py stop
```

`stop` runs Compose teardown with `--volumes`, deleting ordinary Demo volumes.
The external Jaeger volume has separate ownership and survives that teardown.
Use `restore` to undo a fault while keeping the running deployment.

## Comparing single and multi-agent runs

Use the same validated incident window, source revision, model settings,
knowledge snapshot, permissions and budgets. Separate development tuning from
held-out comparisons. Keep every primary failed, inconclusive and timeout
attempt; report diagnostic retries separately. Include investigation, retrieval,
verification and Judge usage in comparisons.

Report service localization, supported findings, completion, false completion,
latency and usage separately. Fixed scripted responses verify control flow;
quality conclusions require labeled cases and review of actual model outputs.
Store experiment inputs, reports, logs and ledgers under ignored `artifacts/`
or in Langfuse, rather than embedding individual run histories in this guide.
