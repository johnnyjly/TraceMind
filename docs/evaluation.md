# Evaluation with local Langfuse

Langfuse owns datasets, experiment results, scores and comparisons. TraceMind
supplies task adapters and versioned evaluators through its Python SDK. The
agent harness still owns execution budgets, authorization, checkpoints and
recovery. An optional online verifier checks completion before acceptance;
Langfuse remains the only experiment/reporting platform.

## Local deployment

```bash
uv sync --locked --extra rag --dev
uv run python scripts/setup_langfuse.py
docker compose --env-file deploy/.env.langfuse -f deploy/compose.langfuse.yaml up -d
```

Open <http://127.0.0.1:3000>. The local administrator is
`tracemind@localhost.invalid`; its generated password is
`LANGFUSE_INIT_USER_PASSWORD` in ignored `deploy/.env.langfuse`. The file has
mode 0600 and contains local project credentials. Bootstrap refuses to replace
an existing file. Keep your DeepSeek key in the existing ignored application
`.env`; bootstrap never copies or prints it.

The independent `tracemind-langfuse` Compose project runs Web, Worker, Postgres,
ClickHouse, Redis and MinIO. Only Web port 3000 and MinIO port 9092 are exposed,
on loopback. Port 9090 remains available to Demo Prometheus. Named volumes retain
Langfuse data. To stop this stack while retaining data:

```bash
docker compose --env-file deploy/.env.langfuse -f deploy/compose.langfuse.yaml stop
```

The stack uses Langfuse v4 and the SDK uses v4 observation reads, v3 score reads
and v4 OTLP ingestion. See the [official v4 compatibility guide](https://langfuse.com/docs/administration/versions-and-compatibility)
and [Public API documentation](https://langfuse.com/docs/api-and-data-platform/features/public-api).
SDK Python evaluators execute in the project process; no server-side Python
code-evaluator dispatcher or AWS Lambda is needed. The UI displays the results;
it does not execute the TraceMind agent/MCP/RAG pipeline itself.

## Models and call admission

The example configuration selects:

| Role | Model | Configuration |
| --- | --- | --- |
| Diagnosis | `deepseek-v4-pro` | `TRACEMIND_LLM_*` |
| Fallback | `deepseek-flash` | `TRACEMIND_AGENT_FALLBACK_*` |
| RAG extraction | `deepseek-flash` | `TRACEMIND_RAG_LLM_*` |
| Online verifier | `deepseek-v4-pro` | `TRACEMIND_VERIFIER_*` |
| Independent Judge invocation | `deepseek-v4-pro` | `TRACEMIND_EVAL_JUDGE_*` |

Diagnosis and Judge use separate instances/configuration even when their model
name is identical. Agreement between the same model family is not independent
ground truth; calibrate Judge scores against human-reviewed cases. An additional
model is optional rather than required to run this system.
Why: Agreement within one model family must not be treated as independent causal
validation.

Every experiment requires `--max-model-calls`. A shared callback admits calls
before provider execution and covers investigator, fallback, retries, RAG
extraction, online verification and Judge, across paired memory variants. Provider SDK retries are
disabled. Calls admitted but failing before/at the provider still consume the
cap. The cap is per CLI invocation; sum separate invocations when enforcing a
campaign limit. Choose a cap explicitly for each authorized campaign. The verifier calibration
shown below uses at most **8 total calls**.
Why: An explicit campaign cap accounts for all admitted provider attempts across
experiment variants.

DeepSeek currently runs with the existing non-thinking configuration. Judge
has a separate timeout (60 s), input byte limit (65,536) and output token limit
(4,096). It makes one attempt per item. Oversized evidence fails evaluation;
observations and counterevidence are never silently dropped to fit the Judge.
Why: A review of selectively truncated observations could approve an unsupported answer.

Self-hosting keeps Langfuse storage local. Calls to DeepSeek still send the
selected prompts/evidence to DeepSeek. Synthetic smoke examples contain only
manufactured data; real incident exports require appropriate authorization.

## Experiment input

Use a JSON list of SDK experiment items, or a dataset already stored in local
Langfuse. `expected_output` and item metadata never enter the investigator or
retriever. Labels enter only the evaluators. Do not embed labels inside `input`.
Why: Held-out answers must not leak into the agent or retriever being measured.

An agent item:

```json
[
  {
    "input": {
      "incident": {
        "title": "Checkout errors",
        "description": "Investigate failures in the specified window",
        "affected_services": ["checkout"]
      },
      "start": "2024-09-01T00:00:00Z",
      "end": "2024-09-01T01:00:00Z"
    },
    "expected_output": {
      "expected_status": "complete",
      "reference_answer": "Human-reviewed root cause and evidential limits"
    }
  }
]
```

```bash
uv run python scripts/run_experiment.py --task agent --name agent-baseline \
  --data artifacts/evaluations/held-out-items.json \
  --env-file .env --env-file deploy/.env.langfuse --judge --max-model-calls 20
```

The alternative `--case evals/cases/payment_failure.json --lookback-minutes 10`
constructs an item from a Demo scenario and a fixed UTC window. It does not
activate faults. Run scenario setup separately, and wait for fresh telemetry.
The case's tool cap still applies. See the Demo procedures below for fault setup.

`--task saved` accepts a saved diagnosis in each item's `input`. Canonical fields
are `diagnosis`, `evidence`, `run_state` (or historical `state`), investigation
window and model/latency metadata. The adapter excludes old `result`, scores,
model history and arbitrary report fields. Re-evaluation judges retained
observations, not evidence that was never collected.

An optional bounded real-Judge integration smoke with two synthetic examples:

```bash
uv run python scripts/run_experiment.py --task saved --name synthetic-judge-smoke \
  --data evals/langfuse/synthetic-diagnoses.json \
  --env-file .env --env-file deploy/.env.langfuse --judge --max-model-calls 2
```

One example deliberately invents a database mechanism/remediation. The intended
outcome is a completed Judge that accepts the supported example and rejects the
unsupported one. Overall experiment exit status should be 1 for this mixed set.
An overall failure alone does not prove the Judge identified the bad answer;
inspect item scores and claim rationales.
Why: A mixed dataset can fail for infrastructure or formatting reasons without
distinguishing supported answers.

For repeatable comparisons in the native dataset experiment UI, create/import
a versioned dataset in Langfuse and use:

```bash
uv run python scripts/run_experiment.py --task agent --name agent-baseline \
  --dataset held-out-incidents --dataset-version 2024-01-01T00:00:00+00:00 \
  --env-file .env --env-file deploy/.env.langfuse --judge --max-model-calls 20
```

Local `--data`/`--case` experiments produce scored observations with experiment
metadata. SDK run-level summaries for local data have no dataset-run target;
TraceMind publishes them on the first item trace with the `experiment.` prefix.
Use hosted datasets for native dataset versioning/comparison, rather than
expecting local files to be automatically uploaded as datasets.
Why: Dataset versions are needed for native repeatable comparisons; local trace
publication has different ownership.

## RAG retrieval

RAG currently returns knowledge chunks; it does not generate a final answer.
Retrieval experiments therefore expose retrieval quality, not invented answer
faithfulness/correctness. Diagnosis answer quality uses the Judge described
below. Adding RAG to answer generation will require attaching those exact chunks
to the answer evaluator.

Build and verify a source-checked snapshot with `scripts/rag.py` first, using the
existing RAG workflow. Query items use stable corpus document IDs, not generated
chunk IDs, for relevance labels:

```json
[
  {
    "input": {"query": "How should connection errors be investigated?", "mode": "mix"},
    "expected_output": {"relevance": {"your.corpus.document.id": 3}}
  }
]
```

```bash
uv run --extra rag python scripts/run_experiment.py --task retrieval --name rag-baseline \
  --data artifacts/evaluations/retrieval-items.json \
  --env-file .env --env-file deploy/.env.langfuse --max-model-calls 20 \
  --min-score final.recall@5=0.8 --min-score final.ndcg@5=0.8
```

Labels map document IDs to grades 0–3. Repeated chunks from the same document
count once. Each available vector/graph, BM25, fused and final stage gets
precision@k, recall@k, MRR@k and nDCG@k. Here k is the configured final result
limit; stages are compared at the same document cutoff. An explicitly empty
relevance map means no documents are relevant and yields `expected_empty`.
Absent labels omit quality scores. Missing stages/labels cannot satisfy an
explicit `--min-score` requirement.
Why: Missing labels or stages are missing evidence of quality and cannot satisfy a
required gate.

RAG observations include query, snapshot/model revisions, document hashes,
chunk provenance and stage rankings; extraction callbacks retain model usage.
Each experiment item initializes/closes its own runtime. `elapsed_ms` measures
the query after initialization, not cold model/index loading. Embedding/reranker
execution and chunking behavior retain their ordinary deterministic tests.

## Scores and acceptance

| Score | Meaning and limit |
| --- | --- |
| `execution_success` | Task produced a diagnosis/retrieval result without operational error |
| `citation_integrity` | Unique cited telemetry IDs exist in the retained in-window evidence; a complete diagnosis cites evidence |
| `required_source_citation` | Labeled service/source provenance only; not causal correctness |
| `trace_integrity`, `tool_budget` | Retained harness accounting/closure/isolation and tool limit |
| `expected_status` | Status matches a held-out case expectation |
| `model_calls`, `tool_calls`, `reported_tokens`, `unreported_model_calls` | Execution accounting; missing usage is explicit |
| `backend_attempts`, `retries`, `recovered`, `diagnosis_repairs`, `fallback_activations`, `context_compressions` | Harness recovery and context behavior |
| `run_status`, `stop_reason` | Retained terminal outcome |
| `judge_completed` | One bounded response parsed and passed schema/reference validation |
| `support_verdict`, `evidence_support` | Claim support against all retained telemetry, including counterevidence |
| `uncertainty_appropriate` | Answer separates observed facts, hypotheses and justified abstention |
| `root_cause_correctness` | Match against held-out reference; omitted when no reference exists |
| `false_completion` | Complete answer with insufficient/contradicted support, inappropriate certainty, or a wrong referenced cause |
| `process_quality` | Judge assessment of necessary/redundant queries, contradictions and stopping; omitted without execution events |
| `evaluation_coverage`, `acceptance` | Required score availability and configured thresholds across every item |
| `model_calls_admitted`, `model_call_cap`, `model_calls_remaining` | Run-level shared call admission, including Judge and previous paired variants |

Default gates require execution success and diagnosis citation integrity. Agent
runs require trace integrity/tool budget. Saved items with retained run state
require those gates individually; mixing in an item without state cannot bypass
another item's failed trace gate. Complete verified runs also require a saved
accepted runtime review. Judge runs additionally require Judge
completion, full claim support and appropriate uncertainty. Per-item status and
correctness labels activate their corresponding gates, including mixed labeled
and unlabeled datasets. `--min-score NAME=MIN` configures/overrides thresholds.
A failed task/evaluator, missing required score/item or unavailable score/trace
publication fails acceptance. Acceptance is limited to the dataset, evaluator version and
thresholds; it is not a causal truth certificate.
Why: Acceptance is meaningful only for the measured dataset, rubric and required score
coverage.

Judge claim rationales, supporting/conflicting evidence IDs and process event
sequences, missing observations,
rubric version and model identity are stored with Langfuse scores. Current
telemetry establishes incident facts; knowledge can explain mechanisms but
cannot prove they occurred. Empty queries and successful individual requests do
not establish absence of a fault. There is no mandatory tool order.

Model callbacks give Langfuse provider-reported usage and latency. Cost depends
on Langfuse model pricing and available usage attribution; configure/verify the
model catalog before treating displayed costs as authoritative. Unknown usage
or price is not a measured zero. No second local pricing/report engine remains.

## Online completion verification

Set `TRACEMIND_VERIFIER_ENABLED=true` in the API/worker/experiment environment.
Default is false for existing deployments. Configure the independent reviewer
with `TRACEMIND_VERIFIER_PROVIDER`, `TRACEMIND_VERIFIER_MODEL`,
`TRACEMIND_VERIFIER_TIMEOUT_SECONDS`, `TRACEMIND_VERIFIER_INPUT_BYTES` and
`TRACEMIND_VERIFIER_OUTPUT_TOKENS`. Defaults are DeepSeek Pro, 60 s, 65,536 bytes
and 4,096 output tokens. Model/provider credentials use the existing settings.
No extra dependency, tool permission or database migration is required.

After citation validation, a complete candidate and **all retained** telemetry,
knowledge excerpts and execution events are reviewed once without tools. Labels
are absent from runtime review. Knowledge cannot establish current incident
facts; empty queries and individual successful requests cannot establish fault
absence. A JSON/schema/reference/proof-obligation validator checks the response.
DeepSeek reviews use provider JSON Output (`response_format=json_object`), as
[documented by DeepSeek](https://api-docs.deepseek.com/guides/json_mode/).
Schema and semantic proof checks still run locally; JSON mode does not establish
claim correctness. `claim.text` must quote a contiguous original root-cause or
remediation excerpt (whitespace is normalized). Invented assertions fail the
review instead of being used to judge the answer. Proposals to inspect/query are
excluded from factual claims; explicit promised remediation effects still need
support. Two generic contrast examples illustrate that distinction without
using calibration item labels. Runtime and offline Judge share these rules.
Why: Review must assess assertions actually made, preserving qualifications and
excluding proposed checks.
Completed reviews from another rubric version cannot be reused on resume.
Why: A saved result from another rubric cannot establish compliance with the current
review contract.

| Review outcome | Runtime outcome |
| --- | --- |
| Supported claims and appropriate uncertainty | Accept the original candidate and emit `diagnosis.accepted` |
| Insufficient/contradicted claims or inappropriate uncertainty | `inconclusive`, reason `verification_rejected`; retain candidate and review |
| Reviewer timeout/provider error, invalid review, oversized input or experiment call-cap denial | Failed run, reason `verification_failed`; retain partial state |
| Shared run deadline or model-call limit exhausted | Existing harness deadline/budget stop applies; never accept complete |
| Candidate already inconclusive | No verifier call |

There is no automatic repair, model fallback or reviewer retry. Verification
uses the existing run deadline and model-call budget, with one model
slot reserved when verification is enabled. Reported tokens enter shared usage
accounting, and review output has its own token cap; there is no aggregate
monetary/token spend gate. Cancellation propagates and SSE
reports review progress. Policy is frozen on run creation and inherited by
follow-ups, so worker configuration changes do not silently bypass a saved gate.
A persisted completed review is reused after checking its subject fingerprint;
an interrupted/ambiguous attempt fails closed instead of issuing another paid
call. Full evidence is never truncated to fit the reviewer.
Why: Recovery must not duplicate an ambiguous paid attempt or silently weaken the
evidence reviewed.

The review record lives in RunState/checkpoints. Langfuse records the review
input/output and model usage when tracing is enabled; a Langfuse outage is not a
second runtime authority. The verifier evaluates retained evidence, not facts
that were never collected, and model agreement does not establish causal truth.

Run the same runtime verifier against eight manufactured cases with predefined
expected verdicts:

```bash
uv run python scripts/run_experiment.py --task verification \
  --name verifier-calibration --data evals/langfuse/verifier-calibration.json \
  --env-file .env --env-file deploy/.env.langfuse --max-model-calls 8
```

This task makes one reviewer attempt per item, with no investigator or extra
Judge call. `expected_output` stays outside the model input. Scores include
`review_completed`, `support_verdict`, `would_accept`, `expected_verdict`,
`expected_decision`, `false_acceptance` and `false_rejection`. A correct rejection
passes the classification experiment. This differs from judging a set of
answers, where an unsupported answer appropriately fails answer acceptance.
Eight cases are a calibration smoke, not a representative quality benchmark.

## Runtime and regression boundaries

Enable runtime tracing by loading the generated Langfuse settings into the
API/worker environment as well as the CLI. Copy the example model/Langfuse
application settings into the application environment. Regular diagnosis remains usable
with tracing disabled; evaluation explicitly requires a configured/authenticated
Langfuse instance. Harness root observations include run ID, status, usage and
events, with nested model/tool spans. Checkpoints, SQLite run state, SSE and JSONL
retain their existing execution responsibilities.

Frozen memory evaluation defaults to off and never reads the live memory DB.
`--memory-snapshot FILE --compare-memory` executes independent off/on variants
with a shared total call cap. Corpus hash and memory mode are retained in
experiment metadata/output. The corpus must predate held-out incident windows.
Why: Frozen pre-incident memory prevents future resolutions and live user data from
leaking into evaluation.

```bash
uv run pytest
```

Replay implementation and recorded telemetry schemas now live under `tests/`.
The six scripted cases verify real harness behavior, failure/cancellation,
isolation and trace accounting without model calls. They do not measure model
accuracy or memory improvement. SDK tests use an in-memory exporter and mocked
HTTP transport; paid integration smoke tests are explicit CLI invocations.

The former evaluation CLIs, substring scorer, custom report/summary engine and
runtime replay API are removed. Historical files remain readable as saved task
inputs; old scores are not reused. 

## OpenTelemetry Demo

Keep the upstream Demo in a sibling checkout, pinned to version 3.1.0:

```bash
git clone https://github.com/open-telemetry/opentelemetry-demo.git ../opentelemetry-demo
git -C ../opentelemetry-demo checkout dedc0178918e260823323b8d95005a8cb924b007
docker info
docker compose version
uv run python scripts/demo_scenario.py prepare
uv run python scripts/demo_scenario.py start
uv run python scripts/demo_scenario.py status
docker compose -f deploy/compose.telemetry-mcp.yaml up -d --build
uv run python scripts/check_mcp.py
```

If already cloned, verify the revision rather than cloning again. `--demo-path`
selects another checkout. Flags live in ignored `artifacts/otel-demo/`; upstream
files stay unchanged and the Demo's agentic mode is excluded. MCP joins the Demo
network; set `OTEL_DEMO_NETWORK` for a different name. Live evaluation calls shared
service assembly directly; only interactive investigations need API and worker.

The Compose override caps checkout and product catalog at 64 MiB each, with
`GOMEMLIMIT=48MiB`. These are ceilings, not preallocated RAM. Measure working
sets, restart counts and OOM events under the intended traffic before
increasing load.

The override gives astronomy-db a 256 MiB limit. Its readiness check uses
the configured Demo's `postgres` role/database rather than the container's
`root` user. Validate restart/OOM counts under the intended evaluation load.

## Live fault cases

| Case | Fault | Expected observations |
| --- | --- | --- |
| `payment_failure` | `paymentFailure=100%` | Checkout failure and Payment Charge error trace |
| `cart_failure` | `cartFailure=100%` | Cart EmptyCart error; items can remain after an order |
| `shipping_slowdown` | `intlShippingSlowdown=5sec` | Slow non-US shipping and shipping delay log |
| `product_catalog_lock_contention` | `productCatalogLockContention=on` | Slow catalog reads and a product-catalog table-lock log |
| `payment_unreachable` | `paymentUnreachable=on` | Checkout payment RPC fails resolving an invalid destination |

After traffic produces payment failures:

```bash
uv run python scripts/run_experiment.py --task agent --name payment-live \
  --case evals/cases/payment_failure.json --lookback-minutes 10 \
  --env-file .env --env-file deploy/.env.langfuse --judge --max-model-calls 20
```

Results and scores are stored in Langfuse. Switch faults with
`scripts/demo_scenario.py switch --scenario <case>`, then use the corresponding
`evals/cases/<case>.json` with the experiment command above. `--verbose` enables
sanitized operational event logging to stderr.

For cart, add an item at `http://localhost:8080`, then clear it. EmptyCart fails,
not every cart operation; checkout can succeed and leave items behind. For shipping,
compare Canadian and US orders: only non-US `/ship-order` is delayed. Explicit
requests help produce evidence. Trace detail includes error spans and twenty
longest successful spans, not a full span tree; corroborate timing with logs.

Each switch starts from pristine flags, saves one previous configuration and
recreates only flagd. Old telemetry remains: **wait at least two minutes for fresh
traffic/export before the one-minute evaluation window** to exclude the old fault.
Why: Old telemetry can otherwise make a new fault appear to have the previous cause.

For catalog lock contention, the Demo periodically holds an ACCESS EXCLUSIVE
lock on PostgreSQL's `catalog.products` for 30 seconds. Browse product pages to
produce requests and corroborate slow spans with the product-catalog lock log.
The case requires a cited product-catalog log. For payment-unreachable,
checkout deliberately uses `badAddress:50051`; inspect the destination resolution
error rather than assuming Payment's Invalid token failure. Provenance gates
check the required service/source; semantic cause claims need independent review.

## Interactive scenarios with SSE progress

API and worker must load the same `.env`. When online knowledge is enabled, set
the published snapshot and grant `search_knowledge` / `read_knowledge` as described
in [RAG setup](rag.md#query-and-enable-online-knowledge). Start these in separate
terminals, restarting both after configuration changes:

```bash
uv run --extra rag uvicorn tracemind.api.app:app --reload
```

```bash
uv run --extra rag python scripts/run_worker.py
```

Switch the running Demo to catalog lock contention:

```bash
uv run --extra rag python scripts/demo_scenario.py switch \
  --scenario product_catalog_lock_contention
```

Wait at least two minutes for fresh traffic/export, then submit the incident and
watch concise SSE progress in another terminal:

```bash
uv run --extra rag python scripts/demo_investigation.py \
  --case evals/cases/product_catalog_lock_contention.json --lookback-minutes 1
```

To try the connection failure, switch and wait again before submitting its case:

```bash
uv run --extra rag python scripts/demo_scenario.py switch \
  --scenario payment_unreachable
```

```bash
uv run --extra rag python scripts/demo_investigation.py \
  --case evals/cases/payment_unreachable.json --lookback-minutes 1
```

The interactive client prints progress, IDs, diagnosis and execution accounting.
Case evaluation is pending in Langfuse; the client does not run a substring scorer.
API limits remain configured independently of case evaluation tool targets.

## Restore and stop

```bash
uv run python scripts/demo_scenario.py restore
uv run python scripts/demo_scenario.py prepare --scenario shipping_slowdown
```

`restore` returns to one previous configuration; `prepare` only generates a
candidate. To remove Demo containers **and volumes** when finished:

```bash
uv run python scripts/demo_scenario.py stop
```

Use restore to undo a fault while preserving data. Status/stop do not rewrite flags.
Experiment names are explicit; old custom JSON report generation is removed.

### Trace-search baseline

`search_traces` accepts `min_duration_ms` (positive, optional) and
`errors_only` (default false). Jaeger applies these filters before its 20-trace
limit. Duration selects matching **spans**, not whole-request duration;
`errors_only` matches indexed `error=true` tags, so it can miss failures without
that tag. Combining the filters requires both. Trace summaries still describe
spans starting inside the investigation window; inspect details before assigning
latency or errors to a particular service/operation.

When no observations remain, the tool returns metadata containing the requested
service, filters, exact catalogue membership and up to 50 known service names.
Names longer than 240 characters are omitted and counted. The catalogue is global,
not restricted to the incident window, and supplies no citable evidence or progress.
No query is automatically broadened or redirected. Catalogue failures remain
visible telemetry failures. These semantics follow the
[Jaeger query parser](https://github.com/jaegertracing/jaeger/blob/v1.62.0/cmd/query/app/query_parser.go)
and [service endpoint](https://github.com/jaegertracing/jaeger/blob/v1.62.0/cmd/query/app/http_handler.go).

Freeze the tool contract, prompt and budgets before comparisons. Preserve
all primary attempts, including inconclusive and failed outcomes, and
report budget-matched comparisons separately. Small diagnostic samples
do not establish general incident accuracy.
