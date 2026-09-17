# Evaluation with local Langfuse

Langfuse owns datasets, experiment results, scores and comparisons. TraceMind
supplies task adapters and versioned evaluators through its Python SDK. The
agent harness still owns execution budgets, authorization, checkpoints and
recovery. The online verifier checks assertions before acceptance and is
enabled by default; single-agent deployments can explicitly disable it;
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

DeepSeek diagnosis thinking is configured by `TRACEMIND_DEEPSEEK_THINKING`
and `TRACEMIND_DEEPSEEK_REASONING_EFFORT`. Verifier and Judge have independent
`*_DEEPSEEK_THINKING` and `*_REASONING_EFFORT` settings, defaulting to thinking
with high effort. A run saves its verifier settings in its verification policy;
changing the main model's effort does not change review effort. Legacy policies
without saved effort use high; historical nondefault settings cannot be inferred.

Online retrieval assumes English input. Query decomposition and keyword
extraction use JSON output with `TRACEMIND_RAG_RETRIEVAL_THINKING=false` and
`TRACEMIND_RAG_RETRIEVAL_OUTPUT_TOKENS=512` by default. Ingestion retains the
existing model thinking and `TRACEMIND_RAG_LLM_OUTPUT_TOKENS` allowance.
Decomposition is optional and separates distinct needs; it does not translate
or produce synonym variants.

Judge has a separate timeout, input byte limit and output token limit, and
allows one bounded format repair. Oversized evidence fails evaluation;
observations and counterevidence are never silently dropped to fit the Judge.
Why: A review of selectively truncated observations could approve an unsupported answer.

For model comparisons, freeze evidence, prompt version and limits, then vary
one profile at a time. Check known-good and known-bad review subjects before
selecting lower effort. Output limits include reasoning tokens; raising them
alone may prolong a request that spends its allowance on reasoning. Record
provider-reported input, output, reasoning and cache tokens, and failed calls
with unknown usage. Reasoning is part of output, not an additional charge.
Saved verifier evaluations preserve their original query events for review;
their fresh execution ledger measures only the new review calls. Provider usage
and elapsed time are measurements; usage multiplied by a published rate is a
cost estimate, not a billing record.

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
The CLI includes `expected_status` only when the case JSON explicitly specifies
it. Omitting it allows evidence-backed localization to pass without a complete
root cause; explicit status requirements remain strict. Use an online verifier
or `--judge` to measure semantic support, rather than citation membership alone.

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
| `diagnosis_status` | Diagnostic certainty, separate from operational run status |
| `verification_completed`, `verification_attempts`, `verification_format_repairs`, `verification_error` | Runtime review completion, paid attempts, local contract repairs and terminal errors; distinct from semantic rejection |
| `grounded_findings`, `supported_finding_count` | Reviewed diagnostic excerpts actually retained with all their supporting telemetry citations; not independent correctness or measured information gain |
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
completion, full claim support and appropriate uncertainty. Explicit per-item
status requirements remain strict. Without an explicit status target, a reviewed
item requires at least one grounded diagnostic finding; a pure abstention or an
empty report cannot pass this usefulness gate. When the item labels a service
and evidence source, their citation-provenance gate also applies. Correctness
against a held-out reference is still reported, but is required by default only
for answers claiming completion. These rules apply per item in mixed datasets.
`--min-score NAME=MIN` configures/overrides thresholds.
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

### Context-cache measurements

`model.completed` records `model`, `purpose`, `call`,
`cache_usage_reported`, `cache_hit_tokens` and `cache_miss_tokens`.
Cache values are null when the provider did not report usable cache usage.
Normalized `input_token_details.cache_read` is preferred; DeepSeek's original
`response_metadata.token_usage.prompt_cache_hit_tokens` is also supported.
Miss tokens are the reported input total minus hit tokens. Cache hits are
already included in input tokens and must not be added to total usage again.

Run usage retains measured hit/miss totals and counts of attempts with and
without cache usage. Langfuse code scores expose `cache_usage_coverage`
(reported cache attempts / all admitted model attempts), measured
`cache_hit_tokens`, `cache_miss_tokens`, and `cache_hit_rate`. The rate is
omitted without measured input tokens. With partial coverage it applies only
to the reported subset, not the entire run. Historical saved runs without
cache fields cannot provide a measured hit rate.

Compare weighted rates: sum(hit) / sum(hit + miss), rather than averaging
per-request percentages. Use event model/purpose and order to distinguish
investigation, verification, knowledge calls, and the first investigation
request from later requests. Match model, tool schemas, workload and context
limits across variants; inspect miss tokens, latency and diagnostic quality
alongside the rate. Service-side cache persistence is best-effort, so a small
live comparison is a smoke check rather than a stable performance estimate.

Investigation requests keep system instructions fixed and append runtime
budget snapshots after complete tool exchanges. Snapshots are retained in
graph history, including retries and diagnosis repairs; the latest snapshot
is authoritative. Context compression still operates on a request-only view,
and finalization still removes investigation tools. These transitions can
interrupt prefix reuse and should be identified when comparing requests.

## Online completion verification

Runtime verification is enabled by default. Set
`TRACEMIND_VERIFIER_ENABLED=false` to explicitly disable it for new runs. The
independent reviewer defaults to DeepSeek Pro, a total 60-second review deadline,
131,072 input bytes and 4,096 output tokens. Existing explicit environment values
continue to apply. The tool allowlist and public `Diagnosis` fields are unchanged.

The investigator submits `DiagnosisProposal`: scoped findings, explicitly
unverified hypotheses, limitations and proposed checks. Each statement cites
run-local `E` telemetry or `K` knowledge handles. The application maintains their
stable mapping to original IDs, numbers statements, validates references and
renders the public diagnosis deterministically. There is no later free-text model
rewrite. Current run/checkpoint contract is `diagnosis-items-v1`; older records
remain readable, but cannot resume the new graph. Start a new run instead.

Rubric `diagnosis-review-v22-items` reviews every numbered item exactly once.
The model returns item IDs and assessments, never copied text. The application
binds each assessment to the complete original item and checks exact ID coverage,
unique valid references, time windows and proof obligations. Provider schemas
constrain item, evidence and query-event references to the actual packet values;
local validation is still authoritative. Model section labels do not exempt
factual premises, claimed completed actions or promised effects from review.

`ReviewPacket` includes every retained telemetry observation and its complete
attributes, including uncited counterevidence. Queries are interned losslessly;
duplicate candidate text, redundant trace indexes and unrelated model/checkpoint
events are omitted. Relevant query events retain their sequence, timestamp and
operational metadata. Original evidence and events remain in the run ledger.
The actual native output schema is included once in input-budget accounting.
Oversize packets are rejected before any paid call; evidence is not silently
truncated. Knowledge cannot establish current incident facts, and an empty query
cannot establish service health.

| Outcome | Runtime behavior |
| --- | --- |
| All items supported | Accept the rendered candidate; inconclusive remains inconclusive |
| Insufficient or contradicted item | Reject complete; preserve supported findings in an inconclusive result |
| Malformed output, missing/duplicate items or invalid references | One targeted format repair at most |
| Semantic rejection | No retry or repair |
| Timeout, provider/call-cap error, exhausted format repair or input limit | Explicit verification failure; never accept complete |

There is no separate model-generated uncertainty veto. Uncertainty is assessed
in the numbered items and its summary is derived from those verdicts. This
prevents contradictory outcomes such as all items supported but rejected merely
because no ritual unknown-cause sentence exists. It does not prove the model's
semantic assessment correct.

Two existing model slots are reserved for review and its optional format repair;
unused capacity is not a paid call. Both calls share the original run budget and
review deadline. Reservations and results are checkpointed before/after external
work. A completed matching review is reused; interrupted/ambiguous attempts fail
closed, including interrupted repairs. Subject fingerprints bind the complete
candidate, evidence packet and relevant query events. Cancellation propagates.

Online and offline review share packets, schemas and proof validation. Offline
Judge has no tool loop and uses one invocation per evaluation; online review may
use the bounded format repair. Runtime inputs never receive held-out labels.
Offline root-cause correctness is separate from evidence support and is null
without a supplied reference. Langfuse is an observability/evaluation sink, not
another runtime authority. Reports separate verification completion/error,
semantic verdict, retained grounded findings and independent correctness; a
correctly rejected wrong diagnosis is a successful verifier decision.

This is a bounded single-agent baseline component for reuse in multi-agent
experiments. Do not tune baseline diagnosis accuracy indefinitely or count an
empty inconclusive result as a useful finding. Compare both architectures with
the same verifier version, data, investigation budgets and separately reported
review overhead.

Run the same runtime verifier against eight manufactured cases with predefined
expected verdicts:

```bash
uv run python scripts/run_experiment.py --task verification \
  --name verifier-calibration --data evals/langfuse/verifier-calibration.json \
  --env-file .env --env-file deploy/.env.langfuse --max-model-calls 16
```

This task uses one review plus at most one format repair per item, without
an investigator or extra Judge call. Semantic rejection is not retried. `expected_output` stays outside the model input. Scores include
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

## Serial multi-agent evaluation

`--task agent` uses `TRACEMIND_AGENT_MODE`; no separate task runner is needed.
Experiment metadata retains agent mode, investigator/fallback/verifier models;
item output retains the frozen run mode, phase, handoffs and execution accounting.

The next real-model comparison should use these three variants:

| Variant | Agent mode | Verifier |
| --- | --- | --- |
| Single-agent baseline | `single` | Disabled |
| Reviewed baseline | `single` | Enabled |
| Serial delegation | `multi` | Always enabled |

Keep RAG/memory disabled and match investigator model, telemetry contracts,
incident windows, total model/tool limits and deadlines. Review consumes the
existing model cap in both reviewed variants. Keep failed and inconclusive runs;
report false completion, supported/correct diagnoses, appropriate abstention,
redundant queries, all admitted calls, reported/unknown usage and elapsed time.
Held-out labels remain outside every role and the runtime verifier.
Why: Larger aggregate budgets, changed models or removed failures would prevent
attributing differences to delegation.

Before any paid campaign, choose its explicit aggregate call cap, including
investigator, coordinator, verifier, retries, fallback and optional offline Judge.
The existing CLI cap covers all these provider calls within one invocation;
separate invocations still require explicit cross-run accounting. No real-model
multi-agent campaign or quality advantage is established by scripted regression.
