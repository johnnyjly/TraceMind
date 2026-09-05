# Demo and evaluation

Commands run from the repository root. Live evaluation needs a model key and
telemetry MCP; offline replay needs neither. See [README](../README.md) for setup.

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
uv run python scripts/run_evaluation.py --lookback-minutes 10
```

The report defaults to `artifacts/evaluations/payment-failure.json`. Use `--output`
to preserve multiple runs and `--verbose` for sanitized operational event logging
to stderr. Reports contain full evidence; JSONL excludes raw prompts/payloads.

Switch faults on the running Demo:

```bash
uv run python scripts/demo_scenario.py switch --scenario cart_failure
uv run python scripts/run_evaluation.py \
  --case evals/cases/cart_failure.json --lookback-minutes 1

uv run python scripts/demo_scenario.py switch --scenario shipping_slowdown
uv run python scripts/run_evaluation.py \
  --case evals/cases/shipping_slowdown.json --lookback-minutes 1
```

For cart, add an item at `http://localhost:8080`, then clear it. EmptyCart fails,
not every cart operation; checkout can succeed and leave items behind. For shipping,
compare Canadian and US orders: only non-US `/ship-order` is delayed. Explicit
requests help produce evidence. Trace detail includes error spans and twenty
longest successful spans, not a full span tree; corroborate timing with logs.

Each switch starts from pristine flags, saves one previous configuration and
recreates only flagd. Old telemetry remains: **wait at least two minutes for fresh
traffic/export before the one-minute evaluation window** to exclude the old fault.

For catalog lock contention, the Demo periodically holds an ACCESS EXCLUSIVE
lock on PostgreSQL's `catalog.products` for 30 seconds. Browse product pages to
produce requests and corroborate slow spans with the product-catalog lock log.
The case requires a cited product-catalog log; a generic slow-checkout answer
does not satisfy the scorer. For payment-unreachable, checkout deliberately uses
`badAddress:50051`. The case expects checkout destination/address resolution as
the cause and a cited checkout trace, rather than payment's Invalid token error.
The existing scorer uses service/term/source checks, not semantic proof of cause.

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

The interactive client prints progress, IDs, the diagnosis and the case score;
it does not write an evaluation report. For a saved evaluation report, use
`scripts/run_evaluation.py --case <case-path> --lookback-minutes 1 --verbose`
with the same RAG extra; it performs a separate run with additional model calls.

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
Default report names follow the case, e.g. `cart-failure.json`.

## Offline replay and aggregation

```bash
uv run python scripts/run_evaluation_suite.py
```

Six synthetic cases cover payment failure, misleading checkout symptoms, missing
telemetry, conflicting evidence, transient recovery and backend failure. Scripted
models execute the real graph/harness with fixed telemetry and recorded schemas;
unrecorded queries are rejected. Ground truth/scripted answers stay outside runtime
prompts. Replay does not contact MCP or measure live-model/network latency.

The batch defaults to `artifacts/evaluations/replay-suite.json`. Six gates should
pass; one case intentionally ends with `tool_error`. Expected inconclusive answers
pass only cases explicitly expecting them. A real model can be injected through
`run_replay_evaluation(..., model=...)` for a separate `model_replay` measurement.

Aggregate saved live reports without another model call:

```bash
uv run python scripts/run_evaluation_suite.py \
  --reports artifacts/evaluations/payment-failure.json \
  --output artifacts/evaluations/live-summary.json
```

Reports separate modes/providers/models and include pass rate, false completion,
failures, attempts/retries, recovery, mean/p95 latency, reported tokens, unknown
usage and trace coverage. Optional `--pricing` accepts a JSON file:

```json
{"provider:model":{"input_per_million":1,"output_per_million":2}}
```

Rates are USD per million tokens. Missing pricing/usage/model attribution keeps
cost unknown; cached-token discounts are not modeled. Live failure and failed
selected suite gates exit 1. Partial state survives diagnosed operational failures;
setup errors outside that lifecycle still propagate.

## Memory evaluation

Evaluation defaults to memory-off even if application memory is on. It never loads
live user guidance or learns from feedback. Export a corpus **before** held-out
incidents, with the API running and output directory created:

```bash
curl -sS --get http://127.0.0.1:8000/api/v1/memories/snapshot \
  --data-urlencode environment=otel-demo-local \
  --data-urlencode deployment_version=demo-3.1.0-local-v1 \
  --data-urlencode service=checkout --data-urlencode limit=10 \
  -o artifacts/evaluations/memory-corpus.json
```

Snapshot time must precede the evaluation window. Same-incident, post-freeze,
scope-mismatched and expired records are excluded. Authors must separately prevent
copied answer text and training/test overlap; timestamps/IDs cannot detect leakage.

```bash
uv run python scripts/run_evaluation_suite.py \
  --memory-snapshot evals/memory/synthetic_training.json --compare-memory \
  --output artifacts/evaluations/memory-comparison.json
```

Paired replays have independent state/identical budgets and report memory mode and
corpus hash. Scripted comparison validates isolation, not accuracy gain. Compare
real models with `scripts/run_evaluation.py` with/without `--memory-snapshot
<frozen-file>` on controlled equivalent faults, keeping model/deployment/limits
fixed. Human-assisted evaluation needs fixed feedback and cumulative cost metrics.

## Interpretation

`demo_investigation.py --case ...` now prints execution status, stop reason,
elapsed time, actual model/tool usage and the case score after the SSE stream.
The API still uses its configured limits; the case's `max_tool_calls` is an
evaluation target, not an API budget override. A completed diagnosis can fail
case scoring, including when the API allows 25 tools and the case allows 12.
The script reports this distinction without changing either limit. Reattaching
by run ID or submitting a follow-up has no case score because the original case
is not inferred from the run.

Reduced audit regressions are separate from the six original synthetic fixtures:

```bash
uv run python scripts/run_evaluation_suite.py \
  --fixtures evals/regressions/live-diagnostic-boundaries \
  --output artifacts/evaluations/diagnostic-replays.json
```

These scripted regressions cover payment name resolution, cart storage access
and catalog lock evidence using sanitized fixtures. They test harness and
scoring contracts, not whether a live model will choose the scripted queries.

For operation scope and cross-trace attribution, the
[fixed-evidence claim checks](../evals/claim-boundaries/README.md) call the real
model with sanitized fixture observations and keep the assessment rubric out of its
input. Review the output manually; this does not replace an autonomous live
investigation. DiagnosisDraft now requests observed failure, counterevidence
and scope, and unknowns within the existing root_cause string. The public result
fields and the case scoring rules are unchanged.

Live scoring checks service/term/source matches and citation provenance, not every
claim or causal reasoning. One passing scenario does not establish broad accuracy.
Replay validates control; real-model recovery, memory inference and RAG benefits
need separate measurements. Retrieval metrics are covered in
[RAG setup](rag.md#evaluate-retrieval). Feature changes are in
[CHANGELOG](../CHANGELOG.md).

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
