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

## Live fault cases

| Case | Fault | Expected observations |
| --- | --- | --- |
| `payment_failure` | `paymentFailure=100%` | Checkout failure and Payment Charge error trace |
| `cart_failure` | `cartFailure=100%` | Cart EmptyCart error; items can remain after an order |
| `shipping_slowdown` | `intlShippingSlowdown=5sec` | Slow non-US shipping and shipping delay log |

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

Live scoring checks service/term/source matches and citation provenance, not every
claim or causal reasoning. One passing scenario does not establish broad accuracy.
Replay validates control; real-model recovery, memory inference and RAG benefits
need separate measurements. Retrieval metrics are covered in
[RAG setup](rag.md#evaluate-retrieval). Feature changes are in
[CHANGELOG](../CHANGELOG.md).
