# Running TraceMind

Start with [README](../README.md) for installation and API/worker/Demo startup.
Commands run from the repository root. Full API schemas are at
`http://127.0.0.1:8000/docs`.

## Requests and lifecycle

Diagnosis, run and investigation creation share this request shape:

```json
{
  "incident": {
    "title": "Checkout HTTP 500 spike",
    "description": "Checkout failures began after a synthetic fault",
    "severity": "high",
    "affected_services": ["checkout"]
  },
  "investigation_start": "2024-01-01T00:00:00Z",
  "investigation_end": "2024-01-01T00:10:00Z"
}
```

The timestamps above are illustrative. Replace them with your actual fault
window. Both require timezones and end must follow start.
Why: Explicit ordered times keep admission and evidence filtering on the same
investigation window.

| Operation | Route and behavior |
| --- | --- |
| Synchronous diagnosis | `POST /api/v1/diagnoses`; waits without persistent recovery |
| Queue a run | `POST /api/v1/runs`; returns 202 with run ID |
| Status/result | `GET /api/v1/runs/{id}` and `/api/v1/runs/{id}/result` |
| Initial triage | `GET /api/v1/runs/{id}/triage` (available before review completes) |
| Cancel | `POST /api/v1/runs/{id}/cancel`; interrupts work and stops new admission |
| Resume | `POST /api/v1/runs/{id}/resume`; requeues after the old worker lease expires |

Run creation accepts `Idempotency-Key`: the same key/body returns the existing
run; a different body returns 409. Workers automatically reclaim expired leases.
`scripts/run_worker.py --once` processes at most one queued run.

Synchronous operational timeouts return 504; tool/model failures and invalid
diagnoses return 502. Error `detail` is a string. Responses with run metadata
include `X-TraceMind-Run-ID` and `X-TraceMind-Stop-Reason` headers.

## Initial evidence ingress

Diagnosis, run and investigation creation accept optional `initial_context`.
An investigating follow-up may supply a new context. The following is an
illustrative addition to the request above, not a live trace:

```json
{
  "initial_context": {
    "prefetch_traces": true,
    "origin": {
      "source": "on-call-engineer",
      "external_id": "incident-42",
      "reported_at": "2024-01-01T00:05:00Z"
    },
    "traces": [{
      "trace_id": "0123456789abcdef0123456789abcdef",
      "span_id": "0123456789abcdef",
      "service": "checkout",
      "operation": "POST /checkout"
    }],
    "error": {
      "type": "TimeoutError",
      "message": "Payment call timed out",
      "file": "checkout.py",
      "line": 42
    },
    "scope": {"environment": "production"},
    "deployments": [{"service": "checkout", "version": "v2"}]
  }
}
```

These are caller-reported hints, stored on each run, not on the incident. They
never receive an evidence ID. `origin.source` does not authenticate the sender
or install an alert connector. A supplied code path/line is not source access,
and a version label is not a verified commit or causal proof of a bad release.

Before the first model call, TraceMind reads supplied trace locators through the
same authorized `get_trace_detail` binding, harness and evidence collector used
by the agent. Exact `span_id` filtering occurs before backend pagination. Service
and operation hints do not restrict this lookup. Deployment hints are scoped to
their named service and do not filter out differently versioned downstream spans.
The first model receives separate reported hints, retained E observations and
lookup results. `backend_coverage` summarizes backend paging before run time/scope
filters; the lookup's E IDs identify only retained matches. An empty coverage
object means coverage is unknown. Existing `read_evidence` retrieves
retained details by E handle.

| Boundary | Behavior |
| --- | --- |
| Input | At most 3 trace/span locators, 10 deployment hints, 32 KiB serialized context; unknown fields rejected |
| Identifiers | Nonzero 16/32 hex trace ID, optional nonzero 16 hex span ID; normalized to lowercase |
| Initial reads | At most 3 actual tool attempts including retries, also charged to the run's existing budget/deadline |
| Scope | Explicit environment/cluster/namespace must match observed resource identity; missing, conflicting, redacted or truncated identity is excluded as unknown |
| Lookup outcomes | `found`, `not_found`, `unavailable`, `outside_window`, `scope_mismatch`, `scope_unknown`, `coverage_incomplete` or `skipped`, with reason when applicable |
| Disable | Omit context for old behavior, or set `prefetch_traces: false` to retain the same hints without initial reads |
| Resume | Attempts are checkpointed before I/O; completed locators are not repeated; an interrupted request can be retried within the remaining cap |
| Follow-up | Parent observations may be retained; new initial hints are supplied explicitly and parent locators are not automatically rerun |
| Explain | New context is rejected; explanation cannot perform initial remote reads |

An explicit scope is a filtering constraint, not an authorization boundary or
backend routing instruction. It applies to all newly retained observations;
metrics without matching resource identity are therefore excluded too. The
current tools do not provide environment/version search selectors. Filtered
results cannot establish the health of the scoped population.

Jaeger process/span tags and exception events, and OpenSearch resource/attribute
fields, retain allowlisted resource identity, version, code location and exception
information when present. Sources, conflicting aliases, truncation and redaction
are recorded. Large model previews omit conflicted selected values and show
`conflicting_fields`; full retained records keep the competing source values.
Exception-only code locations remain available in previews when space permits.

`observed_at` remains the span start time. Span evidence is retained only when
that start lies within the investigation window; an exception inside the window
on an earlier-started span is not currently retained. Retained exception events
keep their own timestamp and get a separate `in_investigation_window` marker,
recomputed when a follow-up changes the window. Missing/invalid event timestamps
remain unknown. An exception event alone does not make a span erroneous.

Each observation retains at most 4 exception source records (not necessarily
distinct failures), with a combined 32 KiB exception allowance; each stack is
bounded to 4 KiB, and log bodies to 16 KiB. Common credential forms are redacted
before storage, but this is not comprehensive secret or PII detection. Do not
submit credentials. Model previews are further bounded in UTF-8 bytes, with
reported-field reduction and omitted-evidence counts. Full sanitized hints stay
in the run snapshot, while `read_evidence` exposes only retained observations.
Extremely small context budgets still fail through the existing context gate.

A found locator means at least one matching retained span, not a complete trace
or confirmed root cause. Initial lookup reads one page; existing coverage and
pagination remain necessary for larger traces. 404 means unavailable telemetry,
not service health. Permissions, cancellation and overall run budgets still
apply; disabling prefetch does not disable tool use later in the investigation.

Offline acceptance is covered by `test_intake.py`, `test_intake_api.py`,
`test_initial_evidence.py`, `test_mcp_evidence_ingress.py` and the existing MCP,
worker, context and persistence tests. These verify contracts and execution,
not diagnosis accuracy or speed. A later paired evaluation can keep incident,
window, hints, model and total budgets fixed, toggle `prefetch_traces`, and compare
total time/tool attempts, useful first evidence and final correctness. Keep
primary failures and timeouts in the denominator. No such production benefit is
claimed by this phase.

## Configuration and budgets

[.env.example](../.env.example) is the settings reference. Its example diagnostic
profile overrides smaller code defaults when copied into `.env`:

| Setting suffix (`TRACEMIND_AGENT_`) | Example profile | Code default |
| --- | --- | --- |
| `CONTEXT_TOKENS` | 65536 | 65536 |
| `OUTPUT_TOKENS` | 16384 | 16384 |
| `OBSERVATION_BYTES` | 8388608 | 8388608 |
| `RECURSION_LIMIT` | 50 | 12 |
| `MAX_MODEL_CALLS` | 15 | 8 |
| `MAX_TOOL_CALLS` | 25 | 12 |
| `TIMEOUT_SECONDS` | none | none |

First-output/idle timeouts default to 180 seconds for models; tool requests have
a 20-second total timeout. These limits
are not model capacity declarations. Retries/repairs consume attempts; evidence
reads consume tool attempts. Evaluation also applies the case's tool-call cap.
Context admission uses the locally bundled `cl100k_base` tokenizer, with
20% headroom plus 1024 framing tokens. This is a proxy for DeepSeek and
other provider tokenizers, not their exact count. Serialized messages,
tools and output schemas are counted; output is reserved within the shared
65536-token envelope for investigation, auxiliary retrieval and review.
The offline judge uses the same configured envelope. Vocabulary loading is
offline and checksum-verified. Actual provider usage remains authoritative
for the ledger; events retain the estimator version. UTF-8 byte limits
remain independent storage and payload guards, never token quantities.

Investigation and knowledge-helper output limits are 16384 tokens. Online
verification allows 32768 output tokens; the offline judge allows 12288.
Both retain the 524288-byte input guard. Thinking tokens consume the output allowance too. These are maxima,
not targets for response length, and they do not extend any deadline.
Existing persisted runs keep their original limits; explicit environment
values override defaults. Restart API/worker processes after configuration
changes.

The cumulative token cap remains unset by default. If enabled, it must also
cover admission reservations: two reviews reserve their maximum input plus
output, and synthesis/repair reserve context allowances. The example optional
cap is 4000000; retaining an old 1000000 cap with the larger defaults can close
investigation before useful work. Unknown usage remains charged; reported
usage is kept separately from these conservative reservations.

`TRACEMIND_AGENT_ALLOWED_TOOLS` is an explicit JSON grant list. Defaults are
`query_metrics`, `discover_metrics`, `search_traces`, `get_trace_detail`,
`search_logs` and `read_evidence`. Extra remote tools need a discovered server
definition and a grant; knowledge tools also need grants. A missing granted remote
tool or local-name collision fails setup. Discovery never authorizes access.

SDK retries are disabled; configure bounded retries with `TRACEMIND_AGENT_RETRY_*`.
Fallback requires both `TRACEMIND_AGENT_FALLBACK_PROVIDER` and
`TRACEMIND_AGENT_FALLBACK_MODEL` plus that provider's key. Invalid output has a
bounded repair allowance. Tool budget, repetition and no progress close tools,
allowing final diagnosis within remaining hard limits.

Restart API/worker after changes and create a new run; saved policies and limits
are retained by existing runs.
Why: Saved policy is an execution contract, so settings changes apply to newly
admitted work.

DeepSeek uses its OpenAI-compatible Chat Completions
endpoint with configurable thinking and tool-based structured diagnoses.
The example configuration enables thinking with high reasoning effort.

## Collaborative multi-agent mode

Enable `TRACEMIND_AGENT_MODE=multi`; single remains the default. All modes use
shared run limits. Multi mode always requires the verifier. Enable knowledge
with `TRACEMIND_AGENT_KNOWLEDGE_ENABLED=true`, a published snapshot ID and explicit
`search_knowledge` / `read_knowledge` grants. Memory writes remain disabled.

Use one local worker. LangGraph runs at most two investigation tasks concurrently
and allows up to two targeted feedback rounds. The coordinator cannot expand
tool permissions. Small model budgets may leave no room for parallel discovery
or feedback; actual model attempts, repairs, auxiliary retrieval and verification
all count against the same cap. Budget enough output tokens/time for DeepSeek
thinking, enabled by default with effort `high`.

Back up both ledger and graph-checkpoint databases. Completed task reports and
query results are reused after recovery; ambiguous external calls can repeat and
remain charged. Recovery requires a compatible saved graph contract.
Follow-up investigations inherit mode; explanations use retained-data single mode.
See [multi-agent architecture and contracts](multi-agent.md) for details.

## Persistence and progress

Queued runs use `TRACEMIND_AGENT_RUN_DATABASE` for ledger/evidence and
`TRACEMIND_AGENT_CHECKPOINT_DATABASE` for checkpoints, defaulting to separate
SQLite files in `artifacts/`. Back up both together before upgrades.
Why: The ledger and graph checkpoints hold different parts of recoverable state. Recovery
restores evidence, attempts, deadline and graph state. Ambiguous external calls
may repeat and still consume budget. Deployment supports one local worker.

Subscribe to `GET /api/v1/runs/{id}/events`. Reconnect using `Last-Event-ID:
<sequence>` or `?after=<sequence>`; the header wins. Deduplicate by run ID/sequence.
Idle heartbeats keep the stream alive. Close on `complete`, emitted after terminal
commit with status and an optional result URL. Disconnecting does not cancel.
Synchronous diagnoses/evaluations are not live SSE runs.
Why: Cursor deduplication and terminal commit boundaries prevent observers from treating
reconnects as new execution.

### Read initial triage while review is running

For queued runs, listen for `triage.available`, then GET the supplied
`triage_url` (`/api/v1/runs/{id}/triage`). The candidate, evidence and event are
checkpointed before invoking the reviewer. Status and investigation-history
responses also provide this URL; its presence alone does not mean a candidate
is ready. Before a current provenance-valid candidate exists, GET returns 409;
an unknown run returns 404. The synchronous `/diagnoses` route still waits.

A 200 response contains `run_id`, `candidate_id`, run `status`,
`cancel_requested`, `diagnosis` and all retained `evidence`. `candidate_id` is
an opaque fingerprint of the review input, not a confidence score. The initial
`diagnosis.review_status` is `pending`, its result status is `inconclusive`,
and every item is `unreviewed` / `not_assessed`. This means citations and the
item contract passed basic checks, not that the claims are semantically true.

Refresh on `verification.completed`, `verification.failed` or terminal events.
The same candidate ID then carries individual review verdicts, or an explicit
review failure with the unreviewed statements preserved. A cancel request or
an interrupted review does not remain pending forever; it is marked
`unavailable` with a reason. A review already completed before cancellation
keeps its assessment. Run lifecycle and review state are independent.

Only the current candidate is projected. Reinvestigation removes the current
candidate until a new one is ready. Changed evidence, candidate, query history
or rubric cannot reuse old support; this endpoint returns 409 for a stale
fingerprint/rubric. Older-version candidates therefore may not be available
through this new endpoint after an upgrade. Persisted historical records and
existing terminal `/result` remain available under their previous semantics.
The endpoint does not create work or resume model calls. SSE carries only URL,
identity and status, not candidate text or raw logs. Clients compare candidate
IDs on refresh, deduplicate events by run/sequence, and use the returned run
status for cancellation/completion rather than treating `pending` as final.

Detailed JSONL defaults to `artifacts/runs/<run_id>.jsonl`: durations, call/span
relationships, evidence IDs and reported usage, without raw prompts, query/result
payloads or exception messages. File errors warn without replacing run outcomes.
JSONL is not a checkpoint. Reports and SQLite can contain full evidence/feedback.

## Follow-up questions

`POST /api/v1/investigations` creates an incident and first queued run. Once it
finishes, `POST /api/v1/investigations/{id}/messages` accepts:

```json
{"message":"Why is payment the root cause?","purpose":"explain"}
```

`explain` reads retained data only; `investigate` can query telemetry. Each round
gets a fresh run/budget and preserves earlier results. An active round rejects a
new question with 409. Creation/follow-up support `Idempotency-Key`. List older
rounds at `GET /api/v1/investigations/{id}?offset=0&limit=20`.

Omitted windows inherit previous bounds. For recurrence, supply both timestamps;
new bounds filter inherited evidence. Explanation keeps the prior window. Prior
conclusions and user hypotheses remain historical claims, not observations.

```bash
uv run python scripts/demo_investigation.py --investigation-id <id> \
  --purpose investigate --message "Check for a network fault" --lookback-minutes 10
uv run python scripts/demo_investigation.py --run-id <id> --after <sequence>
```

The second command only subscribes. Interrupting the demo leaves worker execution
active; it can reconnect after transport failures.

## Optional feedback memory

Enable with a scope matching the telemetry deployment, then restart API/worker:

```dotenv
TRACEMIND_AGENT_MEMORY_ENABLED=true
TRACEMIND_AGENT_MEMORY_ENVIRONMENT=otel-demo-local
TRACEMIND_AGENT_MEMORY_DEPLOYMENT_VERSION=demo-3.1.0-local-v1
```

Ordinary follow-up feedback can propose enduring preferences/skills and
human-confirmed/corrected diagnoses. Learning shares the diagnosis response,
without an extraction call. Temporary requests are not promoted; model inference
can be mistaken and is reviewable. Current instructions override saved habits.
This is a trusted local-user profile, not multi-user authentication.

Historical diagnosis retrieval requires exact environment/version, service overlap
and unexpired confirmation; same-incident/revoked records are excluded. Runs freeze
contents before model execution. Explanation loads guidance but retrieves no new
historical diagnoses. Memory IDs cannot replace current telemetry citations.
Why: Historical resolutions can guide investigation but cannot establish facts about the
present incident.
Inferred confirmations expire after `TRACEMIND_AGENT_MEMORY_AUTO_EXPIRY_DAYS`
(default 30); preferences/skills have no TTL.

| Management route | Action |
| --- | --- |
| `GET /api/v1/guidance?offset=0&limit=20` | Review inferred preferences/skills and provenance |
| `PUT /api/v1/guidance/{id}` | Update `instruction` with `expected_revision` |
| `DELETE /api/v1/guidance/{id}?expected_revision=1` | Remove guidance from future retrieval |
| `POST /api/v1/memories` | Manually confirm a source run |
| `GET /api/v1/memories?offset=0&limit=20` or `/api/v1/memories/{id}` | Review records |
| `PUT /api/v1/memories/{id}` | Replace/reconfirm all fields with `expected_revision` |
| `POST /api/v1/memories/{id}/revoke?expected_revision=1` | Exclude disputed content |
| `DELETE /api/v1/memories/{id}?expected_revision=2` | Remove a library record |

Manual confirmation uses this shape. Replace placeholders and independently
verify the source run/resolution before choosing a future expiry:
Why: Confirmation creates reusable incident memory, so the attestation must come from a
verified resolution.

```json
{
  "source_run_id": "<completed-run-UUID>",
  "environment": "otel-demo-local",
  "deployment_version": "demo-3.1.0-local-v1",
  "services": ["checkout", "payment"],
  "symptom": "Checkout requests returned HTTP 500",
  "resolution": "<human-verified cause and resolution>",
  "diagnostic_queries": ["<useful check>"],
  "remediation": ["<verified recovery action>"],
  "confirmed_by": "<reviewer>",
  "confirmation_note": "<verification method>",
  "expires_at": "<future timezone-aware timestamp>"
}
```

Workflow validates source observations/provenance. Stale revisions return 409.
Correction/revocation/deletion affects future retrieval but keeps copies in prior
snapshots, receipts, checkpoints and reports. Evaluation uses separate
[frozen corpora](evaluation.md#runtime-and-regression-boundaries), never the live user library.

## Online verifier and optional Langfuse

Online verification is enabled by default for newly created runs. Set
`TRACEMIND_VERIFIER_ENABLED=false` consistently in API and worker settings to
explicitly disable it. Restart existing API and worker processes after changes.
One tool-free completion review consumes the existing model-call/time budgets
and shared token accounting; it never automatically repairs a diagnosis.
The product delivers initial triage for an on-call engineer. Completion means
an evidence-supported failure location or propagation finding and a concrete
next inspection. A unique implementation root cause or release attribution is
not required; unsupported factual claims are still rejected individually.

The response retains all numbered `items`, including proposed checks, evidence
gaps and rejected claims. Each item has `review_status`, `evidence_support`,
original citations, separate supporting/conflicting citations and review
rationale. Supported proposals are marked `not_applicable` for factual evidence
support: a sensible inspection is not proof of an incident cause. Supported
uncertainty remains uncertainty. Legacy `root_cause` and `remediation` text also
carry review labels. `confidence` remains an uncalibrated legacy model estimate;
no per-item probability is inferred from model agreement.

Top-level `triage_status` is `ready`, `partial` or `no_findings` independently
of whether every claim passed review. `review_status` distinguishes `completed`,
`pending`, `timed_out`, `unavailable` and `not_requested`. Recoverable review failure returns
an inconclusive initial result with every item explicitly unreviewed, after
citation/window validation; it never counts as passed verification. A current
prior reviewed result may be retained if a later attempt fails. Explicit user
cancellation, invalid candidate citations and stale review contracts retain
their existing failure semantics. Unreviewed/rejected output is not saved as
learned feedback. With verification explicitly disabled, legacy delivery is
unchanged and metadata defaults to `not_requested`.

Persisted policy and review state govern resumed runs. Existing reviews from
older rubric versions are not silently reused. The default overall deadline is
disabled (`TRACEMIND_AGENT_TIMEOUT_SECONDS=none`); a positive value enables it.
Direct and worker runs retain the frozen policy on resume, including any
existing absolute deadline. Model timeouts bound the wait for first meaningful
output and each subsequent idle gap, not the total request duration. Reasoning,
answer text and tool-argument deltas refresh the deadline; heartbeats, empty
chunks and transport headers do not. Continuous output may exceed the configured
seconds. Output-token and model-call budgets still apply, as do cancellation and
an explicitly configured overall deadline. Tool timeouts remain total request
limits.
If a provider's Retry-After exceeds the configured maximum backoff, automatic
retry is declined rather than waiting indefinitely or retrying too early.

Model/tool calls, repeated queries, no-progress rounds and graph steps bound
investigation. Failed attempts also consume the shared budget. Before each
investigation attempt the harness reserves calls for candidate synthesis,
remaining diagnosis repair, and two review attempts (initial plus at most one
output-contract repair). Runtime snapshots show the discovery allocation and
reserved calls. Graph recursion limits count graph steps, not model turns.

The verifier uses `TRACEMIND_VERIFIER_TIMEOUT_SECONDS` as the **first-output and
idle timeout for each request**, independently of `TRACEMIND_AGENT_MODEL_TIMEOUT_SECONDS`.
A format repair receives a new progress monitor but still consumes the shared
model-call budget. The offline Judge likewise uses its timeout per request.
If an overall deadline is explicitly enabled, progress cannot extend it.
`model.started` records `timeout_mode=progress_idle` and the configured interval;
`model.progress` records bounded numeric summaries (first phase output and at
most once per ten seconds), never provider text. Completion events retain first
output times and maximum progress gap; timeout events distinguish `first_output`,
`idle` and `overall_deadline`. The DeepSeek adapter consumes Chat/Responses
streams internally, preserving reasoning for tool turns and terminal usage.
OpenAI/Google use native streaming callbacks; unexposed internal reasoning cannot
refresh a deadline. A stream ending without a terminal response is a transport
failure, not a completed answer;
`verification.started` records input size and remaining run time (null when
disabled). Timed-out review returns an explicitly unreviewed candidate.
The online verifier defaults to `low` reasoning; the investigator and offline
Judge keep their separately configured reasoning levels. The default review
output budget is 32768 tokens, including provider reasoning.
Provider-truncated/incomplete responses return `incomplete_output` without
repeating the same request as a format repair; all candidate items remain
explicitly unreviewed. Complete responses with invalid JSON/contracts can still
use one bounded repair. A larger allowance enables output, not semantic trust.

Initial reports normally contain 2-4 key observations, at most one useful
hypothesis and one essential gap, and 1-2 next checks. This is a prioritization
instruction, not silent truncation: material transaction-state gaps remain.
Every emitted item is still reviewed, with all retained evidence, including
uncited counterevidence. Reported labels and telemetry facts are kept separate.
Repeated evidence attributes remain losslessly shared in the review packet.

`TRACEMIND_AGENT_VERIFICATION_RESERVE_SECONDS` remains an optional single-agent
admission rule for explicitly time-bounded trials. It has no effect without an
overall deadline and cannot guarantee completion of in-flight calls. Admission
freezes policy so later worker settings cannot silently change the run.

Runtime Langfuse tracing is separately opt-in through `TRACEMIND_LANGFUSE_*`;
normal operation does not require the evaluation server. For model limits,
local deployment and scored experiments, see
[Evaluation](evaluation.md#online-completion-verification).


## Demo trace retention

The Demo's upstream Jaeger stores at most 25,000 traces in memory. Continuous
traffic can evict an incident window within hours even while Jaeger remains
healthy. A trace lookup then returns HTTP 404; this is missing telemetry, not
proof that the service is healthy. MCP reports `not_found`, without transport
retries, and the agent can preserve conclusions supported by other evidence.

The TraceMind Demo override uses the existing Jaeger 2.19.0 binary's embedded
Badger backend, with seven-day span retention and the external Docker volume
`tracemind-jaeger-data`. No image or package upgrade is required. The external
volume survives Demo `stop` (which otherwise removes Compose-owned volumes).
`scripts/demo_scenario.py start` creates the volume if needed; direct Compose
users can create it with `docker volume create tracemind-jaeger-data`.
Do not delete it during ordinary teardown. See the
[version-pinned Badger sample](https://github.com/jaegertracing/jaeger/blob/v2.19.0/cmd/jaeger/config-badger.yaml).

Switching an existing memory-only Jaeger requires recreating that container;
its remaining in-memory traces will be lost. Export any needed traces first.
Already-evicted traces cannot be recovered from Jaeger by changing storage.
Capture fresh fault windows after the switch; do not substitute fresh data
into old trial records. Disk usage must be monitored during sustained traffic.

`evaluate_diagnosis.py` now checks every captured trace through the diagnosis
MCP endpoint before creating the model-call budget or Langfuse experiment.
It requires real spans and matching whole-trace fingerprints/counts. A missing
trace, incomplete reply or changed trace blocks the trial before model calls.
A fingerprint can also change after MCP normalization changes; recapture the
window with the current tool version in that case. This check establishes
availability at admission, not a guarantee against later backend outages.
