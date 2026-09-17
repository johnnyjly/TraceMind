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

The timestamps above are illustrative. Replace them with your actual fault window. Both require timezones and
end must follow start.
Why: Explicit ordered times keep admission and evidence filtering on the same
investigation window.

| Operation | Route and behavior |
| --- | --- |
| Synchronous diagnosis | `POST /api/v1/diagnoses`; waits without persistent recovery |
| Queue a run | `POST /api/v1/runs`; returns 202 with run ID |
| Status/result | `GET /api/v1/runs/{id}` and `/api/v1/runs/{id}/result` |
| Cancel | `POST /api/v1/runs/{id}/cancel`; interrupts work and stops new admission |
| Resume | `POST /api/v1/runs/{id}/resume`; requeues after the old worker lease expires |

Run creation accepts `Idempotency-Key`: the same key/body returns the existing
run; a different body returns 409. Workers automatically reclaim expired leases.
`scripts/run_worker.py --once` processes at most one queued run.

Synchronous operational timeouts return 504; tool/model failures and invalid
diagnoses return 502. Error `detail` is a string. Responses with run metadata
include `X-TraceMind-Run-ID` and `X-TraceMind-Stop-Reason` headers.

## Configuration and budgets

[.env.example](../.env.example) is the settings reference. Its example diagnostic
profile overrides smaller code defaults when copied into `.env`:

| Setting suffix (`TRACEMIND_AGENT_`) | Example profile | Code default |
| --- | --- | --- |
| `CONTEXT_TOKENS` | 131072 | 32768 |
| `OUTPUT_TOKENS` | 8192 | 8192 |
| `OBSERVATION_BYTES` | 8388608 | 8388608 |
| `RECURSION_LIMIT` | 50 | 12 |
| `MAX_MODEL_CALLS` | 15 | 8 |
| `MAX_TOOL_CALLS` | 25 | 12 |
| `TIMEOUT_SECONDS` | 600 | 600 |

Attempt timeouts default to 180 seconds for models and 20 for tools. These limits
are not model capacity declarations. Retries/repairs consume attempts; evidence
reads consume tool attempts. Evaluation also applies the case's tool-call cap.
Context estimates are byte-based, distinct from provider-reported token usage.

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
endpoint with thinking disabled and tool-based structured diagnoses.

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
Semantic rejection returns inconclusive, and review failure fails the run.
Persisted policy and review state govern resumed runs.
Why: Admission freezes the gate so later worker settings cannot silently remove it.

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
