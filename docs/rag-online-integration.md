# Online knowledge integration

The shared model budget reserves the final diagnosis and remaining bounded repair
attempts (while allowing a discovery turn when the total budget is at least two).
Auxiliary retrieval cannot consume this reserve. The main loop checks it before
every attempt, including retries and resumed runs, and closes investigation tools
when the reserve is reached. This bounds output attempts; it cannot guarantee a
valid diagnosis from a failing provider.

Knowledge previews expose reranker scores as ranking signals, not probabilities.
Empty retrieval and repeated retained references carry explicit guidance to stop
synonym searches. These metadata messages do not count as investigation progress.
Changing the query can still be justified by a different unresolved hypothesis.

## Module map

| Module | Responsibility |
| --- | --- |
| `agent/runtime/knowledge.py` | Agent tools, Harness admission and retained references |
| `agent/runtime/rag_model_factory.py` | Configured provider/model client assembly |
| `rag/retriever.py` | `SnapshotRetriever`: published-snapshot access and per-search lifetime |
| `rag/engine.py` | `RagRuntime`: LightRAG stores, recall and reranking |
| `rag/qdrant.py` | Async Qdrant adapter using the pinned storage format |
| `rag/model_bridge.py` | `LightRAGModelBridge`: prompt adaptation, call limits and usage |
| `rag/keyword.py` | SQLite canonical records and FTS5/BM25 recall |
| `rag/fusion.py` | LangChain reciprocal-rank fusion of candidate lists |
| `rag/models.py` | Source, parent, child and hit schemas |

LightRAG, Docling and local inference libraries are imported when their
operations run. Agent assembly defers the optional retrieval stack until
knowledge is enabled; lightweight settings and provider assembly use ordinary
module imports. Internal callers use these modules directly; schemas do not
import the fusion algorithm or optional retrieval libraries.

## Flow and boundaries

1. API admission freezes enabled KnowledgePolicy, snapshot and authorized tools.
2. The shared factory selects HTTP retrieval when service_url is configured,
   otherwise embedded SnapshotRetriever; model callbacks stay in the Agent.
3. The Agent optionally calls search_knowledge with a concise English query;
   rewrite=true invokes bounded English planning through the same Harness.
4. All rewritten queries contribute LightRAG and FTS5 candidates. LangChain
   EnsembleRetriever fuses their ranks; one local CrossEncoder reranks the union.
5. Canonical children and full parent sections are retained in durable RunState.
6. read_knowledge pages saved parent text. Explanation needs no retrieval backend.

Knowledge is reference material, separate from current observations. Diagnosis
validation rejects unknown/duplicate knowledge citations and still requires
telemetry evidence for a complete diagnosis. Search does not create evidence.

## Assembly and lifetime

`TRACEMIND_AGENT_KNOWLEDGE_ENABLED` defaults false. Enablement requires an
explicit published snapshot and grants; discovery does not grant local tools.
Run capture persists the policy. Follow-ups inherit it and copied references.
An explanation exposes only authorized retained-data readers.

The application owns credentials, provider selection and request options.
When both RAG provider/model overrides are absent, assembly uses the primary
model. Partial overrides fail. SDK retries stay disabled; Harness owns retries.
Auxiliary responses accept text content blocks and reject incomplete/refused
output before it can enter query planning or graph extraction.

For embedded access, SnapshotRetriever shares local weights and a process admission
lock. Each search owns a fresh RagRuntime and receives the calling run's
completion closure. The runtime verifies immutable hashes, opens SQLite
read-only and copies LightRAG graph/JSON stores into a temporary directory.
Close runs in finally. Cancellation stops and drains framework queues and local
inference before releasing ownership. Qdrant requests use an async transport;
collection setup and local snapshot hashing/copy run outside the event loop.
Expected Qdrant/SQLite failures become recoverable knowledge-unavailable results.
The process rejects one snapshot ID bound to multiple local directories, because
LightRAG's shared KV cache omits the directory from its key. `QDRANT_WORKSPACE`
must be absent or equal to the selected snapshot ID.
An OS file lock protects local snapshots across processes; concurrent
multi-process retrieval is not supported by this deployment contract.

## Injection contract

```python
async def search(queries: list[str], *, question: str, snapshot: str,
                 completion) -> list[RetrievalHit]:
    ...
```

`question` is the ranking query (the combined English plan after rewriting).
`queries` are independently recalled and jointly fused before reranking.
The supplied completion receives messages, max_tokens and keyword_extraction,
and returns AIMessage with provider usage when available. Every actual auxiliary
attempt consumes the run budget; missing usage is unknown. Auxiliary calls
reserve a configurable number of main model calls for final synthesis.

DiagnosisService/build_service retain explicit injection seams for fixed-result
regressions. Disabling online knowledge avoids initializing optional frameworks.
Saved sources remain readable when the retrieval backend is unavailable.

## Context, provenance and progress

Retention has independent count/byte caps; input context has its own byte cap.
Search previews and parent pages use context_index artifacts and the authorized
read_knowledge reader in the existing deterministic compressor. The newest
requested detail page reaches the model intact. Overflow is reported explicitly.
Knowledge text is not replaced by a reference-only index when `read_knowledge`
is unauthorized; the context gate reports overflow if the text cannot fit.

Sources retain snapshot, original file hash, source-registry revision, headings,
Docling element pointers and canonical text. Legacy chunks without parents
remain readable. New parser versions require a new publication; no automatic
migration or mutation of old snapshots occurs.

Durable knowledge.search.started/completed/failed events complement tool/model
spans. Auxiliary purposes distinguish planning, keyword extraction and retrieval.
Public SSE projects friendly stages without prompts, raw documents or queries.
RunState retains full source records for auditing and explanations.

## Validation boundaries

Scripted tests validate control, ownership and contracts. Local Qdrant/model
smoke checks validate actual storage and inference paths. Neither establishes
corpus-wide retrieval quality or diagnostic accuracy. The CLI evaluate action
uses ir-measures with separate snapshot-bound judgments; labels never enter
runtime prompts. Agent RAG off/on comparisons need the same incident window and
frozen corpus. Tenant/source authorization and distributed retrieval are later
work.

## Independent query service

Run `uv run --extra rag python scripts/run_rag_service.py` as a separate long-lived
process before API/worker. This local deployment binds `127.0.0.1:8011` and caps
PyTorch intra-op/inter-op threads at 2/1. It runs as an independent process;
deployment on another host requires network access controls.
It does not install an OS auto-start daemon. Restart the process after changing
its snapshot or model settings, and wait for `GET /ready` before submitting work.

Set `TRACEMIND_RAG_SERVICE_URL=http://127.0.0.1:8011` in API/worker configuration.
Set `TRACEMIND_RAG_SNAPSHOT` to the same **published** snapshot as
`TRACEMIND_AGENT_KNOWLEDGE_SNAPSHOT`; the service rejects mismatches. Startup
verifies the snapshot, loads weights/stores and performs embedding/rerank warmup.
The immutable publication stays unchanged; LightRAG uses a private working copy.
The process owns the snapshot lock until shutdown; do not simultaneously run an
embedded reader of that same snapshot. Offline builds use a different snapshot.

`query_client.py` checks readiness, extracts keywords using the pinned LightRAG
prompt and the investigation's existing counted completion callback, then sends
prepared queries to `POST /search`. `query_contracts.py` bounds and validates
those requests and responses. `query_service.py` performs graph/vector/BM25
retrieval, fusion and reranking once across the prepared queries. It neither
receives API keys nor invokes a provider model. Returned hits pass through the
same Agent reference retention and citation validation. The tool allowlist,
model reserve and investigation budgets are unchanged.

The service rejects overlapping requests with 503 instead of building an
unbounded queue. `TRACEMIND_RAG_SERVICE_QUERY_TIMEOUT=15` bounds retrieval;
the Agent's existing 20-second tool deadline includes keyword extraction and
HTTP retrieval. Readiness failure, mismatch, overload and timeout surface as
knowledge unavailability; the client does not silently fall back to local model
loading. HTTP client cancellation does not itself prove server work has stopped;
server timeout and serialized local inference bound remaining work. Shutdown
drains LightRAG workers before closing stores.

Startup work is outside the investigation deadline. This eliminates per-worker
weight cold starts and per-query store recreation for the configured HTTP path;
it does not guarantee that external LLM latency or large retrieval workloads
will always fit the existing deadline. A service split is useful isolation, not
by itself a claim of production readiness or unlimited concurrency.
