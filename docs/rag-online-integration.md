# Online knowledge integration

## Module map

| Module | Responsibility |
| --- | --- |
| `agent/runtime/knowledge.py` | Agent tools, Harness admission and retained references |
| `agent/runtime/rag_model_factory.py` | Configured provider/model client assembly |
| `rag/retriever.py` | `SnapshotRetriever`: published-snapshot access and per-search lifetime |
| `rag/engine.py` | `RagRuntime`: LightRAG stores, recall and reranking |
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
2. The shared factory wires SnapshotRetriever and a provider callback.
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

SnapshotRetriever shares only local weights and a process admission
lock. Each search owns a fresh RagRuntime and receives the calling run's
completion closure. The runtime verifies immutable hashes, opens SQLite
read-only and copies LightRAG graph/JSON stores into a temporary directory.
Close runs in finally. Cancellation drains local inference before releasing
ownership. An OS file lock protects local snapshots across processes; concurrent
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
