# RAG setup and retrieval

RAG is optional and disabled for baseline diagnosis. Nothing is automatically
imported from `knowledge/`. Commands run from the repository root; internal
contracts are in [online integration](rag-online-integration.md).

## Install and initialize

```bash
uv sync --extra rag --dev
docker compose -f deploy/compose.rag.yaml up -d
uv run --extra rag python scripts/rag.py models
uv run --extra rag python scripts/rag.py init
uv run --extra rag python scripts/rag.py check
```

`models` downloads pinned BGE embedding/reranking weights and Docling layout/table
models and checks inference. Runtime uses cached models only.
Why: Pinned cached weights make runtime initialization reproducible.

OCR is off; CPU is
the default, with `TRACEMIND_RAG_DEVICE=mps` for Apple Silicon. Qdrant is independent
of the Demo and runs on localhost:6333. Workspaces/models default to ignored
`artifacts/rag/`; `bootstrap` is empty and unpublished.

[.env.example](../.env.example) lists `TRACEMIND_RAG_*` settings. Omit both
`TRACEMIND_RAG_LLM_PROVIDER` and `TRACEMIND_RAG_LLM_MODEL` to use the main model,
or set both with the matching application key. Application adapters own provider
selection and credentials. Empty initialization needs no key; builds and populated
queries can make paid calls. Embedding/reranking are local. `--env-file` selects
alternate settings; `--model-cache` shares weights without sharing snapshot files.

## Prepare and publish

The corpus needs English schema-v2 `manifest.json` and `sources.json`, with
profile-selected files in `documents/`. File hashes and referenced Git objects are
verified. HTTPS references need access dates/version policies; live contents and
authored semantic claims are not verified.

```bash
# Inspect normalized text, parents and children without paid calls.
uv run --extra rag python scripts/rag.py prepare --corpus knowledge --profile core

# Example paid-call budget; adjust to the corpus.
uv run --extra rag python scripts/rag.py build --corpus knowledge --profile core \
  --snapshot runbooks-v2 --max-model-calls 512 --build-timeout 1800
uv run --extra rag python scripts/rag.py verify --snapshot runbooks-v2
```

Use `--demo-repo` if the referenced Demo is not at `../opentelemetry-demo`.
Docling converts Markdown, HTML, authored JSON text and PDF. Structural parents
retain full sections; HybridChunker uses the pinned tokenizer with a 320-token
child cap. Provenance retains hashes/revisions, headings and element pointers.
Partial conversions are rejected before chunking; missing pages cannot be
published as a complete document.
`TRACEMIND_RAG_PDF_PIPELINE=native` selects text-only extraction; standard PDF
layout processing fails explicitly if cached models are missing.

Build writes canonical parents/children and FTS5 to SQLite and uses matching
child IDs in LightRAG/Qdrant/graph. Model/time/output limits bound extraction;
`build.json` and document logs record progress/usage. Publication checks canonical
records, vector IDs and graph sources, hashes local files, then atomically writes
`snapshot.json`. Published configuration is immutable: parser/model/chunk changes
require a new ID. Existing legacy snapshots and retained references stay readable.
Why: Changing published content under an existing ID would invalidate saved citations
and hashes.

Incomplete builds cannot be queried. Failed builds require a new ID to retry;
there is no cross-store transaction or automatic partial repair. `verify` checks
local hashes, not remote Qdrant.
Why: Publishing incomplete or partially repaired stores would make retrieval provenance
unreliable.

## Query and enable online knowledge

```bash
uv run --extra rag python scripts/rag.py query --snapshot runbooks-v2 \
  --query "RPC connection errors" --mode mix
```

`mix` combines LightRAG graph/vector and independent FTS5/BM25 recall. LangChain
rank fusion combines all query branches before one BGE rerank. `naive` uses vector
recall with the same lexical/fusion/reranking stages. Results are references,
not a generated diagnosis.

After publication, configure and restart API/worker:

```dotenv
TRACEMIND_AGENT_KNOWLEDGE_ENABLED=true
TRACEMIND_AGENT_KNOWLEDGE_SNAPSHOT=runbooks-v2
TRACEMIND_AGENT_ALLOWED_TOOLS=["query_metrics","discover_metrics","search_traces","get_trace_detail","search_logs","read_evidence","search_knowledge","read_knowledge"]
```

The agent chooses `search_knowledge`; input is assumed to be English. Optional
rewriting decomposes the question while preserving exact identifiers; it does
not translate the input. Auxiliary attempts/retries consume shared harness budgets
and reserve final-synthesis capacity within the shared model budget. Embedded
retrieval can load cold models and exceed the default 20-second tool timeout;
measure locally before adjusting `TRACEMIND_AGENT_TOOL_TIMEOUT_SECONDS`. The
configured HTTP service warms models before readiness; keyword extraction and
retrieval still share the tool deadline.

`read_knowledge` pages full parents saved in RunState, including explanations
without Qdrant. Published SQLite is read-only. Embedded searches each copy mutable
graph/JSON stores; the HTTP service retains a private copy for its process lifetime.
Local admission/snapshot locks serialize access; this is not a distributed
retrieval service. See [service lifetime](rag-online-integration.md#independent-query-service).
Scope metadata is not tenant authorization and
knowledge citations cannot replace current telemetry.
Why: Reference material describes possible mechanisms, not observations of the current
deployment.

Qdrant network calls are asynchronous; collection setup and snapshot hashing/copy
drain their blocking work before cancellation releases ownership. Shutdown stops
LightRAG's background model/embedding queues before closing stores. Do not reuse
one snapshot ID for different corpus directories in the same process, and leave
`QDRANT_WORKSPACE` unset or equal to the selected snapshot ID.
Why: LightRAG caches by snapshot identity, so conflicting directories or workspace names
can mix stores.

## Evaluate retrieval

Judgments stay outside the corpus/runtime queries. Create a suite with document
IDs matching your snapshot, for example:

```json
{
  "snapshot": "runbooks-v2",
  "questions": [{
    "id": "metrics-empty",
    "query": "How should an empty metric query be investigated?",
    "relevant": {"runbooks.empty-metric-results": 1}
  }]
}
```

Save your suite as `artifacts/evaluations/retrieval-suite.json`, then run:

```bash
uv run --extra rag python scripts/rag.py evaluate --snapshot runbooks-v2 \
  --suite artifacts/evaluations/retrieval-suite.json --k 5 --mode mix \
  --env-file .env --env-file deploy/.env.langfuse --max-model-calls 20 \
  --output artifacts/evaluations/retrieval-mix.json
```

Langfuse stores per-stage document precision, recall, MRR and nDCG, query
latency, provenance and internal model usage. Duplicate document hits count once;
empty results remain visible. `--output` writes only the experiment ID,
publication status and acceptance pointer, rather than a second score report.
Labels remain outside retrieval inputs. Compare `mix` and `naive` on the same
frozen suite; diagnosis RAG off/on evaluation is separate. Why: Frozen relevance
judgments measure retrieval without leaking expected answers into queries. See
[Evaluation](evaluation.md#rag-retrieval) for publication and threshold gates.

Use `"relevant": {}` for a question with no relevant document in the selected
snapshot. Its `expected_empty` score checks abstention separately from ranking
quality; retain held-out positive questions so rejecting every query cannot
look successful. The independent RAG query service, shared local model caches,
query planning, parent expansion and configured reranker cutoff are unchanged.

`TRACEMIND_RAG_RERANK_MIN_SCORE` optionally removes candidates below a calibrated
reranker score. It is unset by default: scores are not relevance probabilities
and a universal 0.5 cutoff is inappropriate. Calibrate against the exact snapshot
and pinned reranker, then validate on held-out positive and negative questions.
Clear or recalibrate the cutoff when either changes. This is query policy; it
does not modify the immutable published snapshot.
Why: Reranker scores depend on the model and corpus and are not universal relevance
probabilities.

Keep corpus-specific development and holdout labels under ignored `artifacts/`.
The suites must identify the published snapshot and contain document IDs
from that corpus. Select a cutoff using development questions, then evaluate
it on separate holdout questions. A small corpus does not establish general
retrieval quality.

Timing logs separate `rag.ready` lock wait, verification and initialization from
`rag.search` retrieval and ranking. Retrieval includes embedding and any keyword
model work; it is not a pure vector-store latency measure.
