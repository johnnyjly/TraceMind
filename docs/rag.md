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
models and checks inference. Runtime uses cached models only. OCR is off; CPU is
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
`TRACEMIND_RAG_PDF_PIPELINE=native` selects text-only extraction; standard PDF
layout processing fails explicitly if cached models are missing.

Build writes canonical parents/children and FTS5 to SQLite and uses matching
child IDs in LightRAG/Qdrant/graph. Model/time/output limits bound extraction;
`build.json` and document logs record progress/usage. Publication checks canonical
records, vector IDs and graph sources, hashes local files, then atomically writes
`snapshot.json`. Published configuration is immutable: parser/model/chunk changes
require a new ID. Existing legacy snapshots and retained references stay readable.

Incomplete builds cannot be queried. Failed builds require a new ID to retry;
there is no cross-store transaction or automatic partial repair. `verify` checks
local hashes, not remote Qdrant.

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

The agent chooses `search_knowledge`; optional rewriting translates/decomposes
queries into English. Auxiliary attempts/retries consume shared harness budgets
and reserve at least one model call for final synthesis. Cold models may exceed
the default 20-second tool timeout; measure locally before adjusting
`TRACEMIND_AGENT_TOOL_TIMEOUT_SECONDS`.

`read_knowledge` pages full parents saved in RunState, including explanations
without Qdrant. Published SQLite is read-only; searches privately copy mutable
graph/JSON stores. Local admission/snapshot locks serialize access; this is not a
distributed retrieval service. Scope metadata is not tenant authorization and
knowledge citations cannot replace current telemetry.

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
  --output artifacts/evaluations/retrieval-mix.json
```

`ir-measures` reports document Recall@K, MRR@K and nDCG@K, returned chunk IDs,
latency and internal model usage. Duplicate document hits count once; empty
results remain in the denominator. K must fit the result limit. Compare `mix` and
`naive` on the same frozen suite. Diagnosis RAG off/on evaluation is separate;
local smoke checks do not establish corpus-wide quality.
