# Changelog

Feature milestones for TraceMind. Current setup and behavior are documented in
[README](README.md) and [DESIGN](DESIGN.md).

## Agent coordination

- Bounded parallel investigations with task-scoped inputs, acceptance criteria,
  narrowed tool grants and immutable evidence references.
- Shared run/task token admission with synthesis and review reserves.
- Completion review checks the original objective as well as claim support.
- Single-agent and collaborative workflows share permissions, budgets and
  durable execution state.

## Diagnosis and telemetry

- Structured findings, hypotheses, limitations and proposed checks with validated
  evidence references.
- Independent read-only HTTP MCP service for metrics, traces and logs.
- Fixed investigation windows, trace filters and retained evidence pagination.
- Provider-specific reasoning support, bounded retries, optional fallback,
  context controls and provider-reported usage accounting.

## Persistent execution

- Queued runs, worker leases, graph checkpoints, cancellation and recovery.
- Durable SSE progress, incident follow-ups and retained-evidence explanations.
- Optional scoped feedback memory and frozen-memory evaluation.

## Knowledge and evaluation

- Source-checked corpus preparation, immutable snapshot publication and optional
  online knowledge retrieval.
- Independent RAG query service with graph/vector/BM25 retrieval, rank fusion
  and reranking.
- Langfuse experiments for live diagnosis, saved answers, completion verification
  and retrieval, with held-out labels and explicit call budgets.
- Synthetic and sanitized regression fixtures for execution and claim boundaries.

## Public repository documentation

- TraceMind package names and configuration prefixes throughout the guides.
- Chinese-first bilingual README and focused architecture, operation, RAG and
  evaluation documentation.
- Local credentials, experiment records and corpus-specific labels excluded
  from version control.
