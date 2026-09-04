# Changelog

Feature changes consolidated from the project documentation.
Current behavior is in [README](README.md) and [DESIGN](DESIGN.md).

## Unreleased

- Shortened README and DESIGN to current usage and architecture.
- Consolidated design documentation and feature history.
- Consolidated operational, RAG and evaluation instructions into focused guides.
- Corrected obsolete module paths and removed historical test-blocker claims from
  current documentation.

## Implementation milestones

| Change |
| --- |
| Online RAG in shared API/worker/evaluation assembly; pinned policies, retained parents, harness-counted auxiliary calls, Docling parsing, framework rank fusion and IR evaluation |
| Source-verified corpus preparation/build/publication; English progress assertion alignment |
| Separated runtime/workflow/storage ownership; removed obsolete direct/stdio and investigation/verifier tests; refined prompts and English progress |
| Independent HTTP telemetry MCP and generic discovery/binding; blank-log handling; initial optional Qdrant/NetworkX/BGE RAG |
| Scoped confirmed memory and frozen evaluation, SQLite v3; feedback learning and receipts, v4; trace window/rejection diagnostics |
| Incident follow-ups/SSE, SQLite v2; expanded budgets; tool-free finalization; metric discovery/explanation packs; clean worker interruption |
| DeepSeek and pinned fault evaluation; bounded recovery (S3, `0dfee2f`); persistent queued runs/checkpoints (S4, `dd5a470`); first replay/operations increment (S5, `e7ff7e6`) |
| Execution budgets/trace controls (S1, `6201b0c`); bounded evidence/context controls (S2, `f0a7a66`); structured logging |
| FastAPI/domain foundation, telemetry clients, single-agent loop and pinned Demo evaluation |

The original Phase 1 excluded MCP, RAG, databases and collaboration; subsequent
increments superseded those exclusions. Early RAG used a custom parser, recursive
overlap splitter, homemade fusion and CLI-only retrieval. The framework integration
replaced those implementations while retaining legacy snapshot readability.
