# TraceMind design direction

## Current baseline

The implemented baseline is one bounded tool-calling agent. Telemetry clients
normalize backend-specific responses into `Evidence`, and a complete diagnosis
may cite only evidence returned by an actual tool call. Evaluation cases keep
the incident, tool interfaces, model settings, budgets, and ground truth fixed.

## Investigation state

The next architecture stage will replace chat-history coordination with an
explicit investigation state containing:

- an investigation plan;
- structured hypotheses;
- normalized evidence;
- typed evidence-to-hypothesis relationships;
- tool failures and remaining budgets;
- a verifier decision and targeted next actions.

Raw tool observations will be filtered, normalized, deduplicated, and reduced
to bounded evidence before entering the coordinator context. The verifier will
decide whether evidence supports a root cause, contradicts active hypotheses,
or requires another targeted investigation round.

## Planned workflow

```text
Incident parser -> Coordinator -> Investigation plan
                          |
              +-----------+-----------+
              |                       |
     Telemetry investigator   Code/knowledge investigator
              |                       |
              +-------- Evidence -----+
                          |
                 Hypothesis manager
                          |
                       Verifier
                    /             \
          targeted follow-up      RCA report
```

Specialists will communicate through shared typed state rather than free-form
agent-to-agent conversations. The harness will own budgets, termination,
retries, duplicate-call detection, checkpoints, and verification.

## Tool boundaries

Tools will first be implemented behind stable local schemas and a registry.
Observability, repository, and knowledge adapters can later be exposed through
MCP without changing the workflow contract. MCP is therefore a transport and
isolation boundary, not the source of domain behavior.

## Evaluation sequence

Each orchestration strategy will use the same cases, tools, model providers,
budgets, and scoring rules:

1. single-agent baseline;
2. rule-based multi-agent workflow with a verifier;
3. learned coordinator policy with fixed LLM weights.

Measured outcomes will include root-cause accuracy, evidence grounding, tool
calls, latency, token cost when providers expose it, false completion, and
recovery from tool failures. Features will be described as implemented only
after their code and evaluation evidence exist.
