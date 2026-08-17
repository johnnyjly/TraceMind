"""Deterministic model-input reduction; estimates are not provider usage."""

import json
from dataclasses import dataclass

from langchain.agents.middleware import ModelRequest
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.utils.function_calling import convert_to_openai_tool
from tracemind.models import Evidence
from tracemind.models.run import RunState


def collaboration_context(state: RunState, evidence: list[Evidence]) -> str:
    """Bound prior findings while keeping the current user request intact.

    Complete observations remain available through run-local read_evidence.
    Previous conclusions and user hypotheses are never promoted to evidence.
    """
    prior = state.prior_diagnosis
    payload = {
        "parent_run_id": str(state.parent_run_id),
        "request": state.user_message,
        "recent_user_messages": [m[:800] for m in state.previous_messages],
        "prior_conclusion": {
            "status": prior.status.value,
            "root_cause": prior.root_cause[:1600],
            "confidence": prior.confidence,
            "remediation": [text[:240] for text in prior.remediation[:3]],
            "evidence_ids": [str(i) for i in prior.evidence_ids[:20]],
            "historical_claim_only": True,
        } if prior else None,
        "available_evidence_count": len(evidence),
        "omitted_prior_evidence": state.omitted_prior_evidence,
        "evidence_previews": [{
            "id": str(item.id), "source": item.source.value,
            "service": item.service[:120] if item.service else None,
            "observed_at": item.observed_at.isoformat(),
            "summary": item.summary[:160],
        } for item in evidence[:10]],
        "bounded_history": True,
        "detail_retrieval": (
            "read_evidence lists and pages retained evidence"
            if "read_evidence" in state.allowed_tools
            else "detail retrieval is unavailable under current tool policy"
        ),
    }
    allowance = min(6000, max(
        0, (state.limits.context_tokens - state.limits.output_tokens) // 3
    ))
    while len(json_text(payload).encode()) > allowance:
        if payload["evidence_previews"]:
            payload["evidence_previews"].pop()
        elif payload["recent_user_messages"]:
            payload["recent_user_messages"].pop(0)
        elif payload["prior_conclusion"] is not None:
            payload["prior_conclusion"] = None
        else:
            break  # The existing context gate rejects oversized user input.
    return json_text(payload)


def json_text(value: object) -> str:
    """Serialize JSON deterministically while preserving Unicode characters."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"))


def estimate_input(request: ModelRequest) -> int:
    """Estimate input size from messages, tools, and the response schema.

    Count serialized UTF-8 bytes plus a framing allowance. Exclude tool
    artifacts, which are not sent to the provider. The result is a
    conservative admission estimate, not measured token usage or a
    tokenizer guarantee.
    """
    schema = getattr(
        request.response_format, "schema", request.response_format
    )
    if schema is not None and hasattr(schema, "model_json_schema"):
        schema = schema.model_json_schema()
    payload = {
        "system": (request.system_message.model_dump(mode="json")
                   if request.system_message else None),
        "messages": [m.model_dump(mode="json", exclude={"artifact"})
                     for m in request.messages],
        "tools": [convert_to_openai_tool(tool) for tool in request.tools],
        "response_format": schema,
        "tool_choice": request.tool_choice,
    }
    # One UTF-8 byte per estimated token is deliberately conservative for
    # text/JSON. Allow extra framing space; this is not a tokenizer guarantee.
    return len(json_text(payload).encode("utf-8")) + 1024


@dataclass(frozen=True)
class ContextSelection:
    """Describe a model-input view and its estimated fit within an allowance.

    Before/after values use the same estimator; reduced_messages counts
    replaced tool results and omitted_previews counts hidden references.
    A false fits value requires the caller to stop.
    """

    request: ModelRequest
    before: int
    after: int
    reduced_messages: int
    fits: bool
    omitted_previews: int = 0


def bounded_index(items: list[dict], count: int) -> list[dict]:
    """Retain a prefix of evidence references and all non-evidence records.

    Omission concerns this input view only. The run-local evidence catalogue
    remains pageable, including references no longer displayed here.
    """
    references = [item for item in items if "id" in item]
    metadata = [item for item in items if "id" not in item]
    return [*references[:count], *metadata, {
        "context_omitted_count": len(references) - count,
        "retrieval": (
            "read_evidence without evidence_id pages all run-local evidence; "
            "with evidence_id reads full detail"
        ),
    }]


def select_context(request: ModelRequest, allowance: int) -> ContextSelection:
    """Reduce tool results oldest first to fit the input-only allowance.

    Use application-generated indexes, then bounded previews if authorized
    evidence retrieval is available. Preserve tool-call/result pairs and
    newly requested detail pages. Count omitted references explicitly.
    Return fits=False if reduction is insufficient. Original messages and
    stored evidence are not mutated; output space must be reserved by the
    caller before supplying allowance.
    """
    before = estimate_input(request)
    if before <= allowance:
        return ContextSelection(request, before, before, 0, True)

    messages = list(request.messages)
    last_call = max(
        (i for i, m in enumerate(messages)
         if isinstance(m, AIMessage) and m.tool_calls), default=-1
    )
    reduced = set()
    eligible = []
    after = before
    # Oldest results first; never remove an assistant call or its response.
    for index, message in enumerate(messages):
        if not isinstance(message, ToolMessage):
            continue
        artifact = message.artifact
        if not isinstance(artifact, dict) or "context_index" not in artifact:
            continue
        # A requested detail page must reach the model intact this round.
        if index > last_call and artifact.get("detail_page"):
            continue
        eligible.append(index)
        content = json_text(artifact["context_index"])
        if len(content.encode()) >= len(str(message.content).encode()):
            continue
        messages[index] = message.model_copy(update={"content": content})
        reduced.add(index)
        candidate = request.override(messages=messages)
        after = estimate_input(candidate)
        if after <= allowance:
            return ContextSelection(
                candidate, before, after, len(reduced), True
            )

    # Omitted references must remain discoverable through an authorized tool.
    can_read = any(
        convert_to_openai_tool(tool)["function"]["name"] == "read_evidence"
        for tool in request.tools
    )
    omitted = 0
    if can_read:
        for index in eligible:
            message = messages[index]
            items = message.artifact["context_index"]
            count = sum("id" in item for item in items)
            if not count:
                continue

            def candidate(keep: int) -> ModelRequest:
                """Build a view while preserving messages and artifacts."""
                view = list(messages)
                view[index] = message.model_copy(update={
                    "content": json_text(bounded_index(items, keep)),
                })
                return request.override(messages=view)

            compact = candidate(0)
            size = estimate_input(compact)
            if size >= after:
                continue
            # Binary search accounts for JSON escaping and request framing,
            # retaining as many previews as the actual estimator permits.
            keep = 0
            if size <= allowance:
                low, high = 0, count - 1
                while low <= high:
                    middle = (low + high) // 2
                    probe = candidate(middle)
                    if estimate_input(probe) <= allowance:
                        keep, compact = middle, probe
                        low = middle + 1
                    else:
                        high = middle - 1
            messages = list(compact.messages)
            after = estimate_input(compact)
            reduced.add(index)
            omitted += count - keep
            if after <= allowance:
                return ContextSelection(
                    compact, before, after, len(reduced), True, omitted
                )
    return ContextSelection(
        request.override(messages=messages), before, after, len(reduced),
        False, omitted,
    )
