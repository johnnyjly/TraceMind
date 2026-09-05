"""Deterministic model-input reduction; estimates are not provider usage."""

import json
from dataclasses import dataclass

from langchain.agents.middleware import ModelRequest
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.utils.function_calling import convert_to_openai_tool
from tracemind.models import Evidence
from tracemind.models.run import RunState


def guidance_context(state: RunState) -> str:
    """Expose bounded reusable instructions; saved snapshots retain origins."""
    payload = {
        "profile": state.guidance_snapshot.profile,
        "records": [{
            "id": str(r.id), "key": r.key, "kind": r.kind,
            "revision": r.revision, "instruction": r.instruction[:1000],
            "bounded_preview": True,
        } for r in state.guidance_snapshot.records],
        "omitted_records": 0,
    }
    while (len(json_text(payload).encode()) >
           state.learning_policy.context_bytes and payload["records"]):
        payload["records"].pop()
        payload["omitted_records"] += 1
    return json_text(payload)


def memory_context(state: RunState) -> str:
    """Bound historical hints without turning them into incident evidence.

    Full selected records live in the run snapshot for reproducible audit.
    Each preview marks its truncation; no historical evidence IDs are exposed.
    """
    snapshot = state.memory_snapshot
    payload = {
        "scope": snapshot.scope.model_dump(),
        "snapshot_id": snapshot.fingerprint,
        "historical_experience_only": True,
        "records": [{
            "memory_id": str(r.id), "revision": r.revision,
            "services": r.services, "symptom": r.symptom[:240],
            "resolution": r.resolution[:640],
            "diagnostic_queries": [q[:240] for q in r.diagnostic_queries[:2]],
            "remediation": [q[:240] for q in r.remediation[:2]],
            "source_run_id": str(r.source_run_id),
            "confirmed_by": r.confirmed_by,
            "confirmation_note": r.confirmation_note[:160],
            "confirmed_at": r.confirmed_at.isoformat(),
            "expires_at": r.expires_at.isoformat(), "bounded_preview": True,
        } for r in snapshot.records],
        "omitted_records": 0,
    }
    while (len(json_text(payload).encode()) > state.memory_policy.context_bytes
           and payload["records"]):
        payload["records"].pop()
        payload["omitted_records"] += 1
    if len(json_text(payload).encode()) > state.memory_policy.context_bytes:
        payload.pop("scope")
        payload["scope_omitted"] = True
    return json_text(payload)


def _evidence_preview(item: Evidence) -> dict:
    """Include normalized details, marking any oversized fields omitted.

    Small observations remain complete. Large attributes are bounded
    separately so a long log body cannot hide the service and error fields.
    """
    data = item.model_dump(mode="json")
    if len(json_text(data).encode()) <= 2000:
        return data
    preview = {
        "id": str(item.id), "source": item.source.value,
        "service": item.service[:120] if item.service else None,
        "observed_at": item.observed_at.isoformat(),
        "summary": item.summary[:240],
        "query": item.query[:240] if item.query else None,
        "attributes": {}, "truncated": True,
        "omitted_attribute_keys": [], "truncated_attribute_keys": [],
    }
    priority = (
        "operation", "error_type", "otel_status_description",
        "status_message", "body", "trace_id", "span_id", "labels", "value",
    )
    keys = sorted(item.attributes, key=lambda k: (
        priority.index(k) if k in priority else len(priority), k,
    ))
    for key in keys:
        value = item.attributes[key]
        shortened = isinstance(value, str) and len(value) > 320
        if shortened:
            value = value[:320]
        candidate = {**preview["attributes"], key: value}
        if len(json_text(candidate).encode()) <= 1200:
            preview["attributes"] = candidate
            if shortened:
                preview["truncated_attribute_keys"].append(key)
        else:
            preview["omitted_attribute_keys"].append(key)
    return preview


def _prioritized_evidence(state: RunState, evidence: list[Evidence]):
    """Yield cited observations first, interleaving services and sources.

    This is a coverage heuristic, not a causal ranking. It prevents a large
    batch from one service from displacing every downstream observation.
    """
    cited = set(state.prior_diagnosis.evidence_ids) if (
        state.prior_diagnosis is not None
    ) else set()
    for is_cited in (True, False):
        groups = {}
        for item in evidence:
            if (item.id in cited) == is_cited:
                groups.setdefault((item.service, item.source), []).append(item)
        for offset in range(max(map(len, groups.values()), default=0)):
            for group in groups.values():
                if offset < len(group):
                    yield group[offset]


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
        "evidence_previews": [
            _evidence_preview(item)
            for item in list(_prioritized_evidence(state, evidence))[:20]
        ],
        "omitted_evidence_previews": 0,
        "bounded_history": True,
        "detail_retrieval": (
            "Read a known evidence_id directly for missing details. "
            "Otherwise filter the catalogue by service/source. Do not scan "
            "all catalogue pages unless the question requires that coverage."
            if "read_evidence" in state.allowed_tools
            else "detail retrieval is unavailable under current tool policy"
        ),
    }
    allowance = min(16000, max(
        0, (state.limits.context_tokens - state.limits.output_tokens) // 3
    ))
    payload["omitted_evidence_previews"] = (
        len(evidence) - len(payload["evidence_previews"])
    )
    while len(json_text(payload).encode()) > allowance:
        if payload["recent_user_messages"]:
            payload["recent_user_messages"].pop(0)
        elif payload["evidence_previews"]:
            payload["evidence_previews"].pop()
            payload["omitted_evidence_previews"] += 1
        elif payload["prior_conclusion"] is not None:
            payload["prior_conclusion"] = None
        else:
            break  # The existing context gate rejects oversized user input.
    return json_text(payload)


def json_text(value: object) -> str:
    """Serialize JSON deterministically while preserving Unicode characters."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"))


def estimate_payload(payload: object) -> int:
    """Estimate text/JSON tokens using UTF-8 bytes plus framing allowance."""
    return len(json_text(payload).encode("utf-8")) + 1024


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
    return estimate_payload(payload)


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


def bounded_index(items: list[dict], count: int, *, key: str = "id",
                  reader: str = "read_evidence") -> list[dict]:
    """Retain a prefix of evidence references and all non-evidence records.

    Omission concerns this input view only. The run-local evidence catalogue
    remains pageable, including references no longer displayed here.
    """
    references = [item for item in items if key in item]
    metadata = [item for item in items if key not in item]
    return [*references[:count], *metadata, {
        "context_omitted_count": len(references) - count,
        "retrieval": (
            f"{reader} pages retained references and reads full detail"
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
    readers = {convert_to_openai_tool(tool)["function"]["name"]
               for tool in request.tools}
    # Oldest results first; never remove an assistant call or its response.
    for index, message in enumerate(messages):
        if not isinstance(message, ToolMessage):
            continue
        artifact = message.artifact
        if not isinstance(artifact, dict) or "context_index" not in artifact:
            continue
        reader = artifact.get("context_reader")
        if reader is not None and reader not in readers:
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
    omitted = 0
    if readers:
        for index in eligible:
            message = messages[index]
            key = message.artifact.get("context_reference_key", "id")
            reader = message.artifact.get("context_reader", "read_evidence")
            if reader not in readers:
                continue
            items = message.artifact["context_index"]
            count = sum(key in item for item in items)
            if not count:
                continue

            def candidate(keep: int) -> ModelRequest:
                """Build a view while preserving messages and artifacts."""
                view = list(messages)
                view[index] = message.model_copy(update={
                    "content": json_text(bounded_index(
                        items, keep, key=key, reader=reader)),
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
