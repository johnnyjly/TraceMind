"""Deterministic model-input reduction; estimates are not provider usage."""

import json
from dataclasses import dataclass

from langchain.agents.middleware import ModelRequest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.utils.function_calling import convert_to_openai_tool
from tracemind.models import Evidence
from tracemind.models.run import RunState
from tracemind.agent.workflow.references import model_references
from tracemind.agent.runtime.token_count import estimate_text_tokens


def guidance_context(state: RunState) -> str:
    """Expose bounded reusable instructions; saved snapshots retain origins."""
    payload = {
        "profile": state.guidance_snapshot.profile,
        "records": [
            {
                "id": str(r.id),
                "key": r.key,
                "kind": r.kind,
                "revision": r.revision,
                "instruction": r.instruction[:1000],
                "bounded_preview": True,
            }
            for r in state.guidance_snapshot.records
        ],
        "omitted_records": 0,
    }
    while (
        len(json_text(payload).encode()) > state.learning_policy.context_bytes
        and payload["records"]
    ):
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
        "records": [
            {
                "memory_id": str(r.id),
                "revision": r.revision,
                "services": r.services,
                "symptom": r.symptom[:240],
                "resolution": r.resolution[:640],
                "diagnostic_queries": [
                    q[:240] for q in r.diagnostic_queries[:2]
                ],
                "remediation": [q[:240] for q in r.remediation[:2]],
                "source_run_id": str(r.source_run_id),
                "confirmed_by": r.confirmed_by,
                "confirmation_note": r.confirmation_note[:160],
                "confirmed_at": r.confirmed_at.isoformat(),
                "expires_at": r.expires_at.isoformat(),
                "bounded_preview": True,
            }
            for r in snapshot.records
        ],
        "omitted_records": 0,
    }
    while (
        len(json_text(payload).encode()) > state.memory_policy.context_bytes
        and payload["records"]
    ):
        payload["records"].pop()
        payload["omitted_records"] += 1
    if len(json_text(payload).encode()) > state.memory_policy.context_bytes:
        payload.pop("scope")
        payload["scope_omitted"] = True
    return json_text(payload)


def _bounded_text(value: str, limit: int) -> str:
    return value.encode()[:limit].decode("utf-8", errors="ignore")


def _location_preview(code: dict) -> dict:
    conflicts = code.get("conflicts") or {}
    incomplete = set(code.get("redacted_fields") or ()) | set(
        code.get("truncated_fields") or (),
    )
    result = {
        key: _bounded_text(value, 100) if isinstance(value, str) else value
        for key, value in code.items()
        if key in {"file", "line", "function"}
        and key not in conflicts and key not in incomplete
    }
    if conflicts:
        result["conflicting_fields"] = sorted(conflicts)
    if incomplete:
        result["incomplete_fields"] = sorted(incomplete)
    return result


def _evidence_preview(item: Evidence, *, max_bytes: int = 2000) -> dict:
    """Include normalized details, marking any oversized fields omitted.

    Small observations remain complete. Large attributes are bounded
    separately so a long log body cannot hide the service and error fields.
    """
    data = item.model_dump(mode="json")
    if len(json_text(data).encode()) <= max_bytes:
        return data
    preview = {
        "id": str(item.id),
        "source": item.source.value,
        "service": item.service[:120] if item.service else None,
        "observed_at": item.observed_at.isoformat(),
        "summary": item.summary[:240],
        "query": item.query[:240] if item.query else None,
        "attributes": {},
        "truncated": True,
        "omitted_attribute_keys": [],
        "truncated_attribute_keys": [],
    }
    priority = (
        "trace_id",
        "span_id",
        "diagnostic_source",
        "diagnostic",
        "is_error",
        "resource",
        "code",
        "exceptions",
        "exception_coverage",
        "operation",
        "error_type",
        "otel_status_description",
        "status_message",
        "body",
        "labels",
        "value",
    )
    while len(json_text(preview).encode()) > max_bytes - 300:
        if len(preview["summary"]) > 40:
            length = len(preview["summary"]) // 2
            preview["summary"] = preview["summary"][:length]
        elif preview["query"]:
            preview["query"] = None
        else:
            break
    base_bytes = len(json_text(preview).encode())
    allowance = min(1200, max(0, max_bytes - base_bytes - 120))
    keys = sorted(
        item.attributes,
        key=lambda k: (
            priority.index(k) if k in priority else len(priority),
            k,
        ),
    )
    for key in keys:
        value = item.attributes[key]
        shortened = isinstance(value, str) and len(value) > 320
        if shortened:
            value = value[:320]
        elif (
            key == "diagnostic" and isinstance(value, dict)
            and len(json_text(
                {**preview["attributes"], key: value},
            ).encode()) > allowance
        ):
            fields = {
                "event.name", "upstream.host", "upstream.cluster",
                "destination.address", "source.address", "server.address",
                "server.port", "span.kind", "otel.status_code", "error",
                "rpc.grpc.status_code", "rpc.response.status_code",
                "grpc.status_code", "http.response.status_code",
                "http.status_code",
            }
            fields |= {f + suffix for f in fields
                       for suffix in (".redacted", ".truncated")}
            value = {k: _bounded_text(v, 100) if isinstance(v, str) else v
                     for k, v in value.items() if k in fields}
            shortened = value != item.attributes[key]
        elif key == "code" and isinstance(value, dict):
            value = _location_preview(value)
            shortened = value != item.attributes[key]
        elif key == "resource" and isinstance(value, dict):
            metadata = item.attributes.get("resource_metadata") or {}
            conflicts = metadata.get("conflicts") or {}
            incomplete = set(metadata.get("redacted_fields") or ()) | set(
                metadata.get("truncated_fields") or (),
            )
            value = {k: _bounded_text(v, 80) if isinstance(v, str) else v
                     for k, v in value.items() if k in {
                         "service.version", "service.instance.id",
                         "deployment.environment.name", "k8s.pod.name",
                         "k8s.namespace.name", "k8s.cluster.name",
                     } and k not in conflicts and k not in incomplete}
            if conflicts:
                value["conflicting_fields"] = sorted(conflicts)
            if incomplete:
                value["incomplete_fields"] = sorted(incomplete)
            shortened = value != item.attributes[key]
        elif key == "exceptions" and isinstance(value, list):
            value = [{
                **{k: _bounded_text(v, 100) if isinstance(v, str) else v
                   for k, v in event.items() if k in {
                       "type", "message", "source", "timestamp",
                       "in_investigation_window"}},
                "has_stacktrace": bool(event.get("stacktrace")),
                **({"code": _location_preview(event["code"])}
                   if isinstance(event.get("code"), dict) else {}),
            } for event in value[:2] if isinstance(event, dict)]
            shortened = True
        candidate = {**preview["attributes"], key: value}
        if len(json_text(candidate).encode()) <= allowance:
            preview["attributes"] = candidate
            if shortened:
                preview["truncated_attribute_keys"].append(key)
        else:
            preview["omitted_attribute_keys"].append(key)
    # Metadata is also bounded; the count still records every omission.
    preview["omitted_attribute_count"] = len(preview["omitted_attribute_keys"])
    while len(json_text(preview).encode()) > max_bytes:
        if preview["omitted_attribute_keys"]:
            preview["omitted_attribute_keys"].pop()
        elif preview["truncated_attribute_keys"]:
            preview["truncated_attribute_keys"].pop()
        else:
            break
    return preview


def _prioritized_evidence(
    state: RunState, evidence: list[Evidence], priority_ids=()
):
    """Yield cited observations first, interleaving services and sources.

    This is a coverage heuristic, not a causal ranking. It prevents a large
    batch from one service from displacing every downstream observation.
    """
    cited = (
        set(state.prior_diagnosis.evidence_ids)
        if (state.prior_diagnosis is not None)
        else set()
    )
    if priority_ids:
        cited = set(priority_ids)
    for is_cited in (True, False):
        groups = {}
        for item in evidence:
            if (item.id in cited) == is_cited:
                groups.setdefault((item.service, item.source), []).append(item)
        for offset in range(max(map(len, groups.values()), default=0)):
            for group in groups.values():
                if offset < len(group):
                    yield group[offset]


def collaboration_context(
    state: RunState, evidence: list[Evidence], *, priority_ids=()
) -> str:
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
            "review_status": prior.review_status,
            "review_error_code": prior.review_error_code,
            "triage_status": prior.triage_status,
            "item_review_states": [
                {"id": item.id, "review_status": item.review_status,
                 "evidence_support": item.evidence_support}
                for item in prior.items
            ],
            "remediation": [text[:240] for text in prior.remediation[:3]],
            "evidence_ids": [str(i) for i in prior.evidence_ids[:20]],
            "historical_claim_only": True,
        }
        if prior
        else None,
        "available_evidence_count": len(evidence),
        "omitted_prior_evidence": state.omitted_prior_evidence,
        "evidence_previews": [
            _evidence_preview(item)
            for item in list(
                _prioritized_evidence(state, evidence, priority_ids)
            )[:20]
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
    allowance = max(
        0, (state.limits.context_tokens - state.limits.output_tokens) // 3
    )
    payload["omitted_evidence_previews"] = len(evidence) - len(
        payload["evidence_previews"]
    )
    while (len(json_text(payload).encode()) > 16000
           or estimate_payload(payload) > allowance):
        if payload["recent_user_messages"]:
            payload["recent_user_messages"].pop(0)
        elif payload["evidence_previews"]:
            payload["evidence_previews"].pop()
            payload["omitted_evidence_previews"] += 1
        elif payload["prior_conclusion"] is not None:
            payload["prior_conclusion"] = None
        else:
            break  # The existing context gate rejects oversized user input.
    return json_text(model_references(payload, state))


def _reported_preview(context, text_bytes):
    truncated = []

    def reduce(value, path=""):
        if isinstance(value, dict):
            return {key: reduce(item, f"{path}.{key}".lstrip("."))
                    for key, item in value.items()}
        if isinstance(value, list):
            return [reduce(item, f"{path}.{index}")
                    for index, item in enumerate(value)]
        if (isinstance(value, str)
                and path.rsplit(".", 1)[-1] not in {
                    "trace_id", "span_id", "reported_at",
                }):
            limit = (text_bytes * 3
                     if path == "error.stacktrace" else text_bytes)
            shortened = _bounded_text(value, limit)
            if shortened != value:
                truncated.append(path)
            return shortened
        return value

    reported = reduce(context.model_dump(mode="json", exclude_none=True))
    return reported, truncated


def intake_context(state: RunState, evidence: list[Evidence]) -> str:
    """Present supplied hints and lookup evidence as separate bounded data."""
    reported, truncated = _reported_preview(state.initial_context, 160)
    progress = state.intake_progress
    priority = [key for result in (progress.results if progress else [])
                for key in result.evidence_ids]
    payload = {
        "reported_context": reported,
        "reported_context_is_unverified": True,
        "truncated_reported_fields": truncated,
        "lookup_results": [r.model_dump(mode="json")
                           for r in progress.results] if progress else [],
        "available_evidence_count": len(evidence),
        "evidence_previews": [
            _evidence_preview(item, max_bytes=1200)
            for item in list(
                _prioritized_evidence(state, evidence, priority),
            )[:12]
        ],
        "omitted_evidence_previews": max(0, len(evidence) - 12),
        "detail_retrieval": (
            "Use read_evidence with an E handle for retained full details."
            if "read_evidence" in state.allowed_tools
            else "Detail retrieval is unavailable under current tool policy."
        ),
    }
    allowance = max(
        0, (state.limits.context_tokens - state.limits.output_tokens) // 3,
    )
    text_bytes = 160
    while (len(json_text(payload).encode()) > 12000
           or estimate_payload(payload) > allowance):
        if len(payload["evidence_previews"]) > 1:
            payload["evidence_previews"].pop()
            payload["omitted_evidence_previews"] += 1
        elif text_bytes > 20:
            text_bytes //= 2
            reported, truncated = _reported_preview(
                state.initial_context, text_bytes,
            )
            payload["reported_context"] = reported
            payload["truncated_reported_fields"] = truncated
        else:
            # Keep at least one observation; the context gate rejects budgets
            # too small even for this bounded packet instead of hiding it.
            break
    return json_text(model_references(payload, state))


def json_text(value: object) -> str:
    """Serialize JSON deterministically while preserving Unicode characters."""
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def estimate_payload(payload: object) -> int:
    """Estimate serialized text/JSON tokens with tokenizer headroom."""
    return estimate_text_tokens(json_text(payload))


def estimate_input(request: ModelRequest) -> int:
    """Estimate input size from messages, tools, and the response schema.

    Tokenize serialized model content, excluding tool artifacts. Include
    headroom for provider tokenizer differences and wire framing. This
    is an admission estimate, not measured provider token usage.
    """
    schema = getattr(
        request.response_format, "schema", request.response_format
    )
    if schema is not None and hasattr(schema, "model_json_schema"):
        schema = schema.model_json_schema()
    payload = {
        "system": (
            request.system_message.model_dump(mode="json")
            if request.system_message
            else None
        ),
        "messages": [
            m.model_dump(mode="json", exclude={"artifact"})
            for m in request.messages
        ],
        "tools": [convert_to_openai_tool(tool) for tool in request.tools],
        "response_format": schema,
        "tool_choice": request.tool_choice,
    }
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


def bounded_index(
    items: list[dict],
    count: int,
    *,
    key: str = "id",
    reader: str = "read_evidence",
) -> list[dict]:
    """Retain a prefix of evidence references and all non-evidence records.

    Omission concerns this input view only. The run-local evidence catalogue
    remains pageable, including references no longer displayed here.
    """
    references = [item for item in items if key in item]
    metadata = [item for item in items if key not in item]
    return [
        *references[:count],
        *metadata,
        {
            "context_omitted_count": len(references) - count,
            "retrieval": (
                f"{reader} pages retained references and reads full detail"
            ),
        },
    ]


def select_context(
    request: ModelRequest, allowance: int, *, coordinator_handoff: bool = False
) -> ContextSelection:
    """Reduce tool results oldest first to fit the input-only allowance.

    Use application-generated indexes, then bounded previews if authorized
    evidence retrieval is available. Preserve tool-call/result pairs and
    newly requested pages. Catalogue pages may use full compact indexes,
    but their references must stay reachable through the returned cursor.
    Return fits=False if reduction is insufficient. Original messages and
    stored evidence are not mutated; output space must be reserved by the
    caller before supplying allowance.
    """
    before = estimate_input(request)
    if before <= allowance:
        return ContextSelection(request, before, before, 0, True)

    messages = list(request.messages)
    last_call = max(
        (
            i
            for i, m in enumerate(messages)
            if isinstance(m, AIMessage) and m.tool_calls
        ),
        default=-1,
    )
    reduced = set()
    eligible = []
    after = before
    readers = {
        convert_to_openai_tool(tool)["function"]["name"]
        for tool in request.tools
    }
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
        if not (index > last_call and artifact.get("catalogue_page")):
            eligible.append(index)
        content = json_text(artifact["context_index"])
        view = list(messages)
        view[index] = message.model_copy(update={"content": content})
        candidate = request.override(messages=view)
        size = estimate_input(candidate)
        if size >= after:
            continue
        messages, after = view, size
        reduced.add(index)
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
                view[index] = message.model_copy(
                    update={
                        "content": json_text(
                            bounded_index(items, keep, key=key, reader=reader)
                        ),
                    }
                )
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
    if (
        coordinator_handoff
        and messages
        and isinstance(messages[0], HumanMessage)
    ):
        # This packet is created by the coordinator, not by user/tool text.
        # Measure the outer message escaping, schema and runtime budget too.
        message = messages[0]
        prefix, marker, content = message.text.rpartition(
            "\nCoordinator context:\n"
        )
        if marker:
            packet = json.loads(content)
            originals = packet["originals"]

            def handoff(keep: int) -> ModelRequest:
                view = {
                    **packet,
                    "originals": originals[:keep],
                    "omitted_original_refs": list(
                        dict.fromkeys(
                            [
                                *packet["omitted_original_refs"],
                                *(item["id"] for item in originals[keep:]),
                            ]
                        )
                    ),
                    "originals_reader_available": "read_evidence" in readers,
                    "originals_coverage": (
                        "Only displayed originals were supplied in "
                        "this message. "
                        "Omitted records remain retained, but their content "
                        "is not supplied here. Do not infer their content. "
                        "Use only inspected originals; state missing support."
                    ),
                }
                compact_messages = list(messages)
                compact_messages[0] = message.model_copy(
                    update={"content": prefix + marker + json_text(view)}
                )
                return request.override(messages=compact_messages)

            compact = handoff(0)
            if estimate_input(compact) < after:
                keep = 0
                low, high = 0, len(originals) - 1
                while low <= high:
                    middle = (low + high) // 2
                    probe = handoff(middle)
                    if estimate_input(probe) <= allowance:
                        keep, compact = middle, probe
                        low = middle + 1
                    else:
                        high = middle - 1
                messages = list(compact.messages)
                after = estimate_input(compact)
                reduced.add(0)
                omitted += len(originals) - keep
    return ContextSelection(
        request.override(messages=messages),
        before,
        after,
        len(reduced),
        after <= allowance,
        omitted,
    )
