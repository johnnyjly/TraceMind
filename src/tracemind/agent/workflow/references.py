"""Stable model-facing handles; original evidence remains unchanged."""

import re

from pydantic import BaseModel

from tracemind.models.run import RunState


def register_references(state: RunState) -> None:
    known = set(state.reference_map.values())
    for prefix, values in (
        ("E", state.evidence_ids),
        ("K", [r.id for r in state.knowledge]),
    ):
        number = sum(key.startswith(prefix) for key in state.reference_map)
        for value in values:
            original = str(value)
            if original not in known:
                number += 1
                state.reference_map[f"{prefix}{number}"] = original
                known.add(original)


def model_references(value, state: RunState):
    """Translate reference fields, never arbitrary telemetry text."""
    register_references(state)
    reverse = {v: k for k, v in state.reference_map.items()}
    keys = {
        "id",
        "evidence_id",
        "evidence_ids",
        "knowledge_ref",
        "knowledge_refs",
        "reference_id",
        "supporting_ids",
        "conflicting_ids",
    }

    def convert(item, reference=False):
        if isinstance(item, dict):
            return {k: convert(v, k in keys) for k, v in item.items()}
        if isinstance(item, list):
            return [convert(v, reference) for v in item]
        return reverse.get(str(item), item) if reference else item

    return convert(value)


def original_reference(value, state: RunState):
    return state.reference_map.get(value, value)


def normalize_reference_handles(output: BaseModel, available):
    """Repair an omitted prefix at the model-output boundary only.

    Preserve raw model messages and return a separate audit. No prose, dicts,
    IDs or numerical values are searched or rewritten. Membership is necessary
    for repair, not evidence that the referenced observation supports a claim.
    """
    fields = {
        "evidence_refs": "E", "counterevidence_refs": "E",
        "supporting_refs": "E", "conflicting_refs": "E",
        "knowledge_refs": "K",
    }
    repairs = []

    def visit(model, path):
        updates = {}
        for name in type(model).model_fields:
            value = getattr(model, name)
            location = f"{path}.{name}" if path else name
            if name in fields and isinstance(value, list):
                canonical = []
                for index, ref in enumerate(value):
                    target = fields[name] + ref if (
                        isinstance(ref, str)
                        and re.fullmatch(r"[1-9][0-9]*", ref)
                    ) else ref
                    if target != ref and target in available:
                        repairs.append({"path": f"{location}[{index}]",
                                        "raw": ref, "canonical": target})
                        canonical.append(target)
                    else:
                        canonical.append(ref)
                updates[name] = canonical
            elif isinstance(value, BaseModel):
                updates[name] = visit(value, location)
            elif isinstance(value, list):
                updates[name] = [
                    visit(item, f"{location}[{i}]")
                    if isinstance(item, BaseModel) else item
                    for i, item in enumerate(value)
                ]
        return model.model_copy(update=updates)

    return visit(output, ""), repairs
