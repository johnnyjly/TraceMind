"""Stable model-facing handles; original evidence remains unchanged."""

from tracemind.models.run import RunState


def register_references(state: RunState) -> None:
    known = set(state.reference_map.values())
    for prefix, values in (("E", state.evidence_ids),
                           ("K", [r.id for r in state.knowledge])):
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
    keys = {"id", "evidence_id", "evidence_ids", "knowledge_ref",
            "knowledge_refs", "reference_id"}

    def convert(item, reference=False):
        if isinstance(item, dict):
            return {k: convert(v, k in keys) for k, v in item.items()}
        if isinstance(item, list):
            return [convert(v, reference) for v in item]
        return reverse.get(str(item), item) if reference else item

    return convert(value)


def original_reference(value, state: RunState):
    return state.reference_map.get(value, value)
