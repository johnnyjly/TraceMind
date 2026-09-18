"""Frozen, task-scoped handoffs over the shared immutable evidence store."""

from hashlib import sha256

from tracemind.agent.workflow.context import _evidence_preview, json_text
from tracemind.agent.workflow.references import model_references
from tracemind.models.collaboration import TaskInput
from tracemind.models.knowledge import knowledge_preview


def role_tools(tools, role):
    if role == "knowledge":
        return [t for t in tools if t.name in {
            "search_knowledge", "read_knowledge", "read_evidence"
        }]
    return [t for t in tools if t.name == "read_evidence"
            or (t.metadata or {}).get("tracemind_role") == "investigation"]


def freeze_task_input(state, key, task, evidence, allowed_tools):
    """Select explicit dependencies once, before either sibling starts."""
    hypotheses = [h for h in state.collaboration.hypotheses
                  if h.id in task.hypothesis_ids]
    refs = list(dict.fromkeys([
        *task.context_refs,
        *(r for h in hypotheses for r in
          h.evidence_refs + h.counterevidence_refs + h.knowledge_refs),
    ]))
    originals = {str(e.id): e for e in evidence}
    knowledge = {k.id: k for k in state.knowledge}
    records = []
    for ref in refs:
        identity = state.reference_map[ref]
        item = (originals[identity] if ref.startswith("E")
                else knowledge[identity])
        preview = (_evidence_preview(item) if ref.startswith("E")
                   else knowledge_preview(item))
        records.append({
            "ref": ref,
            "sha256": sha256(item.model_dump_json().encode()).hexdigest(),
            "preview": model_references(preview, state),
        })
    criteria = {f"C{i}": text for i, text in enumerate(
        task.acceptance_criteria or [task.question], 1)}
    packet = {
        "run_id": str(state.id), "task_id": key,
        "incident": state.incident.model_dump(mode="json"),
        "window": {"start": state.investigation_start.isoformat(),
                   "end": state.investigation_end.isoformat()},
        "goal": task.question, "acceptance_criteria": criteria,
        "role": task.role,
        "distinguishing_observations": task.distinguishing_observations,
        "hypotheses": [h.model_dump() for h in hypotheses],
        "context": records, "allowed_tools": list(allowed_tools),
        "context_policy": (
            "Hypotheses and task text are not incident facts. These are the "
            "explicit input references, not the whole evidence catalogue. "
            "Read other authorized run records only when needed. Return "
            "answered_criteria IDs, outcome and missing observations; "
            "never change the goal or criteria. Do not infer that an omitted "
            "observation is absent."
        ),
    }
    prompt = json_text(packet)
    return TaskInput(
        task_id=key, criteria=criteria, allowed_tools=tuple(allowed_tools),
        context_refs=tuple(refs), prompt=prompt,
        sha256=sha256(prompt.encode()).hexdigest(),
    )
