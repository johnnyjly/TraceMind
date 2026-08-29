"""Resolve application-owned tool grants before model or tool dispatch."""

from collections.abc import Sequence

from tracemind.models.run import RunPurpose, ToolName


def resolve_tool_policy(
    allowed_tools: Sequence[ToolName], *,
    purpose: RunPurpose = RunPurpose.AUTOMATIC,
    agent_tools: Sequence[ToolName] | None = None,
) -> tuple[ToolName, ...]:
    """Narrow run grants by task and optional trusted Agent configuration.

    Agent grants can only remove tools from the application/run allowlist.
    Discovery and model output never supply grants. The concrete result is
    stored in RunState, so recovery does not expand permissions.
    """
    selected = dict.fromkeys(allowed_tools)
    if purpose is RunPurpose.EXPLAIN:
        selected = {name: None for name in selected if name == "read_evidence"}
    if agent_tools is not None:
        agent_allowed = set(agent_tools)
        selected = {name: None for name in selected if name in agent_allowed}
    return tuple(selected)
