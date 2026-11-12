"""Resolve published prompts once; persisted content owns resume/revision."""

import asyncio
from contextlib import nullcontext
from hashlib import sha256

from langfuse import propagate_attributes

from tracemind.models.prompts import PromptSnapshot

INVESTIGATOR_CONTRACT = "single-agent-v18-evidence-timing-budget"
INVESTIGATOR = "tracemind/investigator"
VERIFIER = "tracemind/verifier"


def prompt_context(snapshot):
    return (propagate_attributes(prompt={
        "name": snapshot.name, "version": snapshot.version,
    }) if snapshot else nullcontext())


async def pin_prompt(state, role, *, client, label, contract, render=None):
    existing = state.prompts.get(role)
    if existing is not None:
        if existing.contract_version != contract:
            raise ValueError("prompt code contract changed; start a new run")
        return existing
    if label is None:
        return None
    # A pre-feature checkpoint cannot prove its original prompt contents.
    if state.model_calls or state.verification_history or state.verification:
        return None
    if client is None:
        raise ValueError("managed prompts require configured Langfuse")
    name = INVESTIGATOR if role == "investigator" else VERIFIER
    prompt = await asyncio.to_thread(
        client.get_prompt, name, label=label, type="text",
        cache_ttl_seconds=0, max_retries=0, fetch_timeout_seconds=5,
    )
    if prompt.config.get("contract_version") != contract:
        raise ValueError("published prompt has an incompatible code contract")
    content = render(prompt.prompt) if render else prompt.prompt
    snapshot = PromptSnapshot(
        name=name, version=prompt.version, label=label, content=content,
        sha256=sha256(content.encode()).hexdigest(), contract_version=contract,
    )
    state.prompts[role] = snapshot
    return snapshot
