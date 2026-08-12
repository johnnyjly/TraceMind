"""Deterministic model-input reduction; estimates are not provider usage."""

import json
from dataclasses import dataclass

from langchain.agents.middleware import ModelRequest
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.utils.function_calling import convert_to_openai_tool


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
    replaced tool results. A false fits value requires the caller to stop.
    """

    request: ModelRequest
    before: int
    after: int
    reduced_messages: int
    fits: bool


def select_context(request: ModelRequest, allowance: int) -> ContextSelection:
    """Reduce tool results oldest first to fit the input-only allowance.

    Use application-generated indexes while preserving evidence references
    and tool-call/result pairs. Keep newly requested detail pages intact.
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
    reduced = 0
    after = before
    # Oldest results first; never remove an assistant call or its response.
    # Keep every evidence ID, even if it means stopping at the context limit.
    for index, message in enumerate(messages):
        if not isinstance(message, ToolMessage):
            continue
        artifact = message.artifact
        if not isinstance(artifact, dict) or "context_index" not in artifact:
            continue
        # A requested detail page must reach the model intact this round.
        if index > last_call and artifact.get("detail_page"):
            continue
        content = json_text(artifact["context_index"])
        if len(content.encode()) >= len(str(message.content).encode()):
            continue
        messages[index] = message.model_copy(update={"content": content})
        reduced += 1
        candidate = request.override(messages=messages)
        after = estimate_input(candidate)
        if after <= allowance:
            return ContextSelection(candidate, before, after, reduced, True)
    return ContextSelection(
        request.override(messages=messages), before, after, reduced, False
    )
