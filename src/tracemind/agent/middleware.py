from collections.abc import Awaitable, Callable
from langchain.agents.middleware import (
    AgentMiddleware,
    ModelRequest,
    ModelResponse,
)
from langchain_core.messages import SystemMessage, ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import Command

from tracemind.agent.harness import RunHarness


class HarnessMiddleware(AgentMiddleware):
    def __init__(self, harness: RunHarness) -> None:
        self.harness = harness

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        state = self.harness.state
        budget = (
            f"\nRemaining budget including the next call: "
            f"{state.limits.model_calls - state.model_calls} model calls, "
            f"{state.limits.tool_calls - sum(state.tool_calls.values())} "
            "tool calls. "
            "Propose an inconclusive diagnosis if the evidence "
            "is insufficient."
        )
        system = request.system_message.text if request.system_message else ""
        request = request.override(
            system_message=SystemMessage(content=system + budget)
        )
        model_name = getattr(request.model, "model_name", None) or getattr(
            request.model, "model", type(request.model).__name__
        )
        return await self.harness.model_call(
            lambda: handler(request), model_name=str(model_name)
        )

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]],
    ) -> ToolMessage | Command:
        # Also intercept hallucinated/hidden tool names,
        # before framework dispatch.
        self.harness.authorize_tool(request.tool_call["name"])
        result = await handler(request)
        if isinstance(result, ToolMessage) and result.status == "error":
            self.harness.emit(
                "tool.rejected",
                parent_span_id=self.harness.state.id,
                data={
                    "tool": request.tool_call["name"],
                    "reason": "invalid_arguments",
                },
            )
        return result
