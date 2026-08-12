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
    """Adapt LangChain model/tool hooks to a single run's harness controls."""

    def __init__(self, harness: RunHarness) -> None:
        self.harness = harness

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """Append remaining budgets and execute through the run harness.

        Override the request's system message without mutating graph
        message history. The harness owns model-call admission, timeout
        handling, and usage recording.
        """
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
        """Authorize dispatch and trace framework argument errors.

        The tool executor performs budget reservation and rechecks
        authorization. Framework validation errors are returned to the
        agent for bounded repair.
        """
        # Also intercept hallucinated/hidden tool names before dispatch.
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
