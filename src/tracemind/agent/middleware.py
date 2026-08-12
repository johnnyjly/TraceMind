"""LangChain interception hooks for harness-controlled execution."""

from collections.abc import Awaitable, Callable

from langchain.agents.middleware import (
    AgentMiddleware,
    ModelRequest,
    ModelResponse,
)
from langchain_core.messages import SystemMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import Command

from tracemind.agent.context import select_context
from tracemind.agent.harness import RunHarness
from tracemind.models.run import StopReason


class HarnessMiddleware(AgentMiddleware):
    """Adapt LangChain model/tool hooks to a single run's harness controls."""

    def __init__(self, harness: RunHarness) -> None:
        """Bind all interception hooks to the supplied run harness."""
        self.harness = harness

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """Build a bounded request and execute through the run harness.

        Add remaining budgets, reduce tool results when needed, and reserve
        output space. Stop before model admission if the input cannot fit.
        Bind the provider output cap without mutating graph message history.
        """
        self.harness.check_active()
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
        selection = select_context(
            request, state.limits.context_tokens - state.limits.output_tokens
        )
        self.harness.emit(
            "context.selected", parent_span_id=state.id,
            data={"estimator": "utf8-bytes-v1",
                  "estimated_input_before": selection.before,
                  "estimated_input_after": selection.after,
                  "reduced_messages": selection.reduced_messages,
                  "output_reserved": state.limits.output_tokens,
                  "fits": selection.fits},
        )
        if not selection.fits:
            self.harness.stop(StopReason.CONTEXT_LIMIT)
        output_key = (
            "max_output_tokens"
            if isinstance(request.model, ChatGoogleGenerativeAI)
            else "max_tokens"
        )
        request = selection.request.override(model_settings={
            **request.model_settings, output_key: state.limits.output_tokens,
        })
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
