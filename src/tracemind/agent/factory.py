"""Shared service configuration; entry points own resource lifetimes."""

from tracemind.agent.runtime.llm import (
    create_chat_model,
    create_fallback_model,
)
from tracemind.agent.memory.lookup import GuidanceLookup, MemoryLookup
from tracemind.agent.runtime.runner import AgentRunner, LangChainAgentRunner
from tracemind.agent.service import DiagnosisService
from tracemind.config import Settings
from tracemind.mcp.client import ToolClient
from tracemind.models.memory import MemoryPolicy
from tracemind.models.run import RunLimits


def build_runner(settings: Settings, *, checkpointer=None, callbacks=None):
    return LangChainAgentRunner(
        create_chat_model(settings),
        fallback_model=create_fallback_model(settings),
        checkpointer=checkpointer, callbacks=callbacks,
    )


def build_service(
    settings: Settings, runner: AgentRunner, tool_client: ToolClient, *,
    limits: RunLimits | None = None, memory_policy: MemoryPolicy | None = None,
    memory_lookup: MemoryLookup | None = None,
    guidance_lookup: GuidanceLookup | None = None,
) -> DiagnosisService:
    return DiagnosisService(
        runner,
        tool_client=tool_client,
        recursion_limit=settings.agent_recursion_limit,
        limits=limits or settings.run_limits,
        allowed_tools=settings.agent_allowed_tools,
        event_directory=settings.agent_trace_directory,
        memory_policy=memory_policy,
        memory_lookup=memory_lookup,
        guidance_lookup=guidance_lookup,
    )
