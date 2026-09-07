"""Shared service configuration; entry points own resource lifetimes."""

from tracemind.agent.runtime.llm import (
    create_chat_model,
    create_fallback_model,
)
from tracemind.agent.memory.lookup import GuidanceLookup, MemoryLookup
from tracemind.agent.runtime.runner import AgentRunner, LangChainAgentRunner
from tracemind.agent.runtime.knowledge import KnowledgeSearch
from tracemind.agent.runtime.rag_model_factory import create_rag_completion
from tracemind.agent.service import DiagnosisService
from tracemind.agent.verification import DiagnosisVerifier
from tracemind.config import Settings
from tracemind.observability.langfuse import callbacks_for, client_for
from tracemind.mcp.client import ToolClient
from tracemind.models.memory import MemoryPolicy
from tracemind.models.run import RunLimits
from tracemind.models.knowledge import KnowledgePolicy
from tracemind.rag.config import RagSettings
from tracemind.rag.model_bridge import Completion


def build_runner(settings: Settings, *, checkpointer=None, callbacks=None):
    """Build the controlled graph with the configured provider clients."""
    return LangChainAgentRunner(
        create_chat_model(settings),
        fallback_model=create_fallback_model(settings),
        checkpointer=checkpointer,
        callbacks=[*(callbacks or []), *callbacks_for(settings)],
    )


def build_service(
    settings: Settings, runner: AgentRunner, tool_client: ToolClient, *,
    limits: RunLimits | None = None, memory_policy: MemoryPolicy | None = None,
    memory_lookup: MemoryLookup | None = None,
    guidance_lookup: GuidanceLookup | None = None,
    knowledge_policy: KnowledgePolicy | None = None,
    knowledge_search: KnowledgeSearch | None = None,
    knowledge_completion: Completion | None = None,
    knowledge_model_name: str = "rag-auxiliary",
    callbacks=None,
) -> DiagnosisService:
    """Assemble optional knowledge without extending the caller's grants."""
    all_callbacks = [*(callbacks or []), *callbacks_for(settings)]
    if settings.agent_knowledge_enabled:
        # Load the optional retrieval stack only when knowledge is enabled.
        rag = RagSettings()
        knowledge_policy = knowledge_policy or settings.knowledge_policy
        if knowledge_search is None:
            if rag.service_url is not None:
                from tracemind.rag.query_client import (
                    HttpSnapshotRetriever,
                )
                knowledge_search = HttpSnapshotRetriever(rag)
            else:
                from tracemind.rag.retriever import get_snapshot_retriever
                knowledge_search = get_snapshot_retriever(
                    rag.model_dump_json())
        knowledge_completion = (
            knowledge_completion or create_rag_completion(
                settings, rag,
                **({"callbacks": all_callbacks} if all_callbacks else {})))
        if knowledge_model_name == "rag-auxiliary":
            knowledge_model_name = rag.llm_model or settings.llm_model
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
        knowledge_policy=knowledge_policy,
        knowledge_search=knowledge_search,
        knowledge_completion=knowledge_completion,
        knowledge_model_name=knowledge_model_name,
        langfuse_client=client_for(settings),
        verification_policy=settings.verification_policy,
        verifier=DiagnosisVerifier(settings, callbacks=all_callbacks,
                                   client=client_for(settings)),
    )
