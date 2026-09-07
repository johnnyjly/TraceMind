"""Assemble configured RAG model clients for the CLI and agent runtime."""

from tracemind.agent.runtime import llm
from tracemind.agent.errors import ModelConfigurationError
from tracemind.config import Settings
from tracemind.rag.config import RagSettings
from tracemind.rag.model_bridge import Completion


def create_rag_completion(
    settings: Settings, rag: RagSettings, *, callbacks=None,
) -> Completion:
    """Construct a lazy provider adapter with Harness-owned retries."""
    model = None
    configured = None
    if (rag.llm_provider is None) != (rag.llm_model is None):
        raise ModelConfigurationError(
            "RAG provider and model must be set together")
    configuration = {
        **settings.model_dump(),
        "llm_provider": rag.llm_provider or settings.llm_provider,
        "llm_model": rag.llm_model or settings.llm_model,
    }

    async def complete(messages, *, max_tokens, keyword_extraction):
        """Invoke the selected client with explicit output and JSON options."""
        nonlocal model, configured
        if model is None:
            configured = type(settings).model_validate(configuration)
            model = llm.create_chat_model(configured)
        target = model
        if keyword_extraction and configured.llm_provider in {
                "openai", "deepseek"}:
            target = target.bind(response_format={"type": "json_object"})
        return await target.ainvoke(
            messages, max_tokens=max_tokens,
            **({"config": {"callbacks": callbacks}} if callbacks else {}),
        )

    return complete
