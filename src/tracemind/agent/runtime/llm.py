"""Provider client construction and expected model-service error types."""

import httpx
from langchain_core.exceptions import ModelError
from langchain_core.language_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai.chat_models import ChatGoogleGenerativeAIError
from langchain_openai import ChatOpenAI
from langchain_openai.chat_models.base import BaseChatOpenAI
from openai import APIError

from tracemind.agent.errors import ModelConfigurationError
from tracemind.config import Settings


MODEL_SERVICE_ERRORS = (
    ModelError, APIError, ChatGoogleGenerativeAIError, httpx.HTTPError,
)


def create_chat_model(settings: Settings) -> BaseChatModel:
    """Construct the configured provider client without making a model call.

    Raise ModelConfigurationError when its API key is absent. SDK retries
    are disabled so the harness owns call accounting and timeouts.
    """
    if settings.llm_provider == "google":
        if (
            settings.google_api_key is None
            or not settings.google_api_key.get_secret_value()
        ):
            raise ModelConfigurationError(
                "TRACEMIND_GOOGLE_API_KEY is required "
                "when LLM provider is google"
            )
        return ChatGoogleGenerativeAI(
            model=settings.llm_model,
            api_key=settings.google_api_key,
            temperature=1.0,
            max_retries=0,
        )

    if settings.llm_provider == "deepseek":
        if (
            settings.deepseek_api_key is None
            or not settings.deepseek_api_key.get_secret_value()
        ):
            raise ModelConfigurationError(
                "TRACEMIND_DEEPSEEK_API_KEY is required "
                "when LLM provider is deepseek"
            )
        return BaseChatOpenAI(
            model=settings.llm_model,
            api_key=settings.deepseek_api_key,
            base_url="https://api.deepseek.com",
            use_responses_api=False,
            extra_body={"thinking": {"type": "disabled"}},
            profile={"tool_calling": True, "structured_output": False},
            max_retries=0,
        )

    if (
        settings.openai_api_key is None
        or not settings.openai_api_key.get_secret_value()
    ):
        raise ModelConfigurationError(
            "TRACEMIND_OPENAI_API_KEY is required when LLM provider is openai"
        )
    return ChatOpenAI(
        model=settings.llm_model,
        api_key=settings.openai_api_key,
        use_responses_api=True,
        max_retries=0,
    )


def create_fallback_model(settings: Settings) -> BaseChatModel | None:
    """Build an explicitly configured fallback using its own provider key."""
    if settings.agent_fallback_provider is None:
        return None
    return create_chat_model(settings.model_copy(update={
        "llm_provider": settings.agent_fallback_provider,
        "llm_model": settings.agent_fallback_model,
    }))
