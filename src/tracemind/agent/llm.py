import httpx
from langchain_core.exceptions import ModelError
from langchain_core.language_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai.chat_models import ChatGoogleGenerativeAIError
from langchain_openai import ChatOpenAI
from openai import APIError

from tracemind.agent.errors import ModelConfigurationError
from tracemind.config import Settings


MODEL_SERVICE_ERRORS = (ModelError, APIError, ChatGoogleGenerativeAIError, httpx.HTTPError)


def create_chat_model(settings: Settings) -> BaseChatModel:
    if settings.llm_provider == "google":
        if (
            settings.google_api_key is None
            or not settings.google_api_key.get_secret_value()
        ):
            raise ModelConfigurationError(
                "TRACEMIND_GOOGLE_API_KEY is required when LLM provider is google"
            )
        return ChatGoogleGenerativeAI(
            model=settings.llm_model,
            api_key=settings.google_api_key,
            temperature=1.0,
            # S1 owns admission and timeouts; S3 will add accounted retries.
            max_retries=0,
        )

    if settings.openai_api_key is None or not settings.openai_api_key.get_secret_value():
        raise ModelConfigurationError(
            "TRACEMIND_OPENAI_API_KEY is required when LLM provider is openai"
        )
    return ChatOpenAI(
        model=settings.llm_model,
        api_key=settings.openai_api_key,
        use_responses_api=True,
        max_retries=0,
    )
