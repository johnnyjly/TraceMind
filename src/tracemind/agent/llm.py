from langchain_core.language_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

from tracemind.agent.errors import ModelConfigurationError
from tracemind.config import Settings


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
        )

    if settings.openai_api_key is None or not settings.openai_api_key.get_secret_value():
        raise ModelConfigurationError(
            "TRACEMIND_OPENAI_API_KEY is required when LLM provider is openai"
        )
    return ChatOpenAI(
        model=settings.llm_model,
        api_key=settings.openai_api_key,
        use_responses_api=True,
    )
