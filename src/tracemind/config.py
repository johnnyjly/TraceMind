from functools import lru_cache
from typing import Literal

from pydantic import AnyHttpUrl, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="TRACEMIND_",
        extra="ignore",
    )

    environment: str = "development"
    prometheus_url: AnyHttpUrl = "http://localhost:9090"
    jaeger_url: AnyHttpUrl = "http://localhost:16686"
    opensearch_url: AnyHttpUrl = "http://localhost:9200"
    opensearch_logs_index: str = "otel-logs-*"
    llm_provider: Literal["google", "openai"] = "google"
    llm_model: str = "gemini-3.7-flash"
    google_api_key: SecretStr | None = None
    openai_api_key: SecretStr | None = None
    agent_recursion_limit: int = Field(default=12, ge=2, le=50)


@lru_cache
def get_settings() -> Settings:
    return Settings()
