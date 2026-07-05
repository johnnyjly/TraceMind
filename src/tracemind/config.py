from functools import lru_cache

from pydantic import AnyHttpUrl
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


@lru_cache
def get_settings() -> Settings:
    return Settings()
