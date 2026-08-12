"""Environment-backed application settings and run-limit construction."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import AnyHttpUrl, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from tracemind.models.run import TOOL_NAMES, RunLimits, ToolName


class Settings(BaseSettings):
    """Load TRACEMIND_ settings from environment variables and .env."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="TRACEMIND_",
        extra="ignore",
    )

    environment: str = "development"
    prometheus_url: AnyHttpUrl = "http://localhost:9090"
    jaeger_url: AnyHttpUrl = "http://localhost:16686/jaeger/ui"
    opensearch_url: AnyHttpUrl = "http://localhost:9200"
    opensearch_logs_index: str = "otel-logs-*"
    llm_provider: Literal["google", "openai"] = "google"
    llm_model: str = "gemini-3.7-flash"
    google_api_key: SecretStr | None = None
    openai_api_key: SecretStr | None = None
    agent_recursion_limit: int = Field(default=12, ge=2, le=50)
    agent_max_model_calls: int = Field(default=8, ge=1)
    agent_max_tool_calls: int = Field(default=12, ge=1)
    agent_max_identical_tool_calls: int = Field(default=2, ge=1)
    agent_timeout_seconds: float = Field(
        default=120, gt=0, allow_inf_nan=False
    )
    agent_model_timeout_seconds: float = Field(
        default=60, gt=0, allow_inf_nan=False
    )
    agent_tool_timeout_seconds: float = Field(
        default=20, gt=0, allow_inf_nan=False
    )
    agent_allowed_tools: tuple[ToolName, ...] = TOOL_NAMES
    agent_trace_directory: Path | None = Path("artifacts/runs")

    @property
    def run_limits(self) -> RunLimits:
        """Build fresh execution limits from the configured agent controls."""
        return RunLimits(
            model_calls=self.agent_max_model_calls,
            tool_calls=self.agent_max_tool_calls,
            identical_tool_calls=self.agent_max_identical_tool_calls,
            duration_seconds=self.agent_timeout_seconds,
            model_timeout_seconds=self.agent_model_timeout_seconds,
            tool_timeout_seconds=self.agent_tool_timeout_seconds,
        )


@lru_cache
def get_settings() -> Settings:
    """Return cached application settings; clear the cache to reload them."""
    return Settings()
