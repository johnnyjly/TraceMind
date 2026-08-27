"""Environment-backed application settings and run-limit construction."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import AnyHttpUrl, Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from tracemind.models.run import TOOL_NAMES, RunLimits, ToolName
from tracemind.models.memory import MemoryPolicy, MemoryScope


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
    telemetry_transport: Literal["direct", "mcp"] = "mcp"
    telemetry_mcp_url: AnyHttpUrl = "http://127.0.0.1:8012/mcp"
    telemetry_mcp_response_bytes: int = Field(default=16_777_216, ge=1024)
    llm_provider: Literal["google", "openai", "deepseek"] = "google"
    llm_model: str = "gemini-3.7-flash"
    google_api_key: SecretStr | None = None
    openai_api_key: SecretStr | None = None
    deepseek_api_key: SecretStr | None = None
    agent_fallback_provider: Literal[
        "google", "openai", "deepseek"
    ] | None = None
    agent_fallback_model: str | None = Field(default=None, min_length=1)
    agent_recursion_limit: int = Field(default=12, ge=2, le=50)
    agent_context_tokens: int = Field(default=32768, ge=1)
    agent_output_tokens: int = Field(default=4096, ge=1)
    agent_observation_bytes: int = Field(default=8_388_608, ge=1)
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
    agent_run_database: Path = Path("artifacts/runs.sqlite3")
    agent_checkpoint_database: Path = Path("artifacts/checkpoints.sqlite3")
    agent_retry_attempts: int = Field(default=2, ge=0, le=5)
    agent_retry_backoff_seconds: float = Field(
        default=0.25, ge=0, allow_inf_nan=False
    )
    agent_retry_max_backoff_seconds: float = Field(
        default=4, ge=0, allow_inf_nan=False
    )
    agent_diagnosis_repairs: int = Field(default=1, ge=0, le=3)
    agent_no_progress_rounds: int = Field(default=3, ge=1)
    agent_memory_enabled: bool = False
    agent_memory_environment: str | None = Field(
        default=None, min_length=1, max_length=120,
    )
    agent_memory_deployment_version: str | None = Field(
        default=None, min_length=1, max_length=120,
    )
    agent_memory_max_records: int = Field(default=3, ge=1, le=10)
    agent_memory_context_bytes: int = Field(default=4096, ge=512, le=16000)
    agent_memory_auto_expiry_days: int = Field(default=30, ge=1, le=365)

    @model_validator(mode="after")
    def validate_fallback(self) -> "Settings":
        """Require an explicit provider/model pair for opt-in fallback."""
        if (self.agent_fallback_provider is None) != (
            self.agent_fallback_model is None
        ):
            raise ValueError(
                "fallback provider and model must be set together"
            )
        return self

    @model_validator(mode="after")
    def validate_memory_scope(self) -> "Settings":
        """Require explicit deployment identity before enabling memory."""
        if self.agent_memory_enabled:
            self.memory_policy
        return self

    @property
    def memory_policy(self) -> MemoryPolicy | None:
        """Freeze an opt-in policy; disabled memory performs no DB lookup."""
        if not self.agent_memory_enabled:
            return None
        if (not self.agent_memory_environment or
                not self.agent_memory_deployment_version):
            raise ValueError(
                "memory requires environment and deployment version")
        return MemoryPolicy(
            scope=MemoryScope(environment=self.agent_memory_environment,
                              deployment_version=(
                                  self.agent_memory_deployment_version)),
            max_records=self.agent_memory_max_records,
            context_bytes=self.agent_memory_context_bytes,
            auto_expiry_days=self.agent_memory_auto_expiry_days,
        )

    @property
    def run_limits(self) -> RunLimits:
        """Build fresh execution limits from the configured agent controls."""
        return RunLimits(
            context_tokens=self.agent_context_tokens,
            output_tokens=self.agent_output_tokens,
            observation_bytes=self.agent_observation_bytes,
            model_calls=self.agent_max_model_calls,
            tool_calls=self.agent_max_tool_calls,
            identical_tool_calls=self.agent_max_identical_tool_calls,
            duration_seconds=self.agent_timeout_seconds,
            model_timeout_seconds=self.agent_model_timeout_seconds,
            tool_timeout_seconds=self.agent_tool_timeout_seconds,
            retry_attempts=self.agent_retry_attempts,
            retry_backoff_seconds=self.agent_retry_backoff_seconds,
            retry_max_backoff_seconds=self.agent_retry_max_backoff_seconds,
            diagnosis_repairs=self.agent_diagnosis_repairs,
            no_progress_rounds=self.agent_no_progress_rounds,
        )


@lru_cache
def get_settings() -> Settings:
    """Return cached application settings; clear the cache to reload them."""
    return Settings()
