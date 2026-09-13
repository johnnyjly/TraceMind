"""Environment-backed application settings and run-limit construction."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import AnyHttpUrl, Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from tracemind.models.memory import MemoryPolicy, MemoryScope
from tracemind.models.run import RunLimits, TOOL_NAMES, ToolName


class Settings(BaseSettings):
    """Load TRACEMIND_ settings from environment variables and .env."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="TRACEMIND_",
        extra="ignore",
    )

    environment: str = "development"
    langfuse_enabled: bool = False
    langfuse_base_url: AnyHttpUrl = "http://127.0.0.1:3000"
    langfuse_public_key: str | None = None
    langfuse_secret_key: SecretStr | None = None
    eval_judge_provider: Literal["google", "openai", "deepseek"] = "deepseek"
    eval_judge_model: str = "deepseek-v4-pro"
    eval_judge_timeout_seconds: float = Field(default=60, gt=0)
    eval_judge_input_bytes: int = Field(default=65536, ge=1)
    eval_judge_output_tokens: int = Field(default=4096, ge=1)
    verifier_enabled: bool = True
    verifier_provider: Literal["google", "openai", "deepseek"] = "deepseek"
    verifier_model: str = Field(default="deepseek-v4-pro", min_length=1)
    verifier_timeout_seconds: float = Field(
        default=60, gt=0, allow_inf_nan=False)
    verifier_input_bytes: int = Field(default=131072, ge=1)
    verifier_output_tokens: int = Field(default=4096, ge=1)
    telemetry_transport: Literal["mcp"] = "mcp"
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
    agent_knowledge_enabled: bool = False
    agent_knowledge_snapshot: str | None = Field(
        default=None, pattern=r"^[A-Za-z0-9_-]+$",
    )
    agent_knowledge_max_references: int = Field(default=20, ge=1, le=100)
    agent_knowledge_retention_bytes: int = Field(default=262144, ge=1)
    agent_knowledge_context_bytes: int = Field(default=8192, ge=64)
    agent_knowledge_max_queries: int = Field(default=3, ge=1, le=3)
    agent_knowledge_result_limit: int = Field(default=5, ge=1, le=20)
    agent_knowledge_model_output_tokens: int = Field(default=2048, ge=1)
    agent_knowledge_reserved_model_calls: int = Field(default=1, ge=1)
    agent_memory_environment: str | None = Field(
        default=None, min_length=1, max_length=120,
    )
    agent_memory_deployment_version: str | None = Field(
        default=None, min_length=1, max_length=120,
    )
    agent_memory_max_records: int = Field(default=3, ge=1, le=10)
    agent_memory_context_bytes: int = Field(default=4096, ge=512, le=16000)
    agent_memory_auto_expiry_days: int = Field(default=30, ge=1, le=365)

    @property
    def verification_policy(self):
        from tracemind.models.verification import VerificationPolicy
        if not self.verifier_enabled:
            return None
        return VerificationPolicy(
            provider=self.verifier_provider, model=self.verifier_model,
            timeout_seconds=self.verifier_timeout_seconds,
            input_bytes=self.verifier_input_bytes,
            output_tokens=self.verifier_output_tokens,
        )

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
        if self.agent_knowledge_enabled:
            self.knowledge_policy
        return self

    @property
    def knowledge_policy(self):
        """Freeze online run limits independently of CLI configuration."""
        from tracemind.models.knowledge import KnowledgePolicy

        if not self.agent_knowledge_enabled:
            return None
        if not self.agent_knowledge_snapshot:
            raise ValueError(
                "online knowledge requires a published snapshot ID")
        return KnowledgePolicy(
            snapshot=self.agent_knowledge_snapshot,
            max_references=self.agent_knowledge_max_references,
            retention_bytes=self.agent_knowledge_retention_bytes,
            context_bytes=self.agent_knowledge_context_bytes,
            max_queries=self.agent_knowledge_max_queries,
            reserved_model_calls=self.agent_knowledge_reserved_model_calls,
            result_limit=self.agent_knowledge_result_limit,
            model_output_tokens=self.agent_knowledge_model_output_tokens,
        )

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
