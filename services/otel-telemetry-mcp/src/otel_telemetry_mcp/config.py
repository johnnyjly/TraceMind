"""Service settings independent of TraceMind and its model credentials."""

from pydantic import AnyHttpUrl, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Read only TELEMETRY_MCP_ environment variables; never load .env."""

    model_config = SettingsConfigDict(env_prefix="TELEMETRY_MCP_")

    prometheus_url: AnyHttpUrl = "http://localhost:9090"
    jaeger_url: AnyHttpUrl = "http://localhost:16686/jaeger/ui"
    opensearch_url: AnyHttpUrl = "http://localhost:9200"
    opensearch_logs_index: str = "otel-logs-*"
    timeout_seconds: float = Field(default=20, gt=0)
    response_bytes: int = Field(default=16_777_216, ge=1024)
    host: str = "127.0.0.1"
    port: int = Field(default=8012, ge=1, le=65535)
    allowed_hosts: list[str] = [
        "localhost:*", "127.0.0.1:*", "telemetry-mcp:*",
    ]
    allowed_origins: list[str] = ["http://localhost:*", "http://127.0.0.1:*"]
