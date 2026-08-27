FROM ghcr.io/astral-sh/uv:0.12.19-python3.12-trixie-slim
WORKDIR /app
COPY pyproject.toml uv.lock ./
COPY src ./src
RUN uv sync --frozen --no-dev --no-editable \
    && useradd --uid 10001 --create-home telemetry
USER telemetry
ENV TELEMETRY_MCP_HOST=0.0.0.0
EXPOSE 8012
CMD ["/app/.venv/bin/otel-telemetry-mcp"]
