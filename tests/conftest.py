"""Isolate settings and expose the independently packaged service."""
from pathlib import Path
from importlib.util import find_spec
import sys

import pytest

sys.path.insert(
    0,
    str(
        Path(__file__).resolve().parents[1]
        / 'services/otel-telemetry-mcp/src'
    ),
)


def pytest_addoption(parser):
    parser.addoption(
        "--require-rag", action="store_true",
        help="Require all RAG extras for the full acceptance suite")


def pytest_configure(config):
    if config.getoption("--require-rag"):
        missing = [name for name in (
            "docling", "ir_measures", "langchain_classic", "lightrag",
            "qdrant_client", "sentence_transformers", "pypdf",
        ) if find_spec(name) is None]
        if missing:
            raise pytest.UsageError(
                "Install RAG extras for this suite: " + ", ".join(missing))


@pytest.fixture(autouse=True)
def isolated_settings(monkeypatch):
    import os
    from tracemind.config import Settings, get_settings
    from tracemind.rag.config import RagSettings
    for key in tuple(os.environ):
        if key.startswith('TRACEMIND_'):
            monkeypatch.delenv(key)
    monkeypatch.setitem(Settings.model_config, 'env_file', None)
    monkeypatch.setitem(RagSettings.model_config, 'env_file', None)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
