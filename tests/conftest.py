"""Isolate settings and expose the independently packaged service."""
from pathlib import Path
import sys

import pytest

sys.path.insert(
    0,
    str(
        Path(__file__).resolve().parents[1]
        / 'services/otel-telemetry-mcp/src'
    ),
)


@pytest.fixture(autouse=True)
def isolated_settings(monkeypatch):
    import os
    from tracemind.config import Settings, get_settings
    for key in tuple(os.environ):
        if key.startswith('TRACEMIND_'):
            monkeypatch.delenv(key)
    monkeypatch.setitem(Settings.model_config, 'env_file', None)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
