"""Check the complete example configuration and local diagnostic profile."""

import re
from pathlib import Path

from tracemind.config import Settings
from tracemind.rag.config import RagSettings


def test_example_covers_all_settings_and_loads_diagnostic_profile():
    example = Path(__file__).resolve().parents[1] / ".env.example"
    entries = re.findall(r"^(?:# )?(TRACEMIND_\w+)=", example.read_text(),
                         re.MULTILINE)
    assert len(entries) == len(set(entries))
    assert set(entries) == {
        f"TRACEMIND_{name.upper()}" for name in Settings.model_fields
    } | {
        f"TRACEMIND_RAG_{name.upper()}" for name in RagSettings.model_fields
    }
    settings = Settings(_env_file=example)
    assert settings.agent_recursion_limit == 50
    assert settings.run_limits.context_tokens == 131072
    assert settings.run_limits.output_tokens == 8192
    assert settings.run_limits.duration_seconds == 300
    assert settings.run_limits.model_calls == 15
    assert settings.run_limits.tool_calls == 25
    assert settings.llm_provider == "deepseek"
    assert settings.llm_model == "deepseek-v4-pro"
    assert settings.agent_fallback_provider == "deepseek"
    assert settings.agent_fallback_model == "deepseek-flash"
    assert settings.eval_judge_provider == "deepseek"
    assert settings.eval_judge_model == "deepseek-v4-pro"
