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
    assert settings.run_limits.context_tokens == 65536
    assert settings.run_limits.output_tokens == 16384
    assert settings.run_limits.duration_seconds == 600
    assert settings.run_limits.model_calls == 12
    assert settings.run_limits.tool_calls == 25
    assert settings.llm_provider == "deepseek"
    assert settings.llm_model == "deepseek-flash"
    assert settings.agent_fallback_provider is None
    assert settings.agent_fallback_model is None
    assert settings.eval_judge_provider == "deepseek"
    assert settings.eval_judge_model == "deepseek-v4-pro"

    assert settings.verifier_enabled
    assert settings.verification_policy.revise_on_rejection
    assert settings.verification_policy is not None
    assert settings.verification_policy.deepseek_reasoning_effort == "low"
    assert settings.verification_policy.model == "deepseek-flash"
    assert settings.verification_policy.context_tokens == 98304
    assert settings.verification_policy.output_tokens == 32768
    assert settings.verification_policy.timeout_seconds == 180
    assert settings.deepseek_reasoning_effort == "low"
    assert settings.eval_judge_reasoning_effort == "high"


def test_verifier_defaults_on_and_can_be_explicitly_disabled(monkeypatch):
    assert Settings(_env_file=None).verification_policy is not None
    monkeypatch.setenv("TRACEMIND_VERIFIER_ENABLED", "false")
    settings = Settings(_env_file=None)
    assert not settings.verifier_enabled
    assert settings.verification_policy is None


def test_token_defaults_match_direct_policies_and_leave_discovery_budget():
    from tracemind.agent.runtime.harness import RunHarness
    from tracemind.models import Incident
    from tracemind.models.run import RunLimits, RunState
    from tracemind.models.verification import VerificationPolicy
    from tracemind.models.knowledge import KnowledgePolicy
    from datetime import UTC, datetime

    settings = Settings(_env_file=None)
    assert settings.run_limits.context_tokens == RunLimits().context_tokens
    assert settings.run_limits.output_tokens == RunLimits().output_tokens
    assert settings.verification_policy.output_tokens == (
        VerificationPolicy().output_tokens
    )
    assert settings.verification_policy.input_bytes == (
        VerificationPolicy().input_bytes
    )
    assert settings.agent_knowledge_model_output_tokens == (
        KnowledgePolicy(snapshot="test").model_output_tokens
    )
    assert settings.verification_policy.deepseek_reasoning_effort == (
        VerificationPolicy().deepseek_reasoning_effort
    ) == "low"
    assert settings.run_limits.total_tokens is None
    state = RunState(
        incident=Incident(title="test", description="budget admission"),
        investigation_start=datetime.now(UTC),
        investigation_end=datetime.now(UTC), allowed_tools=(),
        limits=settings.run_limits.model_copy(update={"total_tokens": 4000000}),
        verification_policy=settings.verification_policy,
    )
    harness = RunHarness(state)
    assert harness.investigation_token_pool >= state.limits.context_tokens


def test_optional_total_deadline_from_environment(monkeypatch):
    import pytest
    from pydantic import ValidationError

    monkeypatch.setenv("TRACEMIND_AGENT_TIMEOUT_SECONDS", "none")
    assert Settings(_env_file=None).run_limits.duration_seconds is None
    monkeypatch.setenv("TRACEMIND_AGENT_TIMEOUT_SECONDS", "180")
    assert Settings(_env_file=None).run_limits.duration_seconds == 180
    for invalid in ("0", "-1", "inf", "nan"):
        monkeypatch.setenv("TRACEMIND_AGENT_TIMEOUT_SECONDS", invalid)
        with pytest.raises(ValidationError):
            Settings(_env_file=None)
