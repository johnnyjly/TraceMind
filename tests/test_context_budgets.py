"""Role-specific context admission and fixed manual model profiles."""

import json

import pytest
from pydantic import ValidationError

from tracemind.agent.runtime.harness import RunHarness
from tracemind.config import Settings
from tracemind.models.run import RunState
from tracemind.models.verification import VerificationPolicy
from test_recovery import harness
from test_verifier import ReviewModel, diagnose, make_service, review


@pytest.mark.parametrize(
    'thinking,effort,agent_context,agent_output,context,output', [
    (False, 'low', 57344, 8192, 73728, 8192),
    (True, 'low', 65536, 16384, 98304, 32768),
    (True, 'high', 65536, 16384, 98304, 32768),
])
def test_manual_profiles_have_stable_input_ceilings(
    thinking, effort, agent_context, agent_output, context, output,
):
    settings = Settings(
        _env_file=None, deepseek_api_key='fixture-secret',
        deepseek_thinking=thinking, deepseek_reasoning_effort=effort,
        verifier_deepseek_thinking=thinking, verifier_reasoning_effort=effort,
        agent_context_tokens=agent_context, agent_output_tokens=agent_output,
        verifier_context_tokens=context, verifier_output_tokens=output,
    )
    profiles = settings.model_budgets
    assert profiles['investigator']['input_allowance'] == 49152
    assert profiles['verifier']['input_allowance'] == 65536
    for profile in profiles.values():
        assert profile['model'] == 'deepseek-flash'
        assert profile['reasoning_effort'] == (effort if thinking else 'none')
    assert 'fixture-secret' not in json.dumps(profiles)
    assert settings.agent_fallback_provider is None
    assert settings.agent_fallback_model is None


@pytest.mark.parametrize('role', ['agent', 'verifier'])
@pytest.mark.parametrize('output', [100, 101])
def test_invalid_context_configuration_fails_before_running(role, output):
    with pytest.raises(ValidationError, match=f'{role}_output_tokens'):
        Settings(_env_file=None, **{
            f'{role}_context_tokens': 100, f'{role}_output_tokens': output,
        })


def test_old_checkpoints_keep_their_original_context_and_new_ones_keep_override():
    run = harness(context_tokens=73728)
    run.state.verification_policy = VerificationPolicy(context_tokens=114688)
    payload = run.state.model_dump(mode='json')
    explicit = RunState.model_validate(payload)
    assert explicit.verification_policy.context_tokens == 114688
    del payload['verification_policy']['context_tokens']
    restored = RunState.model_validate(payload)
    assert restored.verification_policy.context_tokens == 73728
    assert RunState.model_validate_json(
        restored.model_dump_json()
    ).verification_policy.context_tokens == 73728


@pytest.mark.parametrize('mode,enabled,revise,reserve,finalization', [
    ('single', True, True, 425984, 557056),
    ('single', True, False, 196608, 327680),
    ('multi', True, True, 196608, 327680),
    ('single', False, False, 0, 131072),
])
def test_reserves_count_each_role_with_its_own_context(
    mode, enabled, revise, reserve, finalization,
):
    run = harness(model_calls=12, total_tokens=1_000_000)
    run.state.agent_mode = mode
    run.state.verification_policy = (
        VerificationPolicy(revise_on_rejection=revise) if enabled else None
    )
    assert run.verification_token_reserve == reserve
    assert run.investigation_token_pool == 1_000_000 - finalization


@pytest.mark.parametrize('context,attempts', [(65536, 0), (98304, 1)])
def test_review_admission_is_independent_of_investigator_context(
    monkeypatch, context, attempts,
):
    monkeypatch.setattr(
        'tracemind.agent.verification.review_input_tokens', lambda *args: 40000,
    )
    model = ReviewModel(body=review())
    service, _ = make_service(monkeypatch, model, policy=VerificationPolicy(
        context_tokens=context, revise_on_rejection=False,
    ))
    result = diagnose(service)
    assert result.state.limits.context_tokens == 65536
    assert model.calls == result.state.verification.attempts == attempts
    assert len(result.evidence) == 1
    event = next(e for e in result.state.events
                 if e.kind == 'verification.started')
    assert event.data['context_tokens'] == context
    assert event.data['input_allowance'] == context - 32768
    assert event.data['output_reserved'] == 32768
    assert result.state.verification.error_code == (
        None if attempts else 'context_limit'
    )


def test_format_repair_rechecks_verifier_context(monkeypatch):
    monkeypatch.setattr(
        'tracemind.agent.verification.review_input_tokens',
        lambda messages, provider: 40000 if len(messages) == 2 else 70000,
    )
    model = ReviewModel(body=review(), mode='repair')
    service, _ = make_service(monkeypatch, model)
    result = diagnose(service)
    assert result.state.verification.error_code == 'context_limit'
    assert model.calls == result.state.verification.attempts == 1
    assert len(result.evidence) == 1


def test_review_protects_revision_tokens_with_cumulative_cap(monkeypatch):
    original = RunHarness.model_call
    seen = []

    async def capture(self, *args, **kwargs):
        if kwargs.get('purpose') == 'verification':
            seen.append(kwargs['protected_tokens'])
        return await original(self, *args, **kwargs)

    monkeypatch.setattr(RunHarness, 'model_call', capture)
    model = ReviewModel(body=review())
    service, _ = make_service(
        monkeypatch, model, calls=12, policy=VerificationPolicy(),
    )
    service._limits.total_tokens = 1_000_000
    result = diagnose(service)
    assert result.state.verification.status == 'completed'
    assert seen == [229376]
