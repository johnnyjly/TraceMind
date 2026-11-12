"""Shared local settings, explicit overlays and secret-safe trial profiles."""

import asyncio
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tracemind.api.app import get_request_settings
from tracemind.agent.worker import RunWorker
from tracemind.config import Settings, get_settings, settings_files
from tracemind.rag.config import RagSettings


ROOT = Path(__file__).resolve().parents[1]


def script(name):
    spec = importlib.util.spec_from_file_location(
        name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def local_env(tmp_path, monkeypatch):
    path = tmp_path / ".env"
    path.write_text(
        "TRACEMIND_LLM_PROVIDER=deepseek\n"
        "TRACEMIND_LLM_MODEL=shared-test-model\n"
        "TRACEMIND_DEEPSEEK_API_KEY=fixture-key-no-access\n"
        "TRACEMIND_AGENT_TIMEOUT_SECONDS=611\n"
        "TRACEMIND_AGENT_MAX_MODEL_CALLS=7\n"
        "TRACEMIND_AGENT_MAX_TOOL_CALLS=9\n"
        "TRACEMIND_VERIFIER_REASONING_EFFORT=low\n"
        "TRACEMIND_AGENT_RUN_DATABASE=artifacts/shared-runs.sqlite3\n"
        "TRACEMIND_RAG_DIRECTORY=artifacts/shared-rag\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setitem(Settings.model_config, "env_file", Path(".env"))
    monkeypatch.setitem(RagSettings.model_config, "env_file", Path(".env"))
    return path


def test_api_worker_and_release_lab_share_all_base_settings(local_env):
    lab = script("release_source_lab")
    args = SimpleNamespace(env_file=None, mcp_url=None, credential_file=None)
    api = asyncio.run(get_request_settings())
    worker = RunWorker(get_settings())
    acceptance = lab.evaluation_settings(args)
    assert api is worker.settings
    assert acceptance == api
    assert api.llm_model == "shared-test-model"
    assert api.agent_timeout_seconds == 611
    assert api.verifier_reasoning_effort == "low"
    assert api.agent_run_database.resolve() == (
        local_env.parent / "artifacts/shared-runs.sqlite3")


def test_overlay_keeps_key_and_process_environment_takes_precedence(
    local_env, monkeypatch,
):
    overlay = local_env.parent / "experiment.env"
    overlay.write_text("TRACEMIND_AGENT_MAX_MODEL_CALLS=11\n"
                       "TRACEMIND_RAG_DIRECTORY=artifacts/experiment-rag\n")
    configured = Settings(_env_file=settings_files([overlay]))
    assert configured.deepseek_api_key == Settings().deepseek_api_key
    assert configured.agent_max_model_calls == 11
    assert configured.llm_model == "shared-test-model"
    rag = RagSettings(_env_file=settings_files([overlay]))
    assert rag.directory == Path("artifacts/experiment-rag")
    monkeypatch.setenv("TRACEMIND_AGENT_MAX_MODEL_CALLS", "13")
    configured = Settings(_env_file=settings_files([overlay]))
    assert configured.agent_max_model_calls == 13
    assert Settings(_env_file=settings_files([overlay]),
                    agent_max_model_calls=15).agent_max_model_calls == 15


def test_test_isolation_does_not_reenable_workstation_env():
    assert settings_files() == []
    assert Settings(_env_file=settings_files()).deepseek_api_key is None


@pytest.mark.parametrize("name", ["run_experiment", "evaluate_diagnosis"])
def test_evaluation_cli_uses_shared_settings_without_old_credential_file(
    name, local_env, monkeypatch,
):
    module = script(name)
    data = local_env.parent / "input.json"
    data.write_text("[]")
    captured = []

    class StopBeforeModels(Exception):
        pass

    def client(settings, **kwargs):
        captured.append(settings)
        raise StopBeforeModels

    monkeypatch.setattr(module, "client_for", client)
    argv = [name, "--data", str(data), "--max-model-calls", "1"]
    if name == "run_experiment":
        argv += ["--task", "agent", "--name", "configuration-check"]
    else:
        (local_env.parent / "preflight.json").write_text(json.dumps({
            "reproduced": True,
            "input_sha256": sha256(data.read_bytes()).hexdigest(),
        }))
        (local_env.parent / "telemetry.json").write_text("{}")

        async def preflight(*args):
            return {"checked_traces": 1}

        monkeypatch.setattr(module, "verify_live_window", preflight)
        argv += ["--output", str(local_env.parent / "trial/result.json")]
    monkeypatch.setattr("sys.argv", argv)
    with pytest.raises(StopBeforeModels):
        module.main()
    configured = captured[0]
    base = get_settings()
    assert configured.llm_model == base.llm_model
    assert configured.deepseek_api_key == base.deepseek_api_key
    assert configured.run_limits == base.run_limits
    assert configured.verification_policy == base.verification_policy
    assert configured.agent_mode == base.agent_mode
    assert not configured.langfuse_enabled


def test_release_lab_explicit_overrides_do_not_replace_model_budget(local_env):
    lab = script("release_source_lab")
    legacy = local_env.parent / "legacy.env"
    legacy.write_text("OPS_PILOT_DEEPSEEK_API_KEY=legacy-fixture-key\n"
                      "OPS_PILOT_AGENT_TIMEOUT_SECONDS=1\n")
    args = SimpleNamespace(env_file=None,
                           mcp_url="http://127.0.0.1:9999/mcp",
                           credential_file=legacy)
    configured = lab.evaluation_settings(args)
    assert str(configured.telemetry_mcp_url) == args.mcp_url
    assert configured.agent_timeout_seconds == 611
    assert configured.llm_model == "shared-test-model"
    assert configured.deepseek_api_key != Settings().deepseek_api_key


@pytest.mark.parametrize("field", [
    "TRACEMIND_DEEPSEEK_API_KEY", "TRACEMIND_LANGFUSE_SECRET_KEY",
    "TRACEMIND_deepseek_api_key",
])
def test_profile_rejects_credentials_without_exporting_them(
    field, tmp_path, monkeypatch,
):
    import os
    module = script("evaluate_diagnosis")
    path = tmp_path / "unsafe.env"
    path.write_text("TRACEMIND_AGENT_TIMEOUT_SECONDS=123\n"
                    + field + " = fixture-secret\n")
    with pytest.raises(ValueError, match="credentials belong") as error:
        module.load_profile(path)
    assert "fixture-secret" not in str(error.value)
    assert field not in os.environ
    assert "TRACEMIND_AGENT_TIMEOUT_SECONDS" not in os.environ


def test_source_lab_rejects_multi_before_capture_or_model_calls(
    local_env, monkeypatch,
):
    lab = script("release_source_lab")
    monkeypatch.setenv("TRACEMIND_AGENT_MODE", "multi")
    args = SimpleNamespace(env_file=None, mcp_url=None, credential_file=None,
                           output=local_env.parent / "not-created")
    with pytest.raises(ValueError, match="single-agent mode"):
        asyncio.run(lab.evaluate(args))
    assert not args.output.exists()
