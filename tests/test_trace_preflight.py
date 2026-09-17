"""Expired trace windows never reach paid model admission."""

import asyncio
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tracemind.evaluation.preflight import verify_captured_traces

TRACE = "a" * 32
SPAN = {
    "source": "trace", "summary": "sample", "service": "shipping",
    "observed_at": "2026-10-15T08:42:00Z",
    "attributes": {
        "trace_id": TRACE,
        "coverage": {"trace_fingerprint": "hash", "trace_span_count": 53},
    },
}


class Client:
    def __init__(self, mode):
        self.mode = mode
        self.calls = []

    async def discover_tools(self):
        return []

    async def call(self, name, arguments):
        self.calls.append((name, arguments))
        span = json.loads(json.dumps(SPAN))
        if self.mode == "changed":
            span["attributes"]["coverage"]["trace_fingerprint"] = "different"
        if self.mode == "wrong_id":
            span["attributes"]["trace_id"] = "b" * 32
        body = {"evidence": [span]}
        if self.mode == "empty":
            body = {"evidence": []}
        if self.mode == "metadata":
            body = {"metadata": {"kind": "trace_detail_coverage"}}
        if self.mode == "404":
            body = {"error": {"category": "not_found", "status_code": 404,
                              "retryable": False}}
        return SimpleNamespace(
            structuredContent=body, isError=self.mode == "404",
        )


def test_current_full_trace_identity_passes_with_one_read_per_trace():
    client = Client("ok")
    result = asyncio.run(verify_captured_traces(
        client, {"traces": [SPAN, SPAN]},
    ))
    assert result == {"checked_traces": 1, "fingerprints_match": True}
    assert client.calls == [
        ("get_trace_detail", {"trace_id": TRACE, "limit": 1}),
    ]


@pytest.mark.parametrize(
    "mode", ["404", "empty", "metadata", "changed", "wrong_id"],
)
def test_invalid_live_window_is_rejected(mode):
    client = Client(mode)
    with pytest.raises(ValueError):
        asyncio.run(verify_captured_traces(client, {"traces": [SPAN]}))
    assert len(client.calls) == 1


def test_empty_capture_is_rejected_before_backend_queries():
    client = Client("ok")
    with pytest.raises(ValueError, match="no traces"):
        asyncio.run(verify_captured_traces(client, {"traces": []}))
    assert not client.calls


@pytest.mark.parametrize("mode", ["404", "empty", "metadata", "changed"])
def test_trial_main_never_admits_models_for_invalid_live_window(
    monkeypatch, tmp_path, mode,
):
    spec = importlib.util.spec_from_file_location(
        "evaluate_diagnosis", Path("scripts/evaluate_diagnosis.py"),
    )
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    data = tmp_path / "input.json"
    data.write_text("[]")
    (tmp_path / "preflight.json").write_text(json.dumps({
        "reproduced": True,
        "input_sha256": sha256(data.read_bytes()).hexdigest(),
    }))
    (tmp_path / "telemetry.json").write_text(json.dumps({"traces": [SPAN]}))
    profile = tmp_path / "profile.env"
    profile.write_text("")
    env = tmp_path / ".env"
    env.write_text("")
    output = tmp_path / "trial.json"
    monkeypatch.setattr("sys.argv", [
        "evaluate_diagnosis.py", "--data", str(data), "--output", str(output),
        "--profile", str(profile), "--env-file", str(env), "--mode", "multi",
    ])

    def forbidden(*args, **kwargs):
        pytest.fail("expired window admitted models or created an experiment")

    async def check(url, captured):
        return await verify_captured_traces(Client(mode), captured)

    monkeypatch.setattr(script, "verify_live_window", check)
    monkeypatch.setattr(script, "UsageBudget", forbidden)
    monkeypatch.setattr(script, "client_for", forbidden)
    monkeypatch.setattr(script, "run_experiment", forbidden)
    with pytest.raises(ValueError):
        script.main()
    assert not output.exists()
    assert not output.with_suffix(".ledger.sqlite3").exists()
