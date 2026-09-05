"""Check evaluation isolation, not the quality of a scripted diagnosis."""

import asyncio
import importlib.util
import json
from pathlib import Path

from test_harness import ScriptedModel, tool_call
from tracemind.config import Settings


def test_fixed_evidence_evaluation_holds_out_rubric(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "claim_evaluation", root / "scripts/evaluate_claim_boundaries.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    fixture_path = root / "evals/claim-boundaries/cart-operation-scope.json"
    fixture = json.loads(fixture_path.read_text())

    def respond(messages, calls):
        content = "\n".join(m.text for m in messages)
        assert all(item not in content for item in fixture["rubric"])
        assert fixture["provenance"] not in content
        assert all(item["id"] in content for item in fixture["evidence"])
        assert model.bindings[-1] == ["DiagnosisDraft"]
        return tool_call("DiagnosisDraft", {
            "status": "inconclusive", "root_cause": "Unknown endpoint cause",
            "confidence": .3,
            "evidence_ids": [fixture["evidence"][0]["id"]],
        })

    model = ScriptedModel(respond=respond)
    monkeypatch.setattr(module, "create_chat_model", lambda _: model)
    monkeypatch.setattr(module, "Settings", lambda: Settings(_env_file=None))
    output = tmp_path / "result.json"
    asyncio.run(module.evaluate(fixture_path, output))
    result = json.loads(output.read_text())
    assert result["rubric"] == fixture["rubric"]
    assert result["mode"] == "fixed_evidence_real_model"
    assert result["run"]["state"]["tool_calls"] == {}
    assert result["run"]["state"]["model_calls"] == 1
