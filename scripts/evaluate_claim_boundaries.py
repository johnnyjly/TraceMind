"""Call the configured model on fixed evidence and review against a rubric.

This is a paid model evaluation, not scripted replay or fresh telemetry.
The rubric is saved with the result but never included in the model input.
"""

import argparse
import asyncio
import json
from pathlib import Path

from tracemind.agent.runtime.llm import create_chat_model
from tracemind.agent.runtime.runner import LangChainAgentRunner
from tracemind.agent.service import DiagnosisService
from tracemind.config import Settings
from tracemind.evaluation.replay import ReplayTelemetry
from tracemind.models import Evidence
from tracemind.models.run import RunPurpose, RunState


async def evaluate(path: Path, output: Path) -> None:
    fixture = json.loads(path.read_text())
    settings = Settings()
    state = RunState(
        incident=fixture["incident"], investigation_start=fixture["start"],
        investigation_end=fixture["end"], purpose=RunPurpose.INVESTIGATE,
        user_message="Diagnose the incident from the retained observations.",
        allowed_tools=(), limits=settings.run_limits.model_copy(update={
            "model_calls": 2, "diagnosis_repairs": 1,
        }),
    )
    evidence = [Evidence.model_validate(item) for item in fixture["evidence"]]
    service = DiagnosisService(
        LangChainAgentRunner(create_chat_model(settings)),
        tool_client=ReplayTelemetry([]), allowed_tools=(),
    )
    run = await service.run_persisted(
        state, evidence, persist=lambda *_: None, is_cancelled=lambda: False,
        resume=False, graph_resume=False,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({
        "fixture": fixture["id"], "mode": "fixed_evidence_real_model",
        "model": settings.llm_model, "rubric": fixture["rubric"],
        "assessment": "Manual semantic review required; not auto-scored.",
        "run": run.model_dump(mode="json"),
    }, indent=2) + "\n")
    print(run.diagnosis.model_dump_json(indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    asyncio.run(evaluate(args.fixture, args.output))
