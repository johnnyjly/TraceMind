"""Langfuse experiment tasks and acceptance; no parallel reporting backend."""

import asyncio
import json
from time import monotonic, sleep

from pydantic import AwareDatetime, BaseModel, ConfigDict

from tracemind.evaluation.evaluators import (
    acceptance_evaluator,
    diagnosis_scores,
    retrieval_scores,
    verification_scores,
)
from tracemind.evaluation.judge import diagnosis_judge
from tracemind.evaluation.models import ExperimentOutput
from tracemind.evaluation.runner import execute_agent
from tracemind.models import Incident


class AgentInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    incident: Incident
    start: AwareDatetime
    end: AwareDatetime


def input_of(item):
    return item["input"] if isinstance(item, dict) else item.input


def make_task(
    kind, settings, client, budget, *, rag_settings=None, memory_snapshot=None
):
    async def task(*, item, **kwargs):
        supplied = input_of(item)
        if kind == "saved":
            return ExperimentOutput.from_saved(supplied).model_dump(
                mode="json"
            )
        if kind == "verification":
            from tracemind.agent.errors import DiagnosisError
            from tracemind.agent.runtime.harness import RunHarness
            from tracemind.agent.verification import DiagnosisVerifier
            from tracemind.models.verification import ReviewSubject
            from tracemind.models.run import RunState, RunStatus, StopReason
            from tracemind.observability.langfuse import callbacks_for
            from tracemind.agent.workflow.validation import validate_draft
            from tracemind.models.result import DiagnosisDraft

            subject = ReviewSubject.model_validate(supplied)
            policy = settings.model_copy(
                update={"verifier_enabled": True}).verification_policy
            state = RunState(
                incident=subject.incident, investigation_start=subject.start,
                investigation_end=subject.end, limits=settings.run_limits,
                allowed_tools=(), knowledge=subject.knowledge,
                candidate_items=subject.items,
                verification_policy=policy,
            )
            harness = RunHarness(state)
            error = None
            terminal = RunStatus.FAILED
            reason = StopReason.INTERNAL_ERROR
            try:
                validate_draft(
                    DiagnosisDraft.model_validate(
                        subject.diagnosis.model_dump()),
                    subject.evidence, start=subject.start, end=subject.end,
                    knowledge=state.knowledge,
                )
                async with asyncio.timeout(harness.remaining_seconds):
                    await DiagnosisVerifier(
                        settings, callbacks=[budget, *callbacks_for(settings)],
                        client=client,
                    ).verify(
                        state, subject.diagnosis, subject.evidence, harness)
                terminal, reason = RunStatus.SUCCEEDED, StopReason.COMPLETED
            except asyncio.CancelledError:
                terminal, reason = RunStatus.CANCELLED, StopReason.CANCELLED
                raise
            except DiagnosisError as exc:
                error = type(exc).__name__
                reason = getattr(exc, "reason", StopReason.VERIFICATION_FAILED)
            except TimeoutError:
                error, reason = "TimeoutError", StopReason.DEADLINE
            finally:
                harness.finish(terminal, reason)
            return ExperimentOutput(
                diagnosis=subject.diagnosis, evidence=subject.evidence,
                run_state=state, error_type=error,
                investigation_start=subject.start,
                investigation_end=subject.end,
                knowledge=subject.knowledge,
                model_provider=policy.provider, model_name=policy.model,
            ).model_dump(mode="json")
        if kind == "agent":
            request = AgentInput.model_validate(supplied)
            result = await execute_agent(
                request.incident,
                settings,
                start=request.start,
                end=request.end,
                callbacks=[budget],
                memory_snapshot=memory_snapshot,
            )
            return result.model_dump(mode="json")
        from tracemind.rag.engine import RagRuntime
        from tracemind.rag.config import RagSettings
        from tracemind.observability.langfuse import callbacks_for
        from tracemind.agent.runtime.rag_model_factory import (
            create_rag_completion,
        )

        config = rag_settings or RagSettings()
        completion = create_rag_completion(
            settings, config, callbacks=[budget, *callbacks_for(settings)],
        )
        runtime = RagRuntime(
            config, completion=completion, langfuse_client=client
        )
        try:
            await runtime.initialize()
            started = monotonic()
            await runtime.search(supplied["query"], mode=supplied.get("mode"))
            return ExperimentOutput(
                kind="retrieval",
                retrieval=runtime.last_retrieval,
                elapsed_ms=(monotonic() - started) * 1000,
                model_provider=config.llm_provider or settings.llm_provider,
                model_name=config.llm_model or settings.llm_model,
            ).model_dump(mode="json")
        finally:
            await runtime.close()

    return task


def run_experiment(
    settings,
    client,
    data,
    *,
    kind,
    name,
    budget,
    judge=False,
    thresholds=None,
    rag_settings=None,
    memory_snapshot=None,
    run_name=None,
):
    if not data:
        raise ValueError("experiment dataset must not be empty")
    if kind in {"retrieval", "verification"} and judge:
        raise ValueError(
            "retrieval and verification tasks use their own evaluators; "
            "--judge applies to agent or saved diagnoses"
        )
    gates = {"execution_success": 1.0, **(thresholds or {})}
    if kind not in {"retrieval", "verification"}:
        gates.setdefault("citation_integrity", 1.0)
    if kind in {"agent", "verification"}:
        gates.setdefault("trace_integrity", 1.0)
        gates.setdefault("tool_budget", 1.0)
    if kind == "verification":
        gates.setdefault("review_completed", 1.0)
        gates.setdefault("expected_verdict", 1.0)
        gates.setdefault("expected_decision", 1.0)
    if judge:
        gates.setdefault("judge_completed", 1.0)
        gates.setdefault("evidence_support", 1.0)
        gates.setdefault("uncertainty_appropriate", 1.0)
    evaluators = [
        retrieval_scores if kind == "retrieval" else
        verification_scores if kind == "verification" else diagnosis_scores
    ]
    if judge:
        evaluators.append(diagnosis_judge(settings, client, budget))
    result = client.run_experiment(
        name=name,
        run_name=run_name,
        data=data,
        task=make_task(
            kind,
            settings,
            client,
            budget,
            rag_settings=rag_settings,
            memory_snapshot=memory_snapshot,
        ),
        evaluators=evaluators,
        run_evaluators=[
            acceptance_evaluator(len(data), gates, judge=judge, budget=budget,
                                 saved=kind == "saved")
        ],
        max_concurrency=1,
        metadata={
            "task_kind": kind,
            "model_call_cap": budget.limit,
            "model_calls_before_run": budget.calls,
            "investigator_model": settings.llm_model,
            "fallback_model": settings.agent_fallback_model or "off",
            "judge_model": settings.eval_judge_model if judge else "off",
            "memory_corpus": (
                memory_snapshot.fingerprint if memory_snapshot else "off"
            ),
            "gates": json.dumps(gates, sort_keys=True),
        },
    )
    # Local-data SDK experiments do not persist run-level scores to a dataset
    # run.
    if result.item_results and not result.dataset_run_id:
        for score in result.run_evaluations:
            client.create_score(
                trace_id=result.item_results[0].trace_id,
                name="experiment." + score.name,
                value=score.value,
                data_type=score.data_type,
                metadata={"experiment_id": result.experiment_id},
            )
    client.flush()
    return result


def verify_publication(client, result, *, timeout_seconds=30):
    """Require readable scores and observations, beyond a flushed queue."""
    pending = {
        item.trace_id: {e.name: e.value for e in item.evaluations}
        for item in result.item_results
    }
    if not pending or None in pending:
        return False
    summaries = {score.name: score.value for score in result.run_evaluations}
    if not result.dataset_run_id:
        pending[result.item_results[0].trace_id].update(
            {"experiment." + name: value for name, value in summaries.items()}
        )
        summaries = {}
    deadline = monotonic() + timeout_seconds
    while pending or summaries:
        for trace_id, required in list(pending.items()):
            response = client.api.scores_v3.get_many_v3(
                trace_id=trace_id, limit=100
            )
            available = {score.name: score.value for score in response.data}
            if all(
                name in available and available[name] == value
                for name, value in required.items()
            ):
                observations = client.api.observations.get_many(
                    trace_id=trace_id, limit=1, fields="core"
                )
                if observations.data:
                    del pending[trace_id]
        if summaries:
            response = client.api.scores_v3.get_many_v3(
                experiment_id=result.experiment_id, limit=100
            )
            available = {score.name: score.value for score in response.data}
            if all(
                name in available and available[name] == value
                for name, value in summaries.items()
            ):
                summaries = {}
        if not pending and not summaries:
            return True
        if monotonic() >= deadline:
            return False
        sleep(min(0.5, max(0, deadline - monotonic())))
    return True


def accepted(result):
    return any(
        score.name == "acceptance" and score.value is True
        for score in result.run_evaluations
    )
