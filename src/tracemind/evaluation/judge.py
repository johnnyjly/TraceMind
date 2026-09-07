"""A Langfuse SDK evaluator for diagnosis support and held-out correctness."""

import asyncio

from pydantic import ValidationError

from tracemind.agent.runtime.llm import MODEL_SERVICE_ERRORS, create_chat_model
from tracemind.agent.errors import ModelCallBudgetExceeded
from tracemind.evaluation.evaluators import metric
from tracemind.evaluation.models import ExperimentOutput
from tracemind.observability.langfuse import callbacks_for

from tracemind.agent.verification import (
    VERSION, parse_review, review_messages, review_options,
)
from tracemind.models.verification import ReviewSubject


def diagnosis_judge(settings, client, budget):
    model = create_chat_model(
        settings.model_copy(
            update={
                "llm_provider": settings.eval_judge_provider,
                "llm_model": settings.eval_judge_model,
                "agent_fallback_provider": None,
                "agent_fallback_model": None,
            }
        )
    )

    async def evaluate(*, output, expected_output=None, **kwargs):
        observed = ExperimentOutput.model_validate(output)
        if observed.diagnosis is None:
            return [
                metric("judge_applicable", False),
                metric("judge_completed", False),
            ]
        subject = ReviewSubject(
            diagnosis=observed.diagnosis, evidence=observed.evidence,
            start=observed.investigation_start, end=observed.investigation_end,
            knowledge=observed.knowledge,
            execution_events=(
                [e.model_dump(mode="json") for e in observed.run_state.events]
                if observed.run_state else []),
        )
        messages = review_messages(subject, expected_output)
        if sum(len(m["content"].encode())
               for m in messages) > settings.eval_judge_input_bytes:
            return [metric("judge_completed", False,
                           "input_limit; no evidence dropped")]
        options = review_options(model, settings.eval_judge_provider,
                                 settings.eval_judge_output_tokens)
        try:
            async with asyncio.timeout(settings.eval_judge_timeout_seconds):
                response = await model.ainvoke(
                    messages,
                    config={"callbacks": [budget, *callbacks_for(settings)]},
                    **options,
                )
            review = parse_review(response, subject, expected_output)
        except (
            TimeoutError,
            *MODEL_SERVICE_ERRORS,
            ValueError,
            ValidationError,
            ModelCallBudgetExceeded,
        ) as exc:
            reason = (
                "invalid_review_schema"
                if isinstance(exc, ValidationError)
                else (
                    str(exc)
                    if isinstance(exc, ValueError)
                    else type(exc).__name__
                )
            )
            return [
                metric(
                    "judge_completed",
                    False,
                    reason,
                    rubric_version=VERSION,
                    judge_provider=settings.eval_judge_provider,
                    judge_model=settings.eval_judge_model,
                )
            ]
        verdicts = {c.verdict for c in review.claims}
        verdict = (
            "contradicted"
            if "contradicted" in verdicts
            else "insufficient" if "insufficient" in verdicts else "supported"
        )
        details = {
            "rubric_version": VERSION,
            "judge_provider": settings.eval_judge_provider,
            "judge_model": settings.eval_judge_model,
            "claims": [c.model_dump(mode="json") for c in review.claims],
        }
        scores = [
            metric("judge_completed", True, **details),
            metric("support_verdict", verdict, review.rationale, **details),
            metric(
                "evidence_support",
                sum(c.verdict == "supported" for c in review.claims)
                / len(review.claims),
                **details,
            ),
            metric(
                "uncertainty_appropriate",
                review.uncertainty_appropriate,
                **details,
            ),
        ]
        scores.append(
            metric(
                "false_completion",
                observed.diagnosis.status.value == "complete"
                and (
                    verdict != "supported"
                    or not review.uncertainty_appropriate
                    or review.root_cause_correctness is False
                ),
                **details,
            )
        )
        if review.root_cause_correctness is not None:
            scores.append(
                metric(
                    "root_cause_correctness",
                    review.root_cause_correctness,
                    **details,
                )
            )
        if review.process_quality is not None:
            scores.append(
                metric("process_quality", review.process_quality, **details)
            )
        return scores

    return evaluate
