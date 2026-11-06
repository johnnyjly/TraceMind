"""A Langfuse SDK evaluator for diagnosis support and held-out correctness."""

from pydantic import ValidationError

from tracemind.agent.runtime.llm import MODEL_SERVICE_ERRORS
from tracemind.agent.runtime.progress import model_progress_timeout
from tracemind.agent.errors import ModelCallBudgetExceeded
from tracemind.evaluation.evaluators import finding_scores, metric
from tracemind.evaluation.models import ExperimentOutput
from tracemind.observability.langfuse import callbacks_for

from tracemind.agent.verification import (
    VERSION,
    create_review_model,
    parse_review,
    review_input_bytes,
    review_input_tokens,
    review_messages,
    review_options,
)
from tracemind.models.verification import ReviewSubject


def diagnosis_judge(settings, client, budget):
    model = create_review_model(
        settings.model_copy(
            update={
                "llm_provider": settings.eval_judge_provider,
                "llm_model": settings.eval_judge_model,
                "deepseek_thinking": settings.eval_judge_deepseek_thinking,
                "deepseek_reasoning_effort": (
                    settings.eval_judge_reasoning_effort
                ),
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
        state = observed.run_state
        record = state.verification if state else None
        delivery = observed.diagnosis.triage_status is not None
        subject_kind = "delivery"
        diagnosis = observed.diagnosis
        if delivery:
            fields = {
                "id",
                "section",
                "text",
                "evidence_ids",
                "knowledge_refs",
                "source_refs",
            }
            if (
                record is None
                or record.candidate.incident_id != diagnosis.incident_id
                or [i.model_dump(include=fields) for i in diagnosis.items]
                != [i.model_dump() for i in record.items]
            ):
                return [
                    metric("judge_subject", "unavailable"),
                    metric("judge_completed", False,
                           "triage_delivery_has_no_matching_candidate"),
                ]
            # Delivery labels are application output, not new model claims.
            # Audit the immutable candidate without scoring its labels as facts.
            diagnosis = record.candidate
            items = record.items
            subject_kind = "candidate"
        else:
            items = (
                state.candidate_items if state and (
                    record is None
                    or (record.review and record.review.accepted)
                ) else []
            )
        subject = observed.verification_subject or ReviewSubject(
            incident=(
                observed.run_state.incident if observed.run_state else None
            ),
            purpose=(
                observed.run_state.purpose.value
                if observed.run_state
                else None
            ),
            user_message=(
                observed.run_state.user_message if observed.run_state else None
            ),
            initial_context=(state.initial_context if state else None),
            diagnosis=diagnosis,
            evidence=observed.evidence,
            items=items,
            start=observed.investigation_start,
            end=observed.investigation_end,
            sources=observed.sources or (state.sources if state else []),
            source_snapshot_id=(
                state.source_policy.snapshot_id
                if state and state.source_policy
                else None
            ),
            knowledge=observed.knowledge,
            execution_events=(
                [e.model_dump(mode="json") for e in observed.run_state.events]
                if observed.run_state
                else []
            ),
        )
        messages = review_messages(
            subject, expected_output, provider=settings.eval_judge_provider
        )
        if (
            review_input_bytes(messages, settings.eval_judge_provider)
            > settings.eval_judge_input_bytes
            or review_input_tokens(messages, settings.eval_judge_provider)
            + settings.eval_judge_output_tokens
            > settings.agent_context_tokens
        ):
            return [
                metric(
                    "judge_completed",
                    False,
                    "input_limit; no evidence dropped",
                )
            ]
        options = review_options(
            model,
            settings.eval_judge_provider,
            settings.eval_judge_output_tokens,
            subject,
            expected_output,
        )
        reference_repairs = []
        try:
            for attempt in range(2):
                if (
                    review_input_bytes(
                        messages, settings.eval_judge_provider
                    )
                    > settings.eval_judge_input_bytes
                    or review_input_tokens(
                        messages, settings.eval_judge_provider,
                    ) + settings.eval_judge_output_tokens
                    > settings.agent_context_tokens
                ):
                    raise ValueError("input_limit; no evidence dropped")
                async with model_progress_timeout(
                    settings.eval_judge_timeout_seconds,
                ):
                    response = await model.ainvoke(
                        messages,
                        config={
                            "callbacks": [budget, *callbacks_for(settings)]
                        },
                        **options,
                    )
                attempt_repairs = []
                try:
                    review = parse_review(
                        response, subject, expected_output,
                        reference_repairs=attempt_repairs,
                    )
                    break
                except ValueError as exc:
                    if attempt:
                        raise
                    feedback = (
                        "Correct the output contract: "
                        + str(exc)[:700]
                        + ". Assess each item once. With a held-out "
                        "reference correctness must be boolean; otherwise "
                        "null. Reassess fields and verdict against the "
                        "original assertion; do not change the verdict "
                        "merely to satisfy a field rule. "
                        "Return ItemReview."
                    )
                    messages = [
                        *messages,
                        {"role": "assistant", "content": response.text},
                        {"role": "user", "content": feedback},
                    ]
                finally:
                    reference_repairs.extend(
                        {"attempt": attempt + 1, **repair}
                        for repair in attempt_repairs
                    )
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
                    reference_repairs=reference_repairs,
                    rubric_version=VERSION,
                    judge_provider=settings.eval_judge_provider,
                    judge_model=settings.eval_judge_model,
                )
            ]
        verdict = review.verdict
        details = {
            "judge_subject": subject_kind,
            "semantic_scope": (
                "Original candidate, not the labelled final triage delivery."
                if delivery else "Delivered diagnosis."
            ),
            "reference_repairs": reference_repairs,
            "rubric_version": VERSION,
            "judge_provider": settings.eval_judge_provider,
            "judge_model": settings.eval_judge_model,
            "claims": [c.model_dump(mode="json") for c in review.claims],
            "completion": [c.model_dump(mode="json")
                           for c in review.completion]
            if review.completion is not None else None,
            "reference_version": (
                "demo-rca-v2"
                if review.reference_assessment
                else "service-or-answer-v1"
            ),
        }
        scores = [
            metric("judge_subject", subject_kind),
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
        if not (
            observed.run_state
            and observed.run_state.verification_policy is not None
        ):
            scores.extend(finding_scores(observed.diagnosis, review))
        if review.completion is not None:
            scores.append(metric("task_completed", review.task_completed))
        scores.append(
            metric(
                "false_completion",
                observed.diagnosis.status.value == "complete"
                and (
                    verdict != "supported"
                    or not review.uncertainty_appropriate
                    or review.root_cause_correctness is False
                    or (review.completion is not None
                        and not review.task_completed)
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
        if review.reference_assessment is not None:
            for key in type(review.reference_assessment).model_fields:
                component = getattr(review.reference_assessment, key)
                scores.append(
                    metric(
                        "root_cause_" + key,
                        component.matches,
                        component.rationale,
                        item_ids=component.item_ids,
                        reference_version="demo-rca-v2",
                    )
                )
        if review.process_quality is not None:
            scores.append(
                metric("process_quality", review.process_quality, **details)
            )
        return scores

    return evaluate
