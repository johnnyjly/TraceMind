"""LangChain interception hooks for harness-controlled execution."""

from collections.abc import Awaitable, Callable
import json

from langchain.agents.middleware import (
    AgentMiddleware,
    ModelRequest,
    ModelResponse,
)
from langchain.agents.structured_output import (
    MultipleStructuredOutputsError,
    StructuredOutputError,
)
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    ToolMessage,
)
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import Command
from pydantic import ValidationError

from tracemind.agent.workflow.context import (
    estimate_input, json_text, select_context,
)
from tracemind.agent.workflow.validation import resolve_proposal
from tracemind.agent.workflow.references import (
    normalize_reference_handles, register_references,
)
from tracemind.agent.errors import DiagnosisError, DiagnosisValidationError
from tracemind.agent.runtime.harness import (
    InvestigationClosed, RunHarness, TaskToolBudgetExhausted,
)
from tracemind.agent.runtime.recovery import ModelCallFailed, ToolUnavailable
from tracemind.models.run import StopReason


class TaskBudgetExhausted(Exception):
    """End a specialist without consuming synthesis or sibling reserves."""


class TaskReportInvalid(Exception):
    """A specialist exhausted repairs; its siblings and evidence survive."""


class TaskContextExceeded(Exception):
    """Only this specialist's request cannot fit; other tasks may finish."""


class HarnessMiddleware(AgentMiddleware):
    """Adapt LangChain model/tool hooks to a single run's harness controls."""

    def __init__(
        self,
        harness: RunHarness,
        fallback_model: BaseChatModel | None = None,
        *,
        output_name: str = "DiagnosisProposal",
        validator=None,
        role: str = "diagnosis",
        reserved_calls: int = 0,
        investigation: bool = True,
        allowed_tools: tuple[str, ...] | None = None,
    ) -> None:
        """Bind all interception hooks to the supplied run harness."""
        self.harness = harness
        self._fallback = fallback_model
        self._using_fallback = False
        self._output_name = output_name
        self._validator = validator or (
            self._validate_proposal
            if output_name == "DiagnosisProposal"
            else None
        )
        self._role = role
        self._reserve_policy = reserved_calls
        self._local_finalization = None
        self._last_task_progress = None
        self._stalled = 0
        self._investigation = investigation
        self._allowed_tools = allowed_tools

    @property
    def _reserved_calls(self):
        return (
            self._reserve_policy()
            if callable(self._reserve_policy)
            else self._reserve_policy
        )

    def _validate_proposal(self, proposal):
        draft = resolve_proposal(
            proposal,
            self.harness.state,
            self.harness.evidence(),
        )
        if self.harness.validate_draft is not None:
            self.harness.validate_draft(draft)

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """Build a bounded request and execute through the run harness.

        Add remaining budgets, reduce tool results when needed, and reserve
        output space. Stop before model admission if the input cannot fit.
        Retain runtime snapshots with the model exchanges in graph history.
        """
        self.harness.check_active()
        state = self.harness.state
        if self._investigation:
            if sum(state.tool_calls.values()) >= state.limits.tool_calls:
                self.harness.begin_finalization(StopReason.TOOL_BUDGET)
            elif self.harness.task_tools_remaining() == 0:
                self._local_finalization = StopReason.TOOL_BUDGET
            elif self._output_name == "InvestigationReport":
                progress = sum(
                    int(e.data.get("added", 0))
                    for e in state.events
                    if e.data.get("task_id") == self.harness._task_id.get()
                )
                self._stalled = (
                    self._stalled + 1
                    if progress == self._last_task_progress
                    else 0
                )
                self._last_task_progress = progress
                if self._stalled >= state.limits.no_progress_rounds:
                    self._local_finalization = StopReason.NO_PROGRESS
            else:
                self.harness.check_progress()
        model_history = []
        while True:
            if self._using_fallback:
                request = request.override(model=self._fallback)
            malformed = False
            validation = None
            prepared = None

            def prepare():
                """Rebind before each attempt, including transport retries."""
                nonlocal prepared, request
                if (self._output_name in {"DiagnosisProposal",
                                          "CoordinatorDecision"}
                        and self.harness.tokens_remaining is not None
                        and self.harness.investigation_token_pool
                        < state.limits.context_tokens):
                    self.harness.begin_finalization(StopReason.TOKEN_BUDGET)
                if state.limits.model_calls - state.model_calls <= (
                    self.harness.verification_calls + self._reserved_calls
                ):
                    if self._output_name == "InvestigationReport":
                        raise TaskBudgetExhausted
                    self.harness.stop(StopReason.MODEL_BUDGET)
                if (
                    self._investigation
                    and state.limits.model_calls - state.model_calls
                    <= (self.harness.finalization_calls + self._reserved_calls)
                ):
                    if self._output_name == "InvestigationReport":
                        self._local_finalization = StopReason.MODEL_BUDGET
                    else:
                        self.harness.begin_finalization(
                            StopReason.MODEL_BUDGET
                        )
                prepared = self._prepare_request(request)
                # Keep exactly the snapshot sent before this attempt. Retrying
                # appends a new snapshot without rewriting the earlier prefix.
                snapshot = prepared.messages[-1]
                model_history.append(snapshot)
                request = request.override(
                    messages=[*request.messages, snapshot]
                )

            async def invoke():
                """Retain provider output and usage after schema rejection."""
                nonlocal malformed, validation
                try:
                    return await handler(prepared)
                except StructuredOutputError as exc:
                    malformed = True
                    validation = self._schema_failure(exc, self._output_name)
                    return ModelResponse(result=[exc.ai_message])

            model_name = getattr(request.model, "model_name", None) or getattr(
                request.model, "model", type(request.model).__name__
            )
            try:
                response = await self.harness.model_call(
                    invoke,
                    model_name=str(model_name),
                    before_attempt=prepare,
                    estimated_tokens=lambda: (
                        estimate_input(prepared) + state.limits.output_tokens
                    ),
                    protected_tokens=(
                        (lambda: self.harness.verification_token_reserve)
                        if self._output_name != "InvestigationReport" else 0
                    ),
                    purpose=self._role,
                    required=self._output_name != "InvestigationReport",
                    reserved_calls=(
                        self.harness.verification_calls + self._reserved_calls
                    ),
                    allow_fallback=(
                        self._fallback is not None and not self._using_fallback
                    ),
                )
            except ModelCallFailed as exc:
                if self._output_name == "InvestigationReport" and (
                    self._fallback is None
                    or self._using_fallback
                    or not exc.failure.retryable
                ):
                    raise
                if self._fallback is None or self._using_fallback:
                    self.harness.stop(StopReason.MODEL_BUDGET)
                self._using_fallback = True
                self.harness.state.model_switches += 1
                fallback_name = getattr(
                    self._fallback, "model_name", None
                ) or getattr(
                    self._fallback, "model", type(self._fallback).__name__
                )
                self.harness.emit(
                    "model.switched",
                    parent_span_id=self.harness.state.id,
                    data={
                        "from_model": str(model_name),
                        "to_model": str(fallback_name),
                    },
                )
                continue
            if response.structured_response is not None:
                register_references(state)
                normalized, repairs = normalize_reference_handles(
                    response.structured_response, state.reference_map,
                )
                response.structured_response = normalized
                for repair in repairs:
                    self.harness.emit(
                        "reference.normalized",
                        data={"role": self._role,
                              "output": self._output_name,
                              "response_ids": [m.id for m in response.result
                                               if isinstance(m, AIMessage)
                                               and m.id],
                              **repair},
                    )
                if self._validator is not None:
                    try:
                        self._validator(response.structured_response)
                    except DiagnosisValidationError as exc:
                        malformed = True
                        validation = exc.details()
                    except DiagnosisError:
                        malformed = True
                        validation = {
                            "code": "diagnosis_contract",
                            "message": "Diagnosis contract validation failed.",
                        }
            elif request.response_format is not None and any(
                isinstance(item, AIMessage)
                and (
                    not item.tool_calls
                    or item.invalid_tool_calls
                    or any(
                        call["name"] == self._output_name
                        for call in item.tool_calls
                    )
                )
                for item in response.result
            ):
                malformed = True
                validation = validation or self._output_failure(
                    response,
                    request.response_format,
                    self._output_name,
                )
            finalizing = (
                self._local_finalization is not None
                or self.harness.finalization_reason is not None
            )
            if finalizing and (
                response.structured_response is None
                or any(
                    isinstance(item, AIMessage)
                    and (
                        item.invalid_tool_calls
                        or any(
                            call["name"] != self._output_name
                            for call in item.tool_calls
                        )
                    )
                    for item in response.result
                )
            ):
                malformed = True
                validation = validation or {
                    "code": "finalization_output",
                    "message": (
                        "Investigation tools are disabled; return only one "
                        f"valid {self._output_name} using existing evidence."
                    ),
                }
            if malformed:
                if (
                    self._output_name == "InvestigationReport"
                    and state.limits.model_calls - state.model_calls
                    <= (self.harness.verification_calls + self._reserved_calls)
                ):
                    raise TaskBudgetExhausted
                preview, truncated = self._candidate_preview(
                    response, self._output_name
                )
                if (
                    self._output_name == "InvestigationReport"
                    and state.collaboration.repairs.get(
                        self.harness._task_id.get(), 0
                    )
                    >= state.limits.diagnosis_repairs
                ):
                    self.harness.emit(
                        "collaboration.task_report_rejected",
                        data={
                            "task_id": self.harness._task_id.get(),
                            "validation": json_text(validation),
                        },
                    )
                    raise TaskReportInvalid
                self.harness.repair_diagnosis(
                    validation=validation,
                    candidate_preview=preview,
                    candidate_truncated=truncated,
                )
                feedback = self._repair_feedback(
                    response,
                    finalizing=finalizing,
                    validation=validation,
                    output_name=self._output_name,
                )
                if not finalizing and any(
                    isinstance(item, AIMessage)
                    and any(
                        call["name"] != self._output_name
                        for call in item.tool_calls
                    )
                    for item in response.result
                ):
                    return ModelResponse(
                        result=[*model_history, *feedback.result]
                    )
                # Both output strategies repair here. Only investigation
                # rounds advance the no-progress policy, and all exchanges
                # are returned to graph history when this invocation ends.
                model_history.extend(feedback.result)
                request = request.override(
                    messages=[*request.messages, *feedback.result]
                )
                continue
            return ModelResponse(
                result=[*model_history, *response.result],
                structured_response=response.structured_response,
            )

    @staticmethod
    def _schema_failure(
        error: Exception,
        output_name="DiagnosisProposal",
    ) -> dict:
        """Report field paths and error types without raw provider errors."""
        if isinstance(error, MultipleStructuredOutputsError):
            return {
                "code": "multiple_diagnoses",
                "message": f"Submit exactly one {output_name} per response.",
            }
        source = getattr(error, "source", error)
        seen = set()
        while source is not None and id(source) not in seen:
            seen.add(id(source))
            if isinstance(source, ValidationError):
                errors = source.errors(
                    include_input=False, include_context=False
                )
                return {
                    "code": "schema_validation",
                    "message": f"Correct the listed {output_name} fields.",
                    "fields": [
                        ".".join(str(p) for p in e["loc"])[:160]
                        + ": "
                        + e["type"]
                        for e in errors[:20]
                    ],
                    "omitted_fields": max(0, len(errors) - 20),
                }
            if isinstance(source, json.JSONDecodeError):
                return {
                    "code": "invalid_json",
                    "message": f"Return valid JSON matching {output_name}.",
                }
            source = source.__cause__ or source.__context__
        return {
            "code": "schema_validation",
            "message": f"Return one valid {output_name} matching its schema.",
        }

    @staticmethod
    def _output_failure(
        response: ModelResponse,
        response_format,
        output_name="DiagnosisProposal",
    ) -> dict:
        """Recover field errors already converted to framework tool messages.

        ToolStrategy can return an error ToolMessage instead of raising.
        Revalidate only its rejected arguments to extract safe field metadata.
        """
        calls = [
            call
            for item in response.result
            if isinstance(item, AIMessage)
            for call in item.tool_calls
            if call["name"] == output_name
        ]
        if len(calls) > 1:
            return {
                "code": "multiple_diagnoses",
                "message": f"Submit exactly one {output_name} per response.",
            }
        if any(
            item.invalid_tool_calls
            for item in response.result
            if isinstance(item, AIMessage)
        ):
            return {
                "code": "invalid_tool_arguments",
                "message": "Return valid JSON arguments for each tool call.",
            }
        if calls:
            schema = response_format.schema
            try:
                schema.model_validate(calls[0]["args"])
            except ValidationError as exc:
                return HarnessMiddleware._schema_failure(exc, output_name)
        return {
            "code": "missing_structured_output",
            "message": f"Return a single valid {output_name}.",
        }

    @staticmethod
    def _candidate_preview(
        response: ModelResponse,
        output_name="DiagnosisProposal",
    ) -> tuple[str, bool]:
        """Retain at most 4096 UTF-8 bytes of rejected diagnosis output."""
        candidate = response.structured_response
        if candidate is not None:
            candidate = candidate.model_dump(mode="json")
        else:
            candidate = [
                call.get("args")
                for item in response.result
                if isinstance(item, AIMessage)
                for call in [*item.tool_calls, *item.invalid_tool_calls]
                if call.get("name") == output_name
            ]
            if not candidate:
                candidate = []
                for item in response.result:
                    if isinstance(item, AIMessage):
                        try:
                            candidate.append(json.loads(item.text))
                        except json.JSONDecodeError:
                            candidate.append(item.content)

        # Feedback quotes belong in the ledger, not rejected-output logs.
        def diagnostic_only(value):
            if isinstance(value, dict):
                return {
                    k: diagnostic_only(v)
                    for k, v in value.items()
                    if k != "feedback_updates"
                }
            if isinstance(value, list):
                return [diagnostic_only(item) for item in value]
            if isinstance(value, str) and "feedback_updates" in value:
                return "Unparseable output containing feedback omitted."
            return value

        candidate = diagnostic_only(candidate)
        encoded = json_text(candidate).encode()
        return encoded[:4096].decode(errors="ignore"), len(encoded) > 4096

    @staticmethod
    def _repair_feedback(
        response: ModelResponse,
        *,
        finalizing: bool = False,
        validation: dict | None = None,
        output_name="DiagnosisProposal",
    ) -> ModelResponse:
        """Keep message pairing and send bounded schema/citation feedback."""
        feedback = (
            "Diagnosis rejected by schema or evidence validation. Return one "
            "valid DiagnosisProposal. Cite unique E handles available "
            "in this run within the investigation window, including retained "
            "observations. Use findings, hypotheses, limitations and "
            "proposed_checks; apply the assigned task's completion rule."
        )
        if output_name == "InvestigationReport":
            feedback = (
                f"Structured output rejected. Return one valid {output_name}. "
                "Cite E telemetry and separate K knowledge handles; record "
                "supported findings and gaps in missing_observations."
            )
        elif output_name == "CoordinatorDecision":
            feedback = (
                "Structured output rejected. Return one valid "
                "CoordinatorDecision: investigate with 1-2 tasks and "
                "no proposal, "
                "or submit with a DiagnosisProposal and no tasks. Use valid "
                "hypothesis IDs and E/K handles. If investigation is closed, "
                "submit retained findings with explicit gaps."
            )
        if validation is not None:
            feedback += " Specific validation failure: " + json_text(
                validation
            )
        if finalizing:
            feedback += (
                " All investigation tools are disabled. Submit "
                f"{output_name} using existing evidence only."
            )
        messages = []
        for item in response.result:
            if isinstance(item, ToolMessage) and item.name == output_name:
                continue
            if not isinstance(item, AIMessage):
                messages.append(item)
                continue
            calls = [c for c in item.tool_calls if c["name"] == output_name]
            pending = [c for c in item.tool_calls if c["name"] != output_name]
            if finalizing:
                calls.extend(pending)
                pending = []
            rejected = [*calls, *item.invalid_tool_calls]
            if rejected:
                # LangChain's tools-to-model edge exits if the latest AI
                # exchange includes a structured-output tool response.
                # Close the rejected diagnosis exchange before opening the
                # investigation exchange, preserving every call ID.
                additional = {
                    k: v
                    for k, v in item.additional_kwargs.items()
                    if k != "tool_calls"
                }
                messages.append(
                    item.model_copy(
                        update={
                            "tool_calls": calls,
                            "additional_kwargs": additional,
                        }
                    )
                )
                messages.extend(
                    ToolMessage(
                        content=feedback,
                        tool_call_id=call["id"],
                        name=call["name"],
                        status="error",
                    )
                    for call in rejected
                )
                if pending:
                    messages.append(
                        AIMessage(
                            content="",
                            tool_calls=pending,
                            additional_kwargs=additional,
                        )
                    )
            elif pending:
                messages.extend([HumanMessage(content=feedback), item])
            else:
                messages.extend([item, HumanMessage(content=feedback)])
        return ModelResponse(result=messages)

    def _prepare_request(self, request: ModelRequest) -> ModelRequest:
        """Rebuild bounded context and output settings for the model."""
        state = self.harness.state
        budget = (
            "Application runtime snapshot. Earlier snapshots are historical; "
            "only the latest applies.\n"
            "Remaining budget including the next call: "
            f"{state.limits.model_calls - state.model_calls} model calls, "
            f"{state.limits.tool_calls - sum(state.tool_calls.values())} "
            f"tool calls. Return {self._output_name}; preserve supported "
            "observations and state gaps."
        )
        task_remaining = self.harness.task_tools_remaining()
        if self.harness.tokens_remaining is not None:
            budget += (f"\nRun token admission balance: "
                       f"{self.harness.tokens_remaining}. Input estimates "
                       "plus maximum output are reserved before each attempt; "
                       "unreported usage remains charged.")
            record = self.harness.current_investigation_task()
            if record is not None and record.token_limit is not None:
                budget += (f" Task token balance: "
                           f"{record.token_limit - record.tokens.committed}.")
        if task_remaining is not None:
            budget += (
                f"\nThis task has {task_remaining} tool attempts remaining. "
                "The shared remainder is reserved for other tasks and "
                "follow-up investigation. Prioritize the observation that "
                "best distinguishes the assigned hypotheses."
            )
        reason = self._local_finalization or self.harness.finalization_reason
        if reason is not None:
            budget += (
                f"\nInvestigation is closed ({reason.value}). "
                "Do not call investigation or evidence-reading tools. Submit "
                f"{self._output_name} now using evidence already provided. "
                "Keep conclusions scoped to inspected operations; distinguish "
                "observed failures from unverified mechanisms and address "
                "counterevidence. Do not merge separate traces without a "
                "shared business identifier. "
            )
            if self._output_name == "DiagnosisProposal":
                budget += (
                    "Preserve findings and state gaps in limitations. "
                    "Apply the assigned task's completion rule."
                )
            elif self._output_name == "InvestigationReport":
                budget += (
                    "Return supported findings with E/K handles and gaps in "
                    "missing_observations; findings may be empty."
                )
            else:
                budget += "Choose submit with a proposal and no tasks."
            # The framework still binds the assigned structured-output schema.
            request = request.override(tools=[], tool_choice=None)
        request = request.override(
            messages=[
                *request.messages,
                HumanMessage(
                    content=budget,
                    name="runtime_budget",
                    id=f"runtime-budget-{state.id}-{state.model_calls + 1}",
                ),
            ]
        )
        selection = select_context(
            request,
            state.limits.context_tokens - state.limits.output_tokens,
            coordinator_handoff=self._output_name == "CoordinatorDecision",
        )
        self.harness.emit(
            "context.selected",
            parent_span_id=state.id,
            data={
                "estimator": "utf8-bytes-v1",
                "estimated_input_before": selection.before,
                "estimated_input_after": selection.after,
                "reduced_messages": selection.reduced_messages,
                "omitted_previews": selection.omitted_previews,
                "input_allowance": (
                    state.limits.context_tokens - state.limits.output_tokens
                ),
                "output_reserved": state.limits.output_tokens,
                "fits": selection.fits,
            },
        )
        if not selection.fits:
            if self._output_name == "InvestigationReport":
                raise TaskContextExceeded
            self.harness.stop(StopReason.CONTEXT_LIMIT)
        output_key = (
            "max_output_tokens"
            if isinstance(request.model, ChatGoogleGenerativeAI)
            else "max_tokens"
        )
        request = selection.request.override(
            model_settings={
                **request.model_settings,
                output_key: state.limits.output_tokens,
            }
        )
        return request

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]],
    ) -> ToolMessage | Command:
        """Authorize dispatch and trace framework argument errors.

        The tool executor performs budget reservation and rechecks
        authorization. Framework validation errors are returned to the
        agent for bounded repair.
        """
        try:
            try:
                # Also intercept hallucinated/hidden tool
                # names before dispatch.
                name = request.tool_call["name"]
                if (
                    self._allowed_tools is not None
                    and name not in self._allowed_tools
                ):
                    self.harness.stop(StopReason.TOOL_DENIED)
                self.harness.authorize_tool(name)
                result = await handler(request)
            except TaskToolBudgetExhausted as exc:
                own_task = exc.task_id == self.harness._task_id.get()
                if own_task:
                    self._local_finalization = exc.reason
                self.harness.emit(
                    "tool.skipped",
                    parent_span_id=self.harness.state.id,
                    data={"tool": request.tool_call["name"],
                          "reason": exc.reason.value,
                          "executing_task_id": exc.task_id},
                )
                message = (
                    "This task's tool allocation is exhausted. "
                    f"Submit {self._output_name} using retained evidence "
                    "and explicit gaps. Other tasks can still investigate."
                    if own_task else
                    "The shared call's executing task exhausted its "
                    "allocation before this attempt. Your allocation "
                    "is unchanged; you may retry within your own budget."
                )
                return ToolMessage(
                    content=json.dumps({"error": {
                        "category": "task_tool_budget", "message": message,
                    }}),
                    tool_call_id=request.tool_call["id"],
                    name=request.tool_call["name"], status="error",
                )
            except InvestigationClosed:
                reason = (
                    self._local_finalization
                    or self.harness.finalization_reason
                )
                self.harness.emit(
                    "tool.skipped",
                    parent_span_id=self.harness.state.id,
                    data={
                        "tool": request.tool_call["name"],
                        "reason": reason.value,
                    },
                )
                return ToolMessage(
                    content=json.dumps(
                        {
                            "error": {
                                "category": "investigation_closed",
                                "reason": reason.value,
                                "message": (
                                    "This tool was not executed. "
                                    "Investigation is closed; "
                                    f"submit {self._output_name} "
                                    "using existing evidence."
                                ),
                            }
                        }
                    ),
                    tool_call_id=request.tool_call["id"],
                    name=request.tool_call["name"],
                    status="error",
                )
            except ToolUnavailable as exc:
                self.harness.emit(
                    "tool.unavailable",
                    parent_span_id=self.harness.state.id,
                    data={
                        "tool": request.tool_call["name"],
                        "category": exc.failure.category,
                        "status_code": exc.failure.status_code,
                    },
                )
                return ToolMessage(
                    content=json.dumps(
                        {
                            "error": {
                                "category": exc.failure.category,
                                "status_code": exc.failure.status_code,
                                "message": (
                                    "The requested resource is not available. "
                                    "It may be absent or no longer retained. "
                                    "Do not invent a replacement ID. Use "
                                    "other evidence and report this gap; "
                                    "absence does not establish health."
                                    if exc.failure.category == "not_found"
                                    else "Query failed; correct arguments "
                                    "or use another source. "
                                    "No evidence was returned by this call."
                                ),
                            }
                        }
                    ),
                    tool_call_id=request.tool_call["id"],
                    name=request.tool_call["name"],
                    status="error",
                )
            if isinstance(result, ToolMessage) and result.status == "error":
                self.harness.emit(
                    "tool.rejected",
                    parent_span_id=self.harness.state.id,
                    data={
                        "tool": request.tool_call["name"],
                        "reason": "invalid_arguments",
                    },
                )
            return result
        finally:
            await self.harness.checkpoint(check_cancel=False)
