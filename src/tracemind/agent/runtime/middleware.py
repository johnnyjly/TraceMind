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
    SystemMessage,
    ToolMessage,
)
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import Command
from pydantic import ValidationError

from tracemind.agent.workflow.context import json_text, select_context
from tracemind.agent.workflow.validation import resolve_proposal
from tracemind.agent.errors import DiagnosisError, DiagnosisValidationError
from tracemind.agent.runtime.harness import InvestigationClosed, RunHarness
from tracemind.agent.runtime.recovery import ModelCallFailed, ToolUnavailable
from tracemind.models.run import StopReason


class HarnessMiddleware(AgentMiddleware):
    """Adapt LangChain model/tool hooks to a single run's harness controls."""

    def __init__(
        self, harness: RunHarness, fallback_model: BaseChatModel | None = None
    ) -> None:
        """Bind all interception hooks to the supplied run harness."""
        self.harness = harness
        self._fallback = fallback_model
        self._using_fallback = False

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """Build a bounded request and execute through the run harness.

        Add remaining budgets, reduce tool results when needed, and reserve
        output space. Stop before model admission if the input cannot fit.
        Bind the provider output cap without mutating graph message history.
        """
        self.harness.check_active()
        state = self.harness.state
        if sum(state.tool_calls.values()) >= state.limits.tool_calls:
            self.harness.begin_finalization(StopReason.TOOL_BUDGET)
        else:
            self.harness.check_progress()
        repair_history = []
        while True:
            if self._using_fallback:
                request = request.override(model=self._fallback)
            malformed = False
            validation = None
            prepared = None

            def prepare():
                """Rebind before each attempt, including transport retries."""
                nonlocal prepared
                if (state.limits.model_calls - state.model_calls
                        <= self.harness.finalization_calls):
                    self.harness.begin_finalization(StopReason.MODEL_BUDGET)
                prepared = self._prepare_request(request)

            async def invoke():
                """Retain provider output and usage after schema rejection."""
                nonlocal malformed, validation
                try:
                    return await handler(prepared)
                except StructuredOutputError as exc:
                    malformed = True
                    validation = self._schema_failure(exc)
                    return ModelResponse(result=[exc.ai_message])

            model_name = (
                getattr(request.model, "model_name", None)
                or getattr(request.model, "model",
                           type(request.model).__name__)
            )
            try:
                response = await self.harness.model_call(
                    invoke, model_name=str(model_name),
                    before_attempt=prepare,
                    reserved_calls=self.harness.verification_calls,
                    allow_fallback=(
                        self._fallback is not None and not self._using_fallback
                    ),
                )
            except ModelCallFailed:
                self._using_fallback = True
                self.harness.state.model_switches += 1
                fallback_name = (
                    getattr(self._fallback, "model_name", None)
                    or getattr(self._fallback, "model",
                               type(self._fallback).__name__)
                )
                self.harness.emit(
                    "model.switched", parent_span_id=self.harness.state.id,
                    data={"from_model": str(model_name),
                          "to_model": str(fallback_name)},
                )
                continue
            if response.structured_response is not None:
                if self.harness.validate_draft is not None:
                    try:
                        draft = resolve_proposal(
                            response.structured_response, state,
                            self.harness.evidence(),
                        )
                        self.harness.validate_draft(draft)
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
                isinstance(item, AIMessage) and (
                    not item.tool_calls or item.invalid_tool_calls or any(
                        call["name"] == "DiagnosisProposal"
                        for call in item.tool_calls
                    )
                ) for item in response.result
            ):
                malformed = True
                validation = validation or self._output_failure(
                    response, request.response_format,
                )
            finalizing = self.harness.finalization_reason is not None
            if finalizing and (
                response.structured_response is None or any(
                    isinstance(item, AIMessage) and (
                        item.invalid_tool_calls or any(
                            call["name"] != "DiagnosisProposal"
                            for call in item.tool_calls
                        )
                    ) for item in response.result
                )
            ):
                malformed = True
                validation = validation or {
                    "code": "finalization_output",
                    "message": (
                        "Investigation tools are disabled; return only one "
                        "valid DiagnosisProposal using existing evidence."
                    ),
                }
            if malformed:
                preview, truncated = self._candidate_preview(response)
                self.harness.repair_diagnosis(
                    validation=validation, candidate_preview=preview,
                    candidate_truncated=truncated,
                )
                feedback = self._repair_feedback(
                    response, finalizing=finalizing, validation=validation,
                )
                if not finalizing and any(
                    isinstance(item, AIMessage) and any(
                        call["name"] != "DiagnosisProposal"
                        for call in item.tool_calls
                    ) for item in response.result
                ):
                    return ModelResponse(
                        result=[*repair_history, *feedback.result]
                    )
                # Both output strategies repair here. Only investigation
                # rounds advance the no-progress policy, and all exchanges
                # are returned to graph history when this invocation ends.
                repair_history.extend(feedback.result)
                request = request.override(
                    messages=[*request.messages, *feedback.result]
                )
                continue
            return ModelResponse(
                result=[*repair_history, *response.result],
                structured_response=response.structured_response,
            )

    @staticmethod
    def _schema_failure(error: Exception) -> dict:
        """Report field paths and error types without raw provider errors."""
        if isinstance(error, MultipleStructuredOutputsError):
            return {
                "code": "multiple_diagnoses",
                "message": "Submit exactly one proposal per response.",
            }
        source = getattr(error, "source", error)
        seen = set()
        while source is not None and id(source) not in seen:
            seen.add(id(source))
            if isinstance(source, ValidationError):
                errors = source.errors(include_input=False,
                                       include_context=False)
                return {
                    "code": "schema_validation",
                    "message": "Correct the listed DiagnosisProposal fields.",
                    "fields": [
                        ".".join(str(p) for p in e["loc"])[:160]
                        + ": " + e["type"] for e in errors[:20]
                    ],
                    "omitted_fields": max(0, len(errors) - 20),
                }
            if isinstance(source, json.JSONDecodeError):
                return {
                    "code": "invalid_json",
                    "message": "Return valid JSON matching DiagnosisProposal.",
                }
            source = source.__cause__ or source.__context__
        return {
            "code": "schema_validation",
            "message": "Return one valid proposal matching its schema.",
        }

    @staticmethod
    def _output_failure(response: ModelResponse, response_format) -> dict:
        """Recover field errors already converted to framework tool messages.

        ToolStrategy can return an error ToolMessage instead of raising.
        Revalidate only its rejected arguments to extract safe field metadata.
        """
        calls = [call for item in response.result
                 if isinstance(item, AIMessage) for call in item.tool_calls
                 if call["name"] == "DiagnosisProposal"]
        if len(calls) > 1:
            return {
                "code": "multiple_diagnoses",
                "message": "Submit exactly one proposal per response.",
            }
        if any(item.invalid_tool_calls for item in response.result
               if isinstance(item, AIMessage)):
            return {
                "code": "invalid_tool_arguments",
                "message": "Return valid JSON arguments for each tool call.",
            }
        if calls:
            schema = response_format.schema
            try:
                schema.model_validate(calls[0]["args"])
            except ValidationError as exc:
                return HarnessMiddleware._schema_failure(exc)
        return {
            "code": "missing_structured_output",
            "message": "Return a single valid DiagnosisProposal.",
        }

    @staticmethod
    def _candidate_preview(response: ModelResponse) -> tuple[str, bool]:
        """Retain at most 4096 UTF-8 bytes of rejected diagnosis output."""
        candidate = response.structured_response
        if candidate is not None:
            candidate = candidate.model_dump(mode="json")
        else:
            candidate = [
                call.get("args")
                for item in response.result if isinstance(item, AIMessage)
                for call in [*item.tool_calls, *item.invalid_tool_calls]
                if call.get("name") == "DiagnosisProposal"
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
                return {k: diagnostic_only(v) for k, v in value.items()
                        if k != "feedback_updates"}
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
        response: ModelResponse, *, finalizing: bool = False,
        validation: dict | None = None,
    ) -> ModelResponse:
        """Keep message pairing and send bounded schema/citation feedback."""
        feedback = (
            "Diagnosis rejected by schema or evidence validation. Return one "
            "valid DiagnosisProposal. Cite unique E handles available "
            "in this run within the investigation window, including retained "
            "observations. Include a "
            "nonblank explanation, and use inconclusive if unsupported."
        )
        if validation is not None:
            feedback += (
                " Specific validation failure: " + json_text(validation)
            )
        if finalizing:
            feedback += (
                " All investigation tools are disabled. Submit the final "
                "diagnosis using existing evidence only."
            )
        messages = []
        for item in response.result:
            if isinstance(item, ToolMessage) and (
                    item.name == "DiagnosisProposal"):
                continue
            if not isinstance(item, AIMessage):
                messages.append(item)
                continue
            calls = [c for c in item.tool_calls
                     if c["name"] == "DiagnosisProposal"]
            pending = [c for c in item.tool_calls
                       if c["name"] != "DiagnosisProposal"]
            if finalizing:
                calls.extend(pending)
                pending = []
            rejected = [*calls, *item.invalid_tool_calls]
            if rejected:
                # LangChain's tools-to-model edge exits if the latest AI
                # exchange includes a structured-output tool response.
                # Close the rejected diagnosis exchange before opening the
                # investigation exchange, preserving every call ID.
                additional = {k: v for k, v in item.additional_kwargs.items()
                              if k != "tool_calls"}
                messages.append(item.model_copy(update={
                    "tool_calls": calls, "additional_kwargs": additional,
                }))
                messages.extend(ToolMessage(
                    content=feedback, tool_call_id=call["id"],
                    name=call["name"], status="error",
                ) for call in rejected)
                if pending:
                    messages.append(AIMessage(
                        content="", tool_calls=pending,
                        additional_kwargs=additional,
                    ))
            elif pending:
                messages.extend([HumanMessage(content=feedback), item])
            else:
                messages.extend([item, HumanMessage(content=feedback)])
        return ModelResponse(result=messages)

    def _prepare_request(self, request: ModelRequest) -> ModelRequest:
        """Rebuild bounded context and output settings for the model."""
        state = self.harness.state
        budget = (
            f"\nRemaining budget including the next call: "
            f"{state.limits.model_calls - state.model_calls} model calls, "
            f"{state.limits.tool_calls - sum(state.tool_calls.values())} "
            "tool calls. "
            "Propose an inconclusive diagnosis if the evidence "
            "is insufficient."
        )
        system = request.system_message.text if request.system_message else ""
        reason = self.harness.finalization_reason
        if reason is not None:
            budget += (
                f"\nInvestigation is closed ({reason.value}). "
                "Do not call telemetry or evidence-reading tools. Submit "
                "the final structured diagnosis now using only evidence "
                "already provided. Cite available evidence IDs within the "
                "investigation window. Explain the cause and its evidence, "
                "or return inconclusive with the observed failure, its "
                "citations and the specific missing observation. Do not "
                "discard established findings because the deeper cause "
                "is unknown. Return only DiagnosisProposal."
            )
            # The framework still binds the structured diagnosis schema;
            # DiagnosisProposal only submits the final result.
            request = request.override(tools=[], tool_choice=None)
            # End the investigation exchange explicitly; a system-only budget
            # update can leave the model continuing the last tool result.
            request = request.override(messages=[
                *request.messages,
                HumanMessage(content=(
                    "Investigation is now closed. Submit DiagnosisProposal "
                    "now, using the evidence already available. No further "
                    "tool queries are permitted. Keep conclusions scoped to "
                    "the observed operations and distinguish confirmed "
                    "failures from unverified underlying causes. In "
                    "findings and limitations, discuss supporting and "
                    "contrary "
                    "evidence (including successful "
                    "operations), and state what remains unknown. Do not "
                    "merge events from separate traces into one transaction "
                    "without a shared business identifier."
                )),
            ])
        request = request.override(
            system_message=SystemMessage(content=system + budget)
        )
        selection = select_context(
            request, state.limits.context_tokens - state.limits.output_tokens
        )
        self.harness.emit(
            "context.selected", parent_span_id=state.id,
            data={"estimator": "utf8-bytes-v1",
                  "estimated_input_before": selection.before,
                  "estimated_input_after": selection.after,
                  "reduced_messages": selection.reduced_messages,
                  "omitted_previews": selection.omitted_previews,
                  "input_allowance": (
                      state.limits.context_tokens - state.limits.output_tokens
                  ),
                  "output_reserved": state.limits.output_tokens,
                  "fits": selection.fits},
        )
        if not selection.fits:
            self.harness.stop(StopReason.CONTEXT_LIMIT)
        output_key = (
            "max_output_tokens"
            if isinstance(request.model, ChatGoogleGenerativeAI)
            else "max_tokens"
        )
        request = selection.request.override(model_settings={
            **request.model_settings, output_key: state.limits.output_tokens,
        })
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
                self.harness.authorize_tool(request.tool_call["name"])
                result = await handler(request)
            except InvestigationClosed:
                reason = self.harness.finalization_reason
                self.harness.emit(
                    "tool.skipped", parent_span_id=self.harness.state.id,
                    data={"tool": request.tool_call["name"],
                          "reason": reason.value},
                )
                return ToolMessage(
                    content=json.dumps({"error": {
                        "category": "investigation_closed",
                        "reason": reason.value,
                        "message": (
                            "This tool was not executed. "
                            "Investigation is closed; "
                            "submit a final diagnosis using existing evidence."
                        ),
                    }}),
                    tool_call_id=request.tool_call["id"],
                    name=request.tool_call["name"], status="error",
                )
            except ToolUnavailable as exc:
                self.harness.emit(
                    "tool.unavailable", parent_span_id=self.harness.state.id,
                    data={"tool": request.tool_call["name"],
                          "category": exc.failure.category,
                          "status_code": exc.failure.status_code},
                )
                return ToolMessage(
                    content=json.dumps({"error": {
                        "category": exc.failure.category,
                        "status_code": exc.failure.status_code,
                        "message": (
                            "Query failed; correct arguments or use another "
                            "source. "
                            "No evidence was returned by this call."
                        ),
                    }}),
                    tool_call_id=request.tool_call["id"],
                    name=request.tool_call["name"], status="error",
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
