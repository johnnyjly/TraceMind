"""Bounded LangGraph delegation, parallel investigation and review feedback."""

from hashlib import sha256
from typing import Annotated, TypedDict, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from uuid import UUID

from langchain.agents import create_agent
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from tracemind.agent.errors import DiagnosisValidationError
from tracemind.agent.runtime.recovery import ModelCallFailed
from tracemind.agent.runtime.middleware import (
    HarnessMiddleware,
    TaskBudgetExhausted,
    TaskReportInvalid,
    TaskContextExceeded,
)
from tracemind.agent.runtime.runner import INVESTIGATION_PROMPT, SYSTEM_PROMPT
from tracemind.agent.runtime.knowledge import knowledge_context
from tracemind.agent.workflow.validation import resolve_proposal
from tracemind.agent.workflow.handoffs import freeze_task_input, role_tools
from tracemind.agent.workflow.context import (
    json_text,
    _evidence_preview,
    _prioritized_evidence,
)
from tracemind.agent.workflow.references import model_references
from tracemind.models import Diagnosis, DiagnosisStatus
from tracemind.models.collaboration import (
    CollaborationState,
    InvestigationTask,
    Hypothesis,
    InvestigationReport,
    TaskRecord,
)
from tracemind.models.result import DiagnosisDraft, DiagnosisProposal
from tracemind.observability.langfuse import observation


COORDINATOR_PROMPT = (
    SYSTEM_PROMPT.replace(
        "Submit DiagnosisProposal:", "The nested proposal contains:"
    )
    + """
You coordinate a bounded incident investigation. Return CoordinatorDecision.
Your read_evidence/read_knowledge tools only page records already retained in
this run. They do not search remote telemetry or the knowledge snapshot.
Empty local catalogues are expected before investigation; they do not mean
remote data is absent or searches failed. Delegate a telemetry discovery task
when no incident evidence has yet been collected and investigation is open.
Specialists receive the remote search tools; use investigate to obtain facts.
Maintain up to six competing hypotheses with exact E support/counterevidence
and separate K references. The board and specialist reports are unverified
proposals; check originals before accepting them. Focus on failure owner,
operation, observable mechanism and linked propagation. Seek a distinguishing
observation, not exhaustive service coverage or incidental exact counts.
Choose investigate for one or two independent bounded tasks when a missing
observation could change the diagnosis. Telemetry checks incident facts;
knowledge checks documented contracts and mechanisms when available. Use
hypothesis_ids from your board; an initial discovery task may have none.
Specify acceptance_criteria for each task, context_refs for the retained E/K
records it needs, and requested_tools only to narrow the role's grants.
An empty requested_tools list uses the application's role grants. You cannot
expand permissions. Workers get frozen task packets, not your conversation.
Knowledge questions should use observed operations/errors when available.
When a report identifies a slow/failing owner but leaves competing mechanisms,
prefer a complementary owner-scoped investigation to another broad trace
survey. A concrete available log or knowledge check that could distinguish
those mechanisms should be delegated before submitting with that same gap,
while can_investigate is true. Two parallel tasks must address different
questions; do not send both to reconstruct the same trace path.
Choose submit with a DiagnosisProposal to conclude, revise a contradicted
statement using retained evidence, or stop inconclusively with supported
partial findings and explicit gaps. No tasks accompany submit. Do not repeat
remote queries or delegate when existing evidence answers the question.
If can_investigate is false, submit now. A previous unsupported statement can
be narrowed directly; verification feedback does not require another remote
investigation. Each finding must be atomic and scoped. Keep the main failure
and propagation first, avoid irrelevant factual claims and unneeded counts.
"""
)


class CoordinatorDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")
    hypotheses: list[Hypothesis] = Field(default_factory=list, max_length=6)
    action: Literal["investigate", "submit"]
    tasks: list[InvestigationTask] = Field(default_factory=list, max_length=2)
    proposal: DiagnosisProposal | None = None
    reason: str = Field(min_length=1, max_length=700)

    @model_validator(mode="after")
    def validate_action(self):
        if self.action == "investigate":
            if not self.tasks or self.proposal is not None:
                raise ValueError("investigate needs tasks and no proposal")
        elif self.tasks or self.proposal is None:
            raise ValueError("submit needs a proposal and no tasks")
        return self


INVESTIGATOR_PROMPT = (
    INVESTIGATION_PROMPT
    + """
You are an investigator. Answer ONLY your assigned task. Return
InvestigationReport with at most six atomic findings: one scoped assertion per
finding, its E support, E counterevidence and separate K knowledge
references. Knowledge describes system semantics, never an observed incident.
Cite E/K handles, never UUIDs or trace IDs. An observation may bear on
different assertions; do not duplicate references in one list. Explicitly
label hypotheses. Record gaps in missing_observations; findings may be empty
when nothing is supported. Do not return a diagnosis status. Compare
successful and failing operations where relevant. Do not conclude a fleet-wide
outage from sampled failures. Return the exact C IDs you answered in
answered_criteria. Use answered only when all criteria are answered with
findings; otherwise use partial or blocked with a reason_code and concrete
missing_observations. Tool results from other tasks are available by explicit
read from the shared evidence catalogue; reuse them before repeating queries.
"""
)


def merge_completed(left, right):
    """Idempotent task-keyed reduction, rejecting conflicting completions."""
    merged = dict(left)
    for key, value in right.items():
        if key in merged and merged[key] != value:
            raise ValueError("conflicting task completion")
        merged[key] = value
    return merged


class GraphState(TypedDict, total=False):
    completed: Annotated[dict[str, str], merge_completed]
    task_id: str


def validate_report(report, state, harness, record=None):
    """Check provenance per finding without conflating two assertions."""
    if record is not None:
        answered = report.answered_criteria
        criteria = set(record.inputs.criteria)
        if (len(answered) != len(set(answered))
                or not set(answered) <= criteria):
            raise DiagnosisValidationError(
                "Use each assigned criterion ID at most once.",
                code="task_criteria",
            )
        if report.outcome == "answered":
            if set(answered) != criteria or not report.findings:
                raise DiagnosisValidationError(
                    "Answered requires findings and every assigned criterion. "
                    "Otherwise return partial or blocked with concrete gaps.",
                    code="task_incomplete",
                )
            if report.reason_code is not None:
                raise DiagnosisValidationError(
                    "Answered cannot have a blocking reason.",
                    code="task_outcome",
                )
        elif not report.missing_observations or report.reason_code is None:
            raise DiagnosisValidationError(
                "Partial or blocked needs a reason and missing observations.",
                code="task_outcome",
            )
        if report.outcome == "blocked" and answered:
            raise DiagnosisValidationError(
                "Blocked cannot mark criteria answered; use partial.",
                code="task_outcome",
            )
    for finding in report.findings:
        for refs, prefix in (
            (finding.evidence_refs, "E"),
            (finding.counterevidence_refs, "E"),
            (finding.knowledge_refs, "K"),
        ):
            if len(refs) != len(set(refs)):
                raise DiagnosisValidationError(
                    "Duplicate references within a finding list.",
                    code="duplicate_reference",
                    references=refs,
                )
            unknown = [
                r
                for r in refs
                if not r.startswith(prefix) or r not in state.reference_map
            ]
            if unknown:
                raise DiagnosisValidationError(
                    "Use retained E telemetry and K knowledge references.",
                    code="unknown_reference",
                    references=unknown,
                )
        if finding.kind == "observation" and not finding.evidence_refs:
            raise DiagnosisValidationError(
                "An observation needs current telemetry; otherwise use a "
                "hypothesis or missing observation.",
                code="missing_evidence",
            )
        if finding.kind == "knowledge" and not finding.knowledge_refs:
            raise DiagnosisValidationError(
                "Documented semantics need a retained K reference.",
                code="missing_knowledge",
            )
        ids = list(
            dict.fromkeys(finding.evidence_refs + finding.counterevidence_refs)
        )
        harness.validate_draft(
            DiagnosisDraft(
                status=DiagnosisStatus.INCONCLUSIVE,
                root_cause=finding.text,
                confidence=0,
                evidence_ids=[UUID(state.reference_map[r]) for r in ids],
                knowledge_refs=[
                    state.reference_map[r] for r in finding.knowledge_refs
                ],
            )
        )


def coordinator_context(state, evidence):
    """Prioritize each claim's originals; explicitly expose coverage limits."""
    records = [
        record
        for record in state.collaboration.tasks.values()
        if record.report is not None
        and record.round == state.collaboration.round
    ]
    reports = [
        {
            "task_id": r.id,
            "question": r.task.question,
            "role": r.task.role,
            "status": r.status,
            "acceptance_criteria": r.inputs.criteria if r.inputs else {},
            "input_refs": list(r.inputs.context_refs) if r.inputs else [],
            "input_sha256": r.inputs.sha256 if r.inputs else None,
            "report_sha256": r.report_sha256,
            "report": r.report.model_dump(),
        }
        for r in records
    ]
    refs = list(
        dict.fromkeys(
            ref
            for record in records
            for finding in record.report.findings
            for ref in finding.evidence_refs + finding.counterevidence_refs
        )
    )
    refs = list(
        dict.fromkeys(
            refs
            + [
                r
                for h in state.collaboration.hypotheses
                for r in h.evidence_refs + h.counterevidence_refs
            ]
        )
    )
    # Review advances the round; its candidate and contrary originals must
    # remain prioritized even when no current-round report/board exists.
    revision_ids = (
        [*state.collaboration.draft.evidence_ids]
        if state.collaboration.draft
        else []
    )
    revision_ids.extend(
        i
        for claim in state.collaboration.feedback
        for i in claim.supporting_ids + claim.conflicting_ids
    )
    revision_ids = set(map(str, revision_ids))
    refs = list(
        dict.fromkeys(
            refs
            + [
                ref
                for ref, identity in state.reference_map.items()
                if ref.startswith("E") and identity in revision_ids
            ]
        )
    )
    priority = [UUID(state.reference_map[r]) for r in refs]
    packet = {
        "tasks": reports,
        "review_feedback": [
            c.model_dump(mode="json") for c in state.collaboration.feedback
        ],
        "completion_feedback": [
            c.model_dump(mode="json")
            for c in (state.collaboration.reviews[-1].review.completion or [])
            if not c.met
        ] if state.collaboration.reviews
        and state.collaboration.reviews[-1].review else [],
        "hypotheses": [h.model_dump() for h in state.collaboration.hypotheses],
        "candidate": (
            state.collaboration.draft.model_dump(mode="json")
            if state.collaboration.draft
            else None
        ),
        "knowledge": knowledge_context(state),
        "originals": [],
        "omitted_original_refs": [],
        "available_evidence": len(evidence),
        "query_events": [
            e.model_dump(mode="json")
            for e in state.events
            if e.sequence
            in {
                seq
                for c in state.collaboration.feedback
                for seq in c.supporting_event_sequences
                + c.conflicting_event_sequences
            }
        ],
    }
    allowance = max(
        0, state.limits.context_tokens - state.limits.output_tokens - 14000
    )
    selected = set()
    for item in _prioritized_evidence(state, evidence, priority):
        preview = model_references(_evidence_preview(item), state)
        candidate = {**packet, "originals": [*packet["originals"], preview]}
        if len(json_text(candidate).encode()) <= allowance:
            packet = candidate
            selected.add(preview["id"])
    packet["omitted_original_refs"] = [
        ref for ref in refs if ref not in selected
    ]
    return model_references(packet, state)


class MultiAgentRunner:
    def __init__(
        self,
        model,
        *,
        fallback_model=None,
        checkpointer=None,
        callbacks=(),
        langfuse_client=None,
    ):
        self.model = model
        self.fallback = fallback_model
        self.checkpointer = checkpointer
        self.callbacks = list(callbacks)
        self.client = langfuse_client

    async def run(self, prompt, tools, *, recursion_limit, harness):
        state = harness.state
        if state.verification_policy is None or harness.verifier is None:
            raise ValueError("multi-agent requires a verifier")
        if state.memory_policy or state.learning_policy:
            raise ValueError("multi-agent does not support memory writes")
        if state.collaboration is None:
            state.collaboration = CollaborationState()
            state.prompt_version = "multi-agent-graph-v7-contracts"
            await harness.checkpoint()
        workflow = state.collaboration
        if workflow.phase == "completed":
            return DiagnosisDraft(
                **workflow.draft.model_dump(exclude={"incident_id"})
            )
        available = {t.name for t in tools}
        knowledge_available = "search_knowledge" in available

        async def invoke(
            key, role, system, content, schema, exposed, **kwargs
        ):
            return await self._invoke(
                key,
                role,
                system,
                content,
                schema,
                exposed,
                harness,
                recursion_limit,
                **kwargs,
            )

        def task_revision(hypotheses):
            return sha256(json_text({
                "evidence": state.reference_map,
                "hypotheses": [h.model_dump() for h in hypotheses],
                "snapshot": (state.knowledge_policy.snapshot
                             if state.knowledge_policy else None),
            }).encode()).hexdigest()

        def reusable_task(task, revision):
            return next((r for r in workflow.tasks.values()
                         if r.task == task and r.input_revision == revision
                         and r.status == "completed"), None)

        def batch_tool_budget():
            remaining = max(
                0, state.limits.tool_calls - sum(state.tool_calls.values())
            )
            if workflow.batches == 0 and remaining:
                return max(1, remaining // 2)
            return remaining

        def can_investigate(tasks=1, upcoming_call=0):
            remaining = (
                state.limits.model_calls - state.model_calls - upcoming_call
            )
            return (
                workflow.batches < 3
                and workflow.decision < 5
                and remaining
                >= 2 * tasks
                + 1
                + 2 * state.limits.diagnosis_repairs
                + harness.verification_calls
                and batch_tool_budget() >= tasks
                and (harness.investigation_token_pool is None
                     or harness.investigation_token_pool >=
                     tasks * state.limits.context_tokens)
                and harness.finalization_reason is None
            )

        async def coordinate(_):
            if workflow.phase != "coordinate":
                return {}
            packet = coordinator_context(state, harness.evidence())
            packet.update(
                decision=workflow.decision,
                knowledge_available=knowledge_available,
                can_investigate=can_investigate(upcoming_call=1),
                batch_tool_budget=batch_tool_budget(),
                batch_token_budget=harness.investigation_token_pool,
                max_parallel_tasks=(2 if can_investigate(2, 1)
                                    else int(can_investigate(1, 1))),
                task_history=[
                    {
                        "id": t.id,
                        "question": t.task.question,
                        "status": t.status,
                        "tool_limit": t.tool_limit,
                        "tool_calls": t.tool_calls,
                    }
                    for t in workflow.tasks.values()
                ],
            )

            def validate(value):
                ids = [h.id for h in value.hypotheses]
                if len(ids) != len(set(ids)):
                    raise DiagnosisValidationError(
                        "Hypothesis IDs must be unique.", code="hypothesis_ids"
                    )
                validate_report(
                    InvestigationReport(
                        findings=[
                            h.model_dump(
                                exclude={
                                    "id",
                                    "assessment",
                                    "missing_observations",
                                }
                            )
                            for h in value.hypotheses
                        ],
                        missing_observations=[],
                    ),
                    state,
                    harness,
                )
                if value.action == "investigate":
                    revision = task_revision(value.hypotheses)
                    pending = sum(reusable_task(t, revision) is None
                                  for t in value.tasks)
                    if not can_investigate(tasks=pending):
                        if can_investigate():
                            raise DiagnosisValidationError(
                                "Reduce the number of new tasks to fit "
                                "the remaining batch allocation.",
                                code="task_allocation",
                            )
                        raise DiagnosisValidationError(
                            "Investigation closed; submit retained findings.",
                            code="investigation_closed",
                        )
                    for task in value.tasks:
                        refs = task.context_refs
                        if (len(refs) != len(set(refs))
                                or any(r not in state.reference_map
                                       or not r.startswith(("E", "K"))
                                       for r in refs)):
                            raise DiagnosisValidationError(
                                "Context needs unique retained E/K handles.",
                                code="task_context",
                            )
                        grants = {t.name for t in role_tools(tools, task.role)}
                        if not set(task.requested_tools) <= grants:
                            raise DiagnosisValidationError(
                                "Requested tools exceed this role's grants.",
                                code="task_permissions",
                            )
                        if any(not c.strip()
                               for c in task.acceptance_criteria):
                            raise DiagnosisValidationError(
                                "Acceptance criteria cannot be blank.",
                                code="task_criteria",
                            )
                        if not set(task.hypothesis_ids) <= set(ids):
                            raise DiagnosisValidationError(
                                "Use board hypothesis IDs.",
                                code="hypothesis_ids",
                            )
                        if (
                            task.role == "knowledge"
                            and not knowledge_available
                        ):
                            raise DiagnosisValidationError(
                                "Knowledge unavailable.",
                                code="unavailable_role",
                            )
                else:
                    remote_telemetry = {
                        t.name for t in role_tools(tools, "telemetry")
                        if t.name != "read_evidence"
                    }
                    if (
                        can_investigate()
                        and remote_telemetry
                        and not workflow.tasks
                        and not harness.evidence()
                    ):
                        raise DiagnosisValidationError(
                            "No remote investigation has run. "
                            "Empty local catalogues "
                            "are not search results; delegate a "
                            "telemetry task first.",
                            code="investigation_required",
                        )
                    draft = resolve_proposal(
                        value.proposal, state, harness.evidence()
                    )
                    harness.validate_draft(draft)
                    if draft.feedback_updates:
                        raise DiagnosisValidationError(
                            "Memory writes unavailable.",
                            code="memory_disabled",
                        )

            decision = await invoke(
                f"coordinate-{workflow.decision}",
                "coordinator",
                COORDINATOR_PROMPT,
                prompt + "\nCoordinator context:\n" + json_text(packet),
                CoordinatorDecision,
                [
                    t
                    for t in tools
                    if (t.name == "read_evidence" and harness.evidence())
                    or (t.name == "read_knowledge" and state.knowledge)
                ],
                investigation=False,
                validator=validate,
            )
            workflow.hypotheses = decision.hypotheses
            workflow.decision += 1
            if decision.action == "submit":
                draft = resolve_proposal(
                    decision.proposal, state, harness.evidence()
                )
                workflow.draft = Diagnosis(
                    incident_id=state.incident.id,
                    **draft.model_dump(exclude={"feedback_updates"}),
                )
                workflow.phase = "review"
            else:
                revision = task_revision(workflow.hypotheses)
                previous_tasks = [reusable_task(t, revision)
                                  for t in decision.tasks]
                pending = sum(t is None for t in previous_tasks)
                pool = batch_tool_budget()
                share, extra = divmod(pool, pending) if pending else (0, 0)
                token_pool = harness.investigation_token_pool
                token_share = (token_pool // pending
                               if token_pool is not None and pending else 0)
                ids = []
                for number, (task, previous) in enumerate(
                    zip(decision.tasks, previous_tasks)
                ):
                    key = f"decision-{workflow.decision}-task-{number}"
                    allowance = 0
                    if previous is None:
                        allowance = share + int(extra > 0)
                        extra = max(0, extra - 1)
                    workflow.tasks[key] = TaskRecord(
                        id=key,
                        round=workflow.round,
                        task=task,
                        input_revision=revision,
                        tool_limit=allowance,
                        token_limit=(token_share if previous is None else 0)
                        if token_pool is not None else None,
                        status="completed" if previous else "pending",
                        report=previous.report if previous else None,
                        report_sha256=(previous.report_sha256
                                       if previous else None),
                        inputs=freeze_task_input(
                            state, key, task, harness.evidence(),
                            [t.name for t in role_tools(tools, task.role)
                             if not task.requested_tools
                             or t.name in task.requested_tools],
                        ),
                    )
                    ids.append(key)
                workflow.active_tasks = ids
                workflow.batches += 1
                workflow.phase = "investigate"
            harness.emit(
                "collaboration.decision",
                data={
                    "decision": workflow.decision,
                    "action": decision.action,
                    "tasks": workflow.active_tasks
                    if decision.action == "investigate"
                    else [],
                    "reason": decision.reason,
                },
            )
            await harness.checkpoint()
            return {}

        def dispatch(_):
            if workflow.phase == "investigate":
                return [
                    Send("investigate", {"task_id": key})
                    for key in workflow.active_tasks
                ]
            return "review"

        async def investigate(data):
            record = workflow.tasks[data["task_id"]]
            if record.status in {"completed", "incomplete", "failed"}:
                return {
                    "completed": {record.id: record.report_sha256}
                }
            record.status = "running"
            names = set(record.inputs.allowed_tools)
            exposed = [t for t in tools if t.name in names]
            if (sha256(record.inputs.prompt.encode()).hexdigest()
                    != record.inputs.sha256
                    or {t.name for t in exposed} != names):
                raise ValueError("task input or tool contract changed")

            def reserve():
                return (
                    1
                    + state.limits.diagnosis_repairs
                    + sum(
                        workflow.tasks[key].status in {"pending", "running"}
                        for key in workflow.active_tasks
                        if key != record.id
                    )
                )

            outcome = "completed"
            try:
                report = await invoke(
                    record.id,
                    record.task.role,
                    INVESTIGATOR_PROMPT,
                    record.inputs.prompt,
                    InvestigationReport,
                    exposed,
                    reserved_calls=reserve,
                    validator=lambda value: validate_report(
                        value, state, harness, record
                    ),
                )
            except (TaskBudgetExhausted, TaskReportInvalid,
                    TaskContextExceeded) as exc:
                outcome = "incomplete"
                report = InvestigationReport(
                    findings=[],
                    outcome="blocked",
                    reason_code=("context_limit" if isinstance(
                        exc, TaskContextExceeded) else "invalid_report"
                        if isinstance(exc, TaskReportInvalid) else "budget"),
                    missing_observations=[
                        "Task context, output allocation or schema repairs "
                        "exhausted before a valid "
                        "report. Inspect retained original evidence; no "
                        "specialist finding was accepted."
                    ],
                )
                harness.emit(
                    "collaboration.task_incomplete",
                    data={
                        "task_id": record.id,
                        "reason": (
                            "model_budget_reserved"
                            if isinstance(exc, TaskBudgetExhausted)
                            else "context_limit" if isinstance(
                                exc, TaskContextExceeded)
                            else "invalid_specialist_report"
                        ),
                    },
                )
            except ModelCallFailed as exc:
                outcome = ("incomplete" if exc.failure.category
                           == "token_budget" else "failed")
                report = InvestigationReport(
                    findings=[],
                    outcome="blocked",
                    reason_code=("budget" if exc.failure.category
                                 == "token_budget" else "model_failure"),
                    missing_observations=[
                        "Specialist model failed after bounded retries: "
                        + exc.failure.category
                        + ". Retained tool evidence remains available."
                    ],
                )
                harness.emit(
                    "collaboration.task_incomplete",
                    data={
                        "task_id": record.id,
                        "reason": exc.failure.category,
                    },
                )
            # No await between result publication and task status update.
            record.report = report
            record.status = ("incomplete" if outcome == "completed"
                             and report.outcome != "answered" else outcome)
            record.report_sha256 = sha256(
                report.model_dump_json().encode()
            ).hexdigest()
            await harness.checkpoint()
            return {"completed": {record.id: record.report_sha256}}

        async def join(_):
            if workflow.phase != "investigate":
                return {}
            if not all(
                workflow.tasks[k].status
                in {"completed", "incomplete", "failed"}
                for k in workflow.active_tasks
            ):
                raise RuntimeError("investigation joined before tasks settled")
            workflow.phase = "coordinate"
            await harness.checkpoint()
            return {}

        async def review(_):
            if workflow.phase != "review":
                return {}
            harness.role = "verifier"
            try:
                result = await harness.verifier.verify(
                    state, workflow.draft, harness.evidence(), harness
                )
            finally:
                harness.role = None
            gaps = [c for c in result.claims if c.verdict != "supported"]
            if (
                not gaps
                and workflow.draft.status == DiagnosisStatus.INCONCLUSIVE
            ):
                gaps = [c for c in result.claims if c.kind == "uncertainty"]
            can_continue = (
                (gaps or not result.task_completed)
                and workflow.round < 2
                and workflow.decision < 6
                and state.limits.model_calls - state.model_calls
                >= 3 + state.limits.diagnosis_repairs
            )
            if can_continue:
                workflow.reviews.append(
                    state.verification.model_copy(deep=True)
                )
                state.verification = None
                workflow.feedback = gaps
                workflow.round += 1
                workflow.phase = "coordinate"
                harness.emit(
                    "collaboration.reinvestigation",
                    data={"round": workflow.round, "gaps": len(gaps)},
                )
            else:
                workflow.phase = "completed"
            await harness.checkpoint()
            return {}

        graph = StateGraph(GraphState)
        graph.add_node("coordinate", coordinate)
        graph.add_node("investigate", investigate)
        graph.add_node("join", join)
        graph.add_node("review", review)
        graph.add_edge(START, "coordinate")
        graph.add_conditional_edges("coordinate", dispatch)
        graph.add_edge("investigate", "join")
        graph.add_edge("join", "coordinate")
        graph.add_conditional_edges(
            "review",
            lambda _: "coordinate" if workflow.phase == "coordinate" else END,
        )
        app = graph.compile(checkpointer=self.checkpointer)
        config = {
            "configurable": {"thread_id": f"{state.id}:collaboration-v7"},
            "max_concurrency": 2,
            "recursion_limit": 30,
        }
        saved = (
            await self.checkpointer.aget_tuple(config)
            if self.checkpointer is not None
            else None
        )
        await app.ainvoke(None if saved else {"completed": {}}, config)
        return DiagnosisDraft(
            **workflow.draft.model_dump(exclude={"incident_id"})
        )

    async def _invoke(
        self,
        key,
        role,
        system,
        prompt,
        schema,
        tools,
        harness,
        recursion_limit,
        *,
        reserved_calls=0,
        investigation=True,
        validator=None,
    ):
        state = harness.state
        role_token = harness._role.set(role)
        task_token = harness._task_id.set(key)
        try:
            harness.emit("agent.phase_started", data={"phase": key})
            await harness.checkpoint()
            config = {
                "recursion_limit": recursion_limit,
                "callbacks": self.callbacks,
                "run_name": role,
                "configurable": {"thread_id": f"{state.id}:{key}"},
            }
            saved = (
                await self.checkpointer.aget_tuple(config)
                if self.checkpointer is not None
                else None
            )
            agent = create_agent(
                model=self.model,
                tools=tools,
                system_prompt=system,
                response_format=schema,
                middleware=[
                    HarnessMiddleware(
                        harness,
                        self.fallback,
                        output_name=schema.__name__,
                        validator=validator,
                        role=role,
                        reserved_calls=reserved_calls,
                        investigation=investigation,
                        allowed_tools=tuple(t.name for t in tools),
                    )
                ],
                checkpointer=self.checkpointer,
            )
            with observation(
                self.client,
                f"agent.{role}",
                metadata={
                    "run_id": str(state.id),
                    "task_id": key,
                    "resumed": bool(saved),
                },
            ):
                result = await agent.ainvoke(
                    None
                    if saved
                    else {"messages": [{"role": "user", "content": prompt}]},
                    config,
                )
            output = schema.model_validate(result["structured_response"])
            if validator:
                validator(output)
            harness.emit("agent.phase_completed", data={"phase": key})
            return output
        finally:
            harness._task_id.reset(task_token)
            harness._role.reset(role_token)
