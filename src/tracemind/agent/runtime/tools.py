"""Authorized MCP/local tool bindings and run-local evidence collection."""

from collections.abc import Awaitable, Callable, Sequence
from copy import deepcopy
from datetime import datetime
import hashlib
from time import monotonic
from typing import Annotated

from langchain_core.tools import BaseTool, StructuredTool, tool
from mcp.types import Tool
from pydantic import Field

from tracemind.agent.workflow.context import json_text
from tracemind.agent.workflow.references import (
    model_references, original_reference, register_references,
)
from tracemind.agent.runtime.harness import RunHarness
from tracemind.mcp.client import ToolClient
from tracemind.mcp.contracts import TelemetryReply
from tracemind.mcp.results import (
    reference_content, telemetry_reply, uses_telemetry_contract,
)
from tracemind.models import Evidence, EvidenceSource
from tracemind.models.run import StopReason, ToolName
from tracemind.telemetry.errors import RemoteTelemetryError


LOCAL_TOOL_NAMES = frozenset({
    "read_evidence", "search_knowledge", "read_knowledge",
})
RESERVED_TOOL_NAMES = LOCAL_TOOL_NAMES | {"DiagnosisProposal"}
EVIDENCE_PAGE_SIZE = 20
MAX_EVIDENCE_PREVIEWS = 100


class EvidenceCapacityError(Exception):
    """Signal that retaining an observation would exceed the byte limit."""


class EvidenceCollector:
    """Store run-local observations with exact deduplication and stable IDs.

    The byte limit covers serialized evidence, not total Python memory or
    backend response buffers. A persistent run can snapshot additions through
    the on_change callback.
    """

    def __init__(
        self, *, max_bytes: int = 8_388_608,
        initial: Sequence[Evidence] = (),
        on_change: Callable[[], None] | None = None,
    ) -> None:
        """Initialize an evidence index with a serialized-byte capacity."""
        self._evidence: dict[str, Evidence] = {}
        self._fingerprints: dict[str, str] = {}
        self._documents: dict[str, str] = {}
        self._max_bytes = max_bytes
        self._on_change = None
        self.bytes_used = 0
        self.add(initial)
        self._on_change = on_change

    def add(self, items: Sequence[Evidence]) -> list[dict[str, object]]:
        """Store observations and return previews with canonical IDs.

        Exact content matches excluding the generated ID reuse the first ID.
        New observations are copied. On EvidenceCapacityError, earlier items
        in the batch remain stored; insertion is not transactional.
        """
        ids = []
        for item in items:
            data = item.model_dump(mode="json", exclude={"id"})
            fingerprint = hashlib.sha256(json_text(data).encode()).hexdigest()
            evidence_id = self._fingerprints.get(fingerprint)
            if evidence_id is None:
                evidence_id = str(item.id)
                document = json_text(item.model_dump(mode="json"))
                size = len(document.encode())
                if self.bytes_used + size > self._max_bytes:
                    raise EvidenceCapacityError
                # Own the stored observation; external mutation must not change
                # the evidence behind a previously issued citation.
                self._evidence[evidence_id] = item.model_copy(deep=True)
                self._documents[evidence_id] = document
                self._fingerprints[fingerprint] = evidence_id
                self.bytes_used += size
                if self._on_change is not None:
                    self._on_change()
            if evidence_id not in ids:
                ids.append(evidence_id)
        return [self.preview(key) for key in ids]

    def preview(self, evidence_id: str) -> dict[str, object]:
        """Return complete evidence or a marked preview for a known ID."""
        item = self._evidence[evidence_id]
        document = self._documents[evidence_id]
        if len(document.encode()) <= 1200:
            return item.model_dump(mode="json")
        return {
            "id": evidence_id,
            "source": item.source.value,
            "service": item.service[:120] if item.service else None,
            "observed_at": item.observed_at.isoformat(),
            "summary": item.summary[:240],
            "truncated": True,
            "detail_chars": len(document),
        }

    def read(
        self, evidence_id: str | None, offset: int, *,
        service: str | None = None, source: EvidenceSource | None = None,
    ) -> list[dict]:
        """Page previews by entry offset or JSON by character offset.

        Callers validate nonnegative offsets. Unknown IDs and out-of-range
        detail offsets return error records; next_offset=None ends paging.
        Filters apply only to the catalogue, whose offsets count matches.
        """
        if evidence_id is None:
            ids = [key for key, item in self._evidence.items()
                   if (service is None or item.service == service)
                   and (source is None or item.source == source)]
            end = offset + EVIDENCE_PAGE_SIZE
            result = [self.preview(key) for key in ids[offset:end]]
            result.append({
                "total_evidence": len(ids),
                "next_offset": end if end < len(ids) else None,
            })
            return result
        document = self._documents.get(evidence_id)
        if document is None:
            return [{"error": "unknown_evidence_id"}]
        if offset >= len(document):
            return [{"error": "offset_out_of_range"}]
        end = min(offset + 2048, len(document))
        return [{
            "id": evidence_id, "offset": offset,
            "content": document[offset:end],
            "next_offset": end if end < len(document) else None,
            "total_chars": len(document),
        }]

    @property
    def evidence(self) -> list[Evidence]:
        """Return collected evidence in a new list.

        The Evidence objects are shared with the collector; callers should
        treat them as read-only.
        """
        return list(self._evidence.values())


def context_index(items: list[dict]) -> dict:
    """Build application-owned compact references for context reduction."""
    # Artifact is application-owned metadata, not parsed model/backend text.
    index = []
    for item in items:
        if "id" in item:
            entry = {key: item[key] for key in
                     ("id", "source", "observed_at", "offset", "next_offset")
                     if key in item}
            if item.get("service") is not None:
                entry["service"] = str(item["service"])[:120]
            entry["summary"] = str(item.get("summary", "Stored detail"))[:80]
            entry["truncated"] = True
            index.append(entry)
        else:
            index.append(item)
    return {"context_index": index}


def _query_executor(
    start: datetime, end: datetime,
    collector: EvidenceCollector, harness: RunHarness,
):
    """Reuse evidence retention and preview policy for every transport."""
    harness.evidence = lambda: collector.evidence

    async def execute(
        name: ToolName,
        arguments: dict[str, object],
        call: Callable[[], Awaitable[TelemetryReply]],
        *, max_results: int,
    ) -> tuple[list[dict], dict]:
        """Execute a query and return previews with a compact index."""
        evidence_result = False
        added = 0

        async def collect() -> list[dict]:
            """Store observations, select previews, and trace retention."""
            nonlocal evidence_result, added
            reply = await call()
            harness.check_active()
            if reply.metadata is not None:
                return [reply.metadata]
            evidence_result = True
            observations = reply.evidence
            retained = [item for item in observations if (
                item.observed_at.utcoffset() is not None
                and start <= item.observed_at <= end
            )]
            excluded = len(observations) - len(retained)
            if excluded:
                harness.emit(
                    "observations.filtered", parent_span_id=harness.state.id,
                    data={"tool": name, "received": len(observations),
                          "excluded_outside_window": excluded},
                )
            observations = retained
            before = len(collector.evidence)
            try:
                previews = collector.add(observations)
            except EvidenceCapacityError:
                harness.emit("observations.limit",
                             parent_span_id=harness.state.id,
                             data={"stored_bytes": collector.bytes_used})
                harness.stop(StopReason.CONTEXT_LIMIT)
            finally:
                harness.state.evidence_ids = [e.id for e in collector.evidence]
                register_references(harness.state)
            result = previews[:max_results]
            omitted = len(previews) - len(result)
            if omitted:
                result.append({
                    "omitted_count": omitted,
                    "retrieval": (
                        "Use read_evidence to page stored evidence, filtering "
                        "by service/source when useful."
                        if "read_evidence" in harness.state.allowed_tools
                        else "Detail retrieval is unavailable under current "
                        "tool policy; omitted observations cannot be "
                        "inspected."
                    ),
                })
            if excluded:
                result.append({
                    "excluded_outside_window": excluded,
                    "guidance": (
                        "Observations outside the investigation window were "
                        "excluded and cannot be cited. An empty remaining "
                        "result does not establish service health."
                    ),
                })
            added = len(collector.evidence) - before
            harness.emit("observations.stored",
                         parent_span_id=harness.state.id,
                         data={"received": len(observations),
                               "added": len(collector.evidence) - before,
                               "duplicates": (
                                   len(observations)
                                   - (len(collector.evidence) - before)
                               ),
                               "omitted_previews": omitted,
                               "stored_bytes": collector.bytes_used})
            return result

        result = await harness.tool_call(name, arguments, collect)
        if evidence_result and not added:
            result.append({
                "kind": "query_feedback",
                "new_evidence": 0,
                "guidance": (
                    "No new evidence was added. Reuse the observations "
                    "already available; do not repeat this query unless "
                    "a specific missing detail can change the diagnosis. "
                    "An empty result does not establish service health."
                ),
            })
        result = model_references(result, harness.state)
        return result, (context_index(result) if evidence_result else {})

    return execute


def create_local_tools(
    collector: EvidenceCollector, harness: RunHarness,
    extra_tools: Sequence[BaseTool] = (),
) -> list[BaseTool]:
    """Bind tools that operate solely on the current investigation state."""
    @tool(response_format="content_and_artifact")
    async def read_evidence(
        evidence_id: str | None = None,
        offset: Annotated[int, Field(ge=0)] = 0,
        service: str | None = None,
        source: EvidenceSource | None = None,
    ) -> tuple[list[dict], dict]:
        """List stored previews, or page evidence JSON by ID.

        Pass a known E handle as evidence_id for missing details. Filter the
        catalogue by service and/or source. Follow next_offset only when more
        entries are needed to answer the question.

        Without an ID, offset counts evidence entries. With an ID it counts
        characters in the stored JSON and ignores catalogue filters.
        Only this run's evidence is visible.
        """
        async def read() -> list[dict]:
            """Read a page from this run's evidence collector."""
            return collector.read(
                original_reference(evidence_id, harness.state),
                offset, service=service, source=source)

        result = await harness.tool_call(
            "read_evidence",
            {"evidence_id": evidence_id, "offset": offset,
             "service": service, "source": source}, read, remote=False,
        )
        result = model_references(result, harness.state)
        artifact = context_index(result)
        artifact["detail_page"] = evidence_id is not None
        return result, artifact

    candidates = [read_evidence, *extra_tools]
    if len({t.name for t in candidates}) != len(candidates):
        raise ValueError("duplicate local tool names")
    return [t for t in candidates if t.name in harness.state.allowed_tools]


async def create_mcp_tools(
    client: ToolClient, *, start: datetime, end: datetime,
    collector: EvidenceCollector, harness: RunHarness,
    local_tools: Sequence[BaseTool] = (),
) -> list[BaseTool]:
    """Bind server-discovered schemas to the existing execution policy.

    Application-owned investigation dates are declared by the server schema,
    removed from model inputs and injected at dispatch. Discovery cannot
    expand the local allowlist or replace a local tool.
    """
    local = create_local_tools(collector, harness, local_tools)
    # Disabled local capabilities are never remote discovery requirements.
    allowed = (set(harness.state.allowed_tools) - LOCAL_TOOL_NAMES
               - {t.name for t in local})
    if not allowed or harness.finalization_reason is not None:
        return local
    started = monotonic()
    harness.check_active()
    harness.emit("mcp.discovery.started")
    try:
        await harness.checkpoint()
        definitions = await client.discover_tools()
    except RemoteTelemetryError as exc:
        harness.emit("mcp.discovery.failed", data={
            "error_type": type(exc).__name__},
            duration_ms=(monotonic() - started) * 1000)
        harness.stop(StopReason.TOOL_ERROR)
    tools = bind_mcp_tools(client, definitions, start=start, end=end,
                           collector=collector, harness=harness,
                           local_tools=local_tools)
    harness.emit("mcp.discovery.completed", data={
        "tool_names": [t.name for t in tools]},
        duration_ms=(monotonic() - started) * 1000)
    harness.check_active()
    return tools


def bind_mcp_tools(
    client: ToolClient, definitions: list[Tool], *, start: datetime,
    end: datetime, collector: EvidenceCollector, harness: RunHarness,
    local_tools: Sequence[BaseTool] = (),
) -> list[BaseTool]:
    """Bind discovered or recorded definitions using identical run policy."""
    local = create_local_tools(collector, harness, local_tools)
    local_names = {t.name for t in local}
    allowed = set(harness.state.allowed_tools) - LOCAL_TOOL_NAMES - local_names
    names = [t.name for t in definitions]
    if (len(names) != len(set(names))
            or (RESERVED_TOOL_NAMES | local_names).intersection(names)):
        harness.emit("mcp.discovery.failed", data={
            "category": "tool_contract_mismatch"})
        harness.stop(StopReason.TOOL_ERROR)
    selected = {t.name: t for t in definitions if t.name in allowed}
    if not allowed.issubset(selected):
        harness.emit("mcp.discovery.failed", data={
            "category": "tool_contract_mismatch"})
        harness.stop(StopReason.TOOL_ERROR)
    execute = _query_executor(start, end, collector, harness)

    def bind(definition: Tool) -> BaseTool:
        """Capture one remote definition without writing a query function."""
        schema = deepcopy(definition.inputSchema)
        properties = schema.get("properties", {})
        injected = {}
        for field, spec in list(properties.items()):
            context = spec.get("x-investigation-context")
            if context is not None:
                if context not in {"start", "end"}:
                    raise RemoteTelemetryError("tool_contract_mismatch")
                value = start if context == "start" else end
                injected[field] = value.isoformat()
                del properties[field]
        schema["required"] = [key for key in schema.get("required", [])
                              if key not in injected]
        if not definition.description:
            raise RemoteTelemetryError("tool_contract_mismatch")
        description = definition.description
        if injected:
            description += (
                "\n\nFor this investigation, TraceMind supplies "
                + ", ".join(injected)
                + " from the fixed investigation window. These fields are "
                "omitted from your input schema; do not provide them."
            )
        name = definition.name
        telemetry = uses_telemetry_contract(definition)
        if not telemetry:
            description += (
                "\n\nResults are reference material, not retained incident "
                "observations. IDs in these results are not diagnosis "
                "evidence IDs."
            )
        max_results = (definition.meta or {}).get(
            "tracemind/preview-limit", MAX_EVIDENCE_PREVIEWS,
        )
        if type(max_results) is not int or not (
            1 <= max_results <= MAX_EVIDENCE_PREVIEWS
        ):
            raise RemoteTelemetryError("tool_contract_mismatch")

        async def invoke(**arguments):
            """Execute an admitted RPC and apply run-local result handling."""
            async def remote():
                """Never let model arguments override bound context fields."""
                if set(arguments) & injected.keys():
                    raise RemoteTelemetryError(
                        "invalid_request", status_code=400)
                return await client.call(name, {**arguments, **injected})

            if telemetry:
                async def observations():
                    """Validate an explicitly declared telemetry envelope."""
                    return telemetry_reply(await remote())

                return await execute(name, arguments, observations,
                                     max_results=max_results)

            async def reference():
                """Keep generic text/JSON outside incident evidence storage."""
                return reference_content(await remote())

            result = await harness.tool_call(
                name, arguments, reference, evidence_result=False,
                retry_safe=False,
            )
            return result, {}

        return StructuredTool.from_function(
            coroutine=invoke, name=name, description=description,
            args_schema=schema, response_format="content_and_artifact",
        )

    try:
        tools = [bind(definition) for definition in selected.values()]
    except RemoteTelemetryError as exc:
        harness.emit("mcp.discovery.failed", data={"category": exc.category})
        harness.stop(StopReason.TOOL_ERROR)
    return [*tools, *local]
