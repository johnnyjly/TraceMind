"""Configuration and diagnosis failures shared across service boundaries."""

import asyncio
from collections.abc import Sequence
from typing import TYPE_CHECKING
from uuid import UUID

if TYPE_CHECKING:
    from tracemind.models import Evidence
    from tracemind.models.run import RunState


class ModelConfigurationError(ValueError):
    """Raised when the selected model provider is not configured."""


class DiagnosisCancelled(asyncio.CancelledError):
    """Retain partial run data and task cancellation semantics."""

    def __init__(self, state: "RunState", evidence: "list[Evidence]") -> None:
        """Attach the mutable state finalized by the service's cleanup."""
        super().__init__("Diagnosis cancelled")
        self.state = state
        self.evidence = evidence


class DiagnosisError(RuntimeError):
    """Represent a diagnosis contract violation or an investigation failure.

    Optional state and evidence preserve the run outcome and partial
    findings for API error handling. The service may attach them while
    unwinding; underlying exceptions are retained through exception
    chaining.
    """

    def __init__(
        self,
        message: str,
        *,
        state: "RunState | None" = None,
        evidence: "list[Evidence] | None" = None,
    ) -> None:
        """Attach a message and optional partial evidence and execution
        state.
        """
        super().__init__(message)
        self.state = state
        self.evidence = evidence or []


class DiagnosisValidationError(DiagnosisError):
    """Carry a contract rule and offending citations for targeted repair."""

    def __init__(
        self, message: str, *, code: str, evidence_ids: Sequence[UUID] = (),
        knowledge_refs: Sequence[str] = (),
    ) -> None:
        """Keep validation details separate from provider exception text."""
        super().__init__(message)
        self.code = code
        self.evidence_ids = evidence_ids
        self.knowledge_refs = knowledge_refs

    def details(self) -> dict:
        """Bound citation feedback without hiding that more IDs failed."""
        result = {
            "code": self.code, "message": str(self),
            "evidence_ids": [str(i) for i in self.evidence_ids[:20]],
            "omitted_evidence_ids": max(0, len(self.evidence_ids) - 20),
        }
        if self.knowledge_refs:
            result.update({
                "knowledge_refs": list(self.knowledge_refs[:20]),
                "omitted_knowledge_refs": max(
                    0, len(self.knowledge_refs) - 20),
            })
        return result
