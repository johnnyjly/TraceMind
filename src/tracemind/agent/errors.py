"""Configuration and diagnosis failures shared across service boundaries."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tracemind.models import Evidence
    from tracemind.models.run import RunState


class ModelConfigurationError(ValueError):
    """Raised when the selected model provider is not configured."""


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
