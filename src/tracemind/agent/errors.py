from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tracemind.models import Evidence
    from tracemind.models.run import RunState


class ModelConfigurationError(ValueError):
    """Raised when the selected model provider is not configured."""


class DiagnosisError(RuntimeError):
    """Raised when an agent result violates the diagnosis contract."""

    def __init__(
        self,
        message: str,
        *,
        state: "RunState | None" = None,
        evidence: "list[Evidence] | None" = None,
    ) -> None:
        super().__init__(message)
        self.state = state
        self.evidence = evidence or []
