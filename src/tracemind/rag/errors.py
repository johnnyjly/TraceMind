"""Expected storage failures at the optional retrieval boundary."""


class RetrievalUnavailable(RuntimeError):
    """Allow diagnosis to continue when a knowledge backend is unavailable."""


class RetrievalInputTooLong(ValueError):
    """A query/document pair would require silent model truncation."""
