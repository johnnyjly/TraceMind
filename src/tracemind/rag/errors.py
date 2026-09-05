"""Expected storage failures at the optional retrieval boundary."""


class RetrievalUnavailable(RuntimeError):
    """Allow diagnosis to continue when a knowledge backend is unavailable."""
