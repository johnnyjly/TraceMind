"""A shared admission cap for investigator, fallback, retry and judge calls."""

from threading import Lock
from langchain_core.callbacks import BaseCallbackHandler


from tracemind.agent.errors import ModelCallBudgetExceeded


class ModelCallBudget(BaseCallbackHandler):
    raise_error = True

    def __init__(self, limit: int):
        if limit < 1:
            raise ValueError("model-call limit must be positive")
        self.limit = limit
        self.calls = 0
        self._lock = Lock()

    def on_chat_model_start(self, serialized, messages, **kwargs):
        with self._lock:
            if self.calls >= self.limit:
                raise ModelCallBudgetExceeded(
                    "experiment model-call budget exhausted"
                )
            self.calls += 1
