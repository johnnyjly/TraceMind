"""Bound and log internal LightRAG model calls without hidden SDK retries."""

import asyncio
from collections.abc import Awaitable, Callable
import logging
from time import monotonic

from langchain_core.messages import AIMessage

from tracemind.rag.config import RagSettings


logger = logging.getLogger(__name__)
Completion = Callable[..., Awaitable[AIMessage]]


class RagModel:
    """Own the model budget of one serialized CLI retrieval operation.

    This is not an Agent harness integration. Before exposing RAG as an
    Agent tool, its internal calls must also consume that run's budget.
    """

    def __init__(
        self, settings: RagSettings, completion: Completion | None = None,
    ) -> None:
        """Accept external model execution without choosing a provider."""
        self.settings = settings
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.unreported_calls = 0
        self._completion = completion

    def __deepcopy__(self, memo):
        """Keep counters shared with the callback copied by LightRAG."""
        return self

    def reset(self) -> None:
        """Reset accounting before a new serialized retrieval operation."""
        self.calls = self.input_tokens = self.output_tokens = 0
        self.unreported_calls = 0

    async def complete(
        self, prompt, system_prompt=None, history_messages=None,
        keyword_extraction=False, **kwargs,
    ) -> str:
        """Invoke the supplied model with counted admission and hard limits."""
        if self._completion is None:
            raise RuntimeError("RAG requires an injected model completion")
        if self.calls >= self.settings.llm_max_calls:
            raise RuntimeError("RAG model-call budget exhausted")
        self.calls += 1
        self.unreported_calls += 1
        messages = list(history_messages or [])
        if system_prompt:
            messages.insert(0, {"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        started = monotonic()
        try:
            async with asyncio.timeout(self.settings.timeout_seconds):
                response = await self._completion(
                    messages, max_tokens=self.settings.llm_output_tokens,
                    keyword_extraction=keyword_extraction)
        finally:
            logger.info("rag.model call=%s elapsed_ms=%.1f", self.calls,
                        (monotonic() - started) * 1000)
        usage = response.usage_metadata
        if usage is not None:
            self.unreported_calls -= 1
            self.input_tokens += usage["input_tokens"]
            self.output_tokens += usage["output_tokens"]
        if not isinstance(response.content, str):
            raise ValueError("RAG model must return textual content")
        return response.content
