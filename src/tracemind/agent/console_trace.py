import json
from collections.abc import Callable
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage
from langchain_core.outputs import LLMResult


_SENSITIVE_KEYS = {
    "api_key",
    "authorization",
    "password",
    "secret",
    "token",
}


def _redact(value: Any) -> Any:
    """Recursively mask values under known sensitive dictionary keys.

    Free-text secrets and secrets embedded in serialized strings are not
    recognized by this key-based filter.
    """
    if isinstance(value, dict):
        redacted: dict[Any, Any] = {}
        for key, item in value.items():
            normalized = str(key).casefold().replace("-", "_")
            if normalized in _SENSITIVE_KEYS or any(
                normalized.endswith(f"_{suffix}") for suffix in _SENSITIVE_KEYS
            ):
                redacted[key] = "[REDACTED]"
            else:
                redacted[key] = _redact(item)
        return redacted
    if isinstance(value, (list, tuple)):
        return [_redact(item) for item in value]
    return value


class ConsoleTraceCallback(BaseCallbackHandler):
    """Print development model/tool payloads with per-event truncation.

    This callback is separate from the harness metadata trace. Key-based
    redaction is limited; free-text payloads may still contain sensitive
    data. Truncation affects console output only, not the model context.
    """

    def __init__(
        self,
        *,
        max_chars: int = 12_000,
        write: Callable[[str], None] = print,
    ) -> None:
        if max_chars < 1:
            raise ValueError("max_chars must be positive")
        self._max_chars = max_chars
        self._write = write

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[BaseMessage]],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        self._emit(
            "MODEL INPUT",
            [
                [message.model_dump(mode="json") for message in batch]
                for batch in messages
            ],
        )

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        replies: list[list[Any]] = []
        for batch in response.generations:
            rendered_batch: list[Any] = []
            for generation in batch:
                message = getattr(generation, "message", None)
                if isinstance(message, BaseMessage):
                    rendered_batch.append(message.model_dump(mode="json"))
                else:
                    rendered_batch.append(
                        getattr(generation, "text", str(generation))
                    )
            replies.append(rendered_batch)
        self._emit("MODEL RESPONSE", replies)

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        self._emit(
            "MODEL ERROR",
            {"type": type(error).__name__, "message": str(error)},
        )

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        inputs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        self._emit(
            "TOOL CALL",
            {
                "name": serialized.get("name", "unknown"),
                "input": inputs if inputs is not None else input_str,
            },
        )

    def on_tool_end(
        self,
        output: Any,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        if isinstance(output, BaseMessage):
            output = output.model_dump(mode="json")
        self._emit("TOOL RESULT", output)

    def _emit(self, label: str, payload: Any) -> None:
        rendered = json.dumps(
            _redact(payload),
            indent=2,
            ensure_ascii=False,
            default=str,
        )
        if len(rendered) > self._max_chars:
            omitted = len(rendered) - self._max_chars
            rendered = (
                f"{rendered[: self._max_chars]}\n"
                f"... <{omitted} characters omitted>"
            )
        self._write(f"\n=== {label} ===\n{rendered}")
