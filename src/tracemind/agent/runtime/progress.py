"""Request-local model progress, without retaining provider output text."""

import asyncio
from contextlib import asynccontextmanager
from contextvars import ContextVar

from langchain_core.callbacks import AsyncCallbackHandler


_current = ContextVar("model_progress", default=None)


def report_model_progress(kind, text):
    progress = _current.get()
    if (progress is not None and kind in {"reasoning", "output", "tool_arguments"}
            and isinstance(text, str) and text):
        progress.record(kind, len(text.encode("utf-8")))


class ModelProgress:
    def __init__(self, seconds, remaining_seconds=None, on_progress=None):
        self.loop = asyncio.get_running_loop()
        self.started = self.last = self.loop.time()
        self.seconds = seconds
        self.deadline = (None if remaining_seconds is None else
                         self.started + remaining_seconds)
        self.timer = asyncio.timeout_at(self.next_deadline())
        self.on_progress = on_progress
        self.first = {}
        self.counts = {}
        self.bytes = {}
        self.max_gap = 0.0
        self.last_notice = self.started
        self.closed = False

    def next_deadline(self):
        deadline = self.last + self.seconds
        return (min(deadline, self.deadline)
                if self.deadline is not None else deadline)

    def record(self, kind, size):
        if self.closed or self.timer.expired():
            return
        now = self.loop.time()
        first_kind = kind not in self.first
        self.max_gap = max(self.max_gap, now - self.last)
        self.last = now
        self.first.setdefault(kind, now - self.started)
        self.counts[kind] = self.counts.get(kind, 0) + 1
        self.bytes[kind] = self.bytes.get(kind, 0) + size
        self.timer.reschedule(self.next_deadline())
        # Keep events bounded; never persist individual reasoning fragments.
        if self.on_progress and (first_kind or now - self.last_notice >= 10):
            self.last_notice = now
            self.on_progress(self.summary())

    def summary(self):
        idle = self.loop.time() - self.last
        return {
            "first_progress_seconds": min(self.first.values(), default=None),
            "progress_events": sum(self.counts.values()),
            **{f"first_{kind}_seconds": self.first.get(kind)
               for kind in ("reasoning", "output", "tool_arguments")},
            **{f"{kind}_events": self.counts.get(kind, 0)
               for kind in ("reasoning", "output", "tool_arguments")},
            **{f"{kind}_bytes": self.bytes.get(kind, 0)
               for kind in ("reasoning", "output", "tool_arguments")},
            "last_progress_seconds": (
                self.last - self.started if self.first else None
            ),
            "max_progress_gap_seconds": max(self.max_gap, idle),
            "idle_seconds": idle,
        }


@asynccontextmanager
async def model_progress_timeout(seconds, *, remaining_seconds=None,
                                 on_progress=None):
    progress = ModelProgress(seconds, remaining_seconds, on_progress)
    token = _current.set(progress)
    try:
        async with progress.timer:
            yield progress
    finally:
        progress.closed = True
        _current.reset(token)


class ModelProgressCallback(AsyncCallbackHandler):
    """Observe exposed OpenAI/Google deltas in their native streaming paths."""

    async def on_llm_new_token(self, token, *, chunk=None, **kwargs):
        if chunk is None:
            report_model_progress("output", token)
            return
        message = chunk.message
        if isinstance(message.content, str):
            report_model_progress("output", message.content)
        else:
            for block in message.content:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "text":
                    report_model_progress("output", block.get("text"))
                elif block.get("type") == "thinking":
                    report_model_progress("reasoning", block.get("thinking"))
                elif block.get("type") == "reasoning":
                    report_model_progress("reasoning", block.get("reasoning"))
                    for part in block.get("summary", []):
                        if isinstance(part, dict):
                            report_model_progress("reasoning", part.get("text"))
        for call in message.tool_call_chunks:
            report_model_progress("tool_arguments", call.get("args"))
            report_model_progress("tool_arguments", call.get("name"))
