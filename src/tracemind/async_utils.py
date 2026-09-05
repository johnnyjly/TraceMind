"""Await blocking work without abandoning an in-flight operation."""

import asyncio
from functools import partial
from collections.abc import Callable
from typing import ParamSpec, TypeVar

P = ParamSpec("P")
T = TypeVar("T")


async def finish_task(task: asyncio.Task[T]) -> T:
    """Drain owned work and preserve cancellation if that work also fails."""
    cancelled = None
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError as exc:
            cancelled = exc
        except Exception as exc:
            if cancelled is not None:
                raise cancelled from exc
            raise
    if cancelled is not None:
        try:
            task.result()
        except Exception as exc:
            raise cancelled from exc
        raise cancelled
    return task.result()


async def run_io(
    function: Callable[P, T], /, *args: P.args, **kwargs: P.kwargs,
) -> T:
    """Finish a thread operation before propagating caller cancellation."""
    return await finish_task(asyncio.create_task(
        asyncio.to_thread(partial(function, *args, **kwargs))))
