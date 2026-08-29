"""Await blocking storage work without abandoning an in-flight transaction."""

import asyncio
from functools import partial
from collections.abc import Callable
from typing import ParamSpec, TypeVar

P = ParamSpec("P")
T = TypeVar("T")


async def run_io(
    function: Callable[P, T], /, *args: P.args, **kwargs: P.kwargs,
) -> T:
    """Finish the thread operation before propagating caller cancellation.

    A cancelled to_thread await does not stop SQLite. Waiting for completion
    prevents a late progress write from overtaking terminal finalization.
    """
    task = asyncio.create_task(
        asyncio.to_thread(partial(function, *args, **kwargs)))
    cancelled = None
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError as exc:
            cancelled = exc
    result = task.result()
    if cancelled is not None:
        raise cancelled
    return result
