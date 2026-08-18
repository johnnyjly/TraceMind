"""Verify real SIGINT produces a clean CLI exit in both worker modes."""

import os
import select
import signal
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("once", [False, True])
def test_worker_cli_handles_sigint_without_traceback(tmp_path, once):
    """Interrupt asyncio after startup without touching local run databases."""
    script = Path(__file__).resolve().parents[1] / "scripts" / "run_worker.py"
    # Publish readiness inside the event loop so the signal does not race
    # imports or asyncio's SIGINT handler installation. Continuous mode uses
    # the real empty-queue poll; one-shot mode waits without a model provider.
    code = f"""
import asyncio
import runpy
import sys
from tracemind.agent.worker import RunWorker
original = RunWorker.run_forever
async def ready(self):
    print('READY', flush=True)
    if {once!r}:
        await asyncio.Event().wait()
    else:
        await original(self)
if {once!r}:
    RunWorker.run_once = ready
else:
    RunWorker.run_forever = ready
sys.argv = [{str(script)!r}] + (['--once'] if {once!r} else [])
runpy.run_path({str(script)!r}, run_name='__main__')
"""
    environment = os.environ.copy()
    environment.update({
        "TRACEMIND_AGENT_RUN_DATABASE": str(tmp_path / "runs.sqlite3"),
        "TRACEMIND_AGENT_CHECKPOINT_DATABASE": str(tmp_path / "graph.db"),
    })
    process = subprocess.Popen(
        [sys.executable, "-c", code], cwd=tmp_path, env=environment,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    try:
        readable, _, _ = select.select([process.stdout], [], [], 10)
        assert readable, "worker did not reach its event loop"
        assert process.stdout.readline().strip() == "READY"
        process.send_signal(signal.SIGINT)
        stdout, stderr = process.communicate(timeout=5)
    finally:
        if process.poll() is None:
            process.kill()
        process.communicate()
    assert process.returncode == 0
    assert "Worker stopped by user" in stderr
    assert "Traceback" not in stdout + stderr
    assert "CancelledError" not in stdout + stderr
    assert "KeyboardInterrupt" not in stdout + stderr
