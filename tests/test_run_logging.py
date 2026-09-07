"""Terminal logging views of structured run events."""

import logging
import subprocess
import sys
from datetime import UTC, datetime, timedelta

from tracemind.agent.runtime.harness import RunHarness
from tracemind.models import Incident
from tracemind.models.run import RunLimits, RunState


def make_harness() -> RunHarness:
    """Build a run without opening a JSONL file."""
    start = datetime(2024, 9, 1, tzinfo=UTC)
    state = RunState(
        incident=Incident(title="Checkout failure", description="HTTP 500"),
        investigation_start=start,
        investigation_end=start + timedelta(hours=1),
        limits=RunLimits(), allowed_tools=(),
    )
    return RunHarness(state)


def test_harness_logs_structured_metadata_only_when_debug_enabled(caplog):
    """Terminal filtering must not affect run state events."""
    caplog.set_level(logging.INFO, logger="tracemind.agent.runtime.harness")
    harness = make_harness()
    harness.emit("tool.started", data={"arguments_sha256": "digest"})
    assert len(harness.state.events) == 2
    assert not caplog.records

    caplog.set_level(logging.DEBUG, logger="tracemind.agent.runtime.harness")
    harness.emit("tool.completed", data={"result_count": 1})
    assert len(caplog.records) == 1
    record = caplog.records[0]
    assert record.event_kind == "tool.completed"
    assert record.run_id == str(harness.state.id)
    assert '"result_count":1' in record.getMessage()
    assert "Checkout failure" not in record.getMessage()


def test_cli_selects_terminal_event_detail_without_third_party_debug():
    """Only --verbose enables the harness event logger at DEBUG."""
    result = subprocess.run(
        [sys.executable, "-c", """
import logging
import runpy
from datetime import UTC, datetime, timedelta
from tracemind.agent.runtime.harness import RunHarness
from tracemind.models import Incident
from tracemind.models.run import RunLimits, RunState
configure = runpy.run_path('scripts/run_experiment.py')['configure_logging']
start = datetime(2024, 9, 1, tzinfo=UTC)
state = RunState(incident=Incident(title='failure', description='test'),
                 investigation_start=start,
                 investigation_end=start + timedelta(hours=1),
                 limits=RunLimits(), allowed_tools=())
configure(verbose=False)
harness = RunHarness(state)
logging.getLogger('tracemind.evaluation.cli').info('summary_marker')
configure(verbose=True)
harness.emit('tool.completed', data={'result_count': 1})
logging.getLogger('httpx').debug('third_party_payload')
"""],
        capture_output=True, text=True, check=True,
    )
    assert result.stdout == ""
    assert result.stderr.count("summary_marker") == 1
    assert result.stderr.count("tool.completed") == 1
    assert "run.started" not in result.stderr
    assert "third_party_payload" not in result.stderr
    assert "INFO tracemind.evaluation.cli [run=- parent=-]" in result.stderr
    assert "DEBUG tracemind.agent.runtime.harness [run=" in result.stderr
