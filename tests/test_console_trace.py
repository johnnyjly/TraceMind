import logging
import subprocess
import sys
from uuid import uuid4

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from tracemind.agent import ConsoleTraceCallback


def test_console_trace_logs_model_and_tool_activity(caplog) -> None:
    caplog.set_level(logging.DEBUG, logger="tracemind.agent.console_trace")
    callback = ConsoleTraceCallback()
    run_id = uuid4()

    callback.on_chat_model_start(
        {},
        [[HumanMessage(content="Investigate checkout")]],
        run_id=run_id,
    )
    callback.on_llm_end(
        LLMResult(
            generations=[
                [
                    ChatGeneration(
                        message=AIMessage(
                            content="Checking payment traces",
                            tool_calls=[
                                {
                                    "name": "search_traces",
                                    "args": {"service": "payment"},
                                    "id": "call-1",
                                    "type": "tool_call",
                                }
                            ],
                        )
                    )
                ]
            ]
        ),
        run_id=run_id,
    )
    callback.on_tool_start(
        {"name": "search_traces"},
        "",
        run_id=run_id,
        inputs={"service": "payment", "google_api_key": "not-for-logs"},
    )
    callback.on_tool_end(
        ToolMessage(
            content='[{"error_span_count": 1}]', tool_call_id="call-1"
        ),
        run_id=run_id,
    )

    rendered = caplog.text
    assert len(caplog.records) == 4
    assert all(record.levelno == logging.DEBUG for record in caplog.records)
    assert all(record.run_id == str(run_id) for record in caplog.records)
    assert "=== MODEL INPUT ===" in rendered
    assert "Investigate checkout" in rendered
    assert "=== MODEL RESPONSE ===" in rendered
    assert "search_traces" in rendered
    assert '"service": "payment"' in rendered
    assert "not-for-logs" not in rendered
    assert "[REDACTED]" in rendered
    assert "=== TOOL RESULT ===" in rendered


def test_console_trace_truncates_long_events() -> None:
    output: list[str] = []
    callback = ConsoleTraceCallback(max_chars=20, write=output.append)

    callback.on_tool_end("x" * 100, run_id=uuid4())

    assert "characters omitted" in output[0]
    assert len(output[0]) < 100


def test_logger_filters_details_but_preserves_errors_and_parent_ids(caplog):
    caplog.set_level(logging.INFO, logger="tracemind.agent.console_trace")
    callback = ConsoleTraceCallback(max_chars=120)
    run_id, parent_id = uuid4(), uuid4()
    callback.on_tool_end("hidden detail", run_id=run_id)
    assert not caplog.records
    for method in (callback.on_llm_error, callback.on_tool_error):
        method(RuntimeError("x" * 500), run_id=run_id,
               parent_run_id=parent_id)
    assert len(caplog.records) == 2
    for record in caplog.records:
        assert record.levelno == logging.ERROR
        assert record.run_id == str(run_id)
        assert record.parent_run_id == str(parent_id)
        assert "characters omitted" in record.getMessage()
        assert "x" * 500 not in record.getMessage()


def test_cli_logging_levels_format_and_no_duplicate_handlers():
    result = subprocess.run(
        [sys.executable, "-c", """
import logging
import runpy
from uuid import uuid4
from tracemind.agent import ConsoleTraceCallback
configure = runpy.run_path('scripts/run_evaluation.py')['configure_logging']
configure(verbose=False)
callback = ConsoleTraceCallback()
callback.on_tool_end('hidden_payload', run_id=uuid4())
logging.getLogger('tracemind.evaluation.cli').info('summary_marker')
configure(verbose=True)
callback.on_tool_end('visible_payload', run_id=uuid4())
logging.getLogger('httpx').debug('third_party_payload')
"""],
        capture_output=True, text=True, check=True,
    )
    assert result.stdout == ""
    assert result.stderr.count("summary_marker") == 1
    assert result.stderr.count("visible_payload") == 1
    assert "hidden_payload" not in result.stderr
    assert "third_party_payload" not in result.stderr
    assert "INFO tracemind.evaluation.cli [run=- parent=-]" in result.stderr
    assert "DEBUG tracemind.agent.console_trace [run=" in result.stderr
