from uuid import uuid4

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from tracemind.agent import ConsoleTraceCallback


def test_console_trace_prints_model_and_tool_activity() -> None:
    output: list[str] = []
    callback = ConsoleTraceCallback(write=output.append)
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

    rendered = "\n".join(output)
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
