"""Exercise progress deadlines through real model adapters, without network."""

import asyncio
import json

import httpx
import pytest
from langchain_core.messages import AIMessage
from openai import APIError

from tracemind.agent.runtime.deepseek import DeepSeekChat
from tracemind.agent.runtime.harness import RunHarness, RunStopped, _cache_usage
from tracemind.agent.runtime.progress import (
    model_progress_timeout, report_model_progress,
)
from tracemind.agent.runtime.recovery import ModelCallFailed
from tracemind.models.run import RunLimits, RunState, RunStatus, StopReason
from test_harness import INCIDENT, START, END


def harness(**limits):
    return RunHarness(RunState(
        incident=INCIDENT, investigation_start=START, investigation_end=END,
        status=RunStatus.RUNNING, allowed_tools=(),
        limits=RunLimits(model_timeout_seconds=0.12, retry_attempts=0, **limits),
    ))


def test_progress_outlives_old_total_timeout_without_extra_calls():
    run = harness()

    async def invoke():
        for _ in range(8):
            await asyncio.sleep(0.03)
            report_model_progress("reasoning", "synthetic")
        return AIMessage(content="done")

    result = asyncio.run(run.model_call(invoke, model_name="test"))
    assert result.content == "done"
    assert run.state.model_calls == 1
    event = next(e for e in run.state.events if e.kind == "model.completed")
    assert event.duration_ms > 120
    assert event.data["reasoning_events"] == 8
    assert "synthetic" not in run.state.model_dump_json()


@pytest.mark.parametrize("first", [False, True])
def test_no_output_and_midstream_stall_are_distinct(first):
    run = harness()

    async def invoke():
        if first:
            report_model_progress("reasoning", "synthetic")
        for _ in range(20):
            report_model_progress("output", "")
            await asyncio.sleep(0.03)

    with pytest.raises(ModelCallFailed) as caught:
        asyncio.run(run.model_call(invoke, model_name="test", required=False))
    assert caught.value.timed_out
    assert run.state.model_calls == 1
    assert run.state.usage.unreported_model_calls == 1
    event = next(e for e in run.state.events if e.kind == "model.failed")
    assert event.data["timeout_phase"] == ("idle" if first else "first_output")


def test_parallel_requests_do_not_refresh_each_other():
    async def exercise():
        active, silent = harness(), harness()

        async def busy():
            for _ in range(8):
                await asyncio.sleep(0.03)
                report_model_progress("output", "x")
            return AIMessage(content="done")

        async def stalled():
            await asyncio.sleep(1)

        results = await asyncio.gather(
            active.model_call(busy, model_name="test"),
            silent.model_call(stalled, model_name="test", required=False),
            return_exceptions=True,
        )
        assert isinstance(results[0], AIMessage)
        assert isinstance(results[1], ModelCallFailed)
        assert not any(e.kind == "model.progress" for e in silent.state.events)

    asyncio.run(exercise())


def test_explicit_overall_deadline_still_applies_during_progress():
    run = harness(duration_seconds=0.15)

    async def invoke():
        for _ in range(20):
            await asyncio.sleep(0.02)
            report_model_progress("reasoning", "x")

    with pytest.raises(RunStopped) as caught:
        asyncio.run(run.model_call(invoke, model_name="test"))
    assert caught.value.reason is StopReason.DEADLINE


class Events(httpx.AsyncByteStream):
    def __init__(self, events, *, delay=0, tail_delay=0):
        self.events, self.delay, self.tail_delay = events, delay, tail_delay
        self.closed = False

    async def __aiter__(self):
        for event in self.events:
            await asyncio.sleep(self.delay)
            yield ("data: " + json.dumps(event) + "\n\n").encode()
        if self.tail_delay:
            await asyncio.sleep(self.tail_delay)

    async def aclose(self):
        self.closed = True


def terminal(status="completed"):
    return {
        "type": "response." + status,
        "response": {
            "id": "resp_test", "object": "response", "created_at": 0,
            "model": "deepseek-v4-pro", "status": status,
            "error": ({"code": "server_error", "message": "synthetic"}
                      if status == "failed" else None),
            "incomplete_details": ({"reason": "max_output_tokens"}
                                   if status == "incomplete" else None),
            "output": [{"type": "message", "id": "msg_test",
                        "status": "completed", "role": "assistant",
                        "content": [{"type": "output_text", "text": '{}',
                                     "annotations": []}]}],
            "usage": {"input_tokens": 10, "output_tokens": 6, "total_tokens": 16,
                      "input_tokens_details": {"cached_tokens": 3},
                      "output_tokens_details": {"reasoning_tokens": 4}},
        },
    }


async def responses_call(stream, *, timeout=0.12):
    def respond(request):
        assert request.url.path == "/responses"
        assert json.loads(request.content)["stream"] is True
        return httpx.Response(200, headers={"content-type": "text/event-stream"},
                              stream=stream)

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as http:
        model = DeepSeekChat(model="deepseek-v4-pro", api_key="test",
                             base_url="https://api.deepseek.com",
                             http_async_client=http, use_responses_api=True)
        async with model_progress_timeout(timeout) as progress:
            result = await model.ainvoke("Inspect")
        return result, progress.summary()


def test_responses_reasoning_progress_and_terminal_usage():
    stream = Events([
        *[{"type": "response.reasoning_text.delta", "delta": "推理"} for _ in range(8)],
        {"type": "response.output_text.delta", "delta": '{}'}, terminal(),
    ], delay=0.03)
    result, progress = asyncio.run(responses_call(stream))
    assert result.text == '{}'
    assert result.usage_metadata["total_tokens"] == 16
    assert result.usage_metadata["output_token_details"]["reasoning"] == 4
    assert progress["reasoning_events"] == 8
    assert progress["reasoning_bytes"] == len("推理".encode()) * 8
    assert stream.closed


@pytest.mark.parametrize("event", [
    {"type": "response.in_progress", "response": {"id": "resp_test"}},
    {"type": "response.reasoning_text.delta", "delta": ""},
])
def test_responses_heartbeats_do_not_extend_wait(event):
    stream = Events([event] * 20, delay=0.02)
    with pytest.raises(TimeoutError):
        asyncio.run(responses_call(stream))
    assert stream.closed


@pytest.mark.parametrize("status", ["completed", "incomplete"])
def test_responses_preserves_terminal_status(status):
    result, _ = asyncio.run(responses_call(Events([terminal(status)])))
    assert result.response_metadata["status"] == status
    assert result.usage_metadata["total_tokens"] == 16


def test_responses_failed_and_missing_terminal_are_errors():
    with pytest.raises(APIError):
        asyncio.run(responses_call(Events([terminal("failed")])))
    stream = Events([{"type": "response.reasoning_text.delta", "delta": "x"}])
    with pytest.raises(httpx.RemoteProtocolError):
        asyncio.run(responses_call(stream))
    assert stream.closed


def test_cancellation_closes_provider_stream():
    stream = Events([{"type": "response.reasoning_text.delta", "delta": "x"}],
                    tail_delay=10)

    async def exercise():
        task = asyncio.create_task(responses_call(stream))
        await asyncio.sleep(0.04)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(exercise())
    assert stream.closed


def test_chat_stream_preserves_cache_usage_and_requires_finish():
    async def exercise(finish):
        chunks = [{
            "id": "chat_test", "object": "chat.completion.chunk", "created": 0,
            "model": "deepseek-v4-pro",
            "choices": [{"index": 0, "delta": {"role": "assistant", "content": "ok"},
                         "finish_reason": "stop" if finish else None}],
        }, {
            "id": "chat_test", "object": "chat.completion.chunk", "created": 0,
            "model": "deepseek-v4-pro", "choices": [],
            "usage": {"prompt_tokens": 10, "completion_tokens": 6, "total_tokens": 16,
                      "prompt_cache_hit_tokens": 7,
                      "completion_tokens_details": {"reasoning_tokens": 4}},
        }]
        stream = Events(chunks)
        def respond(request):
            assert json.loads(request.content)["stream_options"]["include_usage"]
            return httpx.Response(200, headers={"content-type": "text/event-stream"},
                                  stream=stream)
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as http:
            model = DeepSeekChat(model="deepseek-v4-pro", api_key="test",
                                 http_async_client=http, use_responses_api=False)
            result = await model.ainvoke("Inspect")
        assert stream.closed
        assert result.text == "ok"
        assert result.response_metadata["finish_reason"] == "stop"
        assert result.usage_metadata["total_tokens"] == 16
        assert _cache_usage(result) == (7, 3)

    asyncio.run(exercise(True))
    with pytest.raises(httpx.RemoteProtocolError):
        asyncio.run(exercise(False))


def test_native_openai_stream_refreshes_shared_progress_monitor():
    from langchain_openai import ChatOpenAI
    from tracemind.agent.runtime.progress import ModelProgressCallback

    async def exercise():
        stream = Events([
            *[{"type": "response.output_text.delta", "delta": "x",
               "item_id": "msg_test", "output_index": 0, "content_index": 0}
              for _ in range(16)],
            terminal(),
        ], delay=0.04)
        def respond(request):
            return httpx.Response(
                200, headers={"content-type": "text/event-stream"}, stream=stream,
            )
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as http:
            model = ChatOpenAI(
                model="test", api_key="test", http_async_client=http,
                use_responses_api=True, streaming=True, stream_chunk_timeout=None,
                callbacks=[ModelProgressCallback()], max_retries=0,
            )
            async with model_progress_timeout(0.5) as progress:
                result = await model.ainvoke("Inspect")
        assert progress.summary()["output_events"] >= 16
        assert result.usage_metadata["total_tokens"] == 16
        assert stream.closed

    asyncio.run(exercise())


def test_native_callback_counts_thought_and_tool_deltas_not_metadata():
    from langchain_core.messages import AIMessageChunk
    from langchain_core.outputs import ChatGenerationChunk
    from tracemind.agent.runtime.progress import ModelProgressCallback

    async def exercise():
        callback = ModelProgressCallback()
        async with model_progress_timeout(1) as progress:
            for message in [
                AIMessageChunk(content="", response_metadata={"status": "in_progress"}),
                AIMessageChunk(content=[{"type": "thinking", "thinking": "thought"}]),
                AIMessageChunk(content="", tool_call_chunks=[
                    {"name": None, "args": "{}", "id": None, "index": 0},
                ]),
            ]:
                await callback.on_llm_new_token(
                    "", chunk=ChatGenerationChunk(message=message),
                )
        assert progress.counts == {"reasoning": 1, "tool_arguments": 1}

    asyncio.run(exercise())
