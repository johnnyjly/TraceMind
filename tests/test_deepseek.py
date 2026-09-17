"""DeepSeek wire protocol tests without external model calls."""

import asyncio
import json

import httpx
from langchain_core.messages import HumanMessage, ToolMessage

from tracemind.agent.runtime.deepseek import DeepSeekChat
from tracemind.agent.runtime.llm import create_chat_model
from tracemind.agent.verification import review_schema, review_packet
from tracemind.config import Settings
from test_verifier import seed


def test_thinking_defaults_and_explicit_opt_out():
    settings = Settings(
        _env_file=None, llm_provider="deepseek", deepseek_api_key="test"
    )
    model = create_chat_model(settings)
    assert isinstance(model, DeepSeekChat)
    assert model.extra_body == {"thinking": {"type": "enabled"}}
    assert model.reasoning_effort == "high"
    disabled = create_chat_model(
        settings.model_copy(update={"deepseek_thinking": False})
    )
    assert disabled.extra_body == {"thinking": {"type": "disabled"}}


def test_reasoning_survives_tool_turn_and_message_serialization():
    from langchain_core.messages import message_to_dict, messages_from_dict

    requests = []

    def respond(request):
        payload = json.loads(request.content)
        requests.append(payload)
        if len(requests) == 1:
            message = {
                "role": "assistant",
                "content": "",
                "reasoning_content": "Synthetic provider reasoning.",
                "tool_calls": [
                    {
                        "id": "read-1",
                        "type": "function",
                        "function": {"name": "read", "arguments": "{}"},
                    }
                ],
            }
        else:
            previous = next(
                m for m in payload["messages"] if m["role"] == "assistant"
            )
            assert (
                previous["reasoning_content"]
                == "Synthetic provider reasoning."
            )
            assert previous["tool_calls"][0]["id"] == "read-1"
            message = {
                "role": "assistant",
                "content": "Done",
                "reasoning_content": "Read completed.",
            }
        return httpx.Response(
            200,
            json={
                "id": "mock",
                "object": "chat.completion",
                "created": 0,
                "model": "deepseek-v4-pro",
                "choices": [
                    {"index": 0, "finish_reason": "stop", "message": message}
                ],
            },
        )

    async def exercise():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(respond)
        ) as http:
            model = DeepSeekChat(
                model="deepseek-v4-pro",
                api_key="test",
                base_url="https://api.deepseek.com",
                http_async_client=http,
                use_responses_api=False,
                extra_body={"thinking": {"type": "enabled"}},
            )
            bound = model.bind_tools(
                [
                    {
                        "name": "read",
                        "description": "Read",
                        "parameters": {"type": "object", "properties": {}},
                    }
                ]
            )
            initial = HumanMessage(content="Inspect")
            response = await bound.ainvoke([initial])
            restored = messages_from_dict([message_to_dict(response)])[0]
            await bound.ainvoke(
                [
                    initial,
                    restored,
                    ToolMessage(content="Observed", tool_call_id="read-1"),
                ]
            )

    asyncio.run(exercise())
    assert len(requests) == 2


def test_stream_chunks_preserve_reasoning():
    from langchain_core.messages import AIMessageChunk

    model = DeepSeekChat(model="deepseek-v4-pro", api_key="test")
    chunk = model._convert_chunk_to_generation_chunk(
        {
            "choices": [
                {
                    "delta": {
                        "role": "assistant",
                        "content": "",
                        "reasoning_content": "synthetic",
                    },
                    "index": 0,
                }
            ]
        },
        AIMessageChunk,
        None,
    )
    assert chunk.message.additional_kwargs["reasoning_content"] == "synthetic"


def test_correctness_schema_matches_held_out_reference_contract():
    subject = seed()
    assert review_schema(review_packet(subject))["properties"][
        "root_cause_correctness"
    ] == {"type": "null"}
    packet = review_packet(subject, {"root_cause_service": "payment"})
    assert review_schema(packet)["properties"]["root_cause_correctness"] == {
        "type": "boolean"
    }


def test_durable_campaign_budget_cannot_reset_or_race(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from tracemind.evaluation.budget import ModelCallBudget
    from tracemind.agent.errors import ModelCallBudgetExceeded

    path = tmp_path / "budget.sqlite3"
    budgets = [ModelCallBudget(5, ledger_path=path) for _ in range(3)]

    def attempt(index):
        try:
            budgets[index % 3].on_chat_model_start({}, [], run_id=str(index))
            return True
        except ModelCallBudgetExceeded:
            return False

    with ThreadPoolExecutor(max_workers=5) as pool:
        assert sum(pool.map(attempt, range(12))) == 5
    assert ModelCallBudget(5, ledger_path=path).calls == 5
    import pytest

    with pytest.raises(ValueError, match="cannot change"):
        ModelCallBudget(6, ledger_path=path)


def test_thinking_structured_output_uses_supported_tool_choice():
    model = DeepSeekChat(
        model="deepseek-v4-pro", api_key="test",
        extra_body={"thinking": {"type": "enabled"}},
    )
    for choice in ("required", {"type": "function",
                                "function": {"name": "Report"}}):
        payload = model._get_request_payload(
            [HumanMessage(content="Return Report")], tool_choice=choice)
        assert payload["tool_choice"] == "auto"
    for choice in ("auto", "none"):
        assert model._get_request_payload(
            [HumanMessage(content="Inspect")], tool_choice=choice
        )["tool_choice"] == choice
    disabled = model.model_copy(
        update={"extra_body": {"thinking": {"type": "disabled"}}})
    assert disabled._get_request_payload(
        [HumanMessage(content="Inspect")], tool_choice="required"
    )["tool_choice"] == "required"


def test_sdk_length_error_is_a_nonretryable_model_failure():
    from openai import LengthFinishReasonError
    from openai.types.chat import ChatCompletion
    from tracemind.agent.runtime.llm import MODEL_SERVICE_ERRORS
    from tracemind.agent.runtime.recovery import classify_failure

    error = LengthFinishReasonError(completion=ChatCompletion(
        id="truncated", object="chat.completion", created=0,
        model="deepseek-flash", choices=[]))
    assert isinstance(error, MODEL_SERVICE_ERRORS)
    failure = classify_failure(error)
    assert failure.category == "output_truncated"
    assert not failure.retryable
