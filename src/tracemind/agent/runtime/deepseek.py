"""DeepSeek reasoning round trips and invocation-scoped async transports."""

import httpx
from langchain_core.language_models.chat_models import agenerate_from_stream
from langchain_core.messages import AIMessageChunk
from langchain_openai.chat_models.base import (
    BaseChatOpenAI, _construct_lc_result_from_responses_api,
)
from tracemind.agent.runtime.progress import report_model_progress
from openai import APIError, AsyncOpenAI


class DeepSeekChat(BaseChatOpenAI):
    """Consume provider streams internally and return complete agent messages."""

    # All async entry points use our complete-response adapter; framework
    # streaming must not bypass its DeepSeek Responses event handling.
    disable_streaming: bool = True

    def _get_request_payload(self, input_, *, stop=None, **kwargs):
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)
        thinking = payload.get("extra_body", {}).get("thinking", {})
        choice = payload.get("tool_choice")
        if thinking.get("type") == "enabled" and (
            choice == "required" or isinstance(choice, dict)
        ):
            # Chat thinking rejects forced calls, including LangChain's
            # structured-output tool. The harness still validates the result.
            payload["tool_choice"] = "auto"
        if "messages" in payload:
            messages = self._convert_input(input_).to_messages()
            for message, wire in zip(messages, payload["messages"]):
                reasoning = message.additional_kwargs.get("reasoning_content")
                if reasoning is not None:
                    wire["reasoning_content"] = reasoning
        return payload

    def _create_chat_result(self, response, generation_info=None):
        result = super()._create_chat_result(response, generation_info)
        raw = response if isinstance(response, dict) else response.model_dump()
        for generation, choice in zip(result.generations, raw["choices"]):
            reasoning = choice["message"].get("reasoning_content")
            if reasoning is not None:
                generation.message.additional_kwargs["reasoning_content"] = (
                    reasoning
                )
        return result

    def _convert_chunk_to_generation_chunk(self, chunk, *args):
        result = super()._convert_chunk_to_generation_chunk(chunk, *args)
        choices = chunk.get("choices") or []
        if result is not None and chunk.get("usage"):
            result.message.response_metadata["token_usage"] = chunk["usage"]
        if result is not None and choices:
            delta = choices[0].get("delta") or {}
            reasoning = delta.get("reasoning_content")
            report_model_progress("reasoning", reasoning)
            report_model_progress("output", delta.get("content"))
            for call in delta.get("tool_calls") or []:
                function = call.get("function") or {}
                report_model_progress("tool_arguments", function.get("name"))
                report_model_progress("tool_arguments", function.get("arguments"))
            if reasoning is not None:
                result.message.additional_kwargs["reasoning_content"] = (
                    reasoning
                )
        return result

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        # Explicit transports belong to the caller (including test transports).
        if self.http_async_client is not None:
            return await self._generate_streamed(
                messages, stop=stop, run_manager=run_manager, **kwargs,
            )
        # Langfuse experiments may use separate event loops. Never reuse a
        # pooled AsyncClient from a closed loop, or close a sibling's client.
        async with AsyncOpenAI(
            api_key=self.openai_api_key.get_secret_value(),
            base_url=self.openai_api_base,
            max_retries=0,
            timeout=self.request_timeout,
        ) as client:
            model = self.model_copy(
                update={
                    "root_async_client": client,
                    "async_client": client.chat.completions,
                }
            )
            return await model._generate_streamed(
                messages, stop=stop, run_manager=run_manager, **kwargs,
            )

    async def _generate_streamed(self, messages, stop=None,
                                 run_manager=None, **kwargs):
        payload = self._get_request_payload(messages, stop=stop, **kwargs)
        payload["stream"] = True
        if self._use_responses_api(payload):
            stream = await self.root_async_client.responses.create(**payload)
            terminal = None
            kinds = {
                "response.reasoning_text.delta": "reasoning",
                "response.reasoning_summary_text.delta": "reasoning",
                "response.output_text.delta": "output",
                "response.function_call_arguments.delta": "tool_arguments",
                "response.custom_tool_call_input.delta": "tool_arguments",
            }
            async with stream:
                async for event in stream:
                    if event.type in kinds:
                        report_model_progress(kinds[event.type], event.delta)
                    if event.type in {"response.completed", "response.incomplete",
                                      "response.failed"}:
                        terminal = event.response
                        break
            if terminal is None:
                raise httpx.RemoteProtocolError(
                    "DeepSeek stream ended without a terminal response"
                )
            if terminal.status == "failed" or terminal.error is not None:
                raise APIError(
                    "DeepSeek returned a failed streaming response",
                    request=httpx.Request(
                        "POST", self.openai_api_base + "/responses",
                    ),
                    body={"code": getattr(terminal.error, "code", None)},
                )
            return _construct_lc_result_from_responses_api(
                terminal, schema=kwargs.get("response_format"),
                output_version=self.output_version,
            )
        payload["stream_options"] = {"include_usage": True}

        async def chunks():
            stream = await self.async_client.create(**payload)
            finished = False
            async with stream:
                async for chunk in stream:
                    raw = chunk.model_dump()
                    generation = self._convert_chunk_to_generation_chunk(
                        raw, AIMessageChunk, {},
                    )
                    finished = finished or any(
                        choice.get("finish_reason") is not None
                        for choice in raw.get("choices", [])
                    )
                    if generation is not None:
                        if run_manager:
                            await run_manager.on_llm_new_token(
                                generation.text, chunk=generation,
                            )
                        yield generation
            if not finished:
                raise httpx.RemoteProtocolError(
                    "DeepSeek stream ended without a finish reason"
                )

        return await agenerate_from_stream(chunks())
