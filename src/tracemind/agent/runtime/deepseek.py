"""DeepSeek reasoning round trips and invocation-scoped async transports."""

from langchain_openai.chat_models.base import BaseChatOpenAI
from openai import AsyncOpenAI


class DeepSeekChat(BaseChatOpenAI):
    """Preserve provider reasoning across tool turns and graph checkpoints."""

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
        if result is not None and choices:
            reasoning = (choices[0].get("delta") or {}).get(
                "reasoning_content"
            )
            if reasoning is not None:
                result.message.additional_kwargs["reasoning_content"] = (
                    reasoning
                )
        return result

    async def _agenerate(self, messages, **kwargs):
        # Explicit transports belong to the caller (including test transports).
        if self.http_async_client is not None:
            return await super()._agenerate(messages, **kwargs)
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
            return await super(DeepSeekChat, model)._agenerate(
                messages, **kwargs
            )
