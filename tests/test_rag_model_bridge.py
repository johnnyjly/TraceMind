"""Provider response contracts remain testable without optional retrieval."""

import asyncio

import pytest
from langchain_core.messages import AIMessage

from tracemind.rag.config import RagSettings
from tracemind.rag.model_bridge import LightRAGModelBridge, completion_text


def test_provider_text_blocks_preserve_usage_and_ignore_reasoning():
    response = AIMessage(
        content=[{"type": "reasoning", "summary": []},
                 {"type": "text", "text": "first"},
                 {"type": "text", "text": "second"}],
        response_metadata={"status": "completed"},
        usage_metadata={"input_tokens": 4, "output_tokens": 2,
                        "total_tokens": 6},
    )

    async def complete(*args, **kwargs):
        return response

    model = LightRAGModelBridge(RagSettings(_env_file=None), complete)
    assert asyncio.run(model.complete("question")) == "firstsecond"
    assert (model.calls, model.input_tokens, model.output_tokens,
            model.unreported_calls) == (1, 4, 2, 0)


@pytest.mark.parametrize("metadata", [
    {"finish_reason": "length"}, {"finish_reason": "MAX_TOKENS"},
    {"status": "incomplete",
     "incomplete_details": {"reason": "max_output_tokens"}},
    {"status": "failed"}, {"status": "cancelled"},
])
def test_incomplete_provider_output_is_never_used(metadata):
    with pytest.raises(RuntimeError, match="incomplete"):
        completion_text(AIMessage(content="partial",
                                  response_metadata=metadata))


@pytest.mark.parametrize("content", [
    " ", [], [{"type": "refusal", "refusal": "refused"}],
    [{"type": "image_url", "image_url": {"url": "https://example.test/x"}}],
])
def test_nontext_provider_output_is_rejected(content):
    with pytest.raises(ValueError, match="textual"):
        completion_text(AIMessage(content=content))
