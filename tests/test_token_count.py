"""Local multilingual token admission never fetches a tokenizer at runtime."""

import socket

import pytest

from tracemind.agent.runtime import token_count


def test_token_count_is_local_and_special_strings_are_data(monkeypatch):
    def network_forbidden(*args, **kwargs):
        raise AssertionError("token admission must remain offline")

    token_count._encoding.cache_clear()
    monkeypatch.setattr(socket, "socket", network_forbidden)
    encoding = token_count._encoding()
    for text in ("checkout failed", "支付失败，需要排查", "🚨🔥", "<|endoftext|>"):
        raw = len(encoding.encode_ordinary(text))
        assert token_count.estimate_text_tokens(text) == (raw * 6 + 4) // 5 + 1024
    text = "支付失败，需要排查" * 100
    assert token_count.estimate_text_tokens(text) < len(text.encode()) + 1024


def test_corrupt_vocabulary_fails_explicitly_without_a_fallback(tmp_path, monkeypatch):
    damaged = tmp_path / "cl100k_base.tiktoken"
    damaged.write_bytes(b"corrupt asset")
    token_count._encoding.cache_clear()
    monkeypatch.setattr(token_count, "VOCABULARY", damaged)
    with pytest.raises(ValueError, match="checksum mismatch"):
        token_count.estimate_text_tokens("hello")
    token_count._encoding.cache_clear()
