"""Local token admission estimates, separate from bytes and provider usage.

cl100k_base is a proxy for non-OpenAI tokenizers, not an exact DeepSeek count.
The margin and framing allowance are admission headroom; provider-reported
usage remains authoritative for settlement. No runtime downloads are used.
"""

from base64 import b64decode
from functools import lru_cache
from hashlib import sha256
from pathlib import Path

from tiktoken import Encoding


ESTIMATOR_VERSION = "cl100k-base-margin20-v1"
VOCABULARY_SHA256 = (
    "223921b76ee99bde995b7ff738513eef100fb51d18c93597a113bcffe865b2a7"
)
VOCABULARY = Path(__file__).with_name("data") / "cl100k_base.tiktoken"
# From tiktoken_ext.openai_public.cl100k_base; see data/LICENSE.tiktoken.
PATTERN = (
    r"'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}++|\p{N}{1,3}+|"
    r" ?[^\s\p{L}\p{N}]++[\r\n]*+|\s++$|\s*[\r\n]|\s+(?!\S)|\s"
)


@lru_cache(maxsize=1)
def _encoding() -> Encoding:
    data = VOCABULARY.read_bytes()
    if sha256(data).hexdigest() != VOCABULARY_SHA256:
        raise ValueError("bundled token vocabulary checksum mismatch")
    ranks = {
        b64decode(token): int(rank)
        for token, rank in (line.split() for line in data.splitlines())
    }
    return Encoding(
        name="tracemind_cl100k_base",
        pat_str=PATTERN,
        mergeable_ranks=ranks,
        special_tokens={},
    )


def estimate_text_tokens(text: str) -> int:
    """Count ordinary text with 20% headroom and 1024 framing tokens."""
    count = len(_encoding().encode_ordinary(text))
    return (count * 6 + 4) // 5 + 1024
