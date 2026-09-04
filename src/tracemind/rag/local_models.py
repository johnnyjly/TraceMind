"""Lazy, revision-pinned BGE embedding, tokenization and reranking."""

import asyncio

from tracemind.rag.config import RagSettings
from tracemind.async_utils import run_io


class LocalModels:
    """Load local model weights only through explicit setup or inference."""

    def __init__(self, settings: RagSettings) -> None:
        """Capture model configuration without downloading weights."""
        self.settings = settings
        self.embedding = None
        self.reranker = None
        self._lock = asyncio.Lock()

    def __deepcopy__(self, memo):
        """Keep one serialized inference owner when LightRAG copies config."""
        return self

    def load(self, *, download: bool = False) -> None:
        """Load pinned models; explicit setup may fetch missing weights."""
        from sentence_transformers import CrossEncoder, SentenceTransformer

        settings = self.settings
        cache = str(settings.models_directory)
        if self.embedding is None:
            self.embedding = SentenceTransformer(
                settings.embedding_model,
                revision=settings.embedding_revision,
                device=settings.device,
                cache_folder=cache,
                local_files_only=not download,
            )
            if (
                self.embedding.get_embedding_dimension()
                != settings.embedding_dimension
            ):
                raise ValueError("embedding dimension differs from settings")
            self.embedding.max_seq_length = settings.embedding_max_tokens
        if self.reranker is None:
            self.reranker = CrossEncoder(
                settings.reranker_model,
                revision=settings.reranker_revision,
                device=settings.device,
                cache_folder=cache,
                local_files_only=not download,
                max_length=settings.reranker_max_tokens,
            )

    async def embed(self, texts: list[str], *, context="document"):
        """Return normalized vectors and reject silent input truncation."""
        async with self._lock:
            self.load()
            values = [
                self.settings.embedding_query_prefix + text for text in texts
            ]
            if context != "query":
                values = texts
            lengths = self.embedding.tokenizer(
                values, truncation=False, add_special_tokens=True
            )["input_ids"]
            if any(
                len(ids) > self.settings.embedding_max_tokens
                for ids in lengths
            ):
                raise ValueError("embedding input exceeds tokenizer limit")
            return await run_io(
                self.embedding.encode,
                values,
                batch_size=self.settings.batch_size,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )

    async def rank(self, query: str, texts: list[str]) -> list[float]:
        """Score bounded query/document pairs without silently truncating."""
        if not texts:
            return []
        async with self._lock:
            self.load()
            lengths = self.reranker.tokenizer(
                [query] * len(texts),
                texts,
                truncation=False,
                add_special_tokens=True,
            )["input_ids"]
            if any(
                len(ids) > self.settings.reranker_max_tokens for ids in lengths
            ):
                raise ValueError("reranker input exceeds tokenizer limit")
            values = await run_io(
                self.reranker.predict,
                [(query, text) for text in texts],
                batch_size=self.settings.batch_size,
                show_progress_bar=False,
            )
            return [float(score) for score in values]


class BgeTokenizer:
    """Expose only tokenization to LightRAG's chunking interface."""

    def __init__(self, tokenizer) -> None:
        """Retain the loaded model's tokenizer without its weights."""
        self._tokenizer = tokenizer

    def encode(self, text: str) -> list[int]:
        """Count text tokens without BERT boundary tokens."""
        return self._tokenizer.encode(
            text, add_special_tokens=False, verbose=False
        )

    def decode(self, tokens: list[int]) -> str:
        """Decode chunk tokens without adding model boundary tokens."""
        return self._tokenizer.decode(tokens, skip_special_tokens=True)


def load_tokenizer(settings: RagSettings):
    """Load only the pinned fast tokenizer for offline preparation."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        settings.embedding_model,
        revision=settings.embedding_revision,
        cache_dir=str(settings.models_directory),
        local_files_only=True,
        use_fast=True,
    )
    return tokenizer
