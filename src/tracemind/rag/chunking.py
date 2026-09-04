"""Docling hierarchical parents and token-bounded retrieval children."""

import re
from dataclasses import dataclass
from typing import Any

from docling.chunking import HybridChunker
from docling_core.transforms.chunker.tokenizer.base import BaseTokenizer

from tracemind.rag.config import RagSettings
from tracemind.rag.corpus import PARSER_VERSION, Corpus, Document
from tracemind.rag.models import KnowledgeChunk, KnowledgeParent


class ModelTokenizer(BaseTokenizer):
    """Adapt the selected embedding tokenizer to Docling's public interface."""

    tokenizer: Any
    max_tokens: int

    def count_tokens(self, text: str) -> int:
        """Count content tokens without model boundary markers."""
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def get_max_tokens(self) -> int:
        """Return the configured maximum for contextualized child text."""
        return self.max_tokens

    def get_tokenizer(self):
        """Expose the tokenizer used by framework split and merge steps."""
        return self.tokenizer


@dataclass(frozen=True)
class PreparedDocument:
    """Canonical source material passed unchanged to all storage backends."""

    document: Document
    parents: tuple[KnowledgeParent, ...]
    chunks: tuple[KnowledgeChunk, ...]


def prepare_document(
    document: Document, corpus: Corpus, tokenizer, settings: RagSettings,
) -> PreparedDocument:
    """Preserve full parents and framework element references."""
    if document.structure is None:
        raise ValueError("new ingestion requires a Docling document")
    parents, chunks, by_heading = [], [], {}
    entry = document.metadata
    position = 0
    for number, section in enumerate(document.sections):
        prefix = f"## {section.heading}\n"
        parent = KnowledgeParent(
            parent_id=f"{document.key}-section-{number:03d}",
            doc_id=entry["doc_id"], title=entry["title"],
            section=section.heading,
            heading_path=list(section.heading_path), text=section.text,
            document_path=entry["path"], document_sha256=entry["sha256"],
            source_refs=entry["source_refs"],
            source_revision="registry-sha256:" + corpus.sources_sha256,
            scope=entry.get("scope", ""),
            extracted_sha256=document.extracted_sha256,
            source_items=list(section.source_items),
            parser_version=PARSER_VERSION,
            source_start=position + len(prefix),
            source_end=position + len(prefix) + len(section.text),
        )
        parents.append(parent)
        by_heading[section.heading_path] = parent
        position = parent.source_end + 2
    adapter = ModelTokenizer(
        tokenizer=tokenizer,
        max_tokens=settings.chunk_tokens)
    chunker = HybridChunker(tokenizer=adapter, merge_peers=True)
    for chunk in chunker.chunk(document.structure):
        path = tuple(chunk.meta.headings or [entry["title"]])
        if any(h.casefold() == "sources" for h in path):
            continue
        parent = by_heading[path]
        text = chunker.contextualize(chunk)
        if adapter.count_tokens(text) > settings.chunk_tokens:
            raise ValueError("framework chunk exceeds embedding token budget")
        matches = re.findall(
            r"`([^`\n]+)`|\b([A-Z][A-Z_]{2,})\b|([\w.-]+[./:_-][\w./:-]+)",
            chunk.text,
        )
        identifiers = " ".join(dict.fromkeys(
            value for match in matches for value in match if value))
        metadata = parent.model_dump(exclude={
            "source_start", "source_end", "text", "source_items",
        })
        chunks.append(KnowledgeChunk(
            **metadata, chunk_id=f"{document.key}-chunk-{len(chunks):03d}",
            text=text, source_text=chunk.text,
            source_items=list(
                dict.fromkeys(
                    i.self_ref for i in chunk.meta.doc_items)),
            identifiers=identifiers,
        ))
    if not chunks:
        raise ValueError("document has no retrieval chunks")
    return PreparedDocument(document, tuple(parents), tuple(chunks))


def prepare_corpus(
    corpus: Corpus, tokenizer, settings: RagSettings,
) -> tuple[PreparedDocument, ...]:
    """Prepare verified documents without embedding or paid model calls."""
    return tuple(prepare_document(doc, corpus, tokenizer, settings)
                 for doc in corpus.documents)
