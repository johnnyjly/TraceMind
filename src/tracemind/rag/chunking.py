"""Structure-aware children with reversible offsets into complete sections."""

import re
from dataclasses import dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter

from tracemind.rag.config import RagSettings
from tracemind.rag.corpus import Corpus, Document
from tracemind.rag.models import KnowledgeChunk, KnowledgeParent


@dataclass(frozen=True)
class PreparedDocument:
    document: Document
    parents: tuple[KnowledgeParent, ...]
    chunks: tuple[KnowledgeChunk, ...]


def _slices(text: str, header: str, tokenizer, settings: RagSettings):
    """Delegate splitting to LangChain; map its text back to source offsets."""

    def token_count(value):
        return len(tokenizer.encode(value, add_special_tokens=False))

    capacity = settings.chunk_tokens - token_count(header)
    if capacity <= settings.chunk_overlap_tokens:
        raise ValueError("section heading leaves insufficient chunk space")
    if not token_count(text):
        raise ValueError("section contains no tokens")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=capacity,
        chunk_overlap=settings.chunk_overlap_tokens,
        length_function=token_count,
        separators=[
            r"\n\s*\n",
            r"\n(?=\s*(?:\d+[.)]\s|[-*+]\s|```|~~~|\|))",
            r"(?<=[.!?])\s+",
            r"\n",
            r"[ \t]+",
            "",
        ],
        is_separator_regex=True,
        keep_separator=True,
        strip_whitespace=False,
    )
    previous_start, previous_end = -1, 0
    previous_text = ""
    for source_text in splitter.split_text(text):
        search_start = previous_end
        if previous_text and settings.chunk_overlap_tokens:
            offsets = tokenizer(
                previous_text,
                add_special_tokens=False,
                return_offsets_mapping=True,
            )["offset_mapping"]
            # LangChain's add_start_index subtracts overlap as characters,
            # even with a token length function. Bound the search in tokens
            # instead, including whitespace before the overlapping tokens.
            overlap = settings.chunk_overlap_tokens
            search_start = previous_start + (
                offsets[-overlap - 1][1] if len(offsets) > overlap else 0
            )
        start = text.find(source_text, max(previous_start + 1, search_start))
        end = start + len(source_text)
        if start < 0 or start > previous_end or end <= previous_end:
            raise ValueError("splitter output cannot be mapped without gaps")
        if token_count(header + source_text) > settings.chunk_tokens:
            raise ValueError("splitter output exceeds chunk token budget")
        yield start, end
        previous_start, previous_end = start, end
        previous_text = source_text
    if previous_end != len(text):
        raise ValueError("splitter output does not cover the complete section")


def prepare_document(
    document: Document, corpus: Corpus, tokenizer, settings: RagSettings
) -> PreparedDocument:
    """Keep full section parents; embed only bounded child windows."""
    parents, chunks = [], []
    entry = document.metadata
    position = 0
    for n, section in enumerate(document.sections):
        prefix = f"## {section.heading}\n"
        parent = KnowledgeParent(
            parent_id=f"{document.key}-section-{n:03d}",
            doc_id=entry["doc_id"],
            title=entry["title"],
            section=section.heading,
            heading_path=list(section.heading_path),
            text=section.text,
            document_path=entry["path"],
            document_sha256=entry["sha256"],
            source_refs=entry["source_refs"],
            source_revision="registry-sha256:" + corpus.sources_sha256,
            scope=entry.get("scope", ""),
            extracted_sha256=document.extracted_sha256,
            source_start=position + len(prefix),
            source_end=position + len(prefix) + len(section.text),
        )
        parents.append(parent)
        position = parent.source_end + 2
        header = f"{entry['title']}\n{section.heading}\n\n"
        for start, end in _slices(section.text, header, tokenizer, settings):
            source_text = section.text[start:end]
            text = header + source_text
            matches = re.findall(
                r"`([^`\n]+)`|\b([A-Z][A-Z_]{2,})\b|([\w.-]+[./:_-][\w./:-]+)",
                source_text,
            )
            identifiers = " ".join(
                dict.fromkeys(
                    value for match in matches for value in match if value
                )
            )
            metadata = parent.model_dump(
                exclude={"source_start", "source_end", "text"}
            )
            chunks.append(
                KnowledgeChunk(
                    **metadata,
                    chunk_id=f"{document.key}-chunk-{len(chunks):03d}",
                    text=text,
                    source_start=start,
                    source_end=end,
                    source_text=source_text,
                    identifiers=identifiers,
                )
            )
    return PreparedDocument(document, tuple(parents), tuple(chunks))


def prepare_corpus(
    corpus: Corpus, tokenizer, settings: RagSettings
) -> tuple[PreparedDocument, ...]:
    """Prepare verified documents without embedding or paid model calls."""
    return tuple(
        prepare_document(doc, corpus, tokenizer, settings)
        for doc in corpus.documents
    )
