"""Manifest-selected parsing and mechanical provenance checks for RAG."""

import json
import re
import subprocess
from dataclasses import dataclass, field
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from urllib.parse import urlparse

PARSER_VERSION = "docling-structure-v2"


def digest(value: bytes) -> str:
    """Hash bytes without conflating serialized and extracted content."""
    return sha256(value).hexdigest()


def canonical_json(value) -> bytes:
    """Serialize registry identity independently of whitespace."""
    return json.dumps(value, sort_keys=True, ensure_ascii=False).encode()


@dataclass(frozen=True)
class Section:
    """Group normalized framework elements under one source heading path."""

    heading: str
    text: str
    heading_path: tuple[str, ...] = ()
    source_items: tuple[str, ...] = ()


@dataclass(frozen=True)
class Document:
    """Preserve verified metadata alongside the parsed framework document."""

    metadata: dict
    sections: tuple[Section, ...]
    extracted_sha256: str
    structure: object = field(default=None, compare=False, repr=False)

    @property
    def key(self) -> str:
        """Use a revision-specific, document-scoped framework identity."""
        return (
            "doc-"
            + digest(
                canonical_json(
                    [self.metadata["doc_id"], self.metadata["sha256"]]
                )
            )[:32]
        )

    @property
    def text(self) -> str:
        """Serialize normalized parents into the canonical ingestion text."""
        return "\n\n".join(f"## {s.heading}\n{s.text}" for s in self.sections)


@dataclass(frozen=True)
class Corpus:
    """Keep selected verified documents and frozen registry identities."""

    documents: tuple[Document, ...]
    profile: str
    manifest_sha256: str
    sources_sha256: str
    sources: dict
    source_checks: dict
    manifest_bytes: bytes = b""
    source_bytes: bytes = b""


def clean_text(text: str) -> str:
    """Normalize line endings and trailing spaces while retaining
    code/caveats.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
    return "\n".join(line.rstrip() for line in text.splitlines()).strip()


def _convert(
    data: bytes,
    fmt: str,
    *,
    pdf_pipeline: str = "standard",
    artifacts_path: Path | None = None,
):
    """Use Docling's converters; PDF processing mode is explicit."""
    from docling.document_converter import (
        DocumentConverter,
        NativePdfFormatOption,
        PdfFormatOption,
    )
    from docling.datamodel.base_models import (
        ConversionStatus,
        DocumentStream,
        InputFormat,
    )
    from docling.datamodel.pipeline_options import PdfPipelineOptions

    formats = {
        "markdown": (InputFormat.MD, "md"),
        "html": (InputFormat.HTML, "html"),
        "pdf": (InputFormat.PDF, "pdf"),
    }
    kind, extension = formats[fmt]
    options = {}
    if kind is InputFormat.PDF:
        if pdf_pipeline == "native":
            options[kind] = NativePdfFormatOption()
        else:
            if artifacts_path is None or not artifacts_path.exists():
                raise RuntimeError(
                    "Docling PDF models missing; run rag.py models"
                )
            options[kind] = PdfFormatOption(
                pipeline_options=PdfPipelineOptions(
                    artifacts_path=artifacts_path, do_ocr=False
                )
            )
    converter = DocumentConverter(
        allowed_formats=[kind], format_options=options
    )
    result = converter.convert(
        DocumentStream(
            name="document." + extension,
            stream=BytesIO(data),
        )
    )
    if result.status is not ConversionStatus.SUCCESS:
        raise ValueError(f"document conversion is incomplete: {result.status}")
    return result.document


def _sections(structure, title: str) -> tuple[Section, ...]:
    """Group framework elements by heading path, preserving source pointers."""
    from docling.chunking import HierarchicalChunker

    groups = {}
    for chunk in HierarchicalChunker().chunk(structure):
        path = tuple(chunk.meta.headings or [title])
        if any(h.casefold() == "sources" for h in path):
            continue
        texts, refs = groups.setdefault(path, ([], []))
        texts.append(chunk.text)
        refs.extend(item.self_ref for item in chunk.meta.doc_items)
    sections = tuple(
        Section(
            " / ".join(path),
            "\n\n".join(texts),
            path,
            tuple(dict.fromkeys(refs)),
        )
        for path, (texts, refs) in groups.items()
        if texts
    )
    if not sections:
        raise ValueError("document contains no extractable text")
    return sections


def markdown_sections(text: str, title: str) -> tuple[Section, ...]:
    """Parse headings, fences and inline markup through Docling."""
    text = clean_text(text)
    if text.startswith("---\n"):
        _, separator, text = text[4:].partition("\n---")
        if not separator:
            raise ValueError("unterminated Markdown frontmatter")
    return _sections(_convert(text.encode(), "markdown"), title)


def parse_document(
    data: bytes,
    entry: dict,
    *,
    pdf_pipeline="standard",
    artifacts_path: Path | None = None,
) -> Document:
    """Convert one verified representation; retain framework provenance."""
    fmt = entry["format"]
    if fmt == "json":
        record = json.loads(data)
        if record["doc_id"] != entry["doc_id"]:
            raise ValueError("JSON document identity differs from manifest")
        text = record["text"]
        if digest(text.encode()) != entry["content_sha256"]:
            raise ValueError("JSON authored content hash mismatch")
        data, fmt = text.encode(), "markdown"
    if fmt not in {"markdown", "html", "pdf"}:
        raise ValueError(f"unsupported document format: {fmt}")
    if fmt == "markdown":
        text = clean_text(data.decode())
        if text.startswith("---\n"):
            _, separator, text = text[4:].partition("\n---")
            if not separator:
                raise ValueError("unterminated Markdown frontmatter")
        data = text.encode()
    structure = _convert(
        data, fmt, pdf_pipeline=pdf_pipeline, artifacts_path=artifacts_path
    )
    sections = _sections(structure, entry["title"])
    extracted = "\n\n".join(f"{s.heading}\n{s.text}" for s in sections)
    document = Document(entry, sections, digest(extracted.encode()), structure)
    if (
        entry.get("content_hash_basis") == "docling-normalized-text-v2"
        and digest(document.text.encode()) != entry["content_sha256"]
    ):
        raise ValueError("normalized authored content hash mismatch")
    return document


def verify_source(source: dict, repositories: dict[str, Path]) -> dict:
    """Verify Git object bytes; web references retain explicit freshness
    limits.
    """
    if source["type"] == "repository":
        revision, path = source["revision"], source["path"]
        if not re.fullmatch(r"[a-f0-9]{40}", revision):
            raise ValueError(
                "repository source needs a full pinned Git revision"
            )
        if Path(path).is_absolute() or ".." in Path(path).parts:
            raise ValueError("repository source path must be relative")
        root = repositories[source["repository"]]
        result = subprocess.run(
            ["git", "-C", str(root), "show", f"{revision}:{path}"],
            capture_output=True,
            check=True,
        )
        if digest(result.stdout) != source["sha256"]:
            raise ValueError(f"pinned source hash mismatch: {path}")
        return {"status": "pinned_git_bytes_verified", "revision": revision}
    if source["type"] == "official_documentation":
        url = urlparse(source["url"])
        if url.scheme != "https" or not url.netloc:
            raise ValueError("official documentation needs an HTTPS URL")
        if not source.get("accessed_on") or not source.get("version_policy"):
            raise ValueError("web reference lacks freshness metadata")
        return {
            "status": "registry_reference_only",
            "live_content_verified": False,
            "accessed_on": source["accessed_on"],
            "version_policy": source["version_policy"],
        }
    raise ValueError("unsupported source type")


def load_corpus(
    root: Path,
    *,
    profile: str,
    repositories: dict[str, Path],
    pdf_pipeline: str = "standard",
    artifacts_path: Path | None = None,
) -> Corpus:
    """Validate explicit profile files and provenance before model
    execution.
    """
    root = root.resolve()
    manifest_bytes = (root / "manifest.json").read_bytes()
    source_bytes = (root / "sources.json").read_bytes()
    manifest, registry = json.loads(manifest_bytes), json.loads(source_bytes)
    if manifest["schema_version"] != 2 or manifest["language"] != "en":
        raise ValueError("expected an English schema-v2 corpus manifest")
    entries = {d["doc_id"]: d for d in manifest["documents"]}
    if len(entries) != len(manifest["documents"]):
        raise ValueError("duplicate document IDs in manifest")
    selected = manifest["profiles"][profile]
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("profile must contain unique document IDs")
    documents, checks, sources, paths = [], {}, {}, set()
    for doc_id in selected:
        entry = entries[doc_id]
        relative = Path(entry["path"])
        path = (root / relative).resolve()
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or not relative.parts
            or relative.parts[0] != "documents"
            or not path.is_relative_to(root / "documents")
        ):
            raise ValueError(
                "document path escapes the manifest documents directory"
            )
        if path in paths:
            raise ValueError("multiple documents point to one file")
        paths.add(path)
        data = path.read_bytes()
        if digest(data) != entry["sha256"]:
            raise ValueError(f"document file hash mismatch: {doc_id}")
        if not entry["source_refs"]:
            raise ValueError(f"document has no sources: {doc_id}")
        for ref in entry["source_refs"]:
            if ref not in sources:
                sources[ref] = registry["sources"][ref]
                checks[ref] = verify_source(sources[ref], repositories)
        documents.append(
            parse_document(
                data,
                entry,
                pdf_pipeline=pdf_pipeline,
                artifacts_path=artifacts_path,
            )
        )
    return Corpus(
        tuple(documents),
        profile,
        digest(manifest_bytes),
        digest(source_bytes),
        sources,
        checks,
        manifest_bytes,
        source_bytes,
    )
