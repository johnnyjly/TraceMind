"""Manifest-selected parsing and mechanical provenance checks for RAG."""

import json
import re
import subprocess
from dataclasses import dataclass
from hashlib import sha256
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlparse

PARSER_VERSION = "tracemind-sections-v1"


def digest(value: bytes) -> str:
    """Hash bytes without conflating serialized and extracted content."""
    return sha256(value).hexdigest()


def canonical_json(value) -> bytes:
    """Serialize registry identity independently of whitespace."""
    return json.dumps(value, sort_keys=True, ensure_ascii=False).encode()


@dataclass(frozen=True)
class Section:
    heading: str
    text: str
    heading_path: tuple[str, ...] = ()


@dataclass(frozen=True)
class Document:
    metadata: dict
    sections: tuple[Section, ...]
    extracted_sha256: str

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
        return "\n\n".join(f"## {s.heading}\n{s.text}" for s in self.sections)


@dataclass(frozen=True)
class Corpus:
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


def markdown_sections(text: str, title: str) -> tuple[Section, ...]:
    """Keep section hierarchy; exclude frontmatter and the source
    bibliography.
    """
    text = clean_text(text)
    if text.startswith("---\n"):
        parts = text.split("\n---", 1)
        if len(parts) != 2:
            raise ValueError("unterminated Markdown frontmatter")
        text = parts[1].lstrip("\n")
    sections, lines, headings = [], [], []
    fenced = False

    def flush():
        body = clean_text("\n".join(lines))
        if body and not any(h.casefold() == "sources" for _, h in headings):
            hierarchy = tuple(h for _, h in headings) or (title,)
            sections.append(Section(" / ".join(hierarchy), body, hierarchy))
        lines.clear()

    for line in text.splitlines():
        if re.match(r"^\s*(```|~~~)", line):
            fenced = not fenced
        match = (
            None if fenced else re.match(r"^(#{1,6})\s+(.+?)\s*#*\s*$", line)
        )
        if match:
            flush()
            level, heading = len(match[1]), match[2]
            headings = [(n, h) for n, h in headings if n < level]
            headings.append((level, heading))
        else:
            lines.append(line)
    flush()
    if not sections:
        raise ValueError("document contains no indexable sections")
    return tuple(sections)


class _HTMLText(HTMLParser):
    """Extract body blocks and headings without navigation, CSS or scripts."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []
        self.body = False
        self.skip = 0

    def handle_starttag(self, tag, attrs):
        if tag == "body":
            self.body = True
        if not self.body:
            return
        if self.skip:
            self.skip += (
                1 if tag not in {"br", "img", "hr", "meta", "input"} else 0
            )
            return
        if tag in {"script", "style", "nav", "footer"} or any(
            k == "class" and "meta" in v.split() for k, v in attrs if v
        ):
            self.skip = 1
        elif re.fullmatch("h[1-6]", tag):
            self.parts.append("\n\n" + "#" * int(tag[1]) + " ")
        elif tag in {"p", "li", "tr", "pre", "br", "div"}:
            self.parts.append("\n")
        elif tag in {"td", "th"}:
            self.parts.append(" | ")

    def handle_endtag(self, tag):
        if self.skip:
            self.skip -= 1
        elif tag in {"p", "li", "tr", "pre"} or re.fullmatch("h[1-6]", tag):
            self.parts.append("\n")
        if tag == "body":
            self.body = False

    def handle_data(self, data):
        if self.body and not self.skip:
            self.parts.append(data)


def _pdf_text(data: bytes) -> str:
    """Extract text PDFs; preserve font-marked headings across page breaks."""
    from io import BytesIO

    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError(
            "PDF ingestion requires pypdf in the rag extra"
        ) from exc

    reader = PdfReader(BytesIO(data))
    parts = []
    previous_heading_size = None
    for page in reader.pages:

        def visit(text, cm, tm, font, size):
            nonlocal previous_heading_size
            value = text.strip()
            if not value:
                return
            if (
                value.startswith(
                    (
                        "TraceMind | authored knowledge reference",
                        "Document text SHA-256:",
                    )
                )
                or value.isdigit()
                or "| verified " in value
            ):
                return
            if size >= 12:
                if previous_heading_size == size:
                    parts[-1] = parts[-1].rstrip() + " " + value + "\n"
                else:
                    parts.append(
                        "\n" + ("# " if size >= 16 else "## ") + value + "\n"
                    )
                previous_heading_size = size
            else:
                parts.append(text)
                previous_heading_size = None

        page.extract_text(visitor_text=visit)
        parts.append("\n")
    text = "".join(parts)
    if not text.strip():
        raise ValueError("PDF has no extractable text; OCR is not configured")
    return text


def parse_document(data: bytes, entry: dict) -> Document:
    """Parse exactly one representation; never duplicate JSON text/sections."""
    fmt = entry["format"]
    if fmt == "pdf":
        text = _pdf_text(data)
    elif fmt == "html":
        parser = _HTMLText()
        parser.feed(data.decode("utf-8"))
        text = "".join(parser.parts)
    elif fmt == "json":
        record = json.loads(data)
        if record["doc_id"] != entry["doc_id"]:
            raise ValueError("JSON document identity differs from manifest")
        text = record["text"]
        if digest(text.encode()) != entry["content_sha256"]:
            raise ValueError("JSON authored content hash mismatch")
    elif fmt == "markdown":
        text = data.decode("utf-8")
    else:
        raise ValueError(f"unsupported document format: {fmt}")
    sections = markdown_sections(text, entry["title"])
    extracted = "\n\n".join(f"{s.heading}\n{s.text}" for s in sections)
    return Document(entry, sections, digest(extracted.encode()))


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
    root: Path, *, profile: str, repositories: dict[str, Path]
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
        documents.append(parse_document(data, entry))
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
