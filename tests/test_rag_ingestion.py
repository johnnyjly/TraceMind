"""Validate preparation, parent recovery and failed/publication boundaries."""

import asyncio
import json
import re
from dataclasses import replace
from io import BytesIO
from itertools import pairwise
from pathlib import Path
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage

from tracemind.rag.chunking import prepare_corpus
from tracemind.rag.config import RagSettings
from tracemind.rag.corpus import (
    digest,
    load_corpus,
    markdown_sections,
    parse_document,
    verify_source,
)
from tracemind.rag.engine import RagRuntime
from tracemind.rag.ingestion import build_snapshot, verify_snapshot
from tracemind.rag.keyword import KeywordIndex
from tracemind.rag.llm import RagModel


class WordTokenizer:
    def encode(self, text, **kwargs):
        return list(range(len(re.findall(r"\S+", text))))

    def __call__(self, text, **kwargs):
        return {
            "offset_mapping": [m.span() for m in re.finditer(r"\S+", text)]
        }


def corpus_fixture(tmp_path, *, text=None):
    root = tmp_path / "knowledge"
    (root / "documents").mkdir(parents=True)
    text = text or (
        "# Connections\n\n## Investigation\n"
        + "\n\n".join(
            " ".join(f"word{i}" for i in range(n, n + 16))
            for n in range(0, 96, 16)
        )
        + "\n\n## Limits\nDo not infer causality from UNAVAILABLE. "
        "Preserve `rpc.deadline_exceeded` and `/get-quote`.\n"
    )
    raw = text.encode()
    (root / "documents" / "connections.md").write_bytes(raw)
    entry = {
        "doc_id": "synthetic.connections",
        "title": "Connections",
        "path": "documents/connections.md",
        "format": "markdown",
        "sha256": digest(raw),
        "source_refs": ["official"],
        "scope": "synthetic",
        "content_sha256": digest(raw),
    }
    manifest = {
        "schema_version": 2,
        "language": "en",
        "documents": [entry],
        "profiles": {"core": [entry["doc_id"]]},
    }
    sources = {
        "sources": {
            "official": {
                "type": "official_documentation",
                "url": "https://example.org/reference",
                "accessed_on": "2026-10-08",
                "version_policy": "Mutable page; recheck.",
            }
        }
    }
    (root / "manifest.json").write_text(json.dumps(manifest))
    (root / "sources.json").write_text(json.dumps(sources))
    return root, manifest


def configured(tmp_path, *, chunk_overlap_tokens=5, **kwargs):
    return RagSettings(
        _env_file=None,
        directory=tmp_path / "rag",
        snapshot="synthetic-v1",
        chunk_tokens=40,
        chunk_overlap_tokens=chunk_overlap_tokens,
        **kwargs,
    )


def prepared_fixture(tmp_path, *, chunk_overlap_tokens=5):
    root, _ = corpus_fixture(tmp_path)
    corpus = load_corpus(root, profile="core", repositories={})
    settings = configured(tmp_path, chunk_overlap_tokens=chunk_overlap_tokens)
    return corpus, prepare_corpus(corpus, WordTokenizer(), settings), settings


def test_structured_children_overlap_and_recover_complete_parent(tmp_path):
    # Each paragraph has 16 tokens; overlap must fit a whole structural unit.
    corpus, prepared, settings = prepared_fixture(
        tmp_path, chunk_overlap_tokens=17
    )
    doc = prepared[0]
    assert len(doc.parents) == 2
    assert doc.parents[0].heading_path == ["Connections", "Investigation"]
    assert len(doc.chunks) > len(doc.parents)
    for parent in doc.parents:
        assert (
            doc.document.text[parent.source_start : parent.source_end]
            == parent.text
        )
        children = [c for c in doc.chunks if c.parent_id == parent.parent_id]
        assert children[0].source_start == 0
        assert children[-1].source_end == len(parent.text)
        for child in children:
            assert (
                parent.text[child.source_start : child.source_end]
                == child.source_text
            )
            assert (
                len(WordTokenizer().encode(child.text))
                <= settings.chunk_tokens
            )
        for first, second in pairwise(children):
            assert first.source_start < second.source_start < first.source_end
            assert first.source_text.endswith(
                parent.text[second.source_start : first.source_end]
            )
    assert "UNAVAILABLE" in doc.parents[1].text
    assert "rpc.deadline_exceeded" in doc.chunks[-1].identifiers
    assert prepare_corpus(corpus, WordTokenizer(), settings) == prepared


@pytest.mark.parametrize("overlap", [0, 5])
@pytest.mark.parametrize(
    "body",
    [
        " ".join(["RepeatedLongIdentifier"] * 200),
        "\n\n".join(
            ["Same paragraph with `/get-quote` and UNAVAILABLE."] * 40
        ),
        "\n".join(["\tVALKEY_ADDR  rpc.deadline_exceeded. Mixed CASE."] * 40),
    ],
)
def test_repeated_text_offsets_preserve_complete_source(
    tmp_path, body, overlap
):
    root, _ = corpus_fixture(tmp_path, text="# Guide\n## Checks\n" + body)
    corpus = load_corpus(root, profile="core", repositories={})
    tokenizer = WordTokenizer()
    settings = configured(tmp_path, chunk_overlap_tokens=overlap)
    doc = prepare_corpus(corpus, tokenizer, settings)[0]
    parent = doc.parents[0]
    covered_end = 0
    for child in doc.chunks:
        assert child.source_start <= covered_end < child.source_end
        assert (
            child.source_text
            == parent.text[child.source_start : child.source_end]
        )
        assert len(tokenizer.encode(child.text)) <= settings.chunk_tokens
        covered_end = child.source_end
    assert covered_end == len(parent.text)
    if overlap and "\n" not in body:
        assert doc.chunks[1].source_start < doc.chunks[0].source_end
    if not overlap:
        assert "".join(c.source_text for c in doc.chunks) == parent.text


def test_whole_build_timeout_prevents_publication(tmp_path):
    corpus, prepared, settings = prepared_fixture(tmp_path)
    runtime = FakeRuntime(settings)

    async def stalled():
        await asyncio.sleep(60)

    runtime.process = stalled
    with pytest.raises(TimeoutError):
        asyncio.run(
            build_snapshot(runtime, corpus, prepared, timeout_seconds=0.02)
        )
    assert not (settings.workspace / "snapshot.json").exists()
    assert (
        json.loads((settings.workspace / "build.json").read_text())["state"]
        == "failed"
    )
    assert runtime.index is None


def test_duplicate_extracted_documents_fail_before_creating_snapshot(tmp_path):
    corpus, _, settings = prepared_fixture(tmp_path)
    original = corpus.documents[0]
    duplicate = replace(
        original, metadata={**original.metadata, "doc_id": "duplicate"}
    )
    corpus = replace(corpus, documents=(original, duplicate))
    prepared = prepare_corpus(corpus, WordTokenizer(), settings)
    with pytest.raises(ValueError, match="explicit deduplication"):
        asyncio.run(
            build_snapshot(
                FakeRuntime(settings), corpus, prepared, timeout_seconds=5
            )
        )
    assert not settings.workspace.exists()


def test_internal_model_respects_summary_cap_and_rejects_truncation(tmp_path):
    requests = []

    async def complete(messages, **kwargs):
        requests.append(kwargs)
        return AIMessage(
            content="partial",
            response_metadata={"finish_reason": "length"},
            usage_metadata={
                "input_tokens": 10,
                "output_tokens": 5,
                "total_tokens": 15,
            },
        )

    model = RagModel(configured(tmp_path), complete)
    with pytest.raises(RuntimeError, match="truncated"):
        asyncio.run(model.complete("summarize", max_tokens=16))
    assert requests[0]["max_tokens"] == 16
    assert model.calls == 1 and model.output_tokens == 5


def test_nested_sections_code_and_source_bibliography():
    text = """---
private_metadata: excluded
---
# Guide
## Checks
Keep this warning.
```text
# This is code, not a heading
rpc.deadline_exceeded
```
### Detail
Preserve mixed CASE and /get-quote.
## Sources
Do not embed this bibliography.
"""
    sections = markdown_sections(text, "Guide")
    assert len(sections) == 2
    assert sections[1].heading_path == ("Guide", "Checks", "Detail")
    assert "# This is code" in sections[0].text
    assert "private_metadata" not in str(sections)
    assert "bibliography" not in str(sections)


@pytest.mark.parametrize(
    "mutation,match",
    [
        ("bytes", "file hash"),
        ("path", "escapes"),
        ("duplicate", "duplicate document"),
        ("empty_refs", "no sources"),
    ],
)
def test_manifest_fails_closed_before_parsing(tmp_path, mutation, match):
    root, manifest = corpus_fixture(tmp_path)
    if mutation == "bytes":
        (root / manifest["documents"][0]["path"]).write_text("tampered")
    elif mutation == "path":
        manifest["documents"][0]["path"] = "../outside.md"
    elif mutation == "duplicate":
        manifest["documents"].append(manifest["documents"][0])
    else:
        manifest["documents"][0]["source_refs"] = []
    (root / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match=match):
        load_corpus(root, profile="core", repositories={})


def test_git_bytes_and_web_reference_are_different_attestations(monkeypatch):
    calls = []

    def git(args, **kwargs):
        calls.append(args)
        return SimpleNamespace(stdout=b"pinned source bytes")

    monkeypatch.setattr("tracemind.rag.corpus.subprocess.run", git)
    source = {
        "type": "repository",
        "repository": "project",
        "revision": "a" * 40,
        "path": "src/module.py",
        "sha256": digest(b"pinned source bytes"),
    }
    check = verify_source(source, {"project": Path("/synthetic")})
    assert check["status"] == "pinned_git_bytes_verified"
    assert calls[0][-1] == "a" * 40 + ":src/module.py"
    source["sha256"] = "b" * 64
    with pytest.raises(ValueError, match="source hash"):
        verify_source(source, {"project": Path("/synthetic")})
    check = verify_source(
        {
            "type": "official_documentation",
            "url": "https://example.org/docs",
            "accessed_on": "2026-10-08",
            "version_policy": "Recheck mutable upstream.",
        },
        {},
    )
    assert check["live_content_verified"] is False


def test_html_noise_removed_and_json_single_representation():
    html = b"""<html><head><style>noise</style></head><body>\
<nav><b>menu</b></nav>
    <h1>Guide</h1><h2>Checks</h2><p>Keep <code>VALKEY_ADDR</code>.</p>
    <footer>footer noise</footer></body></html>"""
    entry = {"format": "html", "title": "Guide"}
    doc = parse_document(html, entry)
    assert doc.sections[0].text == "Keep VALKEY_ADDR."
    text = "# Guide\n## Checks\nOne copy only."
    record = {"doc_id": "a", "text": text, "sections": [{"duplicate": text}]}
    entry = {
        "format": "json",
        "title": "Guide",
        "doc_id": "a",
        "content_sha256": digest(text.encode()),
    }
    doc = parse_document(json.dumps(record).encode(), entry)
    assert doc.text.count("One copy only") == 1
    record["text"] += "tamper"
    with pytest.raises(ValueError, match="authored content hash"):
        parse_document(json.dumps(record).encode(), entry)


def test_text_pdf_headings_and_empty_pdf():
    pypdf = pytest.importorskip("pypdf")
    from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

    writer = pypdf.PdfWriter()
    page = writer.add_blank_page(width=400, height=400)
    font = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        }
    )
    page[NameObject("/Resources")] = DictionaryObject(
        {
            NameObject("/Font"): DictionaryObject(
                {NameObject("/F1"): writer._add_object(font)}
            )
        }
    )
    stream = DecodedStreamObject()
    stream.set_data(
        b"BT /F1 20 Tf 20 360 Td (Guide) Tj /F1 14 Tf 0 -30 Td "
        b"(Limits) Tj /F1 10 Tf 0 -20 Td (Do not infer causality.) Tj ET"
    )
    page[NameObject("/Contents")] = writer._add_object(stream)
    output = BytesIO()
    writer.write(output)
    doc = parse_document(
        output.getvalue(), {"format": "pdf", "title": "Guide"}
    )
    assert doc.sections[0].heading_path == ("Guide", "Limits")
    assert doc.sections[0].text == "Do not infer causality."
    empty = pypdf.PdfWriter()
    empty.add_blank_page(width=400, height=400)
    output = BytesIO()
    empty.write(output)
    with pytest.raises(ValueError, match="no extractable text"):
        parse_document(output.getvalue(), {"format": "pdf", "title": "Empty"})


def test_parent_persistence_rollback_and_readonly_index(tmp_path):
    _, prepared, _ = prepared_fixture(tmp_path)
    doc = prepared[0]
    path = tmp_path / "keyword.sqlite3"
    index = KeywordIndex(path)
    bad = doc.chunks[0].model_copy(update={"source_text": "invented source"})
    with pytest.raises(ValueError, match="parent source"):
        index.register([bad], parents=list(doc.parents))
    assert index.count == 0
    with pytest.raises(ValueError, match="no registered parent"):
        index.get_parent(doc.parents[0].parent_id)
    index.register(list(doc.chunks), parents=list(doc.parents))
    index.close()
    index = KeywordIndex(path, readonly=True)
    try:
        hit = index.search("rpc.deadline_exceeded", limit=1)[0]
        assert index.get_parent(hit.parent_id).text.endswith("`/get-quote`.")
        with pytest.raises(RuntimeError, match="read-only"):
            index.register([])
    finally:
        index.close()


class FakeRuntime:
    """Exercise atomic publication without model downloads or remote
    services.
    """

    def __init__(self, settings, *, failure=None):
        self.settings = settings
        self.model = RagModel(settings)
        self.index = None
        self.rows, self.statuses = {}, {}
        self.failure = failure

    async def initialize(self, *, chunking_func):
        self.index = KeywordIndex(self.settings.workspace / "keyword.sqlite3")
        self.chunker = chunking_func
        self.rag = SimpleNamespace(
            apipeline_enqueue_documents=self.enqueue,
            apipeline_process_enqueue_documents=self.process,
            doc_status=SimpleNamespace(get_by_id=self.status),
            text_chunks=SimpleNamespace(get_by_id=self.row),
            chunks_vdb=SimpleNamespace(
                index_done_callback=self.flush, get_by_ids=self.vectors
            ),
            chunk_entity_relation_graph=SimpleNamespace(
                get_all_nodes=self.nodes, get_all_edges=self.edges
            ),
        )

    async def require_empty_vectors(self):
        pass

    async def enqueue(self, texts, *, ids, file_paths, process_options):
        assert process_options == "C"
        self.doc_key = ids[0]
        self.chunk_rows = self.chunker(WordTokenizer(), texts[0])

    async def process(self):
        self.rows.update({c["chunk_id"]: c for c in self.chunk_rows})
        self.statuses[self.doc_key] = {
            "status": "failed" if self.failure == "pipeline" else "processed",
            "chunks_list": list(self.rows),
        }

    async def status(self, key):
        return self.statuses.get(key)

    async def row(self, key):
        row = self.rows[key]
        return (
            {**row, "content": "tampered"} if self.failure == "text" else row
        )

    async def vectors(self, keys):
        return (
            [] if self.failure == "vectors" else [self.rows[k] for k in keys]
        )

    async def nodes(self):
        return [
            {
                "source_id": "unknown"
                if self.failure == "graph"
                else next(iter(self.rows))
            }
        ]

    async def edges(self):
        return []

    async def flush(self):
        pass

    async def close(self):
        if self.index is not None:
            self.index.close()
            self.index = None


def test_successful_snapshot_publication_and_tampering(tmp_path):
    corpus, prepared, settings = prepared_fixture(tmp_path)
    runtime = FakeRuntime(settings)
    manifest = asyncio.run(
        build_snapshot(runtime, corpus, prepared, timeout_seconds=5)
    )
    assert manifest["state"] == "published"
    assert manifest["parents"] == len(prepared[0].parents)
    assert verify_snapshot(settings.workspace) == manifest
    with pytest.raises(ValueError, match="new snapshot ID"):
        asyncio.run(
            build_snapshot(runtime, corpus, prepared, timeout_seconds=5)
        )
    (settings.workspace / "sources.json").write_text("tampered")
    with pytest.raises(ValueError, match="artifact hash"):
        verify_snapshot(settings.workspace)


@pytest.mark.parametrize("failure", ["pipeline", "text", "vectors", "graph"])
def test_failed_builds_are_never_published(tmp_path, failure):
    corpus, prepared, settings = prepared_fixture(tmp_path)
    runtime = FakeRuntime(settings, failure=failure)
    with pytest.raises((ValueError, RuntimeError)):
        asyncio.run(
            build_snapshot(runtime, corpus, prepared, timeout_seconds=5)
        )
    assert not (settings.workspace / "snapshot.json").exists()
    assert (
        json.loads((settings.workspace / "build.json").read_text())["state"]
        == "failed"
    )
    assert runtime.index is None
    reader = RagRuntime(settings)
    with pytest.raises(ValueError, match="incomplete"):
        asyncio.run(reader.initialize())
