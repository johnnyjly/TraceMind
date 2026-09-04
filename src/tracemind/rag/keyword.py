"""SQLite metadata and independently ranked BM25 recall for one snapshot."""

import re
import sqlite3
from hashlib import sha256
from pathlib import Path

from tracemind.rag.fusion import fuse_candidates
from tracemind.rag.models import KnowledgeChunk, KnowledgeParent


class KeywordIndex:
    """Own one local index; published snapshots will be opened read-only."""

    def __init__(self, path: Path, *, readonly: bool = False) -> None:
        """Create metadata and FTS5 tables without ingesting any documents."""
        self.readonly = readonly
        if readonly:
            self._db = sqlite3.connect(
                path.resolve().as_uri() + "?mode=ro", uri=True
            )
            return
        self._db = sqlite3.connect(path)
        self._db.executescript("""
            CREATE TABLE IF NOT EXISTS parents (
                parent_id TEXT PRIMARY KEY, payload TEXT NOT NULL,
                payload_sha256 TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id TEXT PRIMARY KEY, payload TEXT NOT NULL,
                content_sha256 TEXT NOT NULL
            );
            CREATE VIRTUAL TABLE IF NOT EXISTS prose USING fts5(
                chunk_id UNINDEXED, title, section, text,
                tokenize='porter unicode61'
            );
            CREATE VIRTUAL TABLE IF NOT EXISTS identifiers USING fts5(
                chunk_id UNINDEXED, text,
                tokenize="unicode61 tokenchars '_-./:'"
            );
        """)

    def close(self) -> None:
        """Release the snapshot database connection."""
        self._db.close()

    @property
    def count(self) -> int:
        """Return the number of registered canonical chunks."""
        return self._db.execute("SELECT count(*) FROM chunks").fetchone()[0]

    def register(
        self,
        chunks: list[KnowledgeChunk],
        *,
        parents: list[KnowledgeParent] | None = None,
    ) -> None:
        """Atomically register chunks and update both keyword indexes.

        Ingestion must supply the same chunk IDs used by LightRAG. This
        primitive does not read source files or publish a knowledge snapshot.
        """
        if self.readonly:
            raise RuntimeError("published keyword index is read-only")
        with self._db:
            for parent in parents or []:
                payload = parent.model_dump_json()
                self._db.execute(
                    "INSERT OR REPLACE INTO parents VALUES (?, ?, ?)",
                    (
                        parent.parent_id,
                        payload,
                        sha256(payload.encode()).hexdigest(),
                    ),
                )
            for chunk in chunks:
                if chunk.parent_id is not None:
                    parent = self.get_parent(chunk.parent_id)
                    if chunk.parser_version != "legacy-v1":
                        if (not chunk.source_items
                                or not chunk.source_text
                                or not chunk.text.endswith(chunk.source_text)
                                or not set(chunk.source_items).issubset(
                                    parent.source_items)
                                or chunk.doc_id != parent.doc_id
                                or chunk.document_sha256 != (
                                    parent.document_sha256)
                                or chunk.source_revision != (
                                    parent.source_revision)
                                or chunk.source_refs != parent.source_refs
                                or chunk.section != parent.section
                                or chunk.parser_version != (
                                    parent.parser_version)):
                            raise ValueError(
                                "child chunk differs from its parent source")
                    else:
                        self._validate_legacy_child(chunk, parent)
                self._db.execute(
                    "INSERT OR REPLACE INTO chunks VALUES (?, ?, ?)",
                    (
                        chunk.chunk_id,
                        chunk.model_dump_json(),
                        chunk.content_sha256,
                    ),
                )
                for table in ("prose", "identifiers"):
                    self._db.execute(
                        f"DELETE FROM {table} WHERE chunk_id=?",
                        (chunk.chunk_id,),
                    )
                self._db.execute(
                    "INSERT INTO prose VALUES (?, ?, ?, ?)",
                    (chunk.chunk_id, chunk.title, chunk.section, chunk.text),
                )
                self._db.execute(
                    "INSERT INTO identifiers VALUES (?, ?)",
                    (chunk.chunk_id, chunk.identifiers),
                )

    @staticmethod
    def _validate_legacy_child(chunk: KnowledgeChunk,
                               parent: KnowledgeParent) -> None:
        """Validate pre-Docling snapshots without rewriting their schema."""
        expected = parent.text[chunk.source_start:chunk.source_end]
        canonical = f"{parent.title}\n{parent.section}\n\n{expected}"
        if (chunk.source_end > len(parent.text)
                or chunk.source_end <= chunk.source_start
                or chunk.source_text != expected
                or chunk.doc_id != parent.doc_id
                or chunk.document_sha256 != parent.document_sha256
                or chunk.source_refs != parent.source_refs
                or chunk.source_revision != parent.source_revision
                or chunk.section != parent.section
                or chunk.text != canonical):
            raise ValueError("child chunk differs from its parent source")

    def get(self, chunk_id: str) -> KnowledgeChunk:
        """Require registered provenance and an unchanged canonical payload."""
        row = self._db.execute(
            "SELECT payload, content_sha256 FROM chunks WHERE chunk_id=?",
            (chunk_id,),
        ).fetchone()
        if row is None:
            raise ValueError("retrieved chunk has no registered provenance")
        chunk = KnowledgeChunk.model_validate_json(row[0])
        if chunk.chunk_id != chunk_id or chunk.content_sha256 != row[1]:
            raise ValueError("registered chunk payload failed its hash check")
        return chunk

    def get_parent(self, parent_id: str) -> KnowledgeParent:
        """Return complete section text with checked provenance and
        integrity.
        """
        row = self._db.execute(
            "SELECT payload, payload_sha256 FROM parents WHERE parent_id=?",
            (parent_id,),
        ).fetchone()
        if row is None:
            raise ValueError("chunk has no registered parent section")
        if sha256(row[0].encode()).hexdigest() != row[1]:
            raise ValueError("parent section payload failed its hash check")
        parent = KnowledgeParent.model_validate_json(row[0])
        if parent.parent_id != parent_id:
            raise ValueError("parent section identity mismatch")
        return parent

    def search(self, query: str, *, limit: int) -> list[KnowledgeChunk]:
        """Search escaped literal terms with stemming and identifier recall."""
        terms = re.findall(r"[\w./:-]+", query, flags=re.UNICODE)[:64]
        if not terms:
            return []
        expression = " OR ".join('"' + term + '"' for term in terms)
        branches = {}
        for table, weights in (
            ("prose", "0, 3, 2, 1"),
            ("identifiers", "0, 1"),
        ):
            rows = self._db.execute(
                f"SELECT chunk_id FROM {table} WHERE {table} MATCH ? "
                f"ORDER BY bm25({table}, {weights}), chunk_id LIMIT ?",
                (expression, limit),
            ).fetchall()
            branches[table] = [self.get(row[0]) for row in rows]
        return [hit.chunk for hit in fuse_candidates(branches, limit=limit)]
