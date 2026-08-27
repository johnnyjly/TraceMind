"""SQLite metadata and independently ranked BM25 recall for one snapshot."""

from pathlib import Path
import re
import sqlite3

from tracemind.rag.models import KnowledgeChunk, fuse_candidates


class KeywordIndex:
    """Own one local index; published snapshots will be opened read-only."""

    def __init__(self, path: Path) -> None:
        """Create metadata and FTS5 tables without ingesting any documents."""
        self._db = sqlite3.connect(path)
        self._db.executescript("""
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

    def register(self, chunks: list[KnowledgeChunk]) -> None:
        """Atomically register chunks and update both keyword indexes.

        Ingestion must supply the same chunk IDs used by LightRAG. This
        primitive does not read source files or publish a knowledge snapshot.
        """
        with self._db:
            for chunk in chunks:
                self._db.execute(
                    "INSERT OR REPLACE INTO chunks VALUES (?, ?, ?)",
                    (chunk.chunk_id, chunk.model_dump_json(),
                     chunk.content_sha256))
                for table in ("prose", "identifiers"):
                    self._db.execute(f"DELETE FROM {table} WHERE chunk_id=?",
                                     (chunk.chunk_id,))
                self._db.execute("INSERT INTO prose VALUES (?, ?, ?, ?)",
                                 (chunk.chunk_id, chunk.title, chunk.section,
                                  chunk.text))
                self._db.execute("INSERT INTO identifiers VALUES (?, ?)",
                                 (chunk.chunk_id, chunk.identifiers))

    def get(self, chunk_id: str) -> KnowledgeChunk:
        """Require registered provenance and an unchanged canonical payload."""
        row = self._db.execute(
            "SELECT payload, content_sha256 FROM chunks WHERE chunk_id=?",
            (chunk_id,)).fetchone()
        if row is None:
            raise ValueError("retrieved chunk has no registered provenance")
        chunk = KnowledgeChunk.model_validate_json(row[0])
        if chunk.chunk_id != chunk_id or chunk.content_sha256 != row[1]:
            raise ValueError("registered chunk payload failed its hash check")
        return chunk

    def search(self, query: str, *, limit: int) -> list[KnowledgeChunk]:
        """Search escaped literal terms with stemming and identifier recall."""
        terms = re.findall(r"[\w./:-]+", query, flags=re.UNICODE)[:64]
        if not terms:
            return []
        expression = " OR ".join('"' + term + '"' for term in terms)
        branches = {}
        for table, weights in (("prose", "0, 3, 2, 1"),
                               ("identifiers", "0, 1")):
            rows = self._db.execute(
                f"SELECT chunk_id FROM {table} WHERE {table} MATCH ? "
                f"ORDER BY bm25({table}, {weights}), chunk_id LIMIT ?",
                (expression, limit)).fetchall()
            branches[table] = [self.get(row[0]) for row in rows]
        return [hit.chunk for hit in fuse_candidates(branches, limit=limit)]
