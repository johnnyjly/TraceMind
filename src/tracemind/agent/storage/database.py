"""SQLite schema initialization and explicit transaction ownership."""

from contextlib import contextmanager
import os
from pathlib import Path
import sqlite3
from typing import Iterator

SCHEMA_VERSION = 4


class RunDatabase:
    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        try:
            descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                                 0o600)
        except FileExistsError:
            pass
        else:
            os.close(descriptor)
        with self.transaction() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("BEGIN IMMEDIATE")
            version = connection.execute("PRAGMA user_version").fetchone()[0]
            if version not in (0, 1, 2, 3, SCHEMA_VERSION):
                raise RuntimeError(
                    f"unsupported run database schema version: {version}"
                )
            if version == 0:
                connection.execute("""
                    CREATE TABLE runs (
                        id TEXT PRIMARY KEY,
                        status TEXT NOT NULL,
                        state_json TEXT NOT NULL,
                        evidence_json TEXT NOT NULL,
                        result_json TEXT,
                        request_hash TEXT NOT NULL,
                        idempotency_key TEXT UNIQUE,
                        cancel_requested INTEGER NOT NULL DEFAULT 0,
                        lease_owner TEXT,
                        lease_until REAL,
                        attempt INTEGER NOT NULL DEFAULT 0,
                        created_at REAL NOT NULL,
                        updated_at REAL NOT NULL
                    )
                """)
            if version < 2:
                connection.execute("""
                    CREATE TABLE investigations (
                        id TEXT PRIMARY KEY,
                        incident_json TEXT NOT NULL,
                        created_at REAL NOT NULL
                    )
                """)
                connection.execute(
                    "ALTER TABLE runs ADD COLUMN investigation_id TEXT"
                )
                connection.execute("""
                    CREATE INDEX runs_investigation ON runs(investigation_id)
                """)
                connection.execute("PRAGMA user_version=2")
            if version < 3:
                connection.execute("""
                    CREATE TABLE incident_memories (
                        id TEXT PRIMARY KEY,
                        revision INTEGER NOT NULL,
                        environment TEXT NOT NULL,
                        deployment_version TEXT NOT NULL,
                        source_incident_id TEXT NOT NULL,
                        status TEXT NOT NULL,
                        confirmed_at REAL NOT NULL,
                        expires_at REAL NOT NULL,
                        updated_at REAL NOT NULL,
                        record_json TEXT NOT NULL
                    )
                """)
                connection.execute("""
                    CREATE INDEX memories_scope ON incident_memories
                    (environment,deployment_version,status,expires_at)
                """)
                connection.execute("PRAGMA user_version=3")
            if version < 4:
                connection.execute("""
                    CREATE TABLE agent_guidance (
                        id TEXT PRIMARY KEY,
                        key TEXT NOT NULL UNIQUE,
                        record_json TEXT NOT NULL
                    )
                """)
                connection.execute("""
                    CREATE TABLE memory_feedback (
                        run_id TEXT PRIMARY KEY,
                        result_json TEXT NOT NULL
                    )
                """)
                connection.execute("PRAGMA user_version=4")

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout=10000")
        try:
            yield connection
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()
