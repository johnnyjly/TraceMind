"""A shared admission cap, optionally durable across experiment processes."""

from datetime import UTC, datetime
from pathlib import Path
import sqlite3
from threading import Lock

from langchain_core.callbacks import BaseCallbackHandler

from tracemind.agent.errors import ModelCallBudgetExceeded


class ModelCallBudget(BaseCallbackHandler):
    raise_error = True

    def __init__(self, limit: int, *, ledger_path: Path | None = None):
        if limit < 1:
            raise ValueError("model-call limit must be positive")
        self.limit = limit
        self._calls = 0
        self._lock = Lock()
        self.ledger_path = ledger_path
        if ledger_path is not None:
            ledger_path.parent.mkdir(parents=True, exist_ok=True)
            with sqlite3.connect(ledger_path) as connection:
                connection.execute("CREATE TABLE IF NOT EXISTS budget "
                                   "(id INTEGER PRIMARY KEY, cap INTEGER, "
                                   "calls INTEGER)")
                connection.execute("CREATE TABLE IF NOT EXISTS admissions "
                                   "(number INTEGER PRIMARY KEY, "
                                   "run_id TEXT, admitted_at TEXT)")
                connection.execute("INSERT OR IGNORE INTO budget "
                                   "VALUES (1, ?, 0)", (limit,))
                cap = connection.execute(
                    "SELECT cap FROM budget WHERE id=1").fetchone()[0]
                if cap != limit:
                    raise ValueError("campaign budget cannot change on resume")

    @property
    def calls(self):
        if self.ledger_path is None:
            return self._calls
        with sqlite3.connect(self.ledger_path) as connection:
            return connection.execute(
                "SELECT calls FROM budget WHERE id=1").fetchone()[0]

    def on_chat_model_start(self, serialized, messages, **kwargs):
        with self._lock:
            if self.ledger_path is None:
                if self._calls >= self.limit:
                    raise ModelCallBudgetExceeded(
                        "experiment model-call budget exhausted")
                self._calls += 1
                return
            # Commit admission before provider execution. Unknown outcomes
            # remain charged; reopening the ledger cannot reset the cap.
            with sqlite3.connect(self.ledger_path) as connection:
                connection.execute("BEGIN IMMEDIATE")
                row = connection.execute(
                    "UPDATE budget SET calls=calls+1 "
                    "WHERE id=1 AND calls<cap RETURNING calls").fetchone()
                if row is None:
                    raise ModelCallBudgetExceeded(
                        "experiment model-call budget exhausted")
                connection.execute(
                    "INSERT INTO admissions VALUES (?, ?, ?)",
                    (row[0], str(kwargs.get("run_id", "")),
                     datetime.now(UTC).isoformat()))
