"""Decode database snapshots without depending on execution services."""

from __future__ import annotations

import json
import sqlite3

from tracemind.models import Evidence
from tracemind.models.result import DiagnosisRun
from tracemind.models.run import RunState
from tracemind.models.storage import StoredRun


def decode_run(row: sqlite3.Row) -> StoredRun:
    evidence = [Evidence.model_validate(item) for item in
                json.loads(row["evidence_json"])]
    result = (DiagnosisRun.model_validate_json(row["result_json"])
              if row["result_json"] else None)
    return StoredRun(
        state=RunState.model_validate_json(row["state_json"]),
        evidence=evidence,
        result=result,
        cancel_requested=bool(row["cancel_requested"]),
        attempt=row["attempt"],
        lease_until=row["lease_until"],
    )
