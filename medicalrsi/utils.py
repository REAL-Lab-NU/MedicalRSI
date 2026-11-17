"""Attempt files, scores and workflow keys.

An attempt is one task played by the Solver. Its row holds job, sample_id, harness (sha256 of each harness file),
solved (the outcome), process (the process grader's result, with its score), status, error, seconds and history (the
trajectory, without the harness). A job id is "<tag>|<workflow>/<harness name>|<sample id>"."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

LOST = ("TIMEOUT", "NOT_AVAILABLE")      # attempts lost to the deadline or the server, played once more


def workflow_of(sample_id: str) -> str:
    return sample_id.split("_")[0]


def hkey(row: dict) -> str:
    return hashlib.sha256(row["sample_id"].encode()).hexdigest()


def file_sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_rows(path) -> list:
    if not Path(path).exists():
        return []
    return [json.loads(line) for line in open(path)]


def write_jsonl(path, rows) -> None:
    Path(path).write_text("".join(json.dumps(r) + "\n" for r in rows))


def finished(rows: list) -> list:
    """Attempts played to their end."""
    return [r for r in rows if r.get("error") in (None, "None") and r.get("history")]


def task_scores(rows: list) -> dict:
    """sample id -> (outcome, process score); an attempt past its deadline counts as failed."""
    out = {}
    for r in rows:
        if r.get("error") == "TIMEOUT":
            out[r["sample_id"]] = (0, 0.0)
        elif r.get("error") in (None, "None") and r.get("process") and "score" in r["process"]:
            out[r["sample_id"]] = (int(bool(r["solved"])), r["process"]["score"])
    return out


def clean_success(row: dict) -> bool:
    """Solved, with every check of the process grader passed."""
    groups = (row.get("process") or {}).get("groups") or {}
    return bool(row["solved"]) and row.get("status") == "completed" and all(ok for cs in groups.values() for _, ok in cs)
