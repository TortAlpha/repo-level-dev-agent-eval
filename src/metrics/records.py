"""Run records: the per-run rows written to ``experiments/results/runs.jsonl``.

One row is appended per agent run (see ``run.record_run_result``). This module
loads them into typed ``RunRecord`` objects and, optionally, joins each row
with its repository size bucket from ``collection.csv`` for per-size breakdowns.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_RUNS_PATH = Path("experiments/results/runs.jsonl")
DEFAULT_QUALITY_PATH = Path("experiments/results/quality.jsonl")
DEFAULT_COLLECTION_PATH = Path("repositories/collection.csv")


@dataclass
class RunRecord:
    task_id: str
    run_id: str
    agent_mode: str
    model: str
    status: str
    steps: int
    iterations: int
    test_passed: bool
    hidden_tests_passed: bool | None
    changed_files: list[str] = field(default_factory=list)
    finished_at: str | None = None
    # Efficiency/behavior signals (present on runs recorded since run_metrics).
    duration_s: float | None = None
    llm_calls: int | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    action_counts: dict[str, int] = field(default_factory=dict)
    regressions: int | None = None  # visible tests passing before but not after
    workspace: str | None = None  # repo dir of this run (for quality scoring)
    size: str | None = None  # joined from collection.csv when available
    quality_score: int | None = None  # joined from quality.jsonl when available

    @property
    def solved(self) -> bool:
        return self.status == "solved"

    @property
    def changed_any(self) -> bool:
        return bool(self.changed_files)

    @classmethod
    def from_dict(cls, row: dict) -> RunRecord:
        return cls(
            task_id=row.get("task_id", "?"),
            run_id=row.get("run_id", "?"),
            agent_mode=row.get("agent_mode", "?"),
            model=row.get("model", "?"),
            status=row.get("status", "?"),
            steps=int(row.get("steps", 0)),
            iterations=int(row.get("iterations", 0)),
            test_passed=bool(row.get("test_passed", False)),
            hidden_tests_passed=row.get("hidden_tests_passed"),
            changed_files=list(row.get("changed_files", [])),
            finished_at=row.get("finished_at"),
            duration_s=row.get("duration_s"),
            llm_calls=row.get("llm_calls"),
            input_tokens=row.get("input_tokens"),
            output_tokens=row.get("output_tokens"),
            total_tokens=row.get("total_tokens"),
            action_counts=dict(row.get("action_counts", {})),
            regressions=row.get("regressions"),
            workspace=row.get("workspace"),
        )


def load_runs(path: Path = DEFAULT_RUNS_PATH) -> list[RunRecord]:
    if not path.is_file():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            records.append(RunRecord.from_dict(json.loads(line)))
    return records


def load_sizes(path: Path = DEFAULT_COLLECTION_PATH) -> dict[str, str]:
    """Map ``task_id`` to its size bucket (small/medium) from collection.csv."""
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8") as handle:
        return {
            row["task_id"]: row.get("size", "")
            for row in csv.DictReader(handle)
            if row.get("task_id")
        }


def attach_sizes(records: list[RunRecord], sizes: dict[str, str]) -> None:
    """Fill each record's ``size`` from a task_id->size map, in place."""
    for record in records:
        record.size = sizes.get(record.task_id) or None


def load_quality(path: Path = DEFAULT_QUALITY_PATH) -> dict[str, int]:
    """Map run_id to its latest quality score from quality.jsonl."""
    if not path.is_file():
        return {}
    scores: dict[str, int] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            row = json.loads(line)
            scores[row["run_id"]] = int(row["quality_score"])
    return scores


def attach_quality(records: list[RunRecord], scores: dict[str, int]) -> None:
    """Fill each record's ``quality_score`` from a run_id->score map, in place."""
    for record in records:
        if record.run_id in scores:
            record.quality_score = scores[record.run_id]


def filter_runs(
    records: list[RunRecord],
    *,
    run_id: str | None = None,
    task_id: str | None = None,
    last: int | None = None,
) -> list[RunRecord]:
    """Narrow records to a run, a task, and/or the most recent ``last`` of them."""
    result = records
    if run_id:
        result = [r for r in result if r.run_id == run_id]
    if task_id:
        result = [r for r in result if r.task_id == task_id]
    if last is not None:
        result = sorted(result, key=lambda r: r.finished_at or "")[-last:]
    return result
