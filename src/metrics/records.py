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

_HIDDEN_SUITE_COLUMNS = {
    "hidden_semantic": "hidden_semantic_test_command",
    "hidden_compat": "hidden_compat_test_command",
    "hidden_pr_parity": "hidden_pr_parity_test_command",
}


def _bool_or_none(value: object) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in ("true", "1", "yes"):
            return True
        if lowered in ("false", "0", "no"):
            return False
        if lowered in ("", "none", "null"):
            return None
    return bool(value)


@dataclass
class RunRecord:
    task_id: str
    run_id: str
    agent_mode: str
    model: str
    status: str
    session_id: str  # experiment grouping; see Config.session_id
    steps: int
    iterations: int
    test_passed: bool
    hidden_tests_passed: bool | None
    task_success: bool | None = None
    hidden_required_tests_passed: bool | None = None
    hidden_semantic_tests_passed: bool | None = None
    hidden_compat_tests_passed: bool | None = None
    hidden_pr_parity_tests_passed: bool | None = None
    hidden_suite_results: dict[str, bool] = field(default_factory=dict)
    provider: str | None = None
    action_transport: str | None = None
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
    task_type: str | None = None  # bugfix/feature (recorded, or joined)
    size: str | None = None  # joined from collection.csv when available
    difficulty: str | None = None  # best-available bucket (empirical or static)
    difficulty_estimate: str | None = None  # static (a priori) bucket
    quality_score: int | None = None  # joined from quality.jsonl when available

    @property
    def solved(self) -> bool:
        return self.status == "solved"

    @property
    def changed_any(self) -> bool:
        return bool(self.changed_files)

    @property
    def required_hidden_passed(self) -> bool | None:
        if self.hidden_required_tests_passed is not None:
            return self.hidden_required_tests_passed
        return self.hidden_tests_passed

    @property
    def outcome(self) -> bool | None:
        """Ground-truth task outcome.

        New runs record ``task_success`` as visible + required hidden suites.
        Older runs only have the legacy hidden verdict, so preserve that
        behavior when the new field is absent.
        """
        if self.task_success is not None:
            return self.task_success
        return self.required_hidden_passed

    @classmethod
    def from_dict(cls, row: dict) -> RunRecord:
        hidden_tests_passed = _bool_or_none(row.get("hidden_tests_passed"))
        hidden_suite_results: dict[str, bool] = {}
        for name, value in dict(row.get("hidden_suite_results") or {}).items():
            parsed = _bool_or_none(value)
            if parsed is not None:
                hidden_suite_results[str(name)] = parsed

        def hidden_suite_field(field: str, suite: str) -> bool | None:
            value = row.get(field)
            if value is None:
                value = hidden_suite_results.get(suite)
            return _bool_or_none(value)

        return cls(
            task_id=row.get("task_id", "?"),
            run_id=row.get("run_id", "?"),
            agent_mode=row.get("agent_mode", "?"),
            model=row.get("model", "?"),
            status=row.get("status", "?"),
            session_id=row.get("session_id", "default"),
            provider=row.get("provider"),
            action_transport=row.get("action_transport"),
            steps=int(row.get("steps", 0)),
            iterations=int(row.get("iterations", 0)),
            test_passed=bool(row.get("test_passed", False)),
            hidden_tests_passed=hidden_tests_passed,
            task_success=_bool_or_none(row.get("task_success")),
            hidden_required_tests_passed=_bool_or_none(
                row.get("hidden_required_tests_passed", hidden_tests_passed)
            ),
            hidden_semantic_tests_passed=hidden_suite_field(
                "hidden_semantic_tests_passed", "hidden_semantic"
            ),
            hidden_compat_tests_passed=hidden_suite_field(
                "hidden_compat_tests_passed", "hidden_compat"
            ),
            hidden_pr_parity_tests_passed=hidden_suite_field(
                "hidden_pr_parity_tests_passed", "hidden_pr_parity"
            ),
            hidden_suite_results=hidden_suite_results,
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
            task_type=row.get("task_type"),
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


def _collection_field(path: Path, field: str) -> dict[str, str]:
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8") as handle:
        return {
            row["task_id"]: row.get(field, "")
            for row in csv.DictReader(handle)
            if row.get("task_id")
        }


def load_sizes(path: Path = DEFAULT_COLLECTION_PATH) -> dict[str, str]:
    """Map ``task_id`` to its size bucket (small/medium) from collection.csv."""
    return _collection_field(path, "size")


def load_task_types(path: Path = DEFAULT_COLLECTION_PATH) -> dict[str, str]:
    """Map ``task_id`` to its type (bugfix/feature) from collection.csv."""
    return _collection_field(path, "task_type")


def load_hidden_suite_specs(path: Path = DEFAULT_COLLECTION_PATH) -> dict[str, set[str]]:
    """Map ``task_id`` to the hidden suite names configured in collection.csv."""
    if not path.is_file():
        return {}
    specs: dict[str, set[str]] = {}
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            task_id = row.get("task_id")
            if not task_id:
                continue
            suites = {
                suite
                for suite, column in _HIDDEN_SUITE_COLUMNS.items()
                if (row.get(column) or "").strip()
            }
            specs[task_id] = suites
    return specs


def attach_sizes(records: list[RunRecord], sizes: dict[str, str]) -> None:
    """Fill each record's ``size`` from a task_id->size map, in place."""
    for record in records:
        record.size = sizes.get(record.task_id) or None


def attach_task_types(records: list[RunRecord], types: dict[str, str]) -> None:
    """Backfill ``task_type`` from a task_id->type map for records that lack it
    (runs recorded before task_type was added); recorded values win."""
    for record in records:
        if not record.task_type:
            record.task_type = types.get(record.task_id) or None


def attach_hidden_suite_fallbacks(
    records: list[RunRecord],
    suite_specs: dict[str, set[str]],
) -> None:
    """Backfill legacy hidden verdicts when the split is unambiguous.

    Runs recorded before split-hidden support only have ``hidden_tests_passed``.
    If a task now has exactly one configured hidden suite and it is semantic, the
    legacy verdict is safe to show as ``Semantic Hidden``. For tasks with compat
    or PR-parity suites we keep the split fields unknown rather than guessing.
    """
    for record in records:
        if record.hidden_tests_passed is None:
            continue
        suites = suite_specs.get(record.task_id, set())
        if suites != {"hidden_semantic"}:
            continue
        if record.hidden_semantic_tests_passed is None:
            record.hidden_semantic_tests_passed = record.hidden_tests_passed
            record.hidden_suite_results.setdefault(
                "hidden_semantic", record.hidden_tests_passed
            )


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
    session_id: str | None = None,
    last: int | None = None,
) -> list[RunRecord]:
    """Narrow records to a run, a task, a session, and/or the most recent ``last``."""
    result = records
    if run_id:
        result = [r for r in result if r.run_id == run_id]
    if task_id:
        result = [r for r in result if r.task_id == task_id]
    if session_id:
        result = [r for r in result if r.session_id == session_id]
    if last is not None:
        result = sorted(result, key=lambda r: r.finished_at or "")[-last:]
    return result
