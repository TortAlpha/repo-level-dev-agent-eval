"""Task difficulty.

Two complementary notions:

* **Static estimate** (``difficulty_estimate``) — a cheap, pre-run proxy from
  how much the reference fix changes (patch files/LOC), how hard the fix is to
  localize (repo LOC), and task type. Useful for stratifying a set before any
  model runs, but only a proxy: a one-line patch can still be hard for an agent.

* **Empirical difficulty** (``difficulty``) — the real thing, measured from
  model performance: ``difficulty = 1 - solve_rate`` over the runs on that task,
  plus a ``discrimination`` signal (how much models disagree). A task everyone
  solves — or everyone fails — carries little information regardless of its
  static estimate.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from .records import DEFAULT_COLLECTION_PATH, RunRecord, _collection_field

# --------------------------------------------------------------------------- #
# Static (a priori) estimate
# --------------------------------------------------------------------------- #

STATIC_THRESHOLDS = (25.0, 45.0)  # easy < 25 <= medium < 45 <= hard


def static_score(patch_files: int, patch_loc: int, repo_loc: int, task_type: str) -> float:
    """Cheap pre-run difficulty proxy — see module docstring for the caveat."""
    return (
        patch_files * 3.0
        + patch_loc * 0.3
        + math.log10(max(repo_loc, 1)) * 4.0
        + (5.0 if task_type == "feature" else 0.0)
    )


def bucket(value: float, thresholds: tuple[float, float]) -> str:
    lo, hi = thresholds
    return "easy" if value < lo else "medium" if value < hi else "hard"


def load_static_difficulty(path: Path = DEFAULT_COLLECTION_PATH) -> dict[str, str]:
    """Map ``task_id`` to its static difficulty bucket from collection.csv."""
    pf, pl = _collection_field(path, "patch_files"), _collection_field(path, "patch_loc")
    rl, tt = _collection_field(path, "source_loc"), _collection_field(path, "task_type")
    out: dict[str, str] = {}
    for tid in pf:
        try:
            score = static_score(
                int(pf.get(tid) or 0), int(pl.get(tid) or 0),
                int(rl.get(tid) or 0), tt.get(tid, ""),
            )
        except ValueError:
            continue
        out[tid] = bucket(score, STATIC_THRESHOLDS)
    return out


# --------------------------------------------------------------------------- #
# Empirical (measured) difficulty
# --------------------------------------------------------------------------- #

MIN_RUNS = 3  # below this the empirical estimate is too noisy to trust
EMPIRICAL_THRESHOLDS = (0.34, 0.67)  # on difficulty = 1 - solve_rate


@dataclass
class TaskDifficulty:
    task_id: str
    n_runs: int
    n_models: int
    solve_rate: float             # fraction of runs that resolved@1
    difficulty: float             # 1 - solve_rate
    discrimination: float | None  # std of per-model solve rate (model separation)


def _solved(record: RunRecord) -> bool | None:
    """Ground-truth outcome: task_success, then legacy hidden verdict."""
    if record.outcome is not None:
        return bool(record.outcome)
    if record.status in ("solved", "failed", "handoff"):
        return record.status == "solved"
    return None


def empirical_difficulty(records: list[RunRecord]) -> dict[str, TaskDifficulty]:
    """Per-task difficulty measured from the runs, keyed by ``task_id``."""
    by_task: dict[str, list[RunRecord]] = defaultdict(list)
    for record in records:
        if _solved(record) is not None:
            by_task[record.task_id].append(record)

    out: dict[str, TaskDifficulty] = {}
    for task_id, runs in by_task.items():
        per_model: dict[str, list[float]] = defaultdict(list)
        for record in runs:
            per_model[record.model].append(1.0 if _solved(record) else 0.0)

        outcomes = [o for values in per_model.values() for o in values]
        solve_rate = sum(outcomes) / len(outcomes)

        discrimination = None
        model_rates = [sum(v) / len(v) for v in per_model.values()]
        if len(model_rates) >= 2:
            mean = sum(model_rates) / len(model_rates)
            discrimination = round(
                (sum((r - mean) ** 2 for r in model_rates) / len(model_rates)) ** 0.5, 3
            )

        out[task_id] = TaskDifficulty(
            task_id=task_id,
            n_runs=len(runs),
            n_models=len(per_model),
            solve_rate=round(solve_rate, 3),
            difficulty=round(1 - solve_rate, 3),
            discrimination=discrimination,
        )
    return out


def attach_difficulty(
    records: list[RunRecord], collection_path: Path = DEFAULT_COLLECTION_PATH
) -> dict[str, TaskDifficulty]:
    """Set ``difficulty_estimate`` (static) and ``difficulty`` (best available:
    empirical bucket when the task has >= MIN_RUNS, else the static proxy) on
    each record, in place. Returns the empirical map for reporting."""
    static = load_static_difficulty(collection_path)
    empirical = empirical_difficulty(records)
    for record in records:
        record.difficulty_estimate = static.get(record.task_id)
        measured = empirical.get(record.task_id)
        if measured and measured.n_runs >= MIN_RUNS:
            record.difficulty = bucket(measured.difficulty, EMPIRICAL_THRESHOLDS)
        else:
            record.difficulty = static.get(record.task_id)
    return empirical
