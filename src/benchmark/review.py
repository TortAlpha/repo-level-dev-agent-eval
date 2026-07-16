"""Reviewer-model quality scoring of the agent's diff (``--enable-review``)."""

from __future__ import annotations

import argparse
from pathlib import Path

from ..config import Config
from ..metrics.quality import append_quality, score_solution
from ..run import build_model
from .collection import TaskSpec

# Independent reviewer: a cheap, code-focused model, so the quality score is
# not the agent grading itself.
DEFAULT_REVIEW_MODEL = "openai/gpt-5.1-codex-mini"


def review_solution(
    task: TaskSpec,
    repo_dir: Path,
    config: Config,
    args: argparse.Namespace,
    run_id: str,
) -> int | None:
    """Score the agent's diff with an independent reviewer model and record it."""
    patch_file = repo_dir.parent / "agent.patch"
    diff = patch_file.read_text(encoding="utf-8") if patch_file.is_file() else ""
    if not diff.strip():
        print("\n=== Quality review: skipped (no diff) ===")
        return None

    reviewer = build_model(config, config.provider_spec(args.review_model))
    task_text = task.task_file_path.read_text(encoding="utf-8")
    try:
        score, rationale = score_solution(task_text, diff, reviewer)
    except Exception as exc:  # noqa: BLE001 - review is best-effort
        print(f"\n=== Quality review failed: {exc} ===")
        return None

    print(f"\n=== Quality review: {score}/5 — {rationale} ===")
    append_quality(
        config.results_dir / "quality.jsonl",
        run_id=run_id,
        task_id=task.task_id,
        score=score,
        rationale=rationale,
        reviewer=reviewer.model,
    )
    return score
