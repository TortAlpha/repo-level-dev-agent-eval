"""Reviewer-model quality scoring (spec rubric, 0-5) for a run's solution.

Run after a benchmark run to score the agent's diff against the spec's quality
rubric with an LLM judge. Scores are appended to ``quality.jsonl`` (keyed by
run_id); ``metrics.report`` joins them into the ``quality_score`` metric.

    python -m src.metrics.quality --last 1
    python -m src.metrics.quality --run-id <id> --model <reviewer-model>
"""

from __future__ import annotations

import argparse
import ast
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from ..agents.model import LangChainModel
from ..agents.prompts import load_prompt
from ..config import Config, load_config
from ..run import build_model
from .records import (
    DEFAULT_COLLECTION_PATH,
    DEFAULT_QUALITY_PATH,
    DEFAULT_RUNS_PATH,
    RunRecord,
    filter_runs,
    load_runs,
)

QUALITY_PROMPT = load_prompt("quality_review")


def score_solution(task_text: str, diff: str, model: LangChainModel) -> tuple[int, str]:
    """Ask the reviewer model to score the diff 0-5 with a short rationale."""
    content = f"# Task\n{task_text.strip()}\n\n# Candidate diff\n{diff.strip()}"
    raw = model.summarize(QUALITY_PROMPT, content)
    return _parse_score(raw)


def _parse_score(raw: str) -> tuple[int, str]:
    """Extract (score 0-5, rationale) from a reviewer reply, tolerating
    Markdown fences, Python-style single quotes, and stray prose."""
    text = _strip_fences(raw.strip())
    start, end = text.find("{"), text.rfind("}")
    blob = text[start : end + 1] if start != -1 and end > start else text

    for loader in (json.loads, ast.literal_eval):
        try:
            data = loader(blob)
        except (ValueError, SyntaxError):
            continue
        if isinstance(data, dict) and "score" in data:
            score = max(0, min(5, int(data["score"])))
            return score, str(data.get("rationale", "")).strip()

    # Last resort: the first 0-5 digit near the word "score".
    match = re.search(r"score\D{0,6}([0-5])", text, re.IGNORECASE)
    if not match:
        raise ValueError(f"reviewer returned no parseable score: {raw[:200]}")
    return int(match.group(1)), text.strip()


def _strip_fences(text: str) -> str:
    if text.startswith("```"):
        lines = [ln for ln in text.splitlines() if not ln.strip().startswith("```")]
        return "\n".join(lines).strip()
    return text


def load_agent_patch(record: RunRecord, config: Config) -> str:
    """The agent's diff for a run, saved next to its workspace by the runner."""
    workspace = record.workspace or str(
        config.workspaces_dir / f"{record.task_id}-{record.run_id}" / "repo"
    )
    patch_file = Path(workspace).parent / "agent.patch"
    if patch_file.is_file():
        return patch_file.read_text(encoding="utf-8")
    return ""


def append_quality(
    path: Path,
    *,
    run_id: str,
    task_id: str,
    score: int,
    rationale: str,
    reviewer: str,
) -> None:
    """Append one quality verdict to quality.jsonl (read back by the report)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "run_id": run_id,
        "task_id": task_id,
        "quality_score": score,
        "rationale": rationale,
        "reviewer_model": reviewer,
        "scored_at": datetime.now(timezone.utc).isoformat(),
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    args = parse_args()
    config = load_config(args.provider)

    records = filter_runs(
        load_runs(args.runs),
        run_id=args.run_id,
        task_id=args.task_id,
        last=args.last,
    )
    if not records:
        print("No runs match the given filter.")
        return 1

    from ..benchmark import load_collection  # local import avoids an import cycle

    collection = load_collection(args.collection)
    model = build_model(config, config.provider_spec(args.model))

    scored = 0
    for record in records:
        tag = f"{record.task_id} {record.run_id[:8]}"
        task = collection.get(record.task_id)
        if task is None:
            print(f"skip {tag}: task not in collection")
            continue
        diff = load_agent_patch(record, config)
        if not diff.strip():
            print(f"skip {tag}: no agent.patch found")
            continue

        task_text = task.task_file_path.read_text(encoding="utf-8")
        score, rationale = score_solution(task_text, diff, model)
        print(f"[{tag}] quality = {score}/5 — {rationale}")
        append_quality(
            args.output,
            run_id=record.run_id,
            task_id=record.task_id,
            score=score,
            rationale=rationale,
            reviewer=model.model,
        )
        scored += 1

    print(f"\nScored {scored} run(s) -> {args.output}")
    return 0 if scored else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reviewer-model quality scoring.")
    parser.add_argument("--runs", type=Path, default=DEFAULT_RUNS_PATH)
    parser.add_argument("--collection", type=Path, default=DEFAULT_COLLECTION_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_QUALITY_PATH)
    parser.add_argument("--provider", choices=["local", "openrouter"], default=None)
    parser.add_argument("--model", default=None, help="Reviewer model override.")
    parser.add_argument("--last", type=int, default=1)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--task-id", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
