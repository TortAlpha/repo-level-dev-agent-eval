"""Record a run's outcome to ``runs.jsonl`` and derive its per-run metrics."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from ..agents.model import LangChainModel
from ..agents.state import State
from ..config import Config


def run_metrics(
    model: LangChainModel, final_state: State, duration_s: float
) -> dict:
    """Per-run efficiency/behavior signals recorded alongside the outcome:
    wall-clock time, LLM usage, and a histogram of action kinds (from which
    tool-use validity and hallucinated references are derived)."""
    return {
        "duration_s": round(duration_s, 2),
        "llm_calls": model.calls,
        "input_tokens": model.input_tokens,
        "output_tokens": model.output_tokens,
        "total_tokens": model.total_tokens,
        "action_counts": dict(final_state.action_counts),
    }


def load_task_meta(task_file: Path) -> dict:
    meta_path = task_file.parent / "meta.json"
    if not meta_path.is_file():
        return {}
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def record_run_result(
    config: Config,
    *,
    task_id: str,
    run_id: str,
    model_name: str,
    final_state: State,
    summary: str | None = None,
    test_passed: bool | None = None,
    hidden_tests_passed: bool | None = None,
    extra: dict | None = None,
) -> dict:
    """Append one line per run so post-hoc evaluation (hidden tests,
    quality scoring) can find the matching LangSmith trace by run_id.

    Returns the record that was written."""
    config.results_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "task_id": task_id,
        "run_id": run_id,
        "session_id": config.session_id,
        "agent_mode": config.agent_mode,
        "model": model_name,
        "status": final_state.status,
        "steps": final_state.step,
        "iterations": final_state.iteration,
        "test_passed": final_state.test_passed if test_passed is None else test_passed,
        "changed_files": [str(path) for path in final_state.changed_files],
        "compaction_mode": config.compaction_mode,
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }
    if summary is not None:
        record["summary"] = summary
    if hidden_tests_passed is not None:
        record["hidden_tests_passed"] = hidden_tests_passed
    if extra:
        record.update(extra)
    path = config.results_dir / "runs.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return record
