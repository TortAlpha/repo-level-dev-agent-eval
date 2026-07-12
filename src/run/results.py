"""Record a run's outcome to ``runs.jsonl`` and derive its per-run metrics."""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from ..agents.model import LangChainModel
from ..agents.state import State
from ..config import Config


def run_metrics(
    model: LangChainModel | Sequence[LangChainModel],
    final_state: State,
    duration_s: float,
) -> dict:
    """Per-run efficiency/behavior signals recorded alongside the outcome:
    wall-clock time, LLM usage, and a histogram of action kinds (from which
    tool-use validity and hallucinated references are derived)."""
    models = [model] if isinstance(model, LangChainModel) else list(model)
    unique: list[LangChainModel] = []
    seen: set[int] = set()
    for item in models:
        if id(item) not in seen:
            unique.append(item)
            seen.add(id(item))
    costs = [item.estimated_cost_usd for item in unique]
    cost_usd = (
        sum(cost for cost in costs if cost is not None)
        if not any(cost is None for cost in costs)
        else None
    )
    result = {
        "duration_s": round(duration_s, 2),
        "llm_calls": sum(item.calls for item in unique),
        "input_tokens": sum(item.input_tokens for item in unique),
        "output_tokens": sum(item.output_tokens for item in unique),
        "cached_input_tokens": sum(item.cached_input_tokens for item in unique),
        "total_tokens": sum(item.total_tokens for item in unique),
        "action_counts": dict(final_state.action_counts),
        "compaction_count": final_state.compaction_count,
        "research_guard_violations": final_state.research_guard_violations,
        "compatibility_check_required": final_state.compatibility_check_required,
        "compatibility_check_passed": final_state.compatibility_check_passed,
        "decomposition_required": final_state.decomposition_required,
        "subtask_count": len(final_state.subtasks),
        "completed_subtasks": sum(
            item.status == "completed" for item in final_state.subtasks
        ),
        "workspace_revision": final_state.workspace_revision,
        "full_suite_verified_revision": final_state.full_suite_verified_revision,
        "compatibility_verified_revision": (
            final_state.compatibility_verified_revision
        ),
        "repair_cycles": final_state.repair_cycles,
    }
    if cost_usd is not None:
        result["cost_usd"] = round(cost_usd, 8)
    provider_cost_calls = sum(item.provider_cost_calls for item in unique)
    if provider_cost_calls:
        result["provider_reported_cost_usd"] = round(
            sum(item.provider_reported_cost_usd for item in unique), 8
        )
        result["provider_cost_calls"] = provider_cost_calls
    return result


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
        "finished_at": datetime.now(UTC).isoformat(),
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
