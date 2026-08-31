"""Record a run's outcome to ``runs.jsonl`` and derive its per-run metrics."""

from __future__ import annotations

import importlib
import json
import os
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, BinaryIO

try:  # POSIX (Linux/macOS benchmark hosts).
    _fcntl: Any | None = importlib.import_module("fcntl")
except ImportError:  # pragma: no cover - platform-specific import
    _fcntl = None

try:  # Windows fallback for local exploratory runs.
    _msvcrt: Any | None = importlib.import_module("msvcrt")
except ImportError:  # pragma: no cover - platform-specific import
    _msvcrt = None

from ..agents.model import LangChainModel
from ..agents.state import State
from ..config import Config


@contextmanager
def _advisory_jsonl_lock(path: Path) -> Iterator[None]:
    """Serialize appenders across processes or fail before writing.

    Scientific result rows must not rely on an unlocked best-effort append:
    network/unsupported filesystems can otherwise interleave valid-looking
    JSON. A host without a working OS lock is an infrastructure failure.
    """
    lock_path = path.with_name(f"{path.name}.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as lock_handle:
        acquired: str | None = None
        try:
            if _fcntl is not None:
                _fcntl.flock(lock_handle.fileno(), _fcntl.LOCK_EX)
                acquired = "fcntl"
            elif _msvcrt is not None:  # pragma: no cover - exercised on Windows
                lock_handle.seek(0, os.SEEK_END)
                if lock_handle.tell() == 0:
                    lock_handle.write(b"\0")
                    lock_handle.flush()
                lock_handle.seek(0)
                _msvcrt.locking(lock_handle.fileno(), _msvcrt.LK_LOCK, 1)
                acquired = "msvcrt"
            else:
                raise RuntimeError(
                    "No supported OS file lock is available for benchmark JSONL."
                )
        except OSError as exc:
            raise RuntimeError(
                f"Could not lock benchmark JSONL sidecar {lock_path}: {exc}"
            ) from exc
        try:
            yield
        finally:
            if acquired == "fcntl":
                assert _fcntl is not None
                _fcntl.flock(lock_handle.fileno(), _fcntl.LOCK_UN)
            elif acquired == "msvcrt":  # pragma: no cover - Windows only
                assert _msvcrt is not None
                lock_handle.seek(0)
                _msvcrt.locking(lock_handle.fileno(), _msvcrt.LK_UNLCK, 1)


def _write_all(handle: BinaryIO, payload: bytes) -> None:
    remaining = memoryview(payload)
    while remaining:
        written = handle.write(remaining)
        if written is None or written <= 0:
            raise OSError("short write while appending benchmark JSONL")
        remaining = remaining[written:]


def _append_jsonl_record(path: Path, record: dict) -> None:
    """Durably append exactly one JSON object without interleaving writers."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(record, ensure_ascii=False) + "\n").encode("utf-8")
    with _advisory_jsonl_lock(path):
        with path.open("ab", buffering=0) as handle:
            _write_all(handle, payload)
            os.fsync(handle.fileno())


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
        "cache_write_input_tokens": sum(
            item.cache_write_input_tokens for item in unique
        ),
        "reasoning_tokens": sum(item.reasoning_tokens for item in unique),
        "total_tokens": sum(item.total_tokens for item in unique),
        "usage_accounting_complete": all(
            item.usage_accounting_complete for item in unique
        ),
        "action_counts": dict(final_state.action_counts),
        "test_oracle_tamper_attempts": (
            final_state.test_oracle_tamper_attempts
        ),
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
    total_calls = sum(item.calls for item in unique)
    provider_cost_complete = (
        bool(total_calls)
        and all(item.usage_accounting_complete for item in unique)
        and all(
            item.provider_cost_calls >= item.calls
            for item in unique
            if item.calls
        )
    )
    result["provider_cost_calls"] = provider_cost_calls
    result["provider_cost_complete"] = provider_cost_complete
    if provider_cost_calls:
        result["provider_reported_cost_usd"] = round(
            sum(item.provider_reported_cost_usd for item in unique), 8
        )
    effective_costs = [item.effective_cost_usd for item in unique]
    if not any(cost is None for cost in effective_costs):
        result["effective_cost_usd"] = round(
            sum(cost for cost in effective_costs if cost is not None), 8
        )
        any_complete = any(
            item.calls and item.provider_cost_calls >= item.calls for item in unique
        )
        any_reported = any(item.provider_cost_calls for item in unique)
        result["cost_source"] = (
            "provider_actual"
            if provider_cost_complete
            else "mixed_actual_and_static"
            if any_complete
            else "actual_plus_static_upper_bound"
            if any_reported
            else "static_estimate"
        )
    else:
        result["cost_source"] = "unknown_incomplete_usage"
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
    append: bool = True,
) -> dict:
    """Append one line per run so post-hoc evaluation (hidden tests,
    quality scoring) can find the matching LangSmith trace by run_id.

    Returns the constructed record. ``append=False`` lets the benchmark finish
    required post-processing before committing its completion row."""
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
    if append:
        path = config.results_dir / "runs.jsonl"
        _append_jsonl_record(path, record)
    return record
