"""LangSmith session wiring: env setup, run-level metadata, and feedback."""

from __future__ import annotations

import os
from typing import Any


def configure_langsmith(config: Any) -> None:
    if not config.langsmith_tracing_enabled:
        return

    os.environ["LANGSMITH_TRACING_V2"] = "true"
    os.environ["LANGSMITH_PROJECT"] = config.langsmith_project_name

    if config.langsmith_api_key:
        os.environ["LANGSMITH_API_KEY"] = config.langsmith_api_key
    if config.langsmith_endpoint:
        os.environ["LANGSMITH_ENDPOINT"] = config.langsmith_endpoint


def run_trace_extra(
    *,
    task_id: str,
    agent_mode: str,
    repo_dir: str,
    model_name: str,
    run_id: str | None = None,
    complexity: str | None = None,
    settings: dict[str, Any] | None = None,
) -> dict[str, Any]:
    tags = [f"task:{task_id}", f"agent_mode:{agent_mode}"]
    if complexity:
        tags.append(f"complexity:{complexity}")

    metadata: dict[str, Any] = {
        "task_id": task_id,
        "agent_mode": agent_mode,
        "repo_dir": repo_dir,
        "model": model_name,
    }
    if complexity:
        metadata["complexity"] = complexity
    if settings:
        metadata.update(settings)

    extra: dict[str, Any] = {
        # Name the top-level span after the actual agent so swe-agent / single /
        # multi runs are distinguishable at a glance in LangSmith (tags/metadata
        # also carry agent_mode for filtering).
        "name": f"{agent_mode}.run[{task_id}]",
        "tags": tags,
        "metadata": metadata,
    }
    if run_id is not None:
        extra["run_id"] = run_id
    return extra


def attach_run_feedback(
    run_id: str,
    key: str,
    score: float,
    comment: str = "",
) -> None:
    """Attach a post-run evaluation score to a trace.

    Meant for results that only exist after the agent finishes, e.g. the
    hidden test outcome: ``attach_run_feedback(run_id, "hidden_tests_passed",
    score=1.0)``. The ``run_id`` is the one recorded in runs.jsonl.
    """
    from langsmith import Client

    Client().create_feedback(run_id, key=key, score=score, comment=comment)
