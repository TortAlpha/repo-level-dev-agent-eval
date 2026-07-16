"""LangSmith session wiring: env setup, run-level metadata, and feedback."""

from __future__ import annotations

import os
from typing import Any


def _short_model_name(value: Any) -> str:
    name = str(value or "unknown").rsplit("/", 1)[-1]
    return "".join(char if char.isalnum() or char in ".-" else "-" for char in name)


def model_policy_label(policy: dict[str, Any] | None) -> str | None:
    """Human-readable trace label; the technical type remains a filter tag."""
    if not policy:
        return None
    policy_type = str(policy.get("type") or "legacy")
    base = policy.get("base_model")
    roles = policy.get("roles") if isinstance(policy.get("roles"), dict) else {}
    developer = roles.get("developer") or base
    fallback = policy.get("developer_escalation_model")
    if policy_type == "single":
        return f"single_{_short_model_name(base)}"
    if policy_type == "shared":
        return f"shared_{_short_model_name(base)}"
    if policy_type == "adaptive":
        return (
            f"adaptive_dev_{_short_model_name(developer)}"
            f"_to_{_short_model_name(fallback)}"
        )
    if policy_type == "role_routed":
        return f"dev_{_short_model_name(developer)}"
    return policy_type


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
    policy = (settings or {}).get("model_policy")
    policy_type = policy.get("type") if isinstance(policy, dict) else None
    policy_label = model_policy_label(policy if isinstance(policy, dict) else None)
    if policy_type:
        tags.append(f"model_policy:{policy_type}")
    if policy_label:
        tags.append(f"model_policy_label:{policy_label}")
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
    if policy_label:
        metadata["model_policy_label"] = policy_label

    display_mode = f"{agent_mode}/{policy_label}" if policy_label else agent_mode
    extra: dict[str, Any] = {
        # Name the top-level span after the actual agent so swe-agent / single /
        # multi runs are distinguishable at a glance in LangSmith (tags/metadata
        # also carry agent_mode for filtering).
        "name": f"{display_mode}.run[{task_id}]",
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
