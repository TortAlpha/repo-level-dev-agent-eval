"""Compute the spec's evaluation metrics from run records.

Covers the metrics derivable from ``runs.jsonl`` (see docs/spec.md, section
"Evaluation"). Metrics that need a LangSmith trace (timing, tokens, tool-use
validity, hallucinated references) or a human/reviewer rubric (quality score)
are not computed here; ``EXTERNAL_METRICS`` documents where each comes from.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass

from .pricing import ModelPrice, estimate_cost
from .records import RunRecord

# Every spec metric is now derived from recorded data. quality_score is only
# populated after running the reviewer (python -m src.metrics.quality); until
# then it simply reads n/a, like any metric whose runs lack the field.
EXTERNAL_METRICS: dict[str, str] = {}

# Action-history kinds that count as (in)valid tool use for the validity rate.
_VALID_ACTION_KINDS = frozenset(
    {
        "set_plan",
        "inspect_file",
        "list_dir",
        "search",
        "search_reused",
        "write_file",
        "edit_file",
        "run_shell",
        "run_tests",
        "compatibility_check",
        "set_subtasks",
        "complete_subtask",
        "reopen_subtask",
        "finish",
        "handoff",
    }
)
_INVALID_ACTION_KINDS = frozenset(
    {
        "invalid_action",
        "malformed_action",
        "no_action",
        "repeated_action",
    }
)
_ACTION_FAILURE_KINDS = frozenset(
    {
        "invalid_action",
        "malformed_action",
        "no_action",
    }
)


@dataclass
class MetricSet:
    """Metrics for one group of runs. ``None`` means "no applicable runs"."""

    n_runs: int
    n_tasks: int
    # Outcome
    resolved_at_1: float | None
    task_success_rate: float | None
    visible_test_pass_rate: float | None
    hidden_test_pass_rate: float | None
    hidden_semantic_pass_rate: float | None
    hidden_compat_pass_rate: float | None
    hidden_pr_parity_pass_rate: float | None
    patch_validity_rate: float | None
    handoff_rate: float | None
    repair_success_rate: float | None
    regression_rate: float | None
    mean_regressions: float | None
    # Agent behavior
    test_overfitting_rate: float | None
    tool_use_validity_rate: float | None
    policy_rejections_per_run: float | None
    hallucinated_refs_per_run: float | None
    # Efficiency
    mean_iterations: float | None
    mean_steps: float | None
    mean_duration_s: float | None
    mean_llm_calls: float | None
    mean_total_tokens: float | None
    mean_cost_usd: float | None
    total_cost_usd: float | None
    cost_per_success_usd: float | None
    # Quality
    mean_quality_score: float | None

    def as_dict(self) -> dict:
        return asdict(self)


def _rate(flags: Sequence[bool]) -> float | None:
    return sum(1 for flag in flags if flag) / len(flags) if flags else None


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def compute_metrics(
    records: Sequence[RunRecord],
    *,
    price_override: ModelPrice | None = None,
) -> MetricSet:
    # New runs record task_success (visible + required hidden suites). Older
    # runs only have hidden_tests_passed, exposed via RunRecord.outcome.
    scored = [r for r in records if r.outcome is not None]
    hidden_scored = [r for r in records if r.required_hidden_passed is not None]
    visible_ok_scored = [r for r in scored if r.test_passed]
    # A run needing more than one test iteration started from a failing state;
    # finishing with green visible tests means it was repaired.
    retried = [r for r in records if r.iterations > 1]

    # resolved@1: the first run per task (earliest finish), by hidden verdict.
    first_run: dict[str, RunRecord] = {}
    for record in sorted(records, key=lambda r: r.finished_at or ""):
        first_run.setdefault(record.task_id, record)
    first_scored = [r.outcome for r in first_run.values() if r.outcome is not None]

    # Tool-use validity and hallucinations come from the action histograms.
    with_actions = [r for r in records if r.action_counts]
    valid_calls = sum(
        r.action_counts.get(k, 0) for r in with_actions for k in _VALID_ACTION_KINDS
    )
    invalid_calls = sum(
        r.action_counts.get(k, 0) for r in with_actions for k in _INVALID_ACTION_KINDS
    )
    total_calls = valid_calls + invalid_calls

    # Regression: runs with a before/after visible-test comparison recorded.
    regressed = [r for r in records if r.regressions is not None]

    # Cost: runs with token counts and a known (or overridden) model price.
    costs: list[float] = []
    for r in records:
        if r.effective_cost_usd is not None:
            costs.append(r.effective_cost_usd)
            continue
        if r.input_tokens is None or r.output_tokens is None:
            continue
        cost = estimate_cost(
            r.model, r.input_tokens, r.output_tokens, override=price_override
        )
        if cost is not None:
            costs.append(cost)

    solved_count = sum(1 for r in scored if r.outcome)
    total_cost = sum(costs) if costs else None

    return MetricSet(
        n_runs=len(records),
        n_tasks=len({r.task_id for r in records}),
        resolved_at_1=_rate(first_scored),
        task_success_rate=_rate([bool(r.outcome) for r in scored]),
        visible_test_pass_rate=_rate([r.test_passed for r in records]),
        hidden_test_pass_rate=_rate(
            [bool(r.required_hidden_passed) for r in hidden_scored]
        ),
        hidden_semantic_pass_rate=_rate(
            [
                r.hidden_semantic_tests_passed
                for r in records
                if r.hidden_semantic_tests_passed is not None
            ]
        ),
        hidden_compat_pass_rate=_rate(
            [
                r.hidden_compat_tests_passed
                for r in records
                if r.hidden_compat_tests_passed is not None
            ]
        ),
        hidden_pr_parity_pass_rate=_rate(
            [
                r.hidden_pr_parity_tests_passed
                for r in records
                if r.hidden_pr_parity_tests_passed is not None
            ]
        ),
        patch_validity_rate=_rate([r.changed_any for r in records]),
        handoff_rate=_rate([r.status == "handoff" for r in records]),
        repair_success_rate=_rate([r.test_passed for r in retried]),
        regression_rate=_rate([r.regressions > 0 for r in regressed]),
        mean_regressions=_mean([r.regressions for r in regressed]),
        test_overfitting_rate=_rate([not bool(r.outcome) for r in visible_ok_scored]),
        tool_use_validity_rate=(valid_calls / total_calls if total_calls else None),
        policy_rejections_per_run=_mean(
            [r.action_counts.get("policy_rejection", 0) for r in with_actions]
        ),
        hallucinated_refs_per_run=_mean(
            [
                sum(r.action_counts.get(k, 0) for k in _ACTION_FAILURE_KINDS)
                for r in with_actions
            ]
        ),
        mean_iterations=_mean([r.iterations for r in records]),
        mean_steps=_mean([r.steps for r in records]),
        mean_duration_s=_mean(
            [r.duration_s for r in records if r.duration_s is not None]
        ),
        mean_llm_calls=_mean([r.llm_calls for r in records if r.llm_calls is not None]),
        mean_total_tokens=_mean(
            [r.total_tokens for r in records if r.total_tokens is not None]
        ),
        mean_cost_usd=_mean(costs),
        total_cost_usd=total_cost,
        cost_per_success_usd=(
            total_cost / solved_count
            if total_cost is not None
            and solved_count > 0
            and len(costs) == len(records)
            else None
        ),
        mean_quality_score=_mean(
            [r.quality_score for r in records if r.quality_score is not None]
        ),
    )


def group_by(records: Sequence[RunRecord], key: str) -> dict[str, list[RunRecord]]:
    """Bucket records by an attribute (e.g. ``agent_mode`` or ``size``)."""
    groups: dict[str, list[RunRecord]] = {}
    for record in records:
        bucket = getattr(record, key, None) or "unknown"
        groups.setdefault(str(bucket), []).append(record)
    return dict(sorted(groups.items()))
