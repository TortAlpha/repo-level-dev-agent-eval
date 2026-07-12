"""CLI: print the spec's evaluation metrics from recorded runs.

python -m src.metrics.report                 # overall + per agent_mode
python -m src.metrics.report --group-by size # per repository size bucket
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .compute import EXTERNAL_METRICS, MetricSet, compute_metrics, group_by
from .difficulty import MIN_RUNS, attach_difficulty, empirical_difficulty
from .pricing import ModelPrice
from .records import (
    DEFAULT_COLLECTION_PATH,
    DEFAULT_QUALITY_PATH,
    DEFAULT_RUNS_PATH,
    RunRecord,
    attach_hidden_suite_fallbacks,
    attach_quality,
    attach_sizes,
    attach_task_types,
    filter_runs,
    load_hidden_suite_specs,
    load_quality,
    load_runs,
    load_sizes,
    load_task_types,
)

# (field, label, kind) in spec order; kind picks the number format.
_ROWS: list[tuple[str, str, str]] = [
    ("resolved_at_1", "Resolved@1 (task success)", "rate"),
    ("task_success_rate", "Task success rate", "rate"),
    ("visible_test_pass_rate", "Visible test pass rate", "rate"),
    ("hidden_test_pass_rate", "Required hidden pass rate", "rate"),
    ("hidden_semantic_pass_rate", "Hidden semantic pass rate", "rate"),
    ("hidden_compat_pass_rate", "Hidden compat pass rate", "rate"),
    ("hidden_pr_parity_pass_rate", "Hidden PR parity pass rate", "rate"),
    ("patch_validity_rate", "Patch validity rate", "rate"),
    ("handoff_rate", "Human handoff rate", "rate"),
    ("repair_success_rate", "Repair success rate", "rate"),
    ("regression_rate", "Regression rate (runs w/ any)", "rate"),
    ("mean_regressions", "Regressions (mean/run)", "mean"),
    ("test_overfitting_rate", "Test-overfitting rate", "rate"),
    ("tool_use_validity_rate", "Tool-use validity rate", "rate"),
    ("hallucinated_refs_per_run", "Hallucinated refs (mean/run)", "mean"),
    ("mean_iterations", "Mean iterations", "mean"),
    ("mean_steps", "Mean steps", "mean"),
    ("mean_duration_s", "Mean duration (s)", "mean"),
    ("mean_llm_calls", "Mean LLM calls", "mean"),
    ("mean_total_tokens", "Mean total tokens", "mean"),
    ("mean_cost_usd", "Mean cost (USD)", "usd"),
    ("total_cost_usd", "Total cost (USD)", "usd"),
    ("cost_per_success_usd", "Cost per successful run", "usd"),
    ("mean_quality_score", "Mean quality (0-5)", "mean"),
]

_LABEL_W = 34
_COL_W = 12


def _fmt(value: float | None, kind: str) -> str:
    if value is None:
        return "n/a"
    if kind == "rate":
        return f"{value * 100:.1f}%"
    if kind == "usd":
        return f"${value:.4f}"
    return f"{value:.2f}"


def render(columns: list[tuple[str, MetricSet]]) -> str:
    # Long names (e.g. model ids) collide when truncated, so tag them and
    # spell them out in a legend below the table.
    headers: list[str] = []
    legend: list[str] = []
    for index, (name, _) in enumerate(columns):
        if len(name) <= _COL_W:
            headers.append(name)
        else:
            tag = f"#{index}"
            headers.append(tag)
            legend.append(f"  {tag} = {name}")

    def row(label: str, cells: list[str]) -> str:
        return label.ljust(_LABEL_W) + "".join(c.rjust(_COL_W) for c in cells)

    lines = [
        row("metric", headers),
        "-" * (_LABEL_W + _COL_W * len(columns)),
        row("runs / tasks", [f"{m.n_runs}/{m.n_tasks}" for _, m in columns]),
        "",
    ]
    for field, label, kind in _ROWS:
        lines.append(row(label, [_fmt(getattr(m, field), kind) for _, m in columns]))
    if legend:
        lines.extend(["", *legend])
    return "\n".join(lines)


def _external_note() -> str:
    if not EXTERNAL_METRICS:
        return ""
    lines = ["", "Not derivable from runs.jsonl (source):"]
    for name, source in EXTERNAL_METRICS.items():
        lines.append(f"  - {name}: {source}")
    return "\n".join(lines)


def _print_sessions(records: list) -> None:
    """List sessions with run counts, distinct tasks/agents, and last run time."""
    from collections import defaultdict

    agg: dict[str, dict] = defaultdict(
        lambda: {"n": 0, "last": "", "tasks": set(), "agents": set()}
    )
    for r in records:
        s = agg[r.session_id or "default"]
        s["n"] += 1
        s["last"] = max(s["last"], r.finished_at or "")
        s["tasks"].add(r.task_id)
        s["agents"].add(r.agent_mode)
    print(f"{'session':<28}{'runs':>6}{'tasks':>7}  {'agents':<20}last")
    print("-" * 78)
    for name in sorted(agg, key=lambda k: agg[k]["last"], reverse=True):
        s = agg[name]
        agents = ",".join(sorted(s["agents"]))
        print(
            f"{name:<28}{s['n']:>6}{len(s['tasks']):>7}  {agents:<20}{s['last'][:19]}"
        )


def main() -> int:
    args = parse_args()
    records = load_runs(args.runs)
    if not records:
        print(f"No runs found in {args.runs}.")
        return 1

    if args.list_sessions:
        _print_sessions(records)
        return 0

    records = filter_runs(
        records,
        run_id=args.run_id,
        task_id=args.task_id,
        session_id=args.session,
        last=args.last,
    )
    if not records:
        print("No runs match the given filter.")
        return 1

    attach_sizes(records, load_sizes(args.collection))
    attach_task_types(records, load_task_types(args.collection))
    attach_hidden_suite_fallbacks(records, load_hidden_suite_specs(args.collection))
    attach_difficulty(records, args.collection)
    attach_quality(records, load_quality(args.quality))

    override = None
    if args.price_in is not None and args.price_out is not None:
        override = ModelPrice(args.price_in, args.price_out)

    columns: list[tuple[str, MetricSet]] = [
        ("overall", compute_metrics(records, price_override=override))
    ]
    if args.group_by != "none":
        for name, group in group_by(records, args.group_by).items():
            columns.append((name, compute_metrics(group, price_override=override)))

    print(render(columns))
    if args.difficulty:
        print(_difficulty_table(records))
    note = _external_note()
    if note:
        print(note)
    return 0


def _difficulty_table(records: list[RunRecord]) -> str:
    """Per-task empirical difficulty, hardest first, with flags for tasks that
    carry little signal (everyone solves, everyone fails, or models agree)."""
    measured = sorted(
        empirical_difficulty(records).values(),
        key=lambda d: (-d.difficulty, -(d.discrimination or 0)),
    )
    lines = [
        "",
        "Task difficulty (measured from runs):",
        f"  {'task':30} {'runs':>4} {'models':>6} {'solve':>6} {'diff':>6} {'discrim':>7}  flags",
    ]
    for d in measured:
        flags = []
        if d.n_runs < MIN_RUNS:
            flags.append("low-n")
        elif d.solve_rate == 1.0:
            flags.append("too-easy")
        elif d.solve_rate == 0.0:
            flags.append("unsolved")
        if d.n_models >= 2 and (d.discrimination or 0) == 0:
            flags.append("no-discrim")
        disc = "n/a" if d.discrimination is None else f"{d.discrimination:.2f}"
        lines.append(
            f"  {d.task_id:30} {d.n_runs:>4} {d.n_models:>6} "
            f"{d.solve_rate:>6.2f} {d.difficulty:>6.2f} {disc:>7}  {' '.join(flags)}"
        )
    if not measured:
        lines.append("  (no runs with a known outcome yet)")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Report evaluation metrics.")
    parser.add_argument("--runs", type=Path, default=DEFAULT_RUNS_PATH)
    parser.add_argument("--collection", type=Path, default=DEFAULT_COLLECTION_PATH)
    parser.add_argument("--quality", type=Path, default=DEFAULT_QUALITY_PATH)
    parser.add_argument(
        "--group-by",
        choices=[
            "agent_mode",
            "size",
            "task_type",
            "difficulty",
            "difficulty_estimate",
            "model",
            "session_id",
            "none",
            "action_transport",
        ],
        default="agent_mode",
    )
    parser.add_argument(
        "--difficulty",
        action="store_true",
        help="Also print per-task measured difficulty (solve rate, discrimination).",
    )
    parser.add_argument(
        "--last",
        type=int,
        default=None,
        help="Only the N most recent runs (e.g. --last 1 for the last run).",
    )
    parser.add_argument("--run-id", default=None, help="Only this run_id.")
    parser.add_argument("--task-id", default=None, help="Only this task_id.")
    parser.add_argument("--session", default=None, help="Only runs from this session.")
    parser.add_argument(
        "--list-sessions",
        action="store_true",
        help="List recorded sessions (run counts, agents, last time) and exit.",
    )
    parser.add_argument(
        "--price-in",
        type=float,
        default=None,
        help="Override input price (USD/1M tokens) for all runs.",
    )
    parser.add_argument(
        "--price-out",
        type=float,
        default=None,
        help="Override output price (USD/1M tokens) for all runs.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
