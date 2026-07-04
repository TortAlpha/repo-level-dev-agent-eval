"""CLI: print the spec's evaluation metrics from recorded runs.

    python -m src.metrics.report                 # overall + per agent_mode
    python -m src.metrics.report --group-by size # per repository size bucket
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .compute import EXTERNAL_METRICS, MetricSet, compute_metrics, group_by
from .pricing import ModelPrice
from .records import (
    DEFAULT_COLLECTION_PATH,
    DEFAULT_QUALITY_PATH,
    DEFAULT_RUNS_PATH,
    attach_quality,
    attach_sizes,
    filter_runs,
    load_quality,
    load_runs,
    load_sizes,
)

# (field, label, kind) in spec order; kind picks the number format.
_ROWS: list[tuple[str, str, str]] = [
    ("resolved_at_1", "Resolved@1 (hidden, first patch)", "rate"),
    ("task_success_rate", "Task success rate (hidden)", "rate"),
    ("visible_test_pass_rate", "Visible test pass rate", "rate"),
    ("hidden_test_pass_rate", "Hidden test pass rate", "rate"),
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
        lines.append(
            row(label, [_fmt(getattr(m, field), kind) for _, m in columns])
        )
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


def main() -> int:
    args = parse_args()
    records = load_runs(args.runs)
    if not records:
        print(f"No runs found in {args.runs}.")
        return 1

    records = filter_runs(
        records, run_id=args.run_id, task_id=args.task_id, last=args.last
    )
    if not records:
        print("No runs match the given filter.")
        return 1

    attach_sizes(records, load_sizes(args.collection))
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
    note = _external_note()
    if note:
        print(note)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Report evaluation metrics.")
    parser.add_argument("--runs", type=Path, default=DEFAULT_RUNS_PATH)
    parser.add_argument("--collection", type=Path, default=DEFAULT_COLLECTION_PATH)
    parser.add_argument("--quality", type=Path, default=DEFAULT_QUALITY_PATH)
    parser.add_argument(
        "--group-by",
        choices=["agent_mode", "size", "model", "none"],
        default="agent_mode",
    )
    parser.add_argument(
        "--last",
        type=int,
        default=None,
        help="Only the N most recent runs (e.g. --last 1 for the last run).",
    )
    parser.add_argument("--run-id", default=None, help="Only this run_id.")
    parser.add_argument("--task-id", default=None, help="Only this task_id.")
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
