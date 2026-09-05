"""Generate the frozen benchmark figures from the scored final sessions."""

from __future__ import annotations

import csv
import json
import math
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
RUNS_PATH = ROOT / "experiments/results/runs.jsonl"
OUTPUT_DIR = Path(__file__).resolve().parent
FIGURES_DIR = OUTPUT_DIR / "figures"

ARCH_ORDER = ("single", "multi-graph", "multi-orch-guarded")
ARCH_LABELS = {
    "single": "Single",
    "multi-graph": "Multi-graph",
    "multi-orch-guarded": "Multi-orch-guarded",
}
ARCH_COLORS = {
    "single": "#0072B2",
    "multi-graph": "#E69F00",
    "multi-orch-guarded": "#009E73",
}


@dataclass(frozen=True)
class SessionSpec:
    session_id: str
    block: str
    agent_mode: str
    step_budget: int
    expected_runs: int


SESSION_SPECS = (
    SessionSpec("final_v1_core_small", "Core small", "single", 50, 12),
    SessionSpec("final_v1_core_small", "Core small", "multi-graph", 50, 12),
    SessionSpec(
        "final_v1_core_small",
        "Core small",
        "multi-orch-guarded",
        50,
        12,
    ),
    SessionSpec(
        "final_v1_core_medium_single50",
        "Core medium",
        "single",
        50,
        12,
    ),
    SessionSpec(
        "final_v1_core_medium_multi75",
        "Core medium",
        "multi-graph",
        75,
        12,
    ),
    SessionSpec(
        "final_v1_core_medium_multi75",
        "Core medium",
        "multi-orch-guarded",
        75,
        12,
    ),
    SessionSpec(
        "final_v1_swepro_single50",
        "SWE-bench Pro",
        "single",
        50,
        15,
    ),
    SessionSpec(
        "final_v1_swepro_graph75",
        "SWE-bench Pro",
        "multi-graph",
        75,
        15,
    ),
)


def load_rows() -> list[dict]:
    rows: list[dict] = []
    for line in RUNS_PATH.read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def effective_cost(row: dict) -> float:
    reported = row.get("provider_reported_cost_usd")
    reported_calls = row.get("provider_cost_calls")
    llm_calls = row.get("llm_calls")
    if (
        isinstance(reported, (int, float))
        and reported_calls
        and llm_calls
        and reported_calls >= llm_calls
    ):
        return float(reported)
    estimated = row.get("cost_usd")
    return float(estimated) if isinstance(estimated, (int, float)) else 0.0


def wilson_interval(successes: int, runs: int) -> tuple[float, float]:
    if not runs:
        return 0.0, 0.0
    z = 1.959963984540054
    rate = successes / runs
    denominator = 1 + z * z / runs
    center = (rate + z * z / (2 * runs)) / denominator
    half = z * math.sqrt(rate * (1 - rate) / runs + z * z / (4 * runs * runs))
    half /= denominator
    return max(0.0, center - half), min(1.0, center + half)


def summarize(all_rows: list[dict]) -> tuple[list[dict], dict[tuple[str, str], list[dict]]]:
    selected: dict[tuple[str, str], list[dict]] = {}
    summary: list[dict] = []
    for spec in SESSION_SPECS:
        rows = [
            row
            for row in all_rows
            if row.get("session_id") == spec.session_id
            and row.get("agent_mode") == spec.agent_mode
        ]
        keys = {(row.get("task_id"), row.get("model")) for row in rows}
        if len(rows) != spec.expected_runs or len(keys) != spec.expected_runs:
            raise RuntimeError(
                f"{spec.session_id}/{spec.agent_mode}: expected "
                f"{spec.expected_runs} unique runs, found {len(rows)} rows "
                f"and {len(keys)} unique task-model pairs"
            )
        successes = sum(bool(row.get("task_success")) for row in rows)
        low, high = wilson_interval(successes, len(rows))
        total_cost = sum(effective_cost(row) for row in rows)
        total_tokens = sum(int(row.get("total_tokens") or 0) for row in rows)
        total_steps = sum(int(row.get("steps") or 0) for row in rows)
        total_duration = sum(float(row.get("duration_s") or 0.0) for row in rows)
        summary.append(
            {
                "block": spec.block,
                "agent_mode": spec.agent_mode,
                "step_budget": spec.step_budget,
                "runs": len(rows),
                "successes": successes,
                "success_rate": successes / len(rows),
                "wilson_low": low,
                "wilson_high": high,
                "total_cost_usd": total_cost,
                "mean_cost_usd": total_cost / len(rows),
                "cost_per_success_usd": total_cost / successes if successes else None,
                "total_tokens": total_tokens,
                "mean_tokens": total_tokens / len(rows),
                "total_steps": total_steps,
                "mean_steps": total_steps / len(rows),
                "total_duration_s": total_duration,
                "mean_duration_s": total_duration / len(rows),
            }
        )
        selected[(spec.block, spec.agent_mode)] = rows
    return summary, selected


def configure_plotting():
    os.environ.setdefault("MPLCONFIGDIR", str(OUTPUT_DIR / ".matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#777777",
            "axes.labelcolor": "#222222",
            "axes.titleweight": "bold",
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "grid.color": "#DDDDDD",
            "grid.linewidth": 0.7,
            "text.color": "#222222",
            "xtick.color": "#444444",
            "ytick.color": "#444444",
        }
    )
    return plt


def save_figure(fig, stem: str) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / f"{stem}.png", dpi=220, bbox_inches="tight")
    fig.savefig(FIGURES_DIR / f"{stem}.svg", bbox_inches="tight")


def figure_success_rate(plt, summary: list[dict]) -> None:
    blocks = ("Core small", "Core medium", "SWE-bench Pro")
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2), sharey=True)
    for axis, block in zip(axes, blocks, strict=True):
        items = [row for row in summary if row["block"] == block]
        items.sort(key=lambda row: ARCH_ORDER.index(row["agent_mode"]))
        x = list(range(len(items)))
        rates = [row["success_rate"] * 100 for row in items]
        lower = [(row["success_rate"] - row["wilson_low"]) * 100 for row in items]
        upper = [(row["wilson_high"] - row["success_rate"]) * 100 for row in items]
        bars = axis.bar(
            x,
            rates,
            color=[ARCH_COLORS[row["agent_mode"]] for row in items],
            width=0.68,
            yerr=[lower, upper],
            capsize=4,
            error_kw={"elinewidth": 1.2, "ecolor": "#444444"},
        )
        for bar, row in zip(bars, items, strict=True):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                min(103, bar.get_height() + 2.2),
                f"{row['successes']}/{row['runs']}",
                ha="center",
                va="bottom",
                fontweight="bold",
            )
        axis.set_title(block)
        axis.set_xticks(
            x,
            [f"{ARCH_LABELS[row['agent_mode']]}\n{row['step_budget']} steps" for row in items],
        )
        axis.set_ylim(0, 112)
        axis.grid(axis="y")
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Task success rate (%)")
    fig.suptitle("Task success by evaluation block", fontsize=14, fontweight="bold")
    fig.text(
        0.5,
        -0.01,
        "Error bars: 95% Wilson intervals. Outcome is evaluator task_success.",
        ha="center",
        color="#555555",
    )
    fig.tight_layout()
    save_figure(fig, "01-task-success-rate")
    plt.close(fig)


def figure_cost_efficiency(plt, summary: list[dict]) -> None:
    blocks = ("Core small", "Core medium", "SWE-bench Pro")
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8))
    metrics = (
        ("total_cost_usd", "Total model cost (USD)"),
        ("cost_per_success_usd", "Cost per successful task (USD)"),
    )
    width = 0.23
    for axis, (metric, ylabel) in zip(axes, metrics, strict=True):
        for offset_index, agent in enumerate(ARCH_ORDER):
            offset = (offset_index - 1) * width
            xs: list[float] = []
            values: list[float] = []
            for block_index, block in enumerate(blocks):
                row = next(
                    (
                        item
                        for item in summary
                        if item["block"] == block and item["agent_mode"] == agent
                    ),
                    None,
                )
                if row is None or row[metric] is None:
                    continue
                xs.append(block_index + offset)
                values.append(float(row[metric]))
            bars = axis.bar(
                xs,
                values,
                width=width,
                color=ARCH_COLORS[agent],
                label=ARCH_LABELS[agent],
            )
            for bar, value in zip(bars, values, strict=True):
                axis.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(values + [1]) * 0.025,
                    f"${value:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    rotation=90 if value >= 1 else 0,
                )
        axis.set_xticks(range(len(blocks)), blocks)
        axis.set_ylabel(ylabel)
        axis.grid(axis="y")
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, loc="upper left")
    fig.suptitle("Cost and cost efficiency", fontsize=14, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, "02-cost-efficiency")
    plt.close(fig)


def figure_resource_use(plt, summary: list[dict]) -> None:
    blocks = ("Core small", "Core medium", "SWE-bench Pro")
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8))
    metrics = (
        ("mean_steps", "Mean agent steps per task", lambda value: f"{value:.1f}"),
        ("mean_tokens", "Mean tokens per task", lambda value: f"{value / 1000:.0f}k"),
    )
    width = 0.23
    for axis, (metric, ylabel, formatter) in zip(axes, metrics, strict=True):
        for offset_index, agent in enumerate(ARCH_ORDER):
            offset = (offset_index - 1) * width
            xs: list[float] = []
            values: list[float] = []
            for block_index, block in enumerate(blocks):
                row = next(
                    (
                        item
                        for item in summary
                        if item["block"] == block and item["agent_mode"] == agent
                    ),
                    None,
                )
                if row is None:
                    continue
                xs.append(block_index + offset)
                values.append(float(row[metric]))
            bars = axis.bar(
                xs,
                values,
                width=width,
                color=ARCH_COLORS[agent],
                label=ARCH_LABELS[agent],
            )
            for bar, value in zip(bars, values, strict=True):
                axis.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(values + [1]) * 0.025,
                    formatter(value),
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )
        axis.set_xticks(range(len(blocks)), blocks)
        axis.set_ylabel(ylabel)
        axis.grid(axis="y")
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, loc="upper left")
    fig.suptitle("Resource use", fontsize=14, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, "03-resource-use")
    plt.close(fig)


def task_short_label(task_id: str) -> str:
    family = "A" if "_ansible_" in task_id else "O" if "_openlibrary_" in task_id else "Q"
    return f"{family}:{task_id.rsplit('_', 1)[-1]}"


def figure_swe_outcomes(plt, selected: dict[tuple[str, str], list[dict]]) -> None:
    single_rows = selected[("SWE-bench Pro", "single")]
    graph_rows = selected[("SWE-bench Pro", "multi-graph")]
    single = {row["task_id"]: row for row in single_rows}
    graph = {row["task_id"]: row for row in graph_rows}
    task_ids = [row["task_id"] for row in single_rows]
    outcomes = [
        [int(bool(single[task].get("task_success"))) for task in task_ids],
        [int(bool(graph[task].get("task_success"))) for task in task_ids],
    ]
    cost_delta = [effective_cost(graph[task]) - effective_cost(single[task]) for task in task_ids]

    from matplotlib.colors import ListedColormap

    fig, (outcome_axis, cost_axis) = plt.subplots(
        2,
        1,
        figsize=(14.5, 6.4),
        gridspec_kw={"height_ratios": [1.0, 1.55], "hspace": 0.38},
    )
    outcome_axis.imshow(
        outcomes,
        aspect="auto",
        cmap=ListedColormap(["#D9D9D9", "#009E73"]),
        vmin=0,
        vmax=1,
    )
    for row_index, values in enumerate(outcomes):
        for column_index, value in enumerate(values):
            outcome_axis.text(
                column_index,
                row_index,
                "PASS" if value else "FAIL",
                ha="center",
                va="center",
                fontsize=7,
                color="white" if value else "#333333",
                fontweight="bold",
            )
    outcome_axis.set_yticks([0, 1], ["Single · 50 steps", "Multi-graph · 75 steps"])
    outcome_axis.set_xticks(range(len(task_ids)))
    outcome_axis.tick_params(axis="x", bottom=False, labelbottom=False)
    outcome_axis.set_title("SWE-bench Pro paired task outcomes (15/15 agreement)")
    outcome_axis.spines[:].set_visible(False)
    for boundary in (4.5, 9.5):
        outcome_axis.axvline(boundary, color="white", linewidth=3)

    colors = ["#E69F00" if value >= 0 else "#0072B2" for value in cost_delta]
    bars = cost_axis.bar(range(len(task_ids)), cost_delta, color=colors, width=0.72)
    cost_axis.axhline(0, color="#555555", linewidth=1)
    for bar, value in zip(bars, cost_delta, strict=True):
        cost_axis.text(
            bar.get_x() + bar.get_width() / 2,
            value + (0.008 if value >= 0 else -0.008),
            f"{value:+.2f}",
            ha="center",
            va="bottom" if value >= 0 else "top",
            fontsize=7,
        )
    cost_axis.set_xticks(range(len(task_ids)), [task_short_label(task) for task in task_ids])
    cost_axis.tick_params(axis="x", rotation=55)
    cost_axis.set_ylabel("Graph − single cost (USD)")
    cost_axis.set_title("Per-task cost difference")
    cost_axis.grid(axis="y")
    cost_axis.set_axisbelow(True)
    cost_axis.spines[["top", "right"]].set_visible(False)
    fig.text(
        0.5,
        0.015,
        "A = Ansible, O = OpenLibrary, Q = qutebrowser. Positive values mean multi-graph cost more.",
        ha="center",
        color="#555555",
    )
    fig.subplots_adjust(left=0.13, right=0.99, top=0.91, bottom=0.22)
    save_figure(fig, "04-swe-task-outcomes")
    plt.close(fig)


def summarize_task_types(
    selected: dict[tuple[str, str], list[dict]],
) -> list[dict]:
    all_selected = [row for rows in selected.values() for row in rows]
    summary: list[dict] = []
    for task_type in ("bugfix", "feature"):
        for agent in ("single", "multi-graph"):
            rows = [
                row
                for row in all_selected
                if row.get("task_type") == task_type
                and row.get("agent_mode") == agent
            ]
            successes = sum(bool(row.get("task_success")) for row in rows)
            low, high = wilson_interval(successes, len(rows))
            total_cost = sum(effective_cost(row) for row in rows)
            summary.append(
                {
                    "task_type": task_type,
                    "agent_mode": agent,
                    "runs": len(rows),
                    "successes": successes,
                    "success_rate": successes / len(rows),
                    "wilson_low": low,
                    "wilson_high": high,
                    "total_cost_usd": total_cost,
                    "cost_per_success_usd": total_cost / successes if successes else None,
                }
            )
    return summary


def figure_task_types(plt, task_types: list[dict]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.4))
    for column, task_type in enumerate(("bugfix", "feature")):
        items = [row for row in task_types if row["task_type"] == task_type]
        items.sort(key=lambda row: ARCH_ORDER.index(row["agent_mode"]))
        x = range(len(items))
        rates = [row["success_rate"] * 100 for row in items]
        lower = [(row["success_rate"] - row["wilson_low"]) * 100 for row in items]
        upper = [(row["wilson_high"] - row["success_rate"]) * 100 for row in items]
        rate_axis = axes[0, column]
        bars = rate_axis.bar(
            x,
            rates,
            color=[ARCH_COLORS[row["agent_mode"]] for row in items],
            width=0.62,
            yerr=[lower, upper],
            capsize=5,
            error_kw={"elinewidth": 1.2, "ecolor": "#444444"},
        )
        for bar, row in zip(bars, items, strict=True):
            rate_axis.text(
                bar.get_x() + bar.get_width() / 2,
                min(104, bar.get_height() + 2.5),
                f"{row['successes']}/{row['runs']}",
                ha="center",
                va="bottom",
                fontweight="bold",
            )
        title = "Bug fixes · 18 core tasks" if task_type == "bugfix" else "Features · 21 core + SWE Pro tasks"
        rate_axis.set_title(title)
        rate_axis.set_xticks(x, [ARCH_LABELS[row["agent_mode"]] for row in items])
        rate_axis.set_ylim(0, 112)
        rate_axis.set_ylabel("Task success rate (%)" if column == 0 else "")
        rate_axis.grid(axis="y")
        rate_axis.set_axisbelow(True)
        rate_axis.spines[["top", "right"]].set_visible(False)

        cost_axis = axes[1, column]
        costs = [float(row["cost_per_success_usd"] or 0.0) for row in items]
        cost_bars = cost_axis.bar(
            x,
            costs,
            color=[ARCH_COLORS[row["agent_mode"]] for row in items],
            width=0.62,
        )
        for bar, value in zip(cost_bars, costs, strict=True):
            cost_axis.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(costs) * 0.035,
                f"${value:.3f}",
                ha="center",
                va="bottom",
                fontweight="bold",
            )
        cost_axis.set_xticks(x, [ARCH_LABELS[row["agent_mode"]] for row in items])
        cost_axis.set_ylabel("Cost per success (USD)" if column == 0 else "")
        cost_axis.grid(axis="y")
        cost_axis.set_axisbelow(True)
        cost_axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Outcome and cost efficiency by task type", fontsize=14, fontweight="bold")
    fig.text(
        0.5,
        0.01,
        "Guarded is excluded because SWE-bench Pro was evaluated only with single and multi-graph.",
        ha="center",
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    save_figure(fig, "05-task-type-split")
    plt.close(fig)


def write_summary_csv(summary: list[dict]) -> None:
    path = OUTPUT_DIR / "summary.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)


def write_task_type_csv(task_types: list[dict]) -> None:
    path = OUTPUT_DIR / "task_type_summary.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(task_types[0]))
        writer.writeheader()
        writer.writerows(task_types)


def write_report(summary: list[dict], task_types: list[dict]) -> None:
    lookup = {(row["block"], row["agent_mode"]): row for row in summary}
    type_lookup = {
        (row["task_type"], row["agent_mode"]): row for row in task_types
    }
    swe_single = lookup[("SWE-bench Pro", "single")]
    swe_graph = lookup[("SWE-bench Pro", "multi-graph")]
    report = f"""# Final Benchmark Figures

Generated from the frozen scored sessions in `experiments/results/runs.jsonl`.
The primary outcome is evaluator `task_success`.

## Main result

- Core small: all three architectures resolved 11/12 tasks. Single cost
  ${lookup[("Core small", "single")]["total_cost_usd"]:.4f}; multi-graph cost
  ${lookup[("Core small", "multi-graph")]["total_cost_usd"]:.4f}; guarded cost
  ${lookup[("Core small", "multi-orch-guarded")]["total_cost_usd"]:.4f}.
- Core medium: single resolved 11/12, while both multi variants resolved 10/12.
- SWE-bench Pro: single and multi-graph both resolved {swe_single["successes"]}/15,
  with identical per-task outcomes. Single cost ${swe_single["total_cost_usd"]:.4f};
  multi-graph cost ${swe_graph["total_cost_usd"]:.4f}.
- Bug fixes: single resolved {type_lookup[("bugfix", "single")]["successes"]}/18;
  multi-graph resolved {type_lookup[("bugfix", "multi-graph")]["successes"]}/18.
- Features: single resolved {type_lookup[("feature", "single")]["successes"]}/21;
  multi-graph resolved {type_lookup[("feature", "multi-graph")]["successes"]}/21.
  Their feature cost per success was nearly identical at
  ${type_lookup[("feature", "single")]["cost_per_success_usd"]:.3f} and
  ${type_lookup[("feature", "multi-graph")]["cost_per_success_usd"]:.3f}.
- The evaluated multi-agent configurations did not improve aggregate resolve
  rate or cost efficiency over the strengthened single loop.

## Figures

1. `figures/01-task-success-rate` — resolve rate with 95% Wilson intervals.
2. `figures/02-cost-efficiency` — total cost and cost per successful task.
3. `figures/03-resource-use` — mean agent steps and tokens per task.
4. `figures/04-swe-task-outcomes` — paired SWE Pro outcomes and cost deltas.
5. `figures/05-task-type-split` — bug-fix versus feature outcomes and cost efficiency.

Each figure is available as PNG and SVG. Exact plotted values are in
`summary.csv` and `task_type_summary.csv`.

## Interpretation constraint

Core small used a common 50-step cap. Core medium and SWE-bench Pro used 50
steps for single and 75 for multi architectures. Blocks must be reported
separately; a pooled 39-task aggregate would mix different architecture
coverage and step policies.
"""
    (OUTPUT_DIR / "README.md").write_text(report, encoding="utf-8")


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    summary, selected = summarize(load_rows())
    task_types = summarize_task_types(selected)
    write_summary_csv(summary)
    write_task_type_csv(task_types)
    write_report(summary, task_types)
    plt = configure_plotting()
    figure_success_rate(plt, summary)
    figure_cost_efficiency(plt, summary)
    figure_resource_use(plt, summary)
    figure_swe_outcomes(plt, selected)
    figure_task_types(plt, task_types)
    print(f"Wrote {len(summary)} summary rows and 5 figures to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
