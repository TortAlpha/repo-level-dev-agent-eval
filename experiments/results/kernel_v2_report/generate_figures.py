"""Figures and tables for the kernel v2 core-small re-measurement.

Mirrors the style of ``experiments/results/final_report/generate_figures.py``
but reads the kernel v2 sessions. The historical ``final_v1`` report is left
untouched; ``final_v1_core_small`` appears here only as the old-harness
baseline it is being compared against.

Run with matplotlib available, for example::

    PYTHONPATH=/path/to/matplotlib .venv/bin/python \\
        experiments/results/kernel_v2_report/generate_figures.py
"""

from __future__ import annotations

import csv
import json
import math
import os
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

PASS_COLOR = "#2C7355"
FAIL_COLOR = "#A8402A"
ABSENT_COLOR = "#DDDDDD"

STEP_CAP = 50


@dataclass(frozen=True)
class RunSet:
    """One comparable column: a model on a harness under one protocol."""

    key: str
    label: str
    short: str
    model: str
    harness: str
    sessions: tuple[str, ...]
    expected_per_arch: int


RUN_SETS = (
    RunSet(
        key="glm_old",
        label="glm-5.2 · harness 6c0af80",
        short="glm\nold harness",
        model="z-ai/glm-5.2",
        harness="6c0af80",
        sessions=("final_v1_core_small",),
        expected_per_arch=12,
    ),
    RunSet(
        key="glm_new",
        label="glm-5.2 · harness fd00518",
        short="glm\nnew harness",
        model="z-ai/glm-5.2",
        harness="fd00518",
        sessions=("kernel_v2_glm_5_2_core_small_control",),
        expected_per_arch=12,
    ),
    RunSet(
        key="deepseek",
        label="DeepSeek V4 Flash · harness fd00518",
        short="DeepSeek\nnew harness",
        model="deepseek/deepseek-v4-flash-0731:nitro",
        harness="fd00518",
        sessions=(
            "kernel_v2_ds_v4_flash_0731_nitro_low_core_small_v4",
            "kernel_v2_ds_v4_flash_0731_nitro_low_core_small_v4_tenacity_pin",
        ),
        expected_per_arch=12,
    ),
)

# Probes are deliberately outside the comparable block: a different step cap,
# or only a subset of tasks.
PROBE_SESSIONS = {
    "glm_steps100": "kernel_v2_glm_5_2_core_small_steps100",
    "deepseek_steps100": "kernel_v2_ds_v4_flash_0731_nitro_low_parse165_steps100",
    "terra_pilot": "kernel_v2_terra_high_pilot",
    "terra_parse165": "kernel_v2_terra_high_parse165",
}

TASK_ORDER = (
    "h11_pr_181",
    "humanize_pr_329",
    "pluggy_pr_646",
    "w3lib_pr_272",
    "parse_pr_165",
    "parse_pr_227",
    "cachetools_pr_57d2e48",
    "tinydb_pr_616",
    "python_dotenv_pr_640",
    "tenacity_pr_628",
    "freezegun_pr_546",
    "croniter_pr_235",
)


def load_rows() -> list[dict]:
    rows: list[dict] = []
    with RUNS_PATH.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def effective_cost(row: dict) -> float:
    """Provider-reported billing when complete, else the static estimate.

    Identical rule to the final_v1 report, so the two are comparable.
    """
    recorded = row.get("effective_cost_usd")
    if isinstance(recorded, (int, float)):
        return float(recorded)
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


def collect(all_rows: list[dict]) -> tuple[list[dict], dict]:
    """Per run-set and architecture aggregates, plus the raw selection."""
    summary: list[dict] = []
    selected: dict[tuple[str, str], list[dict]] = {}
    for run_set in RUN_SETS:
        for arch in ARCH_ORDER:
            rows = [
                row
                for row in all_rows
                if row.get("session_id") in run_set.sessions
                and row.get("agent_mode") == arch
            ]
            tasks = {row.get("task_id") for row in rows}
            if len(rows) != run_set.expected_per_arch or len(tasks) != len(rows):
                raise RuntimeError(
                    f"{run_set.key}/{arch}: expected {run_set.expected_per_arch} "
                    f"unique task rows, found {len(rows)} rows over {len(tasks)} tasks"
                )
            successes = sum(bool(row.get("task_success")) for row in rows)
            low, high = wilson_interval(successes, len(rows))
            cost = sum(effective_cost(row) for row in rows)
            tokens = sum(int(row.get("total_tokens") or 0) for row in rows)
            steps = sum(int(row.get("steps") or 0) for row in rows)
            duration = sum(float(row.get("duration_s") or 0.0) for row in rows)
            capped = sum(1 for row in rows if int(row.get("steps") or 0) >= STEP_CAP)
            summary.append(
                {
                    "run_set": run_set.key,
                    "label": run_set.label,
                    "model": run_set.model,
                    "harness": run_set.harness,
                    "agent_mode": arch,
                    "runs": len(rows),
                    "successes": successes,
                    "success_rate": successes / len(rows),
                    "wilson_low": low,
                    "wilson_high": high,
                    "total_cost_usd": cost,
                    "mean_cost_usd": cost / len(rows),
                    "cost_per_success_usd": cost / successes if successes else None,
                    "total_tokens": tokens,
                    "mean_tokens": tokens / len(rows),
                    "mean_steps": steps / len(rows),
                    "runs_at_step_cap": capped,
                    "total_duration_s": duration,
                }
            )
            selected[(run_set.key, arch)] = rows
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


def pick(summary: list[dict], run_set: str, arch: str) -> dict:
    for item in summary:
        if item["run_set"] == run_set and item["agent_mode"] == arch:
            return item
    raise KeyError(f"{run_set}/{arch}")


def figure_harness_effect(plt, summary: list[dict]) -> None:
    """Success rate by architecture for the three comparable columns."""
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.4), sharey=True)
    for axis, run_set in zip(axes, RUN_SETS):
        positions = range(len(ARCH_ORDER))
        rates, lows, highs = [], [], []
        for arch in ARCH_ORDER:
            item = pick(summary, run_set.key, arch)
            rates.append(item["success_rate"] * 100)
            lows.append((item["success_rate"] - item["wilson_low"]) * 100)
            highs.append((item["wilson_high"] - item["success_rate"]) * 100)
        axis.bar(
            positions,
            rates,
            color=[ARCH_COLORS[a] for a in ARCH_ORDER],
            width=0.62,
            yerr=[lows, highs],
            capsize=4,
            ecolor="#555555",
        )
        for x, arch in zip(positions, ARCH_ORDER):
            item = pick(summary, run_set.key, arch)
            # Sit above the upper Wilson whisker, not on top of it.
            axis.text(
                x,
                item["wilson_high"] * 100 + 2.5,
                f"{item['successes']}/{item['runs']}",
                ha="center",
                fontsize=9,
                fontweight="bold",
            )
        total_ok = sum(pick(summary, run_set.key, a)["successes"] for a in ARCH_ORDER)
        total_runs = sum(pick(summary, run_set.key, a)["runs"] for a in ARCH_ORDER)
        total_cost = sum(
            pick(summary, run_set.key, a)["total_cost_usd"] for a in ARCH_ORDER
        )
        axis.set_title(
            f"{run_set.label}\n{total_ok}/{total_runs} · ${total_cost:.2f}",
            fontsize=10,
        )
        axis.set_xticks(list(positions))
        axis.set_xticklabels([ARCH_LABELS[a] for a in ARCH_ORDER], rotation=18, ha="right")
        axis.set_ylim(0, 118)
        axis.grid(axis="y", alpha=0.6)
        axis.set_axisbelow(True)
    axes[0].set_ylabel("task_success rate, %")
    fig.suptitle(
        "Core-small: the same model loses four combinations on the new harness",
        fontsize=12,
        fontweight="bold",
    )
    fig.text(
        0.5,
        -0.04,
        "Error bars are 95% Wilson intervals at n=12 per bar. Only the two right "
        "panels share a harness and may be compared directly.",
        ha="center",
        fontsize=8.5,
        color="#555555",
    )
    fig.tight_layout()
    save_figure(fig, "01-harness-effect")
    plt.close(fig)


def figure_cost(plt, summary: list[dict]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.6))
    width = 0.26
    for panel, (metric, title, ylabel) in enumerate(
        (
            ("mean_cost_usd", "Cost per run", "USD per run"),
            ("cost_per_success_usd", "Cost per resolved task", "USD per success"),
        )
    ):
        axis = axes[panel]
        for index, arch in enumerate(ARCH_ORDER):
            values = [pick(summary, rs.key, arch)[metric] or 0.0 for rs in RUN_SETS]
            offsets = [i + (index - 1) * width for i in range(len(RUN_SETS))]
            bars = axis.bar(
                offsets,
                values,
                width=width,
                color=ARCH_COLORS[arch],
                label=ARCH_LABELS[arch],
            )
            for rect, value in zip(bars, values):
                axis.text(
                    rect.get_x() + rect.get_width() / 2,
                    value,
                    f"{value:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=7.5,
                )
        axis.set_xticks(range(len(RUN_SETS)))
        axis.set_xticklabels([rs.short for rs in RUN_SETS], fontsize=9)
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", alpha=0.6)
        axis.set_axisbelow(True)
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle(
        "Actual provider billing, not the static price-table estimate",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout()
    save_figure(fig, "02-cost")
    plt.close(fig)


def figure_step_pressure(plt, summary: list[dict]) -> None:
    """Why the new harness loses tasks: it spends more steps."""
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.6))
    width = 0.26
    for index, arch in enumerate(ARCH_ORDER):
        offsets = [i + (index - 1) * width for i in range(len(RUN_SETS))]
        means = [pick(summary, rs.key, arch)["mean_steps"] for rs in RUN_SETS]
        capped = [pick(summary, rs.key, arch)["runs_at_step_cap"] for rs in RUN_SETS]
        axes[0].bar(
            offsets, means, width=width, color=ARCH_COLORS[arch], label=ARCH_LABELS[arch]
        )
        axes[1].bar(offsets, capped, width=width, color=ARCH_COLORS[arch])
        for x, value in zip(offsets, means):
            axes[0].text(x, value, f"{value:.0f}", ha="center", va="bottom", fontsize=7.5)
        for x, value in zip(offsets, capped):
            axes[1].text(x, value, f"{value}", ha="center", va="bottom", fontsize=7.5)
    axes[0].axhline(STEP_CAP, color=FAIL_COLOR, linestyle="--", linewidth=1)
    axes[0].text(
        len(RUN_SETS) - 0.5,
        STEP_CAP + 1,
        f"{STEP_CAP}-step cap",
        color=FAIL_COLOR,
        fontsize=8,
        ha="right",
    )
    axes[0].set_title("Mean steps per run")
    axes[0].set_ylabel("steps")
    axes[1].set_title(f"Runs reaching the {STEP_CAP}-step cap")
    axes[1].set_ylabel("runs out of 12")
    for axis in axes:
        axis.set_xticks(range(len(RUN_SETS)))
        axis.set_xticklabels([rs.short for rs in RUN_SETS], fontsize=9)
        axis.grid(axis="y", alpha=0.6)
        axis.set_axisbelow(True)
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle(
        "Step consumption rises on the new harness and meets a fixed cap",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout()
    save_figure(fig, "03-step-pressure")
    plt.close(fig)


def figure_task_matrix(plt, selected: dict) -> None:
    """Task × (run set, architecture) outcome grid."""
    columns: list[tuple[str, str]] = [
        (rs.key, arch) for rs in RUN_SETS for arch in ARCH_ORDER
    ]
    grid = []
    for task in TASK_ORDER:
        row = []
        for key, arch in columns:
            match = [r for r in selected[(key, arch)] if r.get("task_id") == task]
            row.append(1.0 if match and match[0].get("task_success") else 0.0)
        grid.append(row)

    fig, axis = plt.subplots(figsize=(11.6, 6.4))
    for y, row in enumerate(grid):
        for x, value in enumerate(row):
            axis.add_patch(
                plt.Rectangle(
                    (x, y),
                    1,
                    1,
                    facecolor=PASS_COLOR if value else FAIL_COLOR,
                    edgecolor="white",
                    linewidth=1.6,
                )
            )
            axis.text(
                x + 0.5,
                y + 0.5,
                "✓" if value else "✗",
                ha="center",
                va="center",
                color="white",
                fontsize=11,
                fontweight="bold",
            )
    for index in range(1, len(RUN_SETS)):
        axis.axvline(index * len(ARCH_ORDER), color="#333333", linewidth=2)
    axis.set_xlim(0, len(columns))
    axis.set_ylim(len(TASK_ORDER), 0)
    axis.set_xticks([i + 0.5 for i in range(len(columns))])
    axis.set_xticklabels(
        [ARCH_LABELS[arch].replace("Multi-", "M-") for _, arch in columns],
        rotation=40,
        ha="right",
        fontsize=8.5,
    )
    axis.set_yticks([i + 0.5 for i in range(len(TASK_ORDER))])
    axis.set_yticklabels(TASK_ORDER, fontsize=9)
    for index, run_set in enumerate(RUN_SETS):
        axis.text(
            index * len(ARCH_ORDER) + len(ARCH_ORDER) / 2,
            -0.45,
            run_set.label,
            ha="center",
            fontsize=9.5,
            fontweight="bold",
        )
    axis.set_xlabel("")
    for spine in axis.spines.values():
        spine.set_visible(False)
    axis.tick_params(length=0)
    fig.suptitle(
        "Every core-small outcome by task, architecture and run set",
        fontsize=12,
        fontweight="bold",
        y=0.99,
    )
    fig.tight_layout()
    save_figure(fig, "04-task-matrix")
    plt.close(fig)


def figure_step_cap_probe(plt, all_rows: list[dict], selected: dict) -> None:
    """Raising the cap to 100 on the runs that exhausted 50."""
    probe = [
        row
        for row in all_rows
        if row.get("session_id")
        in (PROBE_SESSIONS["glm_steps100"], PROBE_SESSIONS["deepseek_steps100"])
    ]
    if not probe:
        return
    entries = []
    for row in probe:
        base_key = (
            "glm_new"
            if row.get("session_id") == PROBE_SESSIONS["glm_steps100"]
            else "deepseek"
        )
        before = [
            item
            for item in selected[(base_key, row["agent_mode"])]
            if item.get("task_id") == row.get("task_id")
        ]
        if not before:
            continue
        model = "glm-5.2" if base_key == "glm_new" else "DeepSeek"
        entries.append(
            {
                "label": f"{row['task_id']}\n{ARCH_LABELS[row['agent_mode']]} · {model}",
                "steps_before": int(before[0].get("steps") or 0),
                "steps_after": int(row.get("steps") or 0),
                "ok_after": bool(row.get("task_success")),
            }
        )
    entries.sort(key=lambda item: (not item["ok_after"], item["label"]))

    fig, axis = plt.subplots(figsize=(12.6, 5.2))
    positions = range(len(entries))
    width = 0.38
    axis.bar(
        [p - width / 2 for p in positions],
        [e["steps_before"] for e in entries],
        width=width,
        color="#999999",
        label="steps used at the 50-step cap (all failed)",
    )
    axis.bar(
        [p + width / 2 for p in positions],
        [e["steps_after"] for e in entries],
        width=width,
        color=[PASS_COLOR if e["ok_after"] else FAIL_COLOR for e in entries],
        label="steps used at the 100-step cap",
    )
    for position, entry in zip(positions, entries):
        axis.text(
            position + width / 2,
            entry["steps_after"] + 1.5,
            "resolved" if entry["ok_after"] else "failed",
            ha="center",
            fontsize=8,
            fontweight="bold",
            color=PASS_COLOR if entry["ok_after"] else FAIL_COLOR,
        )
    axis.axhline(STEP_CAP, color="#555555", linestyle="--", linewidth=1)
    axis.text(-0.45, STEP_CAP + 1.5, f"{STEP_CAP}-step cap", fontsize=8, ha="left")
    axis.set_xticks(list(positions))
    axis.set_xticklabels(
        [e["label"] for e in entries], fontsize=7.4, rotation=16, ha="right"
    )
    axis.set_ylabel("steps used")
    axis.set_ylim(0, 118)
    axis.grid(axis="y", alpha=0.6)
    axis.set_axisbelow(True)
    axis.legend(frameon=False, fontsize=8.5, loc="upper left")
    fig.suptitle(
        "Both recoveries finish below the cap that had failed them",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout()
    save_figure(fig, "05-step-cap-probe")
    plt.close(fig)


def write_summary_csv(summary: list[dict]) -> None:
    fields = [
        "run_set",
        "label",
        "model",
        "harness",
        "agent_mode",
        "runs",
        "successes",
        "success_rate",
        "wilson_low",
        "wilson_high",
        "total_cost_usd",
        "mean_cost_usd",
        "cost_per_success_usd",
        "total_tokens",
        "mean_tokens",
        "mean_steps",
        "runs_at_step_cap",
        "total_duration_s",
    ]
    with (OUTPUT_DIR / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in summary:
            writer.writerow({key: item.get(key) for key in fields})


def write_task_csv(selected: dict) -> None:
    columns = [(rs.key, arch) for rs in RUN_SETS for arch in ARCH_ORDER]
    fields = ["task_id"] + [f"{key}:{arch}" for key, arch in columns]
    with (OUTPUT_DIR / "task_matrix.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(fields)
        for task in TASK_ORDER:
            row = [task]
            for key, arch in columns:
                match = [r for r in selected[(key, arch)] if r.get("task_id") == task]
                row.append(
                    "resolved"
                    if match and match[0].get("task_success")
                    else "failed"
                    if match
                    else ""
                )
            writer.writerow(row)


def probe_rows(all_rows: list[dict], session: str) -> list[dict]:
    return [row for row in all_rows if row.get("session_id") == session]


def write_report(summary: list[dict], selected: dict, all_rows: list[dict]) -> None:
    lines: list[str] = []
    add = lines.append
    add("# Kernel v2 Core-Small Re-measurement")
    add("")
    add("Generated by `generate_figures.py`. Costs are actual provider billing.")
    add("")
    add("## Run sets")
    add("")
    add("| Run set | Model | Harness | Resolved | Cost | Tokens | Mean steps | At 50-step cap |")
    add("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |")
    for run_set in RUN_SETS:
        items = [pick(summary, run_set.key, arch) for arch in ARCH_ORDER]
        ok = sum(i["successes"] for i in items)
        runs = sum(i["runs"] for i in items)
        cost = sum(i["total_cost_usd"] for i in items)
        tokens = sum(i["total_tokens"] for i in items)
        steps = sum(i["mean_steps"] * i["runs"] for i in items) / runs
        capped = sum(i["runs_at_step_cap"] for i in items)
        add(
            f"| {run_set.label} | `{run_set.model}` | `{run_set.harness}` | "
            f"**{ok}/{runs}** | ${cost:.4f} | {tokens:,} | {steps:.1f} | {capped}/{runs} |"
        )
    add("")
    add("## By architecture")
    add("")
    add("| Run set | Architecture | Resolved | 95% Wilson | Cost | Cost per success |")
    add("| --- | --- | ---: | ---: | ---: | ---: |")
    for run_set in RUN_SETS:
        for arch in ARCH_ORDER:
            item = pick(summary, run_set.key, arch)
            per = item["cost_per_success_usd"]
            add(
                f"| {run_set.label} | {ARCH_LABELS[arch]} | "
                f"{item['successes']}/{item['runs']} | "
                f"{item['wilson_low'] * 100:.0f}–{item['wilson_high'] * 100:.0f}% | "
                f"${item['total_cost_usd']:.4f} | "
                f"{'$%.4f' % per if per else 'n/a'} |"
            )
    add("")
    add("## Task matrix")
    add("")
    header = "| Task | " + " | ".join(
        f"{rs.short.replace(chr(10), ' ')} · {ARCH_LABELS[a][:3]}"
        for rs in RUN_SETS
        for a in ARCH_ORDER
    ) + " |"
    add(header)
    add("| --- | " + " | ".join([":---:"] * (len(RUN_SETS) * len(ARCH_ORDER))) + " |")
    for task in TASK_ORDER:
        cells = []
        for run_set in RUN_SETS:
            for arch in ARCH_ORDER:
                match = [
                    r for r in selected[(run_set.key, arch)] if r.get("task_id") == task
                ]
                cells.append("✅" if match and match[0].get("task_success") else "❌")
        add(f"| `{task}` | " + " | ".join(cells) + " |")
    add("")
    add("## Probes outside the comparable block")
    add("")
    add("| Session | Runs | Resolved | Cost | Note |")
    add("| --- | ---: | ---: | ---: | --- |")
    notes = {
        "glm_steps100": "glm-5.2 at a 100-step cap, only the runs that exhausted 50",
        "deepseek_steps100": "DeepSeek at a 100-step cap, `parse_pr_165` only",
        "terra_pilot": "`openai/gpt-5.6-terra` at `high`, `h11_pr_181` only",
        "terra_parse165": "`openai/gpt-5.6-terra` at `high`, `parse_pr_165` only",
    }
    for key, session in PROBE_SESSIONS.items():
        rows = probe_rows(all_rows, session)
        if not rows:
            continue
        ok = sum(bool(r.get("task_success")) for r in rows)
        cost = sum(effective_cost(r) for r in rows)
        add(f"| `{session}` | {len(rows)} | {ok}/{len(rows)} | ${cost:.4f} | {notes[key]} |")
    add("")
    add("## Figures")
    add("")
    for stem, caption in (
        ("01-harness-effect", "Success rate by architecture for the three run sets"),
        ("02-cost", "Cost per run and cost per resolved task"),
        ("03-step-pressure", "Mean steps and runs reaching the 50-step cap"),
        ("04-task-matrix", "Every outcome by task, architecture and run set"),
        ("05-step-cap-probe", "Steps used before and after raising the cap to 100"),
    ):
        add(f"### {caption}")
        add("")
        add(f"![{caption}](figures/{stem}.png)")
        add("")
    (OUTPUT_DIR / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    all_rows = load_rows()
    summary, selected = collect(all_rows)
    plt = configure_plotting()
    figure_harness_effect(plt, summary)
    figure_cost(plt, summary)
    figure_step_pressure(plt, summary)
    figure_task_matrix(plt, selected)
    figure_step_cap_probe(plt, all_rows, selected)
    write_summary_csv(summary)
    write_task_csv(selected)
    write_report(summary, selected, all_rows)
    print(f"wrote figures and tables to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
