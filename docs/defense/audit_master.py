"""Read-only reaggregation of the saved origin/master experiment records."""
from pathlib import Path
import collections
import csv
import json
import math

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "source"
rows = [json.loads(line) for line in (SOURCE / "experiments/results/runs.jsonl").read_text().splitlines()]
specs = [
    ("Core small", "final_v1_core_small", "single", 12, 50),
    ("Core small", "final_v1_core_small", "multi-graph", 12, 50),
    ("Core small", "final_v1_core_small", "multi-orch-guarded", 12, 50),
    ("Core medium", "final_v1_core_medium_single50", "single", 12, 50),
    ("Core medium", "final_v1_core_medium_multi75", "multi-graph", 12, 75),
    ("Core medium", "final_v1_core_medium_multi75", "multi-orch-guarded", 12, 75),
    ("SWE-bench Pro", "final_v1_swepro_single50", "single", 15, 50),
    ("SWE-bench Pro", "final_v1_swepro_graph75", "multi-graph", 15, 75),
]

def cost(row):
    if (isinstance(row.get("provider_reported_cost_usd"), (int, float))
        and row.get("llm_calls") and row.get("provider_cost_calls", 0) >= row["llm_calls"]):
        return row["provider_reported_cost_usd"]
    return row.get("cost_usd") or 0.0

selected = []
blocks = []
for block, session, mode, n, cap in specs:
    group = [row for row in rows if row.get("session_id") == session and row.get("agent_mode") == mode]
    assert len(group) == n
    assert len({row["task_id"] for row in group}) == n
    selected.extend(group)
    blocks.append(dict(block=block, agent_mode=mode, runs=n,
                       successes=sum(row["task_success"] for row in group),
                       total_cost_usd=sum(map(cost, group)),
                       total_steps=sum(row["steps"] for row in group),
                       total_tokens=sum(row["total_tokens"] for row in group),
                       step_cap=cap))

published = list(csv.DictReader((SOURCE / "experiments/results/final_report/summary.csv").open()))
for result in blocks:
    original = next(p for p in published if p["block"] == result["block"] and p["agent_mode"] == result["agent_mode"])
    for key in ("runs", "successes", "total_steps", "total_tokens"):
        assert result[key] == int(original[key]), (key, result)
    assert math.isclose(result["total_cost_usd"], float(original["total_cost_usd"]), abs_tol=1e-10)

paired = collections.defaultdict(dict)
for row in selected:
    if row["agent_mode"] in ("single", "multi-graph"):
        paired[row["task_id"]][row["agent_mode"]] = bool(row["task_success"])
assert all(set(outcomes) == {"single", "multi-graph"} for outcomes in paired.values())
disagreements = {task: outcomes for task, outcomes in paired.items() if len(set(outcomes.values())) > 1}

manifest = json.loads((SOURCE / "eval/task_sets/final_v1.json").read_text())
registry = {row["task_id"]: row for row in csv.DictReader((SOURCE / "repositories/collection.csv").open())}
composition = {}
for name, file_min, file_max, loc_min, loc_max in [
    ("core_small", 5, 30, 500, 3000), ("core_medium", 31, 120, 3001, 15000)
]:
    tasks = [registry[task] for task in manifest["groups"][name]]
    composition[name] = dict(tasks=len(tasks), repositories=len({t["repo_url"] for t in tasks}),
        strict_raw_loc_and_files=sum(file_min <= int(t["source_files"]) <= file_max and loc_min <= int(t["source_loc"]) <= loc_max for t in tasks),
        strict_nonblank_loc_and_files=sum(file_min <= int(t["source_files"]) <= file_max and loc_min <= int(t["source_loc_nonblank"]) <= loc_max for t in tasks),
        single_source_file_patches=sum(int(t["patch_files"]) == 1 for t in tasks))

aggregates = {}
for mode in ("single", "multi-graph", "multi-orch-guarded"):
    group = [row for row in selected if row["agent_mode"] == mode]
    solved = sum(row["task_success"] for row in group)
    aggregates[mode] = dict(runs=len(group), successes=solved,
        total_cost_usd=sum(map(cost, group)), cost_per_success_usd=sum(map(cost, group))/solved)

audit = dict(commit=json.loads((ROOT / "SOURCE.json").read_text())["commit"],
    published_summary_matches_raw_records=True, historical_records=len(rows),
    selected_records=len(selected), unique_tasks=len(paired), blocks=blocks,
    aggregates=aggregates, paired_disagreements=disagreements,
    paired_outcomes=dict(collections.Counter(f"single={v['single']},graph={v['multi-graph']}" for v in paired.values())),
    primary_cost_usd=sum(map(cost, selected)),
    campaign_cost_usd=sum(cost(r) for r in rows if r.get("campaign_id") == "final_v1_main"),
    historical_recorded_cost_usd=sum(map(cost, rows)),
    composition=composition,
    provider_record_counts=dict(collections.Counter(r.get("provider", "absent") for r in rows)),
    primary_models=sorted({r["model"] for r in selected}),
    primary_status_evaluation_disagreements=sum((r["status"] == "solved") != r["task_success"] for r in selected),
    missing_cost_records=sum(r.get("cost_usd") is None and r.get("provider_reported_cost_usd") is None for r in rows),
    scope="Arithmetic verification only; does not re-run model calls or validate original test execution.")
(ROOT / "audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n")
with (ROOT / "audit_summary.csv").open("w") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(blocks[0]))
    writer.writeheader()
    writer.writerows(blocks)
print(json.dumps(audit, indent=2, ensure_ascii=False))
