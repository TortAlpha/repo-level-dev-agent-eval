"""Reproduce the descriptive audit of three paired DeepSeek core-small attempts.

Read-only inputs: existing experiment journals and archived artifacts.
Outputs are confined to this script's analysis directory. No tests or LLM calls.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
from pathlib import Path
from statistics import mean, median

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[3]
RESULTS = ROOT / "experiments/results"
BASE = "kernel_v2_ds_v4_flash_0731_nitro_low_core_small_v4"
SESSIONS = {
    "original": (BASE, BASE + "_tenacity_pin"),
    "r1": ("kernel_v2_ds_defense_repeat_20260904_r1",),
    "r2": ("kernel_v2_ds_defense_repeat_20260904_r2",),
}
MODES = ("single", "multi-graph")
MODEL = "deepseek/deepseek-v4-flash-0731:nitro"
VALID = {"set_plan", "inspect_file", "list_dir", "search", "search_reused", "write_file",
         "edit_file", "run_shell", "run_tests", "compatibility_check", "set_subtasks",
         "complete_subtask", "reopen_subtask", "finish", "handoff"}
INVALID = {"invalid_action", "malformed_action", "no_action", "repeated_action"}
ACTION_FAILURE = {"invalid_action", "malformed_action", "no_action"}
EFFICIENCY = ("steps", "iterations", "llm_calls", "input_tokens", "output_tokens",
              "cached_input_tokens", "cache_write_input_tokens", "reasoning_tokens",
              "total_tokens", "duration_s")


def artifact(path):
    return {"path": str(path.relative_to(ROOT)), "exists": path.exists(),
            "bytes": path.stat().st_size if path.is_file() else None,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None}


def journal(name):
    return [json.loads(line) for line in (RESULTS / name).read_text().splitlines() if line.strip()]


def amount(rows, field):
    values = [Decimal(str(r[field])) for r in rows if r.get(field) is not None]
    return float(sum(values, Decimal(0))) if values else None


def proportion(numerator, denominator):
    return {"numerator": numerator, "denominator": denominator,
            "rate": numerator / denominator if denominator else None}


def aggregate(rows):
    n = len(rows)
    solved = sum(r["task_success"] for r in rows)
    actions = Counter()
    for row in rows:
        actions.update(row["action_counts"])
    valid = sum(actions[k] for k in VALID)
    invalid = sum(actions[k] for k in INVALID)
    multi_iteration = [r for r in rows if r["iterations"] > 1]
    visible = [r for r in rows if r["test_passed"]]
    regressions = [r for r in rows if r.get("regressions") is not None]
    cost = amount(rows, "provider_reported_cost_usd")
    efficiency = {}
    for field in EFFICIENCY:
        values = [r[field] for r in rows if r.get(field) is not None]
        efficiency[field] = {"sum": sum(values), "mean": mean(values), "median": median(values),
                             "minimum": min(values), "maximum": max(values), "recorded_runs": len(values)}
    return {
        "n_observations": n, "n_distinct_tasks": len({r["task_id"] for r in rows}),
        "task_success": proportion(solved, n),
        "visible_test_pass": proportion(len(visible), n),
        "required_hidden_suite_pass": proportion(sum(r["hidden_required_tests_passed"] for r in rows), n),
        "semantic_hidden_suite_pass": proportion(sum(r["hidden_semantic_tests_passed"] for r in rows), n),
        "compatibility_hidden_suite_pass": proportion(sum(r["hidden_compat_tests_passed"] for r in rows
                                                           if r.get("hidden_compat_tests_passed") is not None),
                                                      sum(r.get("hidden_compat_tests_passed") is not None for r in rows)),
        "regression_run_rate": proportion(sum(r["regressions"] > 0 for r in regressions), len(regressions)),
        "regressions_sum": sum(r["regressions"] for r in regressions),
        "patch_presence_proxy": proportion(sum(bool(r["changed_files"]) for r in rows), n),
        "handoff_rate": proportion(sum(r["status"] == "handoff" for r in rows), n),
        "repair_success_proxy": proportion(sum(r["test_passed"] for r in multi_iteration), len(multi_iteration)),
        "visible_success_final_failure_proxy": proportion(sum(not r["task_success"] for r in visible), len(visible)),
        "tool_use_validity_histogram": proportion(valid, valid + invalid),
        "invalid_action_events_including_repeats": invalid,
        "malformed_invalid_no_action_events": sum(actions[k] for k in ACTION_FAILURE),
        "hallucinated_reference_proxy_events_per_run": sum(actions[k] for k in ACTION_FAILURE) / n,
        "policy_rejection_events": actions["policy_rejection"],
        "policy_rejection_events_per_run": actions["policy_rejection"] / n,
        "detected_oracle_tampering_runs": sum(bool(r.get("test_oracle_tampered")) for r in rows),
        "oracle_tamper_attempt_events": sum(r.get("test_oracle_tamper_attempts", 0) for r in rows),
        "setup_artifact_tampering_runs": sum(bool(r.get("setup_artifact_tampered")) for r in rows),
        "provider_reported_cost_usd": cost, "effective_cost_usd": amount(rows, "effective_cost_usd"),
        "static_estimate_usd_not_used_for_comparison": amount(rows, "cost_usd"),
        "mean_provider_cost_usd": cost / n, "provider_cost_per_success_usd": cost / solved if solved else None,
        "cost_source_counts": dict(Counter(r["cost_source"] for r in rows)),
        "complete_provider_cost_runs": sum(r.get("provider_cost_complete") is True for r in rows),
        "provider_cost_call_coverage": proportion(sum(r["provider_cost_calls"] for r in rows), sum(r["llm_calls"] for r in rows)),
        "complete_usage_runs": sum(r.get("usage_accounting_complete") is True for r in rows),
        "efficiency": efficiency, "action_counts": dict(sorted(actions.items())),
        "internal_status_counts": dict(Counter(r["status"] for r in rows)),
        "internal_solved_disagrees_with_final_score_runs": sum((r["status"] == "solved") != r["task_success"] for r in rows),
    }


def paired(rows):
    keyed = {(r["repetition"], r["task_id"], r["agent_mode"]): r for r in rows}
    counts = Counter()
    for repetition, task in {(r["repetition"], r["task_id"]) for r in rows}:
        single = keyed[repetition, task, "single"]["task_success"]
        graph = keyed[repetition, task, "multi-graph"]["task_success"]
        counts["both_solved" if single and graph else "both_failed" if not single and not graph
               else "single_only" if single else "graph_only"] += 1
    return dict(counts)


def main():
    runs, attempts, infrastructure = map(journal, ("runs.jsonl", "sweep_attempts.jsonl", "infrastructure_failures.jsonl"))
    session_to_repetition = {s: rep for rep, sessions in SESSIONS.items() for s in sessions}
    selected = [dict(row, repetition=session_to_repetition[row["session_id"]]) for row in runs
                if row.get("session_id") in session_to_repetition and row.get("agent_mode") in MODES]
    tasks = json.loads((ROOT / "eval/task_sets/final_v1.json").read_text())["groups"]["core_small"]
    expected = {(rep, task, mode) for rep in SESSIONS for task in tasks for mode in MODES}
    actual = Counter((r["repetition"], r["task_id"], r["agent_mode"]) for r in selected)
    assert len(selected) == 72 and actual == Counter({key: 1 for key in expected}), "Incomplete or duplicate matrix"
    assert len({r["run_id"] for r in selected}) == 72
    assert all(r["model"] == MODEL and r["run_record_complete"] is True for r in selected)
    assert all(isinstance(r["task_success"], bool) and r["cost_source"] == "provider_actual"
               and r["provider_cost_complete"] is True and r["usage_accounting_complete"] is True
               and r["provider_cost_calls"] >= r["llm_calls"] for r in selected)
    selected.sort(key=lambda r: (list(SESSIONS).index(r["repetition"]), tasks.index(r["task_id"]), MODES.index(r["agent_mode"])))
    arch = {mode: aggregate([r for r in selected if r["agent_mode"] == mode]) for mode in MODES}
    repetitions = {rep: {"architectures": {mode: aggregate([r for r in selected if r["repetition"] == rep
                                                           and r["agent_mode"] == mode]) for mode in MODES},
                         "paired_outcomes": paired([r for r in selected if r["repetition"] == rep]),
                         "sessions": list(SESSIONS[rep])} for rep in SESSIONS}
    task_outcomes = []
    for task in tasks:
        entry = {"task_id": task}
        for mode in MODES:
            by_rep = {r["repetition"]: r for r in selected if r["task_id"] == task and r["agent_mode"] == mode}
            outcomes = [by_rep[rep]["task_success"] for rep in SESSIONS]
            entry[mode] = {"outcomes_original_r1_r2": outcomes, "successes_of_three": sum(outcomes),
                           "consistent_outcome": len(set(outcomes)) == 1,
                           "run_ids": [by_rep[rep]["run_id"] for rep in SESSIONS]}
        entry["graph_minus_single_successes"] = entry["multi-graph"]["successes_of_three"] - entry["single"]["successes_of_three"]
        task_outcomes.append(entry)
    consistency = {mode: {"success_count_distribution_over_12_tasks": {str(k): sum(t[mode]["successes_of_three"] == k for t in task_outcomes) for k in range(4)},
                          "all_three_solved": sum(t[mode]["successes_of_three"] == 3 for t in task_outcomes),
                          "all_three_failed": sum(t[mode]["successes_of_three"] == 0 for t in task_outcomes),
                          "mixed_outcomes": sum(not t[mode]["consistent_outcome"] for t in task_outcomes),
                          "ever_solved_in_three_observed_attempts": sum(t[mode]["successes_of_three"] > 0 for t in task_outcomes)} for mode in MODES}
    archive_evidence = []
    for r in selected:
        rid = r["run_id"]
        patch, metrics = RESULTS / "patches" / f"{rid}.patch", RESULTS / "metrics" / f"{rid}.json"
        assert patch.is_file() and metrics.is_file(), f"Missing patch/metrics for {rid}"
        metric_payload = json.loads(metrics.read_text())
        assert metric_payload["task_success"] == r["task_success"] and metric_payload["run_id"] == rid
        matched = [a for a in attempts if a.get("run_id") == rid]
        assert len(matched) == 1 and matched[0]["classification"] == ("task_success" if r["task_success"] else "task_failed")
        archive_evidence.append({"run_id": rid, "patch": artifact(patch), "metrics": artifact(metrics),
                                 "setup_artifact_directory_exists": (RESULTS / "setup_artifacts" / rid).is_dir(),
                                 "attempt_classification": matched[0]["classification"], "run_record_complete": r["run_record_complete"]})
    relevant_infra = [r for r in infrastructure if r.get("session_id") in session_to_repetition and r.get("agent_mode") in MODES]
    new_attempts = [r for r in attempts if r.get("session_id") in (*SESSIONS["r1"], *SESSIONS["r2"])]
    assert len(new_attempts) == 48 and all(a["classification"] in ("task_success", "task_failed") for a in new_attempts)
    session_paths = [RESULTS / "sessions" / session / "sweep.json" for session in session_to_repetition]
    snapshots = {p.parent.name: json.loads(p.read_text()) for p in session_paths if p.is_file()}
    config_summary = {s: {"harness": x["harness"], "resolved_config": x["resolved_config"],
                          "order_seed": x["settings"]["seed"], "fingerprint": x["fingerprint"]} for s, x in snapshots.items()}
    for path in session_paths:
        if not path.is_file():
            config_summary[path.parent.name] = {"sweep_snapshot_missing": True,
                                                "note": "Recovery was recorded without a session sweep.json; use per-run provenance and retain this limitation."}
    quality_rows = journal("quality.jsonl")
    quality_matches = [r for r in quality_rows if r.get("run_id") in {x["run_id"] for x in selected}]
    provenance_path = OUT / "provenance_review.json"
    provenance_review = json.loads(provenance_path.read_text())
    assert provenance_review["n_pairwise_repeat_to_baseline_comparisons"] == 48
    assert provenance_review["all_compared_provenance_fields_match"] is True
    assert provenance_review["all_archived_setup_artifacts_match"] is True
    assert len(provenance_review["archived_setup_artifact_directories"]) == 72
    facts = {
        "scope": {"observations": 72, "new_observations": 48, "repetitions_including_original": 3,
                  "architectures": list(MODES), "observations_per_architecture": 36, "distinct_shared_tasks": 12,
                  "paired_task_repetition_cells": 36, "independence_warning": "The 72 observations reuse 12 tasks and are paired by task and repetition; they are not 72 independent tasks."},
        "architectures": arch, "repetitions": repetitions, "paired_outcomes_all_three": paired(selected),
        "task_consistency": consistency, "task_outcomes": task_outcomes,
        "effect": {"graph_minus_single_successes": arch["multi-graph"]["task_success"]["numerator"] - arch["single"]["task_success"]["numerator"],
                   "graph_minus_single_success_rate_pp": 100 * (arch["multi-graph"]["task_success"]["rate"] - arch["single"]["task_success"]["rate"]),
                   "graph_cost_increase_percent": 100 * (arch["multi-graph"]["provider_reported_cost_usd"] / arch["single"]["provider_reported_cost_usd"] - 1)},
        "new_repetitions_combined": {m: aggregate([r for r in selected if r["repetition"] != "original" and r["agent_mode"] == m]) for m in MODES},
        "infrastructure": {"record_count_original_two_architectures": len(relevant_infra),
                           "new_repetition_failure_count": sum(r["session_id"] in (*SESSIONS["r1"], *SESSIONS["r2"]) for r in relevant_infra),
                           "recorded_effective_cost_usd": amount(relevant_infra, "effective_cost_usd"),
                           "records": [{k: r.get(k) for k in ("run_id", "session_id", "task_id", "agent_mode", "classification", "effective_cost_usd", "finished_at")} for r in relevant_infra],
                           "note": "Original tenacity setup failures are infrastructure records, followed by the predeclared pinned-dependency recovery; guarded-mode records are outside this audit."},
        "archive_evidence": archive_evidence, "quality_rubric_records_found": len(quality_matches),
        "quality_rubric_note": "Automated correctness/behavior metrics do not measure maintainability, minimality, robustness or safety as a human quality rubric.",
        "configuration_snapshots": config_summary,
        "completed_provenance_review": {
            "source": artifact(provenance_path), "repeat_to_scored_baseline_comparisons": 48,
            "all_recorded_inputs_match": True,
            "matched_fields": provenance_review["comparison_fields"],
            "archived_setup_artifact_trees_rehashed_and_matching": 72,
            "retained_full_run_workspaces": provenance_review["retained_run_workspaces"],
            "interpretation": "All 48 paired comparisons match setup commands and frozen evaluator dependency fingerprints, including the scored tenacity recovery. No full retained pip inventory or independently rehashable complete dependency workspace was identified.",
        },
        "metric_definitions": {
            "success": "Final task_success: visible plus required hidden suites and evaluator safeguards; internal agent status is not the scoring outcome.",
            "resolved_at_1": "Each repetition has one scored run per task/architecture; its success fraction can be called resolved@1. The pooled 29/36 and 32/36 are per-attempt success across repeated tasks, not resolved@1 over 36 distinct tasks.",
            "hidden_rates": "Fraction of runs whose whole required/semantic/compatibility suite passed, not fraction of individual assertions.",
            "hidden_compatibility": "All 12 core tasks reuse the visible result for hidden compatibility; this is not an independent withheld compatibility suite.",
            "tool_use_validity_histogram": "Valid event count divided by valid plus invalid event count, using the action-kind sets from src/metrics/compute.py. Repeated actions count as invalid; policy rejections and role reports are separately counted/excluded. Syntactic/policy acceptance does not prove useful or semantically correct tool use.",
            "hallucinated_reference_proxy": "Counts invalid_action, malformed_action and no_action events; it does not independently verify hallucinated repository references.",
            "repair_success_proxy": "Final visible-test pass fraction among runs with iterations > 1. It does not establish a failed-then-repaired trajectory.",
            "visible_success_final_failure_proxy": "Final failure among visible-pass runs; useful generalization-gap indicator, not proof of deliberate test overfitting.",
            "patch_presence_proxy": "At least one changed file in the run record. It does not validate patch syntax, minimality or maintainability.",
            "duration": "Recorded duration_s measures agent.run only. It excludes prior evaluator dependency preparation and subsequent final scoring. Sums are cumulative agent execution time, not full run or campaign wall time. Repetition workers share host/provider resources.",
            "cost": "Complete provider_reported_cost_usd for every selected run. Static estimates are retained only for comparison and are not substituted into the headline.",
            "stability": "Number of observed successes out of three attempts per task. This is descriptive observed consistency, not an independent confidence estimate or an extrapolated pass@k metric.",
        },
        "limitations": [
            "Existing core-small tasks were previously used to develop/evaluate the harness. Repeats test stability on familiar tasks, not unseen-task generalization.",
            "Twelve shared tasks and three attempts per task are a small paired sample. No independence-based significance claim is made.",
            "Single and graph retain architecture-specific prompts, guards and role quotas. Effects apply to complete tested configurations, not agent count in isolation.",
            "The original scored baseline combines its main session with tenacity_pin recovery. Raw main-session tenacity setup failures are retained separately, not silently scored as patch failures.",
            "The tenacity_pin recovery has no saved session sweep.json; individual scored records contain run/model/policy provenance, but a complete frozen session snapshot is unavailable.",
            "The completed per-run provenance review supersedes the earlier startup-only caveat: all 48 repeat-to-scored-baseline comparisons match setup commands and frozen evaluator dependency fingerprints, including the tenacity pinned recovery. The failed original tenacity setup is not the scored baseline.",
            "All 72 archived setup-artifact trees were independently rehashed and match their recorded fingerprints. Complete run workspaces were removed, so full dependency trees cannot now be rehashed independently and no full retained pip inventory was identified. Matching recorded fingerprints must not be described as an independent inventory of every installed package.",
            "Recorded runtime_dependencies describe the host LLM/harness libraries; target dependencies are covered by the separate recorded frozen_dependency_environment_tree identity, which matches within each paired condition.",
            "Hidden compatibility reuses the visible result on all 12 tasks; it is not an independent withheld compatibility test suite.",
            "New repetition runs executed in parallel across two serial workers on one host; durations can include resource/provider contention.",
            "Provider caching and billing conditions may differ across repetitions; the lower later prices must not be attributed solely to architecture changes.",
            "No manual quality-rubric score is inferred from automated metrics. No new tests or model calls are made by this audit.",
        ],
    }
    retained = ("repetition", "task_id", "run_id", "session_id", "campaign_id", "agent_mode", "task_success", "status",
                "test_passed", "hidden_required_tests_passed", "hidden_semantic_tests_passed", "hidden_compat_tests_passed",
                "regressions", "changed_files", "action_counts", "provider_reported_cost_usd", "effective_cost_usd", "cost_usd",
                "cost_source", "provider_cost_complete", "provider_cost_calls", "usage_accounting_complete", "run_record_complete",
                "finished_at", "run_fingerprint", "experiment_fingerprint", *EFFICIENCY)
    facts["selected_records"] = [{key: row.get(key) for key in retained} for row in selected]
    facts["provenance"] = {"generated_at_utc": datetime.now(timezone.utc).isoformat(), "audit_script": artifact(Path(__file__)),
                           "sources": [artifact(RESULTS / name) for name in ("runs.jsonl", "sweep_attempts.jsonl", "infrastructure_failures.jsonl", "quality.jsonl")]
                                      + [artifact(p) for p in session_paths] + [artifact(ROOT / "src/metrics/compute.py"), artifact(provenance_path)],
                           "assertions": {"unique_complete_scored_matrix_72": True, "one_attempt_record_per_scored_run": True,
                                          "all_patches_and_metrics_present_and_metrics_scores_agree": True,
                                          "all_costs_complete_provider_actual": True, "new_attempts_exactly_48_without_infrastructure_failures": True}}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "audit_repetitions.json").write_text(json.dumps(facts, indent=2) + "\n")
    total_single = arch["single"]["task_success"]["numerator"]
    total_graph = arch["multi-graph"]["task_success"]["numerator"]
    pairs = facts["paired_outcomes_all_three"]
    lines = ["# DeepSeek kernel/v2 repetition audit", "",
             f"The three paired attempts produced **{total_graph}/36 successful graph observations versus {total_single}/36 for single**, at **{facts['effect']['graph_cost_increase_percent']:.1f}% higher recorded provider cost** for graph. The sample contains **12 shared tasks**, not 36 independent tasks per architecture.", "",
             "| Attempt | Single success | Graph success | Single cost, USD | Graph cost, USD |",
             "|---|---:|---:|---:|---:|"]
    for rep, value in repetitions.items():
        s, g = (value["architectures"][m] for m in MODES)
        lines.append(f"| {rep} | {s['task_success']['numerator']}/12 | {g['task_success']['numerator']}/12 | {s['provider_reported_cost_usd']:.8f} | {g['provider_reported_cost_usd']:.8f} |")
    lines += [f"| All three | {total_single}/36 | {total_graph}/36 | {arch['single']['provider_reported_cost_usd']:.8f} | {arch['multi-graph']['provider_reported_cost_usd']:.8f} |", "",
              f"The two new repetitions completed all 48 planned observations with no infrastructure failures. All 72 selected observations have complete provider-reported costs, unique scored records, matching attempt records, archived patches and matching metric files. {len(relevant_infra)} original infrastructure records for these two architectures had USD {facts['infrastructure']['recorded_effective_cost_usd'] or 0:.8f} recorded cost and are retained separately; the scored baseline uses the pinned tenacity recovery.", "",
              "## Observed task consistency", "",
              f"Graph solved {consistency['multi-graph']['all_three_solved']}/12 tasks in all three attempts; single solved {consistency['single']['all_three_solved']}/12 consistently. {consistency['single']['mixed_outcomes']} single {'task' if consistency['single']['mixed_outcomes'] == 1 else 'tasks'} and {consistency['multi-graph']['mixed_outcomes']} graph {'task' if consistency['multi-graph']['mixed_outcomes'] == 1 else 'tasks'} varied. Single solved {consistency['single']['ever_solved_in_three_observed_attempts']}/12 distinct tasks at least once; graph solved {consistency['multi-graph']['ever_solved_in_three_observed_attempts']}/12 at least once.", "",
              "| Task | Single successes / 3 | Graph successes / 3 | Single original, r1, r2 | Graph original, r1, r2 |",
              "|---|---:|---:|---|---|"]
    for t in task_outcomes:
        fmt = lambda m: "/".join("pass" if x else "fail" for x in t[m]["outcomes_original_r1_r2"])
        lines.append(f"| {t['task_id']} | {t['single']['successes_of_three']} | {t['multi-graph']['successes_of_three']} | {fmt('single')} | {fmt('multi-graph')} |")
    lines += ["", f"Across 36 paired task-by-attempt cells: {pairs.get('both_solved', 0)} both solved, {pairs.get('both_failed', 0)} both failed, {pairs.get('graph_only', 0)} graph-only and {pairs.get('single_only', 0)} single-only. These repeated cells are clustered by task; they are not 36 independent task draws.", "", "## Automated metrics and efficiency", "",
              "| Metric | Single, 36 observations | Graph, 36 observations |", "|---|---:|---:|"]
    for name, key in (("Visible suite pass", "visible_test_pass"), ("Required hidden suite pass", "required_hidden_suite_pass"),
                      ("Regression runs", "regression_run_rate"), ("Patch presence proxy", "patch_presence_proxy"),
                      ("Tool-use histogram validity", "tool_use_validity_histogram"), ("Repair proxy", "repair_success_proxy"),
                      ("Visible-pass/final-failure proxy", "visible_success_final_failure_proxy")):
        def fmt(mode):
            value = arch[mode][key]
            return f"{value['numerator']}/{value['denominator']} ({100 * value['rate']:.2f}%)" if value['rate'] is not None else "N/A"
        lines.append(f"| {name} | {fmt('single')} | {fmt('multi-graph')} |")
    for name, key in (("Invalid/malformed/no-action proxy events", "malformed_invalid_no_action_events"),
                      ("Policy rejection events", "policy_rejection_events")):
        lines.append(f"| {name} | {arch['single'][key]} | {arch['multi-graph'][key]} |")
    for label, key in (("LLM calls, total", "llm_calls"), ("Tokens, total", "total_tokens"),
                       ("Agent steps, total", "steps"), ("Test iterations, total", "iterations"),
                       ("Agent execution time, cumulative seconds", "duration_s")):
        fmt = lambda mode: (f"{arch[mode]['efficiency'][key]['sum']:,.2f}" if key == "duration_s"
                            else f"{arch[mode]['efficiency'][key]['sum']:,}")
        lines.append(f"| {label} | {fmt('single')} | {fmt('multi-graph')} |")
    lines.append(f"| Provider cost per successful observation, USD | {arch['single']['provider_cost_per_success_usd']:.6f} | {arch['multi-graph']['provider_cost_per_success_usd']:.6f} |")
    lines += ["", "Hidden pass rates count whole-suite verdicts per run. Hidden compatibility reuses the visible result on all 12 tasks; it is not an independent withheld suite. Tool validity is an action-histogram proxy; accepted events can still be unhelpful. The hallucinated-reference label counts malformed/invalid/no-action events rather than verified fictitious references. Repair means visible tests passed after more than one iteration, not a confirmed failed-then-repaired trajectory. The visible/final gap is not proof of overfitting. Patch presence is not patch quality. Full numerators, denominators, action sets and per-attempt efficiency are in the JSON.", "",
              "Agent execution time is measured around `agent.run`. It excludes initial evaluator dependency preparation and subsequent final scoring, and its sum is not campaign wall time.", "",
              f"Quality-rubric records found for these runs: **{len(quality_matches)}**. Automated functional checks do not replace maintainability, minimality, robustness or safety assessment.", "", "## Interpretation and limitations", ""]
    lines += ["The [completed provenance review](provenance_review.md) confirms matching recorded task, model, harness and frozen evaluator inputs in all 48 repeat-to-baseline comparisons, including setup commands and evaluator dependency fingerprints. All 72 retained setup-artifact trees were rehashed successfully. This is stronger than the initial snapshot-only audit, but does not independently inventory every installed package.", ""]
    lines.extend(f"- {item}" for item in facts["limitations"])
    lines += ["", f"Graph produced {total_graph - total_single:+d} successful observations and {consistency['multi-graph']['all_three_solved'] - consistency['single']['all_three_solved']:+d} consistently solved tasks relative to single in this model/configuration sample. Interpret these differences descriptively together with cost. They do not establish that multi-agent systems generally outperform single-agent systems or that their advantage increases with task complexity.", "", "## Reproduction", "", "```bash", "python3 experiments/plans/defense_repeats_20260904/analysis/audit_repetitions.py", "```", "", "The script reads existing journals, archives and the completed `provenance_review.json`, asserts coverage and billing completeness, and writes only `audit_repetitions.json` and this Markdown report. It performs no test execution or model calls.", ""]
    (OUT / "audit_repetitions.md").write_text("\n".join(lines))
    print(json.dumps({"status": "audited", "observations": 72, "json": str(OUT / "audit_repetitions.json"),
                      "markdown": str(OUT / "audit_repetitions.md"), "effect": facts["effect"]}, indent=2))


if __name__ == "__main__":
    main()
