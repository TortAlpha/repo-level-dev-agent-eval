# Kernel v2: new-task architecture comparison

This prospective study compares complete single-agent and multi-graph systems on 12 previously unattempted tasks. The primary selection contains four tasks from each of Ansible, Open Library and qutebrowser: two localized and two cross-module changes. Classification describes the reference change and interacting components, not an established difficulty rating or repository-size category.

The selected task IDs are the `PRIMARY` list in `preflight.py`. Candidate reports record selection rationale and fixed-order backups. Before any solver results, the complete selected set must pass strict preflight. A replacement requires a documented environment, oracle or specification problem and must preserve the repository/class balance. Solver success must never determine selection or replacement. The final collection, task set, commands, evidence hashes, source identity and model configuration are frozen together by the launcher.

## Preflight gate

Every `strict_preflight/<task_id>.json` must identify the correct task and report all of the following as true:

- `verified`
- `base_visible_passed`
- `base_hidden_semantic_failed`
- `reference_visible_passed`
- `reference_hidden_semantic_passed`
- `reference_hidden_compat_passed`

The launcher compares current registry rows, descriptions, reference patches, repository trees and hidden-test trees with the strict-preflight hashes. It checks the resolved Docker image ID and the presence and hashes of raw evidence files. Missing, failed or stale evidence blocks the entire experiment. Imported `pr_task_verified` metadata alone is insufficient. The strict validator must require executed test assertions, reject collection/import errors and inappropriate skips, and distinguish expected semantic failure from infrastructure failure. Selection and gold/hidden artifacts are evaluator-only and must never enter solver context.

## Fixed treatment and budget

There are two independent repetition sweeps. Each runs all 12 tasks with `single` and `multi-graph`, giving 24 planned observations per repetition and 48 total. Both use `deepseek/deepseek-v4-flash-0731:nitro` through OpenRouter, including every graph role. There is no stronger-model escalation.

| Setting | Value |
|---|---|
| Maximum steps / iterations | 75 / 5 per run |
| Temperature / reasoning effort | 0 / low |
| Maximum completion tokens per call | 16,384, including model reasoning where billed as output |
| Working context / model context setting | 48,000 / 1,048,576 tokens |
| Compaction / action transport | checkpoint / text JSON |
| Request timeout / API retries | 300 seconds / 0 |
| Test / shell timeout | 600 / 300 seconds |
| Infrastructure retries | 0 |
| Cost guard per run | USD 0.30 |
| Cost guard per repetition campaign | USD 8.00 |
| Maximum active benchmark runs | 2, one per repetition |

The total authorized campaign threshold is USD 16 across the two campaigns. These are soft cost guards, not a billing guarantee: the agent checks known token cost around calls, and the sweep checks recorded campaign cost before starting the next run. A call or final run can overshoot; unrecorded interrupted work can be missing. Campaign accounting includes recorded infrastructure failures. Report provider cost when available, otherwise label static estimates. Do not hide failed attempts or infrastructure expense. Reaching a cap may leave the planned 48 observations incomplete; do not silently raise limits or rerun failures.

The common kernel, frozen visible/hidden evaluation, regression checks, Docker isolation and network restrictions remain enabled. Patches are archived and workspaces are cleaned by the existing harness. Single research and compatibility guards are explicitly enabled, with research warning/hard limits of 12/20 before planning and 5/8 after planning. Graph retains its architecture-specific role transitions and quotas. These settings are fixed across repetitions; the study does not isolate agent count from every other workflow difference.

Task order is fixed by the frozen manifest. Seeds `20260904` and `20260905` deterministically shuffle the two architecture orders within each task using the existing sweep algorithm. They are ordering seeds, not provider or model sampling seeds. Temperature zero does not guarantee identical model execution.

## Launch and records

Run from the repository root using its installed virtual environment:

```bash
.venv/bin/python experiments/plans/defense_new_tasks_20260904/launch_repetitions.py --plan-only
.venv/bin/python experiments/plans/defense_new_tasks_20260904/launch_repetitions.py --launch
```

`--plan-only` starts no tests and no models. After all strict gates pass, it freezes the selected collection/task set and runs the underlying sweep's read-only plan validation for both repetitions. `--launch` repeats those checks before dispatch. The launcher refuses dirty harness sources, incompatible ambient routing/escalation settings, recorded exposure of selected tasks, or an existing launch record. It never uses `--allow-dirty-harness` and never automatically resumes or duplicates an experiment.

Each repetition uses a distinct session and campaign, `kernel_v2_defense_new12_20260904_repeat1` or `kernel_v2_defense_new12_20260904_repeat2`, and `--concurrency 1`. Detached supervisors keep both sweeps alive after the launching terminal exits and record final return codes. `launch_metadata.json` contains supervisor PIDs, process group IDs and log/status paths. `repeat1_status.json` and `repeat2_status.json` record each sweep PID and completion state. `repeat1.log` and `repeat2.log` capture execution output; corresponding `_plan.log` files contain sweep validation output. Global run, attempt, infrastructure, trace and patch files remain managed by the existing harness under `experiments/results`.

The sweeps are independent repetitions but share one physical host, Docker service and provider account. CPU, memory, disk and provider contention can affect duration; report timing with this limitation. Image preparation must finish before launch. No additional parallelism or automatic per-repository batching is introduced.

## Analysis commitments

Use final `task_success` from frozen scoring as the primary outcome, not the agent's internal `status`. Report per-task paired results, both repetitions separately, and success by localized/cross-module class. Report visible and hidden test outcomes, regressions, valid tool use, token/call counts, elapsed time, cost per task and cost per solved task with their operational definitions. Preserve unsuccessful attempts and distinguish infrastructure failures from evaluated failures.

Interpret the 12-task sample as a small, purposively selected extension. Tasks are unattempted in the recorded local experiment journals, not proven unseen during model pretraining or all prior human work. Imported augmented specifications sometimes name interfaces. Avoid claims about original SWE-bench Pro scores or a clean causal effect of agent count. Two repetitions provide limited evidence about variability; they do not establish a universal architecture ranking.
