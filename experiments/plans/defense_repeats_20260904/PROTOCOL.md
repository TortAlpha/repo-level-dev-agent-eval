# Two additional repetitions of the existing core-small comparison

This bounded experiment repeats the existing DeepSeek kernel/v2 comparison on the same 12 `core_small` tasks. It measures repeatability and variability on familiar tasks. It is not a new held-out benchmark and must not be described as unseen-task generalization.

The historical reference is `experiments/results/sessions/kernel_v2_ds_v4_flash_0731_nitro_low_core_small_v4/sweep.json`. A read-only startup identity audit confirmed the same 12 task source trees, descriptions, hidden tests, visible/hidden commands, base commits and Docker image IDs, including tenacity. The active harness is clean at commit `928d899da31bb1c3510196a50c4fd91c0a904df1`; its source-tree digest `9ccfc8d1757d3c3ebfef7f94d0a6f892929c9dd6c9a6d32f85dee8559bf7b695` matches the historical harness at `fd00518589b9ba9ec9c4a4a5de81530355fe38ab`. The comparison is recorded in `baseline_comparison.json`; it was a startup audit, not an additional pre-dispatch gate inside `start.py`. No new task tests were run for this identity audit. The existing sweep CLI validates and freezes runtime inputs before solving.

This identity comparison does not fully freeze dependency resolution: the historical per-task snapshot does not include `setup_commands`. The current tenacity setup adds a `pytest<9` constraint that differs from the original main sweep setup and matches the later recovery setup. Compare saved dependency/environment artifacts after completion and disclose any difference. Matching source/test/image hashes must not be presented as proof of identical installed dependencies.

Two repetitions run concurrently on the same host. Each is a separate serial sweep containing 12 tasks × `single,multi-graph` = 24 observations, or 48 additional planned observations in total. Distinct session and campaign names are `kernel_v2_ds_defense_repeat_20260904_r1` and `kernel_v2_ds_defense_repeat_20260904_r2`. Each worker has `--concurrency 1`, so at most two benchmark tasks execute simultaneously. Shared CPU, Docker and provider contention limits the interpretation of execution times.

## Fixed configuration

| Setting | Value |
|---|---|
| Model, including every graph role | `deepseek/deepseek-v4-flash-0731:nitro` via OpenRouter |
| Reasoning / temperature | low / 0 |
| Maximum steps / iterations | 50 / 5 |
| Maximum output tokens | 16,384 |
| Model context setting / working context | 256,000 / 48,000 tokens |
| Compaction / action transport | checkpoint / text JSON |
| Request timeout / API retries | 300 seconds / 0 |
| Test timeout / shell timeout | 600 / 300 seconds |
| Infrastructure retries | 1, matching the historical sweep |
| Per-run cost guard | USD 0.15 |
| Per-repetition campaign guard | USD 2.00 |

There is no model escalation or role-specific model/effort override. Single research and compatibility guards remain enabled with 12/20 warning/hard research steps before planning and 5/8 after planning. Graph keeps its existing role quotas and transitions. Both use the same frozen kernel and final scoring procedure, regression checks, no-network Docker policy and archived patches. Workspaces are cleaned through the existing sweep behavior. These are whole-system comparisons; architecture-specific controls remain part of the treatments.

The new campaign threshold is USD 2 per repetition (USD 4 combined), reduced from the historical USD 3 campaign threshold. Cost guards are soft: a call can overshoot the per-run threshold, and a final run or recorded infrastructure attempt can overshoot a campaign threshold. Recorded infrastructure costs are included; interrupted work without complete records can be absent. A budget stop may leave fewer than 48 completed observations. Do not silently increase limits or selectively rerun task failures.

The fixed task order matches `final-v1/core_small`. Seeds `20260904` and `20260905` shuffle architecture order within each task. These are ordering seeds, not model sampling seeds. Temperature zero does not guarantee deterministic provider execution.

## Invocation and records

The existing CLI is invoked as `.venv/bin/python -m src.benchmark.sweep --task-set final-v1 --task-groups core_small --agents single,multi-graph`, with the fixed settings above, a distinct session/campaign and the corresponding order seed. The complete commands and environment overrides are recorded by the plan-local `start.py` in `launch.json`. A CLI `--plan-only` validation occurs before live dispatch and starts no models or task tests. It prints the 24 ordered combinations for each repetition.

Each live session creates `experiments/results/sessions/<session_id>/sweep.json` with the complete settings, inputs and provenance fingerprint. The CLI enforces separate session/campaign leases and refuses an incompatible existing snapshot. `launch.json` records the launched processes and output paths: `r1.log` and `r2.log` contain live output, and `r1.plan.log` and `r2.plan.log` contain the pre-dispatch CLI plan output. Global runs, attempts, infrastructure failures, traces and archived patches remain under `experiments/results`. Read process/log metadata before considering any manual resume; never dispatch a duplicate campaign.

## Reporting

Report the historical baseline and each new repetition separately before aggregating. Use final `task_success`, not the agent's internal `status`. Preserve paired per-task outcomes, count completed/scored observations, identify infrastructure failures and budget-censored combinations, and report success variability, cost, tokens, tool use, regressions and timing. Price estimates must be labelled if provider billing is unavailable. With 12 previously used tasks and two added repetitions, conclusions concern stability of the tested configurations, not a universal multi-agent advantage or a held-out generalization result.
