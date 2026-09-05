# Independent content review for the defense update

Scope: read-only audit of the English deck builder, the frozen master artifacts, and the two completed defense repetition sessions. No benchmark tests, model calls, or source/task edits were performed.

## Confirmed master budget confound

The defended source is `origin/master` at `8da3f3f890340caac306af062e2adc21acfdb943`. The master audit selects 102 primary scored rows. Its SWE Pro slice contains 15 single rows and 15 graph rows.

- `source/experiments/results/sessions/final_v1_swepro_single50/sweep.json:417`: `resolved_config.max_tokens = 16384`; 50 steps.
- `source/experiments/results/sessions/final_v1_swepro_graph75/sweep.json:417`: `resolved_config.max_tokens = 4096`; 75 steps.
- All 15 single rows match session fingerprint `d5286edb1a58598f9eb836c5e408a0651fa91c4f785c950a37e9edcbd6408647`.
- All 15 graph rows match session fingerprint `0d764d533b61ccdd3c293af6c7112d0607c87bf8c59890fc128fd8f92c42bae2`.
- The runner passes `config.max_tokens` into its model configuration. Run rows themselves do not contain modern per-route maximum-output metadata; the saved session is the direct configuration evidence.

Important qualification: these are **initial completion-budget settings**, not a guarantee that every call stays below that value. In frozen `src/agents/model.py:346`, `_escalated_max_tokens()` returns `min(configured * 4, 65536)` for the one-shot truncation retry. The text-action loop applies the raised `max_tokens` at line 205. Therefore the potential retry budgets are 65,536 and 16,384 respectively. The audit establishes the differing policy, not whether truncation caused any particular task outcome.

Suggested visible wording:

> SWE Pro budgets differ: single 50 steps / 16,384 initial output tokens; graph 75 steps / 4,096. Truncation retries can raise the output budget.

Suggested speaker-note qualification:

> These are descriptive comparisons of the complete recorded systems. They do not isolate agent count, and the output-budget difference is an additional confound beyond the step limits.

The current builder omits this output-budget confound. Add it to the execution-budget slide and update the success/cost/limitations/SWE appendix notes so they do not imply that step count is the only resource mismatch.

## Independently recalculated new repetitions

Data source: `experiments/results/runs.jsonl`, selected by exact session IDs `kernel_v2_ds_defense_repeat_20260904_r1` and `kernel_v2_ds_defense_repeat_20260904_r2`, using final `task_success`.

All 48 scored rows are present: 12 task IDs x 2 architectures x 2 repetitions. All run IDs and session/task/architecture combinations are unique. The matching attempts journal has 48 rows: 41 task successes and 7 task failures. There are no matching infrastructure-failure rows, and the session logs report zero skipped combinations.

All 48 rows have complete provider billing, `cost_source=provider_actual`, and equal `effective_cost_usd` and `provider_reported_cost_usd`. Use these fields for billed totals. The separate `cost_usd` field is an estimate and gives much larger totals; do not mix it with actual billing.

| Repetition | Architecture | Success | Provider cost | Estimated cost_usd |
|---|---|---:|---:|---:|
| R1 | Single | 10/12 | $0.18732158 | $0.89724964 |
| R1 | Graph | 10/12 | $0.30577711 | $1.46970824 |
| R2 | Single | 10/12 | $0.22878596 | $1.06560224 |
| R2 | Graph | 11/12 | $0.31368073 | $1.51033892 |
| New repetitions pooled | Single | 20/24 | $0.41610754 | $1.96285188 |
| New repetitions pooled | Graph | 21/24 | $0.61945784 | $2.98004716 |

New-run billed total: **$1.03556538**. Graph costs 48.87% more overall, with one additional success among 24 repeated observations. Billed cost per resolved observation is $0.020805377 for single and $0.0294979924 for graph; graph is 41.78% higher on that measure. These are 24 observations per architecture, not 24 different tasks.

Across the two repetitions, both systems solve the **same 11 unique tasks** at least once. Both fail `parse_pr_165` twice. Single solves 9 tasks in both repetitions; graph solves 10 in both. These are descriptive consistency counts, not an independently estimated probability of future reliability.

Paired outcomes over the 24 task/repetition pairs:

| Outcome | Pairs |
|---|---:|
| Both succeed | 19 |
| Graph only | 2 |
| Single only | 1 |
| Neither succeeds | 2 |

The graph-only pairs are `w3lib_pr_272` in R1 and `cachetools_pr_57d2e48` in R2. The single-only pair is `humanize_pr_329` in R1. Do not treat repeated observations of the same task as independent new benchmark tasks.

## Historical baseline and three-attempt presentation

The historical main session `kernel_v2_ds_v4_flash_0731_nitro_low_core_small_v4` contains only 11 scored tasks per architecture. Complete its 12-task block with `tenacity_pr_628` from `kernel_v2_ds_v4_flash_0731_nitro_low_core_small_v4_tenacity_pin`:

- Single recovery success, billed cost $0.00342052.
- Graph recovery success, billed cost $0.00525449.

The complete historical baseline is therefore 9/12 single at $0.36847612 and 11/12 graph at $0.42419292. Baseline + R1 + R2 yields 29/36 versus 32/36 and historical-inclusive billed totals $0.78458366 versus $1.04365076. This is **36 observations per architecture on 12 tasks**; show the three blocks separately before any aggregate.

The protocol explicitly records an environment caveat: historical task snapshots omit setup_commands, and current tenacity setup adds pytest<9, matching the later recovery setup. Identical source/test/image hashes do not prove identical installed dependency trees. The root analyst is checking runtime artifacts; until resolved, describe this as a repeatability extension with the recorded setup caveat, not an exact controlled replication. Both new sweeps also share one host concurrently, which limits execution-time comparisons. Provider billing across dates can differ in rate/cache behavior, so lower new billed totals are not proof of a harness efficiency improvement.

## Scientific wording to update

1. **H1:** retain “Supported descriptively in the tested master core-small block.” The new DeepSeek repeats show lower billed cost per solved observation for single while graph has one extra success; this supports a cost/coverage tradeoff in these configurations. It does not prove single is always preferable on simple tasks.
2. **H2:** retain “Not supported by the master results; not disproved.” The new repetitions use only the same core-small tasks, so **they do not test a complexity interaction**. Do not strengthen H2 conclusions from these repeats.
3. **Repeatability:** the historical 9/12 vs 11/12 becomes 10/12 vs 10/12 and 10/12 vs 11/12. Suggested wording: “The size of the observed advantage varies across attempts. Graph does not lose on aggregate success in these three blocks, but a tie and only a few paired disagreements do not establish a universal advantage.”
4. **Generalization:** say “previously used tasks” or “same-task repetitions”; never “new held-out benchmark.” The prepared unseen SWE Pro candidate experiment was not the experiment run here.
5. **Task diversity:** the same 11 unique tasks are solved at least once by either architecture in the new repeats. An extra successful observation is not an additional newly solved task family.
6. **Master scope:** the master-only hypothesis slide and primary metric tables should remain clearly labeled master; new repetitions belong to the kernel/v2 extension and must not change the master 102-run denominator.
7. **Quality:** repeated executable correctness results do not fill the missing human rubric for maintainability/minimality/robustness/safety. Keep rubric quality N/A unless separately assessed.
8. **Cost:** avoid presenting cost_usd estimates as billed cost. The fresh reports have complete actual provider costs for all 48 scored rows.
9. **No causal attribution:** comparisons still include architecture-specific guards, role quotas and transitions. Matching model, reasoning and step settings does not isolate agent count alone.
