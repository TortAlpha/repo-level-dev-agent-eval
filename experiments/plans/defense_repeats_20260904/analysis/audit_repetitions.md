# DeepSeek kernel/v2 repetition audit

The three paired attempts produced **32/36 successful graph observations versus 29/36 for single**, at **33.0% higher recorded provider cost** for graph. The sample contains **12 shared tasks**, not 36 independent tasks per architecture.

| Attempt | Single success | Graph success | Single cost, USD | Graph cost, USD |
|---|---:|---:|---:|---:|
| original | 9/12 | 11/12 | 0.36847612 | 0.42419292 |
| r1 | 10/12 | 10/12 | 0.18732158 | 0.30577711 |
| r2 | 10/12 | 11/12 | 0.22878596 | 0.31368073 |
| All three | 29/36 | 32/36 | 0.78458366 | 1.04365076 |

The two new repetitions completed all 48 planned observations with no infrastructure failures. All 72 selected observations have complete provider-reported costs, unique scored records, matching attempt records, archived patches and matching metric files. 4 original infrastructure records for these two architectures had USD 0.00000000 recorded cost and are retained separately; the scored baseline uses the pinned tenacity recovery.

## Observed task consistency

Graph solved 10/12 tasks in all three attempts; single solved 7/12 consistently. 4 single tasks and 1 graph task varied. Single solved 11/12 distinct tasks at least once; graph solved 11/12 at least once.

| Task | Single successes / 3 | Graph successes / 3 | Single original, r1, r2 | Graph original, r1, r2 |
|---|---:|---:|---|---|
| h11_pr_181 | 2 | 3 | fail/pass/pass | pass/pass/pass |
| humanize_pr_329 | 3 | 2 | pass/pass/pass | pass/fail/pass |
| pluggy_pr_646 | 2 | 3 | fail/pass/pass | pass/pass/pass |
| w3lib_pr_272 | 2 | 3 | pass/fail/pass | pass/pass/pass |
| parse_pr_165 | 0 | 0 | fail/fail/fail | fail/fail/fail |
| parse_pr_227 | 3 | 3 | pass/pass/pass | pass/pass/pass |
| cachetools_pr_57d2e48 | 2 | 3 | pass/pass/fail | pass/pass/pass |
| tinydb_pr_616 | 3 | 3 | pass/pass/pass | pass/pass/pass |
| python_dotenv_pr_640 | 3 | 3 | pass/pass/pass | pass/pass/pass |
| tenacity_pr_628 | 3 | 3 | pass/pass/pass | pass/pass/pass |
| freezegun_pr_546 | 3 | 3 | pass/pass/pass | pass/pass/pass |
| croniter_pr_235 | 3 | 3 | pass/pass/pass | pass/pass/pass |

Across 36 paired task-by-attempt cells: 28 both solved, 3 both failed, 4 graph-only and 1 single-only. These repeated cells are clustered by task; they are not 36 independent task draws.

## Automated metrics and efficiency

| Metric | Single, 36 observations | Graph, 36 observations |
|---|---:|---:|
| Visible suite pass | 36/36 (100.00%) | 36/36 (100.00%) |
| Required hidden suite pass | 29/36 (80.56%) | 32/36 (88.89%) |
| Regression runs | 0/36 (0.00%) | 0/36 (0.00%) |
| Patch presence proxy | 31/36 (86.11%) | 36/36 (100.00%) |
| Tool-use histogram validity | 1002/1033 (97.00%) | 1342/1384 (96.97%) |
| Repair proxy | 23/23 (100.00%) | 29/29 (100.00%) |
| Visible-pass/final-failure proxy | 7/36 (19.44%) | 4/36 (11.11%) |
| Invalid/malformed/no-action proxy events | 30 | 31 |
| Policy rejection events | 22 | 36 |
| LLM calls, total | 1,089 | 1,550 |
| Tokens, total | 6,908,844 | 7,683,363 |
| Agent steps, total | 1,068 | 1,547 |
| Test iterations, total | 83 | 93 |
| Agent execution time, cumulative seconds | 7,288.57 | 10,779.90 |
| Provider cost per successful observation, USD | 0.027055 | 0.032614 |

Hidden pass rates count whole-suite verdicts per run. Hidden compatibility reuses the visible result on all 12 tasks; it is not an independent withheld suite. Tool validity is an action-histogram proxy; accepted events can still be unhelpful. The hallucinated-reference label counts malformed/invalid/no-action events rather than verified fictitious references. Repair means visible tests passed after more than one iteration, not a confirmed failed-then-repaired trajectory. The visible/final gap is not proof of overfitting. Patch presence is not patch quality. Full numerators, denominators, action sets and per-attempt efficiency are in the JSON.

Agent execution time is measured around `agent.run`. It excludes initial evaluator dependency preparation and subsequent final scoring, and its sum is not campaign wall time.

Quality-rubric records found for these runs: **0**. Automated functional checks do not replace maintainability, minimality, robustness or safety assessment.

## Interpretation and limitations

The [completed provenance review](provenance_review.md) confirms matching recorded task, model, harness and frozen evaluator inputs in all 48 repeat-to-baseline comparisons, including setup commands and evaluator dependency fingerprints. All 72 retained setup-artifact trees were rehashed successfully. This is stronger than the initial snapshot-only audit, but does not independently inventory every installed package.

- Existing core-small tasks were previously used to develop/evaluate the harness. Repeats test stability on familiar tasks, not unseen-task generalization.
- Twelve shared tasks and three attempts per task are a small paired sample. No independence-based significance claim is made.
- Single and graph retain architecture-specific prompts, guards and role quotas. Effects apply to complete tested configurations, not agent count in isolation.
- The original scored baseline combines its main session with tenacity_pin recovery. Raw main-session tenacity setup failures are retained separately, not silently scored as patch failures.
- The tenacity_pin recovery has no saved session sweep.json; individual scored records contain run/model/policy provenance, but a complete frozen session snapshot is unavailable.
- The completed per-run provenance review supersedes the earlier startup-only caveat: all 48 repeat-to-scored-baseline comparisons match setup commands and frozen evaluator dependency fingerprints, including the tenacity pinned recovery. The failed original tenacity setup is not the scored baseline.
- All 72 archived setup-artifact trees were independently rehashed and match their recorded fingerprints. Complete run workspaces were removed, so full dependency trees cannot now be rehashed independently and no full retained pip inventory was identified. Matching recorded fingerprints must not be described as an independent inventory of every installed package.
- Recorded runtime_dependencies describe the host LLM/harness libraries; target dependencies are covered by the separate recorded frozen_dependency_environment_tree identity, which matches within each paired condition.
- Hidden compatibility reuses the visible result on all 12 tasks; it is not an independent withheld compatibility test suite.
- New repetition runs executed in parallel across two serial workers on one host; durations can include resource/provider contention.
- Provider caching and billing conditions may differ across repetitions; the lower later prices must not be attributed solely to architecture changes.
- No manual quality-rubric score is inferred from automated metrics. No new tests or model calls are made by this audit.

Graph produced +3 successful observations and +3 consistently solved tasks relative to single in this model/configuration sample. Interpret these differences descriptively together with cost. They do not establish that multi-agent systems generally outperform single-agent systems or that their advantage increases with task complexity.

## Reproduction

```bash
python3 experiments/plans/defense_repeats_20260904/analysis/audit_repetitions.py
```

The script reads existing journals, archives and the completed `provenance_review.json`, asserts coverage and billing completeness, and writes only `audit_repetitions.json` and this Markdown report. It performs no test execution or model calls.
