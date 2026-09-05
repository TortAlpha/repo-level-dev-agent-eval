# Provenance and metric interpretation of the DeepSeek repetitions

## Defensible conclusion

The historical baseline and both new repetitions are comparable on the recorded experimental inputs for each task/architecture pair. All 48 repeat-to-baseline comparisons match the complete saved policy configuration, model routes, prompts, tool schemas, task/source identity, harness sources, harness runtime package versions, and frozen evaluator input identities. This includes **setup commands and the frozen evaluator dependency fingerprints**, which were not covered by the earlier startup-only snapshot comparison.

Use this wording in presentation notes:

> We repeated the same 12-task comparison twice. Recorded task, model, harness, and frozen evaluator dependency fingerprints matched the historical scored baseline, including its tenacity recovery. The results measure repeatability on familiar tasks; shared-host execution and provider behavior limit timing comparisons.

Do not say that every installed package was independently inventoried or that the environment was globally identical. The saved dependency fingerprint covers the frozen user dependency tree; it excludes `__pycache__`. The Docker image ID also matches. All 72 working directories have been removed, so their full dependency directories cannot now be independently rehashed and no full retained pip inventory was identified. The saved setup artifacts are still present and **all 72 archived setup-artifact trees were rehashed and match their recorded fingerprints**.

Machine-readable comparisons and exact run IDs: [provenance_review.json](provenance_review.json).

## Included observations

| Block | Sources | Single | Graph |
|---|---|---:|---:|
| Historical scored baseline | 22 selected runs in the original main session plus 2 tenacity recovery runs | 12 | 12 |
| Repetition 1 | `kernel_v2_ds_defense_repeat_20260904_r1` | 12 | 12 |
| Repetition 2 | `kernel_v2_ds_defense_repeat_20260904_r2` | 12 | 12 |

The original main session also contains guarded runs; those are outside this two-architecture repetition comparison. Its separate tenacity recovery session has three architectures, of which only single and graph are included here. No extra model attempts were run for this provenance review.

There are 72 scored observations but only **12 unique tasks**. They are repeated observations on previously used tasks, not 72 independent tasks and not an unseen-task generalization benchmark. The third architecture has not been repeated. Preserve the block and paired task structure when calculating uncertainty; do not treat all repeated task outcomes as independent benchmark items.

## What matches

- All 12 per-task inputs in the historical, r1, and r2 sweep snapshots: base commits, source working trees, task text, hidden fixtures, visible/hidden commands, and Docker image identities.
- Harness source digest `9ccfc8d1757d3c3ebfef7f94d0a6f892929c9dd6c9a6d32f85dee8559bf7b695`; historical commit `fd00518589b9ba9ec9c4a4a5de81530355fe38ab` and repeat commit `928d899da31bb1c3510196a50c4fd91c0a904df1` record the same 75-file harness digest.
- Full per-run `reproducibility.policy_kernel.configuration`, including 50 steps, 5 iterations, DeepSeek V4 Flash 0731 Nitro, low reasoning, temperature 0, 16,384 output tokens, 48,000 working context, checkpoint compaction, text JSON actions, 300-second request timeout, zero API retries, USD 0.15 per-run guard, architecture-specific controls, editing/evaluation network policy, and setup commands.
- Per-run frozen scoring-oracle, dependency-environment, setup-artifact and pytest-bootstrap tree identities, compared within each task/architecture condition.
- Host harness packages: langchain-core 1.4.8, langchain-openai 1.3.3, openai 2.45.0, packaging 26.2, pydantic 2.13.4. These are harness packages, not a full list of task-container dependencies.

The historical sweep's `task_inputs` omit setup commands, but the per-run policy configuration contains them. Therefore the protocol's earlier warning was appropriate before the repetitions; the completed per-run records now provide a stronger comparison.

## Tenacity dependency correction

The original main sweep's tenacity preflight encountered pytest 9.1.1 and typeguard 4.6.0. Its output reported 118 passing tests, but trusted scoring rejected inconsistent JUnit accounting: the testsuite declared 130 tests and contained only 118 cases. These were **pre-agent infrastructure failures**, not scored solver failures.

The historical scored tenacity observations come from `kernel_v2_ds_v4_flash_0731_nitro_low_core_small_v4_tenacity_pin`:

- Graph: `7859b591-8bfb-41de-b589-568fbaa95551`.
- Single: `5bf74c73-940b-425a-ad38-7dd6662a1158`.

Both record the setup command `python -m pip install -q 'pytest<9' typeguard`, identical to both new repetitions. Repeat logs explicitly show pytest 8.4.2. All six included tenacity observations record the same 527-file dependency tree, digest `8b91a74d9b…`; the full digest is in the JSON review. Their archived `tenacity/_version.py` bytes also match, SHA256 `dbbf39139d4ae6d03da07658667aebe90ff11da5c7e235af36936844424d1b3e`.

Do not compare the repeats against the failed original tenacity environment, or suggest that the scored baseline used a different pytest constraint. Keep the recovery disclosure, because the baseline is a documented combined block rather than a single uninterrupted sweep.

## Differences and limits

- r1 and r2 ran concurrently on one host, while the historical main sweep was serial. Within each repetition concurrency remained 1. Shared CPU/Docker/provider contention can affect durations and timeouts.
- Order seeds changed from 20260831 to 20260904 and 20260905. They shuffle architecture order; they are not model sampling seeds. Temperature zero does not prove deterministic execution.
- The campaign guard was reduced from USD 3 to USD 2 per repetition; all 24 selected observations completed in each repetition. This campaign-level limit therefore did not censor the comparison, but should remain in the protocol.
- A sweep CLI flag changed from an implicit/default network setting to explicit `--no-network`; the resolved per-run editing and evaluation network policies match and report isolation after setup. This does not mean dependency installation itself had no network access.
- Vendor serving implementation and cache behavior are not frozen by the experiment. All 72 scored observations report complete token accounting and complete provider-reported cost, with `cost_source=provider_actual`. Use `effective_cost_usd` for these observed costs, not the larger uncached token-price estimate in `cost_usd`.
- The recorded `duration_s` is measured around `agent.run`. It does not include the prior evaluator dependency preparation or subsequent final scoring. Label it agent execution time, not complete campaign wall time.
- Equal step ceilings are not equal token spend. Graph and single still have different internal role controls and verification policies; the comparison concerns the tested systems and does not isolate agent count alone.

## Quality and AI metric caveats

All 72 metrics artifacts exist, all have `quality_score=null`, and none of these run IDs has an entry in `quality.jsonl`. All sweep settings disable `enable_review`. Thus maintainability, minimality, robustness and safety rubric scores remain **not evaluated**; test success and tool validity must not be presented as replacement rubric scores.

| Recorded metric | What it actually measures | Reporting limitation |
|---|---|---|
| `task_success` | Final visible and required hidden success, zero regressions and no detected oracle/setup tampering | Test-measured correctness, not proof of all requested behavior or code quality |
| `resolved_at_1` | The earliest finished scored run per task in the records given to `compute_metrics` | Applied to the pooled repetitions, this selects the historical attempt rather than averaging three attempts; report per-block resolved rate or pooled per-attempt success separately |
| Hidden test pass rate | Fraction of scored runs passing the whole required hidden suite | Not the proportion of individual hidden tests passed |
| Hidden compatibility | Reuses the visible result on all 12 selected core tasks | Not an independent withheld compatibility suite |
| Patch validity | Whether `changed_files` is nonempty | Does not assess syntactic validity, minimality or maintainability |
| Tool-use validity | Recognized valid action kinds divided by recognized valid plus invalid/repeated kinds | A recognized shell action can still fail; high validity is not high software quality |
| Hallucinated references | Counts `invalid_action`, `malformed_action`, and `no_action` | A proxy for action failures, not a semantic audit of invented symbols or paths |
| Repair success | Final visible success among runs with `iterations > 1` | Repeated test iterations need not imply a prior failure or a genuine repair |
| Test overfitting | Final task failure among visible-passing scored runs | Can include oracle/policy failures; not direct evidence that a model overfit the tests |
| Handoff rate | Fraction with internal final `status == handoff` | Not recorded human labor; internal status can differ from final evaluator success |

Several runs have an internal status of `failed` or `handoff` while their saved patch passes the final evaluator. Use `task_success` for success claims, not the agent's internal stopping label.

## Evidence paths

- [Historical sweep](/Users/tortalpha/Code/repo-level-dev-agent-eval/experiments/results/sessions/kernel_v2_ds_v4_flash_0731_nitro_low_core_small_v4/sweep.json).
- [Repetition 1 sweep](/Users/tortalpha/Code/repo-level-dev-agent-eval/experiments/results/sessions/kernel_v2_ds_defense_repeat_20260904_r1/sweep.json).
- [Repetition 2 sweep](/Users/tortalpha/Code/repo-level-dev-agent-eval/experiments/results/sessions/kernel_v2_ds_defense_repeat_20260904_r2/sweep.json).
- [Per-run records](/Users/tortalpha/Code/repo-level-dev-agent-eval/experiments/results/runs.jsonl) and [infrastructure failures](/Users/tortalpha/Code/repo-level-dev-agent-eval/experiments/results/infrastructure_failures.jsonl).
- [Repeat 1 pytest evidence](/Users/tortalpha/Code/repo-level-dev-agent-eval/experiments/plans/defense_repeats_20260904/r1.log:6227) and [repeat 2 pytest evidence](/Users/tortalpha/Code/repo-level-dev-agent-eval/experiments/plans/defense_repeats_20260904/r2.log:6356).
- [Tree identity implementation](/Users/tortalpha/Code/repo-level-dev-agent-eval/src/run/reproducibility.py:317), [metric definitions](/Users/tortalpha/Code/repo-level-dev-agent-eval/src/metrics/compute.py:126), and [agent timer](/Users/tortalpha/Code/repo-level-dev-agent-eval/src/benchmark/runner.py:1367).
