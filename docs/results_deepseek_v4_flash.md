# DeepSeek V4 Flash — Core-Small Results

Status: complete. All 36 core-small task-agent combinations scored.

Date: 2026-09-02

Model: `deepseek/deepseek-v4-flash-0731:nitro` via OpenRouter, reasoning
effort `low`.

## Executive Summary

The core-small block was run end to end on DeepSeek V4 Flash under the
kernel/v2 harness. The primary outcome is evaluator **`task_success`**, not the
agent's internal `status`.

| Architecture | Resolved | Success rate | Effective cost | Cost/success |
| --- | ---: | ---: | ---: | ---: |
| Multi-graph | **11/12** | 91.7% | $0.4242 | $0.0386 |
| Multi-orch-guarded | **11/12** | 91.7% | $0.4760 | $0.0433 |
| Single | 9/12 | 75.0% | $0.3685 | $0.0409 |

Overall 31/36 (86.1%), actual billed cost $1.26867261 over 7,412,402 tokens.
The static price-table estimate for the same runs is $3.19668888 — see
*Token and Cost Accounting* for why the two differ.

Both multi-agent architectures resolved two more tasks than the single loop at
this reasoning effort, and the entire difference sits on two combinations. This
is the opposite direction from the `z-ai/glm-5.2` result in the README, but the
two runs are not a controlled comparison — see *Comparability* below.

## Run Configuration

| Setting | Value |
| --- | --- |
| Task set | `final-v1`, matrix `core-small` (12 tasks × 3 architectures) |
| Provider | OpenRouter, route `:nitro` |
| Reasoning effort | `low` (requested and effective) |
| Action transport | `text_json` |
| Step cap | 50 for all architectures |
| Per-run cost cap | $0.15 |
| Campaign cost cap | $3.00 (measured on effective cost) |
| Compaction | `checkpoint`, context budget 48000 tokens |
| Temperature | 0 |
| Request timeout / retries | 300 s / 0 |
| Order / seed | `task-major` / 20260831 |

The run spans two sessions:

| Session | Runs | Harness commit | Fingerprint |
| --- | ---: | --- | --- |
| `…_nitro_low_core_small_v4` | 33 | `fd00518` | `f42c638f…` |
| `…_core_small_v4_tenacity_pin` | 3 | `0ca9587` | `e8226f66…` |

The split exists because `tenacity_pr_628` was initially unscoreable; see
*Infrastructure failures*.

## Results by Task

| Task | single | multi-graph | multi-orch-guarded | Resolved | Effective cost | Tokens |
| --- | :---: | :---: | :---: | ---: | ---: | ---: |
| `h11_pr_181` | ✗ | ✓ | ✓ | 2/3 | $0.0628 | 563,234 |
| `humanize_pr_329` | ✓ | ✓ | ✓ | 3/3 | $0.2783 | 1,077,143 |
| `pluggy_pr_646` | ✗ | ✓ | ✓ | 2/3 | $0.0398 | 427,285 |
| `w3lib_pr_272` | ✓ | ✓ | ✓ | 3/3 | $0.1401 | 646,674 |
| `parse_pr_165` | ✗ | ✗ | ✗ | 0/3 | $0.2120 | 873,267 |
| `parse_pr_227` | ✓ | ✓ | ✓ | 3/3 | $0.0582 | 336,077 |
| `cachetools_pr_57d2e48` | ✓ | ✓ | ✓ | 3/3 | $0.0960 | 591,178 |
| `tinydb_pr_616` | ✓ | ✓ | ✓ | 3/3 | $0.1259 | 843,219 |
| `python_dotenv_pr_640` | ✓ | ✓ | ✓ | 3/3 | $0.1189 | 611,791 |
| `tenacity_pr_628` | ✓ | ✓ | ✓ | 3/3 | $0.0174 | 503,958 |
| `freezegun_pr_546` | ✓ | ✓ | ✓ | 3/3 | $0.0638 | 448,685 |
| `croniter_pr_235` | ✓ | ✓ | ✓ | 3/3 | $0.0556 | 489,891 |

Ten of twelve tasks were resolved by all three architectures. `parse_pr_165`
was resolved by none while being the second most expensive task in the block —
the agents work long and still fail to converge rather than giving up cheaply.

`tenacity_pr_628`, the task that blocked the experiment longest, turned out to
be the cheapest in the block at $0.0174 for three runs.

## The Five Failures

| Task | Architecture | Steps used | Terminal status | Cause |
| --- | --- | ---: | --- | --- |
| `h11_pr_181` | single | 24 / 50 | `handoff` | voluntary early stop |
| `pluggy_pr_646` | single | 12 / 50 | `handoff` | voluntary early stop |
| `parse_pr_165` | single | 31 / 50 | `handoff` | voluntary early stop |
| `parse_pr_165` | multi-graph | 50 / 50 | `failed` | step cap reached |
| `parse_pr_165` | multi-orch-guarded | 50 / 50 | `failed` | step cap reached |

No run approached the $0.15 per-run cost cap; the most expensive single run was
$0.1144. Three of the five failures are voluntary handoffs well below the step
cap, so they are genuine give-ups rather than truncation.

## Token and Cost Accounting

| Metric | Value |
| --- | ---: |
| `input_tokens` | 6,448,818 |
| of which cached | 2,466,816 |
| `output_tokens` | 963,584 |
| of which reasoning | 815,340 |
| `cache_write_input_tokens` | 0 |
| `total_tokens` | 7,412,402 |
| `effective_cost_usd` (actual billing) | 1.26867261 |
| `cost_usd` (static price-table estimate) | 3.19668888 |

**The two cost fields are not "actual" and "discounted" — they are "actual" and
"estimated".** `effective_cost_usd` is the provider's own reported billing
whenever every call reported it, which holds for all 36 runs here
(`cost_source = provider_actual`, `provider_cost_complete = true`), and it
equals `provider_reported_cost_usd` exactly. `cost_usd` is the static estimate
computed from token counts against `static/pricing.csv`.

The estimate overshoots by 2.5× because the `:nitro` row in `pricing.csv`
($0.44 / $1.32 per 1M) is far above what was actually billed; the base
`deepseek-v4-flash-0731` row is $0.065 / $0.18. The campaign budget guard sums
`effective_cost_usd`, so it was measuring real money and the $3.00 limit was
never actually approached. The price table, not the guard, is what needs
attention.

## Infrastructure Failures

Nine infrastructure failures occurred, all before agent execution, so all cost
zero tokens and zero dollars. There were **no provider-side failures** in the
entire experiment: with `MAX_RETRIES=0` any API error would have failed the run
immediately, and all 36 records carry `cost_source = provider_actual` and
`usage_accounting_complete = true`.

### Sandbox keep-alive race — 1 failure, not fixed

`humanize_pr_329 / multi-orch-guarded` failed with `Could not capture the
sandbox keep-alive process id`. The container starts with `--tmpfs /tmp`; its
entrypoint writes its PID to a file and then `exec tail -f /dev/null`, while the
harness reads that file with a single `docker exec` immediately after
`docker run -d` returns, with no wait and no retry. `docker run -d` returns when
the container is created, not when the write has happened. The retry succeeded.
Non-deterministic: one occurrence in 45 attempts.

### `tenacity_pr_628` JUnit evidence — 8 failures, fixed

The task's pristine tests passed (118 passed), but the evaluator rejected the
evidence with `JUnit testsuite declared 130 tests but contains 118 cases`.

The tenacity tests use `unittest.subTest` in six places inside loops. pytest 9
counts *passing* subtest reports in the JUnit `testsuite` `tests` attribute but
emits no matching `<testcase>` elements, so the strict equality check in
`_parse_junit_evidence` rejected a correct result:

```
pytest 9.1.1 → 118 passed, 12 subtests passed → tests="130", <testcase> × 118  reject
pytest 8.4.2 → 118 passed                    → tests="118", <testcase> × 118  accept
```

The task setup installed `pytest` unpinned, so the version drifted since the
task was last validated. The fix pins `'pytest<9'` in `setup_commands`
(commit `0ca9587`), restoring the environment the task was verified in without
weakening the evidence check. `src.benchmark.validate` reports VERIFIED, and all
three architectures then passed on the first attempt.

## Comparability with the README Result

The README reports core-small on `z-ai/glm-5.2`: single 11/12 $0.3765,
multi-graph 11/12 $0.7642, multi-orch-guarded 11/12 $0.9707 — 33/36 at $2.1113.
Those numbers reproduce exactly from `runs.jsonl`.

**These two runs share the block design but are not a controlled comparison.**

Identical: the same 12 tasks with identical `collection_sha256`
(`0d3d1807…`), the same three architectures, the same 50-step cap, the same
`text_json` transport, the same `final-v1` task set and the same
`task_success` metric.

Different, simultaneously:

| | README run | this run |
| --- | --- | --- |
| Model | `z-ai/glm-5.2` | `deepseek-v4-flash-0731:nitro` |
| Reasoning effort | provider default (`high`) | `low` |
| Harness | `6c0af80` (12 Jul) | `fd00518` (31 Aug) |
| Policy kernel | absent from the snapshot | `9467ca76…` |
| Compaction mode | `summarize` | `checkpoint` |
| Request timeout / retries | 600 s / 2 | 300 s / 0 |
| Per-run cost cap | none | $0.15 |
| Seed | 20260712 | 20260831 |

Between the two harness commits, 37 files changed in fingerprinted paths
(+9474/−785), including all four multi-agent role prompts and
`single_agent.txt`. The agents were literally given different instructions.

`z-ai/glm-5.2` supports only the `xhigh` and `high` reasoning efforts, so a glm
run at `low` is not possible: the reasoning-effort difference between the two
runs cannot be closed by re-running either side.

### Where the difference actually sits

The delta between 33/36 and 31/36 is exactly two combinations, both `single`:

| Combination | glm-5.2 | DeepSeek V4 Flash |
| --- | --- | --- |
| `h11_pr_181 / single` | solved, 15 steps | `handoff`, 24 steps |
| `pluggy_pr_646 / single` | solved, 17 steps | `handoff`, 12 steps |

`parse_pr_165` failed 3/3 in **both** runs identically — a property of the task,
not of the model.

Step usage rose across the board: the old `single` never reached the 50-step cap
(maximum 29 steps, mean 21.4), while the new one reaches it in 4 of 12 runs
(mean 31.6). That did not cause these two failures, but it means the cap became
binding where it previously was not.

Voluntary handoffs went from 0/36 to 4/36. The `handoff` status existed in the
older harness as well, so this is a behavioural difference, not a new mechanism.

### Measured: the harness accounts for most of the gap

A control run re-ran `z-ai/glm-5.2` under its original protocol on the current
harness — see [results_glm_5_2_harness_control.md](results_glm_5_2_harness_control.md).
The same model on the new harness drops from 33/36 to **29/36** and costs
**2.6× more** ($2.1113 → $5.5807), with four regressions and zero improvements.

Against that correct baseline the ranking reverses:

| | glm-5.2, old harness | glm-5.2, new harness | DeepSeek V4 Flash |
| --- | ---: | ---: | ---: |
| Resolved | 33/36 | 29/36 | **31/36** |
| Actual billed cost | $2.1113 | $5.5807 | **$1.2687** |

DeepSeek V4 Flash resolves two more combinations than glm-5.2 on the same
harness at 4.4× less money. Comparing it against the README instead makes it
look worse than glm, which is an artifact of the harness difference, not a
property of the model. Reasoning effort still differs — glm-5.2 has no `low` —
so part of the cost gap is effort rather than model.

## Limitations

1. **Two collection versions.** 33 rows predate the pytest pin, 3 tenacity rows
   follow it, so the two sessions carry different `collection_sha256` and
   therefore different fingerprints. Rows are distinguishable by
   `experiment_fingerprint`, but aggregating over the campaign silently mixes
   them.
2. **Null `task_set_id` on three rows.** The follow-up session used `--tasks`
   because no `final-v1` group isolates a single task, so the tenacity rows
   carry `task_set_id = null`. Filter by `task_id` or campaign, not by task set.
3. **Dependency versions are still outside the fingerprint.** Fingerprints cover
   harness source, task set, repository trees and the docker image id, but not
   PyPI package versions. Only tenacity is pinned; the other eleven tasks can
   break the same way at any time.
4. **n = 1 per combination.** No variance estimate, and `TEMPERATURE=0` does not
   guarantee provider-side determinism. The 11/12 versus 9/12 gap rests on two
   tasks and is not statistically significant.
5. **The agent's internal `status` does not track success.** Of 31 successful
   runs only 14 report `status = solved`; 16 report `failed` and one `handoff`
   while the task was in fact solved. Only `task_success` is a valid outcome.
6. **The block is close to saturation.** Ten of twelve tasks are resolved by all
   three architectures, so the set barely discriminates between them.
7. **Results are bound to the `:nitro` route** with reasoning effort `low`, a
   50-step cap and a $0.15 per-run cap.
8. **Two harness tests fail in the current environment**
   (`test_search_retries_basic_grep_alternation` and
   `test_repeated_search_result_survives_outside_context`). They also fail with
   the pytest pin reverted, so they are unrelated to it — but the 207-passed
   baseline was recorded earlier, so workspace search behaviour changed at some
   point and all 36 runs used the changed behaviour.
9. **Policy-boundary noise.** 35 malformed actions, 34 policy rejections, 66
   quota warnings and 8 research-guard violations. No test-oracle tampering and
   no setup-artifact tampering occurred. None of this affects `task_success`,
   but part of the step budget goes to rejected actions.

## Source Data

`experiments/results/runs.jsonl` (filter on the two session ids above),
with per-run `metrics/`, `patches/`, `setup_artifacts/` and the session
snapshots under `sessions/`.
