# glm-5.2 Harness Control — Core-Small

Status: complete. 36/36 combinations scored, no infrastructure failures.

Date: 2026-09-02

Model: `z-ai/glm-5.2` via OpenRouter, reasoning effort `high`.

## Why This Run Exists

The README reports the core-small block on `z-ai/glm-5.2` under harness
`6c0af80` (12 July 2026). The DeepSeek V4 Flash run in
[results_deepseek_v4_flash.md](results_deepseek_v4_flash.md) used harness
`fd00518` (31 August 2026), 37 files and +9474/−785 lines later, including
changes to all four multi-agent role prompts and `single_agent.txt`.

Comparing those two runs would attribute the harness change to the model. This
control re-runs **the same model under the same protocol on the current
harness**, so the only variable is the harness itself.

`z-ai/glm-5.2` supports only the `xhigh` and `high` reasoning efforts, so it
cannot be run at DeepSeek's `low`. The original run left `reasoning_effort`
unset and therefore resolved to the provider default `high`; this control
passes `high` explicitly so the value is fingerprinted rather than inherited.

## Result

**The harness change alone costs glm-5.2 four tasks and 2.6× the money.**

| | glm-5.2, harness `6c0af80` | glm-5.2, harness `fd00518` |
| --- | ---: | ---: |
| Resolved | **33/36** | **29/36** |
| Actual billed cost | $2.1113 | $5.5807 |
| Cost per run | $0.0586 | $0.1550 |
| Tokens | 5,309,440 | 7,900,067 |
| Wall-clock | 7,856 s (2.2 h) | 20,057 s (5.6 h) |
| Mean steps per run | 31.1 | 38.1 |
| Runs reaching the 50-step cap | 6/36 | 14/36 |

By architecture:

| Architecture | old harness | new harness |
| --- | ---: | ---: |
| Single | 11/12, $0.3765 | 10/12, $1.6271 |
| Multi-graph | 11/12, $0.7642 | **9/12**, $1.6354 |
| Multi-orch-guarded | 11/12, $0.9707 | 10/12, $2.3182 |

### What regressed

Four combinations that the old harness resolved now fail. Nothing that failed
before now passes.

| Combination | old harness | new harness |
| --- | --- | --- |
| `h11_pr_181 / multi-graph` | solved | fails, 50/50 steps |
| `parse_pr_227 / single` | solved | fails |
| `croniter_pr_235 / multi-graph` | solved | fails |
| `croniter_pr_235 / multi-orch-guarded` | solved | fails |

`parse_pr_165` failed 3/3 on both harnesses, as it does for DeepSeek at both the
50- and 100-step caps. It is the one task in the block that no tested
configuration resolves.

The regressions concentrate in the multi-agent architectures (3 of 4), which are
also where the prompt changes landed and where the new
`developer_escalate_after_failed_tests` and
`developer_escalate_after_no_edit_episodes` defaults apply — those settings did
not exist in the old harness at all.

Two of these four turn out to be artifacts of the fixed 50-step cap rather than
lost capability; see *Step-Cap Probe* below.

## Results by Task

Outcomes on the current harness. The `was` column flags where the old harness
differed.

| Task | single | multi-graph | multi-orch-guarded | Resolved | Actual cost | Tokens |
| --- | :---: | :---: | :---: | ---: | ---: | ---: |
| `h11_pr_181` | ✓ | ✗ *(was ✓)* | ✓ | 2/3 | $0.4130 | 637,501 |
| `humanize_pr_329` | ✓ | ✓ | ✓ | 3/3 | $0.8168 | 697,590 |
| `pluggy_pr_646` | ✓ | ✓ | ✓ | 3/3 | $0.4410 | 833,079 |
| `w3lib_pr_272` | ✓ | ✓ | ✓ | 3/3 | $0.4524 | 464,867 |
| `parse_pr_165` | ✗ | ✗ | ✗ | 0/3 | $0.7984 | 926,695 |
| `parse_pr_227` | ✗ *(was ✓)* | ✓ | ✓ | 2/3 | $0.4045 | 598,288 |
| `cachetools_pr_57d2e48` | ✓ | ✓ | ✓ | 3/3 | $0.4555 | 619,451 |
| `tinydb_pr_616` | ✓ | ✓ | ✓ | 3/3 | $0.3563 | 771,290 |
| `python_dotenv_pr_640` | ✓ | ✓ | ✓ | 3/3 | $0.3149 | 527,857 |
| `tenacity_pr_628` | ✓ | ✓ | ✓ | 3/3 | $0.3271 | 647,607 |
| `freezegun_pr_546` | ✓ | ✓ | ✓ | 3/3 | $0.5059 | 675,929 |
| `croniter_pr_235` | ✓ | ✗ *(was ✓)* | ✗ *(was ✓)* | 1/3 | $0.2950 | 499,913 |

Eight of twelve tasks are still resolved by all three architectures, against ten
of twelve for DeepSeek V4 Flash.

## The Seven Failures

| Task | Architecture | Steps used | Terminal status |
| --- | --- | ---: | --- |
| `croniter_pr_235` | multi-graph | 50 / 50 | `failed` |
| `croniter_pr_235` | multi-orch-guarded | 50 / 50 | `failed` |
| `h11_pr_181` | multi-graph | 50 / 50 | `failed` |
| `parse_pr_165` | multi-graph | 50 / 50 | `failed` |
| `parse_pr_165` | multi-orch-guarded | 50 / 50 | `failed` |
| `parse_pr_165` | single | 25 / 50 | `handoff` |
| `parse_pr_227` | single | 23 / 50 | `handoff` |

**Five of seven failures exhausted the step cap.** That is the opposite profile
from DeepSeek V4 Flash, where three of five failures were voluntary handoffs
well below the cap. glm-5.2 on this harness fails by running out of budget, not
by giving up — which is consistent with mean steps rising from 31.1 to 38.1 and
with cap hits rising from 6/36 to 14/36. A higher cap might recover some of
these; the `parse_pr_165` probe on DeepSeek suggests it would not recover that
task specifically.

## Token and Cost Accounting

| Metric | Value |
| --- | ---: |
| `input_tokens` | 6,754,693 |
| of which cached | 3,832,087 |
| `output_tokens` | 1,145,374 |
| of which reasoning | 836,722 |
| `cache_write_input_tokens` | 0 |
| `total_tokens` | 7,900,067 |
| `effective_cost_usd` (actual billing) | 5.58074255 |
| `cost_usd` (static estimate) | 4.05894044 |
| Most expensive single run | $0.5527 (`humanize_pr_329 / multi-orch-guarded`, 2151 s) |

## Step-Cap Probe — Half the Regression Is a Cap Artifact

The five failures that exhausted the 50-step cap were re-run with the cap raised
to 100, everything else in the glm protocol unchanged (session
`kernel_v2_glm_5_2_core_small_steps100`, $1.1989).

| Task | Architecture | at 50 steps | at 100 steps |
| --- | --- | --- | --- |
| `croniter_pr_235` | multi-graph | fails, 50/50, $0.0501 | **resolves, 30 steps**, $0.0431 |
| `h11_pr_181` | multi-graph | fails, 50/50, $0.2622 | **resolves, 35 steps**, $0.0591 |
| `croniter_pr_235` | multi-orch-guarded | fails, 50/50, $0.2095 | fails, 100/100, $0.2205 |
| `parse_pr_165` | multi-graph | fails, 50/50, $0.2347 | fails, 100/100, $0.2588 |
| `parse_pr_165` | multi-orch-guarded | fails, 50/50, $0.3252 | fails, 100/100, $0.6174 |

**Both recoveries finished well below the original 50-step cap** — 30 and 35
steps. So this is not "the task needed more room". The step budget is announced
to the agent in its role prompt (`src/agents/roles.py:104`: *"Your role budget
is N steps; report before it runs out"*), so a larger cap is an input that
changes behaviour, not a neutral ceiling. Two readings fit the data equally
well at n = 1: ordinary run-to-run variance, or a larger announced budget
producing a better strategy. This run cannot separate them.

Against the four harness regressions:

| Regression | at 100 steps |
| --- | --- |
| `h11_pr_181 / multi-graph` | recovered, 35 steps |
| `croniter_pr_235 / multi-graph` | recovered, 30 steps |
| `croniter_pr_235 / multi-orch-guarded` | still fails at 100/100 |
| `parse_pr_227 / single` | not probed — it handed off at 23 steps, never cap-bound |

So **two of the four regressions are step-cap artifacts**, not capability loss.
The harness change makes agents consume more steps, and under a fixed 50-step
cap that alone accounts for half the observed damage. The remaining half is
`croniter_pr_235 / multi-orch-guarded`, which fails even with double the budget,
and `parse_pr_227 / single`, which never reached the cap at all.

Counting the probe, glm-5.2 under a 100-step cap would reach **31/36** — level
with DeepSeek V4 Flash. That figure mixes two step caps and must not be merged
into the 50-step block.

The two probes are also not symmetric. DeepSeek's own 100-step probe recovered
**none** of its two cap-bound failures, but both of those were on
`parse_pr_165`, the one task that no tested configuration resolves. Its other
three failures were voluntary handoffs that never approached the cap and were
not probed.

## Three-Way Comparison

Only the middle and right columns are a valid comparison; the left column is on
a different harness.

| Architecture | glm-5.2, old harness | glm-5.2, new harness | DeepSeek V4 Flash, new harness |
| --- | ---: | ---: | ---: |
| Single | 11/12, $0.3765 | 10/12, $1.6271 | 9/12, $0.3685 |
| Multi-graph | 11/12, $0.7642 | 9/12, $1.6354 | **11/12**, $0.4242 |
| Multi-orch-guarded | 11/12, $0.9707 | 10/12, $2.3182 | **11/12**, $0.4760 |
| **Total** | 33/36, $2.1113 | 29/36, $5.5807 | **31/36**, **$1.2687** |

Against the correct baseline — both models on the current harness — DeepSeek V4
Flash resolves two more combinations at **4.4× less money**. Read against the
README instead, DeepSeek appears to *lose* 31 to 33; that reading is an artifact
of comparing across harness versions.

The two models also disagree about which architecture wins. On the new harness
glm is weakest on `multi-graph` (9/12) while DeepSeek is strongest there and on
`multi-orch-guarded` (11/12 each) and weakest on `single` (9/12). These are
one- and two-task differences at n = 1 and should not be read as a stable
ordering.

## Run Configuration

Reproduces the original glm protocol from the `final_v1_core_small` snapshot,
on the current harness.

| Setting | Value | Source |
| --- | --- | --- |
| Compaction | `summarize`, budget 48000 | original snapshot |
| Request timeout / retries | 600 s / 2 | original snapshot |
| Per-run cost cap | none | original snapshot |
| Step cap | 50 (default, unchanged between harnesses) | original snapshot |
| `max_iterations` | 5 | original snapshot |
| Order / seed | `task-major` / 20260712 | original snapshot, same combination order |
| Action transport | `text_json` | original resolved config |
| Reasoning effort | `high` | explicit; original resolved to the same value by default |
| Campaign cost guard | $6.00 | safety only, never approached ($5.5807) |

Session `kernel_v2_glm_5_2_core_small_control`, campaign
`kernel_v2_glm_5_2_control_main`, fingerprint `0f396d52…`.

## Integrity

All 36 rows verified: 36 unique combinations, no duplicate `run_id`, one exact
model across every role route, `reasoning_effort` and effective reasoning both
`high`, `action_transport = text_json`, a single fingerprint, complete usage
accounting, and metrics, patch index, patch archive with matching sha256 and
setup artifacts present for every run. No setup-artifact tampering and no
test-oracle tamper attempts.

Terminal status against `task_success`: 19 `solved`+true, 10 `failed`+true,
5 `failed`+false, 2 `handoff`+false — so 10 of 29 successes report an internal
status of `failed`, confirming again that only `task_success` is a valid
outcome.

## Cost Field Semantics

`effective_cost_usd` is the provider's **actual reported billing** whenever
every call reported it, which holds for all 36 runs here; it equals
`provider_reported_cost_usd`. `cost_usd` is a **static estimate** computed from
token counts against `static/pricing.csv`.

The two diverge in opposite directions for the two models:

| Model | static estimate | actual billing |
| --- | ---: | ---: |
| glm-5.2 | $4.0589 | $5.5807 |
| DeepSeek V4 Flash `:nitro` | $3.1967 | $1.2687 |

`pricing.csv` has no cache-read rate for `z-ai/glm-5.2` and prices
`deepseek-v4-flash-0731:nitro` at $0.44/$1.32 per 1M against $0.065/$0.18 for
the base `deepseek-v4-flash-0731` row. Both rows are worth revisiting. Budget
guards sum `effective_cost_usd` and therefore measure real money correctly; it
is the estimate that is unreliable.

## Limitations

1. **Reasoning effort cannot be equalised across the two models.** glm-5.2 has
   no `low`; DeepSeek ran at `low`. Part of the cost and success difference
   between them is reasoning effort, not model capability, and no re-run can
   remove that.
2. **`tenacity_pr_628` ran with the `pytest<9` pin** in this control, unlike the
   original July run. Without the pin the task cannot be scored at all, so this
   deviation was unavoidable.
3. **The task-set file changed** by 8 lines since July, so `task_set_sha256`
   differs from the original snapshot even though the 12 core-small task ids
   are identical and `collection_sha256` matches for the 11 unpinned tasks.
4. **n = 1 per combination.** A four-task regression out of 36 at n = 1 is a
   strong signal but not a measured effect size; the direction is consistent
   (four regressions, zero improvements) which is what makes it credible.
5. **The harness change is not one variable.** Prompts, the policy kernel,
   developer escalation defaults and the decomposition configuration all moved
   together. This control shows the change is costly; it does not identify which
   part of it is responsible.

## Consequence for the README

The README's core-small numbers describe harness `6c0af80` and cannot be
compared with any run on the current harness. Either re-run the remaining blocks
(core-medium, SWE-bench Pro) on the current harness before comparing, or label
the README numbers explicitly with the harness revision they belong to.
