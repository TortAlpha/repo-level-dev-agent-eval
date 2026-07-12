# Final Benchmark Task Set

Status: frozen candidate set as of 2026-07-11.

This document defines the task set used for the final architecture comparison.
The benchmark is split into a core small/medium set and a separately reported
large-repository extension. Results must be aggregated within these groups
rather than combined into a single unstratified success rate.

The primary outcome is evaluator `task_success`, not the agent's internal
`status`.

## Core Benchmark

The core benchmark follows `docs/spec.md`: 24 verified local tasks, consisting
of 12 small-repository tasks and 12 medium-repository tasks.

### Small repositories (12)

- `h11_pr_181`
- `humanize_pr_329`
- `pluggy_pr_646`
- `w3lib_pr_272`
- `parse_pr_165`
- `parse_pr_227`
- `cachetools_pr_57d2e48`
- `tinydb_pr_616`
- `python_dotenv_pr_640`
- `tenacity_pr_628`
- `freezegun_pr_546`
- `croniter_pr_235`

### Medium repositories (12)

- `toolz_pr_603`
- `markdown_pr_1606`
- `markdown_pr_1546`
- `sqlparse_pr_642`
- `sqlparse_pr_749`
- `sqlparse_pr_579`
- `sqlparse_pr_746`
- `sqlparse_pr_664`
- `more_itertools_pr_1166`
- `more_itertools_pr_1153`
- `boltons_pr_403`
- `pydash_pr_cdd84b4`

## Large-Repository Extension

Large tasks are an extension bucket. Their results are reported separately
from the core benchmark because repository size and task-specification style
differ materially from the small/medium local tasks.

### Local large tasks (2)

- `sqlglot_pr_7660486c`
- `sqlglot_pr_45931d44`

### SWE-bench Pro: Ansible (5)

- `swepro_ansible_11c1777d`
- `swepro_ansible_a6e671db`
- `swepro_ansible_9a21e247`
- `swepro_ansible_d62496fe`
- `swepro_ansible_77658704`

### SWE-bench Pro: OpenLibrary (5)

- `swepro_openlibrary_2abe28b4`
- `swepro_openlibrary_43f9e7e0`
- `swepro_openlibrary_6afdb09d`
- `swepro_openlibrary_9cd47f4d`
- `swepro_openlibrary_7f7e53aa`

### SWE-bench Pro: qutebrowser (5)

- `swepro_qutebrowser_50efac08`
- `swepro_qutebrowser_54bcdc1e`
- `swepro_qutebrowser_44e64199`
- `swepro_qutebrowser_3e21c821`
- `swepro_qutebrowser_36ade4bb`

The SWE-bench Pro subset intentionally uses verified tasks with readable task
text, focused test commands, hidden evaluation, and relatively bounded
reference patches. Imported tasks with quoted or literal escaped-newline task
formatting are excluded.

## Calibration Tasks Excluded from Final Scoring

The following tasks were used heavily while developing or diagnosing the
architectures and must not be included in the final aggregate result:

- `sqlparse_pr_768`
- `sqlglot_pr_ed58b9bd`
- `more_itertools_pr_1135`
- `more_itertools_pr_1135_stats`

Their runs remain useful as development evidence and architecture ablations,
but not as held-out final evaluation.

## Evaluation Exposure

The manifest records 21 selected tasks that had at least one exploratory run
before the final architecture freeze. They remain valid paired comparison
tasks, but are not described as held-out. The other 20 tasks form the primary
held-out slice. Report both slices separately, with the complete 41-task result
as a secondary aggregate. The exposure split is frozen in
`eval/task_sets/final_v1.json` and must not be changed after final runs begin.

## Architecture Coverage

### Full core comparison

Run every core task with the same GLM model, transport, evaluator, step limit,
and iteration limit:

- `single`
- `multi-graph`
- `multi-orch-guarded`

This is the primary 72-run comparison: 24 tasks times three architectures.
`single` is the strengthened baseline, `multi-graph` is the deterministic
role-based control, and `multi-orch-guarded` is the main candidate system.

The scored `single` baseline uses the strong progress guard introduced after
the exploratory Sonnet runs:

- warning after 12 consecutive read-only steps and a hard limit at 20 before
  an implementation plan;
- warning after 5 read-only steps and a hard limit at 8 after a plan;
- two ignored hard-limit warnings cause human handoff;
- repeated searches reuse a durable cached result across context compaction;
- compaction preserves a deterministic recent-search ledger;
- after the latest edit, a distinct focused compatibility probe must pass;
  the full suite and a newly created regression test cannot satisfy this gate.

The earlier single configurations remain reproducible with
`--no-single-research-guard` and/or `--no-single-compatibility-guard`, but
their runs are ablations and must not be mixed into the final baseline
aggregate.

`single-decomposed` remains reproducible, but is deferred to a separately
reported small ablation after the primary core comparison. It must not be
mixed into the primary core aggregate.

### SWE-bench Pro comparison

SWE-bench Pro architecture coverage is deliberately deferred until the core
comparison is complete. The selected tasks remain frozen, but no executable
SWE matrix is present in the manifest yet. The next comparison may use
`single` versus `multi-orch-guarded`; that decision must be recorded before
any scored SWE runs begin.

### Role-model policy extension

Sonnet is not used as a single-agent baseline. On a small hard-task subset,
compare these guarded policies separately from the shared-GLM architecture
comparison:

- Sonnet only for the developer role;
- adaptive developer escalation from GLM to Sonnet.

The hard-task subset is selected after the shared-GLM runs using a fixed rule:
choose tasks where at least one shared-GLM architecture produced a valid patch
but failed required hidden evaluation, plus at least one task where no
architecture produced a patch.

## Execution and Budget Rules

- Total experimental budget: USD 50.
- Run in blocks: small, medium, local large, then each SWE-bench Pro family.
- Inspect actual spend and setup reliability after every block.
- Do not use a common high per-run cost cap as a target; it is only a safety
  cutoff and may be exceeded by the final model call.
- Preserve full visible, required hidden, compatibility, regression, token,
  duration, and cost metrics for every scored run.
- Use new session names for final runs and do not mix exploratory runs into the
  final report.
- Select tasks through the machine-readable `final-v1` manifest in
  `eval/task_sets/final_v1.json`; never use `--tasks all` for final scoring.
- Use deterministic task-major ordering with the recorded seed so architecture
  order is interleaved within each task.
- Final task-set sweeps run sequentially, support `--resume`, archive patches,
  remove completed workspaces, and record a session fingerprint over the task
  set, collection, pricing table, harness tree, and all experiment settings.
- The agent container may access the network during dependency setup only. It
  is disconnected before the first model-controlled action.
- Infrastructure/provider/evaluation failures are logged separately from
  `task_failed` and do not silently become benchmark failures.

### Reproducible core sweep blocks

The current manifest fixes core task membership and architecture coverage.
Use a new session for each block and the same model/transport/limits
throughout. Both sessions share one campaign so their costs and final core
report can be aggregated without rerunning tasks:

```text
python -m src.benchmark.sweep --task-set final-v1 --matrix core-small --session final_v1_core_small --campaign final_v1_main --provider openrouter --models z-ai/glm-5.2 --max-total-cost-usd 50
python -m src.benchmark.sweep --task-set final-v1 --matrix core-medium --session final_v1_core_medium --campaign final_v1_main --provider openrouter --models z-ai/glm-5.2 --max-total-cost-usd 50
```

Do not run `core-all` after these two blocks: it contains the same 24 tasks and
would duplicate all 72 scored combinations. Aggregate the two sessions by
their shared campaign instead.

Interrupted blocks resume with the identical command plus `--resume`. A final
task-set sweep refuses a dirty harness; `--allow-dirty-harness` is reserved for
explicitly non-final exploratory runs.

## Set Size

- Core: 24 tasks.
- Local large extension: 2 tasks.
- SWE-bench Pro extension: 15 tasks.
- Total selected: 41 tasks.
