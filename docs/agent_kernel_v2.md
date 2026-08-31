# Pseudo-SWE-Agent Kernel v2

This revision hardens the built-in agent runtime without changing the research
object. The built-in baseline remains a pseudo-SWE-agent: one model chooses one
structured repository action per turn, receives the observation, and repeats
until it finishes or exhausts a bound. Multi-agent variants reuse the same
action contract, executor, workspace, sandbox, and evaluator.

It deliberately does **not** import AI-autent's production manager, DAG,
template selection, repository maps, or dynamic workflow routing. Those would
change the independent variable from agent architecture to an unequal bundle
of architecture and capabilities. The changes below are measurement,
reliability, and protocol controls shared by the compared built-in agents.

## What changed

### 1. Accounted and deterministic context

- The context budget counts the complete replay envelope sent to the provider,
  including retained assistant tool calls and provider reasoning metadata that
  are not printed in the visible state rendering.
- `checkpoint` is the default compaction mode. It creates a bounded,
  deterministic ledger of edits, tests, searches, and inspected files.
- `summarize` remains available as a stochastic, model-generated ablation;
  `drop` remains the minimal ablation.
- Provider-native assistant messages and reasoning details are stored on the
  immutable `State`, so compaction and replay do not depend on a hidden
  process-local side cache.
- File snapshots are invalidated after both structured edits and audited shell
  mutations, so compaction cannot rescue an inspection from an older workspace
  revision.

### 2. Auditable shell execution and test-oracle protection

- Shell and test actions snapshot the Git worktree before and after execution.
- Mutations performed indirectly through the shell are registered in the same
  `changed_files` and workspace-revision state as explicit edit actions.
- Changes to tracked benchmark tests are restored and rejected. A durable
  `test_oracle_tamper_attempts` counter survives rollback and compaction, so a
  clean final diff cannot hide an attempted oracle change.
- Tracked test roots are restored and mounted read-only as bounded directory
  mounts; root-level runners and controls outside those roots use exact
  immutable file mounts with bind-pinned ancestors. This scales to large
  SWE-Pro repositories without weakening overwrite, unlink, path-replacement,
  symlink, or hardlink protection. Structured host actions may still add a
  regression test, while shell commands cannot mutate the mounted oracle.
- The complete agent patch is archived before scoring. Agent-added regression
  tests remain in that patch but are removed from the scoring checkout;
  agent-added control files such as `conftest.py` are both removed and recorded
  as tampering. Deleted or modified ignored test runners are restored from the
  pristine task input.
- Action-supplied timeouts are bounded separately for shell and test commands.
  Snapshot failures fail closed and invalidate earlier verification.
- Final scoring reconstructs `HEAD + archived agent.patch` after deleting all
  ignored/untracked runtime residue. It reads neither a mutable task source nor
  the agent's post-run dependency state.
- Dependencies are prepared once from the pristine checkout, fingerprinted,
  and mounted read-only for final scoring. A trusted pytest bootstrap starts
  with `python -I -S`, loads and attests the evaluator runtime before adding
  repository `PYTHONPATH` and conventional `/workspace/src` entries, and
  therefore ignores submitted `pytest`, `sitecustomize`, and executable
  editable-install `.pth` launch shims while supporting normal src layouts.
- Direct pytest, `xvfb-run`, and immutable shell runners all pass through a
  read-only evaluator PATH shim. JUnit paths are injected inside that shim,
  and an isolated one-shot marker rejects wrappers which invoke pytest zero or
  multiple times. Pytest-affecting config options are compared with the frozen
  baseline and restored on change; unrelated `pyproject.toml`/`setup.cfg`
  edits and marker declarations remain eligible solution changes.
- Hidden fixtures use one exact manifest for inventory and overlay, including
  legitimate support paths named `target`, `build`, or `dist`. Existing files
  retain their visible bytes until the visible suite finishes, then receive the
  withheld overlay under a read-only directory mount.
- Visible and hidden runs require structurally complete JUnit evidence read
  from container-local tmpfs. Hidden scoring rejects missing/malformed reports,
  failures, errors, any skipped required case, zero execution, and evidence
  covering fewer cases than the command's explicit selectors.

### 3. Hardened, persistent Docker sandbox

The setup and agent phases use the same container so editable installs remain
available, but the model-controlled phase runs with network access disabled.
The container uses the host user's numeric UID/GID, a read-only root filesystem
and `.git`, dropped Linux capabilities, `no-new-privileges`, CPU/memory/PID
limits, and explicit writable tmpfs locations for temporary files and caches.
Every action is followed by fail-closed process reaping. The keep-alive PID is
captured on the host before agent execution, so a background process cannot
survive by removing an environment marker or forging a writable PID file.
The executable search path prefers immutable system locations; Python safe-path
mode is enabled, and VCS metadata is rejected by host file tools even through a
symlink alias.

### 4. Run fingerprints

Each built-in run records hashes for the exact prepared source tree, task,
prompts, action schemas, policy configuration, relevant policy source files,
resolved model routes, model-profile registry, hidden fixtures/evaluator
configuration, and the resolved immutable Docker image ID. These are combined
into `run_fingerprint`. Sweeps also record the existing task-set and experiment
fingerprints and reject a Docker tag whose ID changed after the sweep snapshot.

Concurrent sweep writers use process-safe, durable JSONL appends. A host that
cannot acquire an OS file lock fails before writing instead of risking a
valid-looking but interleaved result row. Parallel runs receive their IDs in
advance and only attribute or clean up their own workspace.

Session and campaign leases cover selection, resume claims, and budget checks,
preventing two sweep processes from scheduling the same observation. A sweep
passes expected source/task/hidden/collection/harness/policy/pricing hashes to
every child; the runner validates them before and after execution. A scientific
row is appended last and is resumable only when `run_record_complete=true`.

A run fingerprint identifies the executable protocol; it is not a quality
score. Two rows should be pooled only when every intended control matches or
the differing field is the declared experimental variable.

### 5. Reasoning replay and cost provenance

- Reasoning effort accepts the canonical scale `none`, `minimal`, `low`,
  `medium`, `high`, `xhigh`, and `max`.
- `reasoning_max_tokens` is a gated alternative to effort. It is mutually
  exclusive with effort, may not exceed the completion budget, and is accepted
  only when the frozen route profile explicitly verifies an end-to-end exact
  token guarantee. A provider accepting or translating the field is not enough.
- Known model routes are validated against a dated OpenRouter capability
  snapshot. Unknown routes remain usable with effort controls and are marked
  capability-unknown; exact token budgets fail closed. The launcher may refresh
  the public model catalog, but a live claim does not replace the frozen profile
  required for a comparable exact-budget run.
- One global reasoning setting is shared by all roles by default. Explicit
  role or escalation overrides are recorded as a heterogeneous policy and must
  be treated as a separate ablation.
- Usage records include cached-read, cache-write, visible output, and reasoning
  tokens. Complete provider-reported billing is preferred; incomplete billing
  uses the explicit conservative upper bound `reported partial cost + complete
  static token estimate`. Missing token usage makes accounting unknown rather
  than silently recording a zero-dollar call.

OpenRouter documents both its [reasoning controls](https://openrouter.ai/docs/guides/best-practices/reasoning-tokens)
and [usage accounting](https://openrouter.ai/docs/cookbook/administration/usage-accounting).
The local registry is intentionally fingerprinted because provider aliases and
capabilities can change over time.

### 6. Stable repository tools

- Search paths and results are repository-relative.
- Search cache keys include query, path, mode, and result limit.
- `auto` search uses regex semantics when valid and literal fallback for a
  malformed expression; explicit `regex` and `literal` modes are available.
- Tracked files remain visible to mutation snapshots even inside ignored
  directories, while generated artifacts such as `.egg-info` stay excluded.
- Read/search/list observations keep their beginning; shell and test output
  keeps the diagnostically useful tail.
- All harness Git calls strip inherited `GIT_*` routing, disable hooks, external
  diff/filter drivers and includes, validate local config, and verify the
  expected HEAD and clean index before patch capture.

### External SWE-agent is an explicit ablation

The built-in agents remain the comparable pseudo-SWE-agent implementation.
The external `swe-agent`/`multi-swe` adapter is separately identified and is
accepted only with a content-hashed Python interpreter and standard library,
the complete marker-resolved installed dependency closure rooted at `sweagent`
and `swe-rex` (including LiteLLM), and an immutable execution-image ID. The
runtime and dependency closure are checked before and after each episode.
Inherited Python/LiteLLM/loader injection variables are removed before launch.
Arbitrary CLI passthrough, unaccounted trajectories, per-run cost caps that
cannot be enforced inside an episode, offline claims, and reasoning controls
that the adapter cannot forward are rejected. External execution is therefore
an explicitly all-online ablation, not silently pooled with the built-in
protocol.

## Remaining boundary limitations

The scorer now isolates the launcher, test/control files, dependency state, and
JUnit transport, but tested production code necessarily executes in the pytest
process. A deliberately adversarial submission could still try to alter test
framework behavior after normal project imports begin. Eliminating that entire
class requires a stronger out-of-process/per-test attestation protocol and
would be a separately versioned research kernel.

Dependency setup may resolve mutable network packages unless a task pins and
hashes its dependency graph. The resulting frozen environment is fingerprinted,
so drift produces a different run identity but cannot retroactively make two
different resolutions identical. OpenRouter can also route an alias to an
upstream backend that is not exposed in the response; the requested route and
dated provider capability snapshot are fingerprinted, but an undisclosed
upstream implementation cannot be.

## Controlled comparison protocol

For a homogeneous single-vs-multi comparison, keep these fixed:

1. task-set revision and repetition policy;
2. provider model ID and model-profile snapshot;
3. temperature, completion budget, reasoning control, and action transport;
4. prompt/tool/policy kernel fingerprint except for the declared architecture
   prompt or routing policy;
5. compaction mode and context budget;
6. Docker image digest, setup commands, test commands, step/test/time/cost
   bounds, and evaluator version.

Use a new session and campaign name for kernel-v2 experiments. The published
`final_v1` rows are historical evidence produced by the earlier executable
kernel. They must not be pooled with kernel-v2 rows or silently regenerated;
compare them as separate protocol versions.

The design follows the fairness concern in Xu et al.,
["Rethinking the Value of Multi-Agent Workflow: A Strong Single Agent Baseline"](https://arxiv.org/abs/2601.12307):
workflow architecture should be compared against a strong baseline while
holding model and test-time controls explicit. Repository tools, sandbox
integrity, and accounting are shared controls here, not benefits assigned to
one architecture.

## Examples

Use one shared effort across architectures:

```bash
python -m src.benchmark.runner \
  --task-id parse_pr_227 \
  --agent single \
  --model openai/gpt-5.6-terra-pro \
  --reasoning-effort high \
  --session kernel_v2_single_high
```

The v1 bundled registry does not currently mark any benchmark route as having
a verified exact-token guarantee, so published comparisons should use
`--reasoning-effort`. The `--reasoning-max-tokens` interface is retained for a
future reviewed profile with `supports_reasoning_max_tokens=True`; it rejects
known-but-unverified, unknown, and local routes instead of silently treating a
translated provider hint as exact.

Do not pass both reasoning flags. For a primary architecture comparison, do
not add `--role-model`, `--role-reasoning-effort`, or developer-escalation
overrides; those intentionally define a different, heterogeneous experiment.
