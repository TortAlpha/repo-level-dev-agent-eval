# Repository-Level Dev Agent Evaluation

A reproducible evaluation of single-agent and multi-agent LLM architectures on
repository-level software engineering tasks.

The project asks a practical question:

> Can agent architecture make autonomous repository development both cheaper
> and more successful than a strong single-agent loop?

The repository contains the agent implementations, Docker benchmark harness,
hidden evaluation, experiment runner, metrics pipeline, result artifacts, and a
local web console. The final primary comparison is complete.

The recorded `final_v1` results below are historical and remain immutable. The
current runtime is the separately fingerprinted
[pseudo-SWE-agent kernel v2](docs/agent_kernel_v2.md); new kernel-v2 runs must
use new sessions and must not be pooled silently with `final_v1`.

## Result at a Glance

The main benchmark contains **102 scored runs over 39 unique tasks**:

- 24 local tasks from small and medium Python repositories;
- 15 selected SWE-bench Pro tasks;
- one shared hosted model: `z-ai/glm-5.2`;
- a strengthened `single` baseline;
- deterministic `multi-graph`;
- phase-constrained `multi-orch-guarded`.

Across the 39 tasks evaluated by both single and graph:

| Architecture | Resolved | Success rate | Total cost | Cost/success |
| --- | ---: | ---: | ---: | ---: |
| **Single** | **31/39** | **79.5%** | **$4.1605** | **$0.1342** |
| Multi-graph | 30/39 | 76.9% | $4.9328 | $0.1644 |

Graph cost **18.6% more** and resolved one fewer task. On SWE-bench Pro, the
architectures had identical outcomes on all 15 tasks: both resolved **9/15**.

The final result does not support the hypothesis that multi-agent architecture
alone makes repository-level development both better and cheaper. The
strengthened single loop is the best default in the evaluated setting.

There is one narrow signal in favor of graph decomposition:

- bug fixes: single 18/18, graph 16/18;
- features: single 13/21, graph 14/21.

The feature difference comes from one core task. SWE-bench Pro outcomes were
identical, and the confidence intervals overlap substantially, so this is a
future research direction rather than evidence of a general advantage.

Read the complete interpretation, limitations, and cost accounting in
[Final Benchmark Results](docs/final_benchmark_results.md).

![Task success rate by benchmark block](experiments/results/final_report/figures/01-task-success-rate.png)

## What Was Built

This is an executable benchmark system, not only a project specification.

### Agent runtime

- immutable agent state and explicit structured actions;
- repository inspection, search, editing, shell, and test tools;
- text-JSON and native tool-call transports;
- fully accounted bounded context with deterministic checkpointing by default;
- model-generated summarization and context dropping as explicit ablations;
- deterministic reuse of previous searches after compaction;
- provider-native reasoning replay stored in durable agent state;
- per-run cost limits and campaign-wide sweep budgets;
- provider billing provenance and executable prompt/tool/policy fingerprints;
- canonical reasoning effort (`none` through `max`) or a capability-gated,
  mutually exclusive exact reasoning-token budget, with per-role settings
  recorded as heterogeneous ablations;
- LangSmith tracing with agent, role, model, and session metadata.

### Agent architectures

- **`single`** — one loop performs research, planning, implementation, testing,
  and repair;
- **`single-decomposed`** — single-agent execution with explicit subtasks and
  repair cycles;
- **`multi-graph`** — deterministic planner, developer, tester, and reviewer
  routing;
- **`multi-orch`** — model-supervised role delegation;
- **`multi-orch-guided`** — orchestration with phase guidance;
- **`multi-orch-guarded`** — orchestration with enforced phase invariants,
  quotas, and transition guards;
- **`swe-agent` / `multi-swe`** — adapters for external SWE-agent-based
  execution.

The multi-agent runtime also supports fixed role-specific models and adaptive
developer escalation to a stronger model.

The external SWE-agent modes are separately fingerprinted all-online
ablations. Comparable built-in runs keep the pseudo-SWE-agent action kernel;
the adapter fingerprints and revalidates its Python runtime and complete
resolved dependency closure, sanitizes inherited Python/LiteLLM injection
settings, and rejects arbitrary passthrough flags, missing usage trajectories,
and reasoning controls it cannot faithfully forward.

### Strengthened single-agent baseline

The final single baseline was improved during system development to avoid
comparing multi-agent workflows against an artificially weak loop:

- warning and hard limits for extended read-only research;
- stricter progress limits after an implementation plan exists;
- repeated-search reuse across context compaction;
- a deterministic ledger of recent repository evidence;
- a focused compatibility gate after the latest edit;
- explicit handoff when repeated no-progress limits are ignored.

Legacy variants remain reproducible through CLI flags and are treated as
ablations rather than final baseline runs.

### Benchmark and evaluation

- versioned task collection and frozen final task manifest;
- isolated Docker workspace for every attempt;
- network access during dependency setup only;
- visible tests available to the agent;
- hidden semantic and compatibility suites;
- before/after regression evaluation;
- scalable read-only test-root protection plus exact root-level runner mounts;
- shell-side mutation auditing with durable attempted-oracle-tamper metrics;
- final reconstruction from `HEAD + archived patch`, with section-aware
  pytest-config attestation, a pristine frozen dependency environment, and a
  trusted direct/wrapper pytest bootstrap;
- exact hidden-fixture manifests and single-invocation, fail-closed JUnit
  execution evidence kept outside the writable checkout;
- infrastructure failures recorded separately from task failures;
- archived patches, run metadata, model usage, and experiment fingerprints;
- resumable sweeps with deterministic ordering, session/campaign leases, and
  parent-to-child input hash contracts.

### Analysis and web console

- outcome, cost, token, duration, action, and role metrics;
- session and campaign aggregation;
- task-type, repository-size, model, difficulty, and architecture breakdowns;
- generated CSV, PNG, and SVG benchmark figures;
- local React console with sessions, active jobs, logs, run details, role usage,
  costs, and metric explanations.

## Architecture Overview

All built-in architectures use the same executor, workspace, sandbox,
evaluation, and result-recording layers.

```text
task + repository
        |
        v
architecture policy
  |        |              |
  |        |              +--> guarded supervisor -> role episodes
  |        +-----------------> deterministic role graph
  +--------------------------> strengthened single loop
        |
        v
structured actions -> policy validation -> Docker workspace
        |
        v
visible tests -> hidden evaluation -> regression check
        |
        v
runs.jsonl + metrics + patches + traces + web console
```

The primary multi-agent workflow separates the following responsibilities:

```text
planner -> developer -> tester -> reviewer -> final decision
```

Role separation is an experimental variable. It does not receive privileged
repository access or hidden-test information.

## Final Benchmark Design

The primary outcome is evaluator **`task_success`**, not the agent's internal
`status`. A run succeeds only when its final workspace passes visible tests and
all required hidden semantic or compatibility suites.

| Block | Tasks | Architectures | Step policy |
| --- | ---: | --- | --- |
| Core small | 12 | single, graph, guarded | 50 for all |
| Core medium | 12 | single, graph, guarded | single 50; multi 75 |
| SWE-bench Pro | 15 | single, graph | single 50; graph 75 |

Core-small is the strict equal-step pilot. Multi-agent architectures receive a
higher step cap on core-medium and SWE-bench Pro because role handoffs consume
workflow steps. Results are reported by block; pooled results are secondary.

### Block results

| Block | Single | Multi-graph | Multi-orch-guarded |
| --- | ---: | ---: | ---: |
| Core small | **11/12**, $0.3765 | **11/12**, $0.7642 | **11/12**, $0.9707 |
| Core medium | **11/12**, $0.4764 | 10/12, $0.7274 | 10/12, $0.9331 |
| SWE-bench Pro | **9/15**, $3.3077 | **9/15**, $3.4412 | not run |

The primary 102-run comparison cost **$10.9971**. Total recorded spend for the
`final_v1_main` campaign, including an incomplete local-large attempt, was
**$11.1070**. All 291 exploratory and final runs currently stored in
`experiments/results/runs.jsonl` account for **$20.9396** under the repository's
effective-cost rule.

The two selected local-large SQLGlot tasks did not produce a paired final
comparison: one task failed environment setup because editable installation
could not derive package metadata without a Git checkout, and the second has
only an unpaired single result. These attempts are excluded from the primary
tables.

## Requirements

- Python 3.11 or newer;
- Docker;
- an OpenRouter API key or an OpenAI-compatible local model server;
- Node.js 20.19+ or 22.12+ for the optional web console frontend.

## Installation

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[graph,dev]" pytest
cp .env.example .env
```

For OpenRouter, set at least:

```dotenv
MODEL_PROVIDER=openrouter
OPENROUTER_API_KEY=...
OPENROUTER_MODEL_NAME=z-ai/glm-5.2
```

For a local OpenAI-compatible endpoint:

```dotenv
MODEL_PROVIDER=local
LOCAL_API_URL=http://localhost:1234/v1
LOCAL_API_KEY=dummy
LOCAL_MODEL_NAME=your-model-name
```

Docker networking is enabled only for setup by default. The model-controlled
agent phase runs offline.

## Running One Benchmark Task

Validate the setup without calling the model:

```bash
python -m src.benchmark.runner \
  --task-id parse_pr_227 \
  --provider openrouter \
  --agent single \
  --model z-ai/glm-5.2 \
  --session example_single \
  --max-steps 50 \
  --dry-run
```

Remove `--dry-run` to execute the task. Runs are appended to
`experiments/results/runs.jsonl`; metrics and patches are stored alongside the
run ID.

Compare another architecture by changing the agent:

```bash
python -m src.benchmark.runner \
  --task-id parse_pr_227 \
  --provider openrouter \
  --agent multi-graph \
  --model z-ai/glm-5.2 \
  --session example_graph \
  --max-steps 75
```

Available architecture names can be listed with:

```bash
python -m src.benchmark.runner --help
```

## Running a Reproducible Sweep

Inspect the frozen core-small matrix without spending model credits:

```bash
python -m src.benchmark.sweep \
  --task-set final-v1 \
  --matrix core-small \
  --session example_core_small \
  --campaign example_campaign \
  --provider openrouter \
  --models z-ai/glm-5.2 \
  --max-total-cost-usd 10 \
  --plan-only
```

Remove `--plan-only` only when ready to execute the paid sweep. Interrupted
sessions can be resumed with the identical command plus `--resume`.

The exact final campaign commands are preserved in
[Final Benchmark Task Set](docs/final_benchmark_task_set.md).

## Metrics and Reports

List recorded sessions:

```bash
python -m src.metrics.report --list-sessions
```

Report one session by architecture:

```bash
python -m src.metrics.report \
  --session final_v1_core_small \
  --group-by agent_mode
```

Regenerate the final CSV files and figures:

```bash
python experiments/results/final_report/generate_figures.py
```

Generated artifacts:

- [`summary.csv`](experiments/results/final_report/summary.csv);
- [`task_type_summary.csv`](experiments/results/final_report/task_type_summary.csv);
- [`figures/`](experiments/results/final_report/figures);
- [`Final Benchmark Results`](docs/final_benchmark_results.md).

## Web Console

Build the frontend and start the same-origin local server:

```bash
cd web-console
npm ci
npm run build
cd ..
python -m src.console.server
```

Open [http://127.0.0.1:8765](http://127.0.0.1:8765).

The console can:

- group sessions by creation date and time;
- show completed and currently running jobs;
- stream available run logs;
- inspect task outcomes and hidden-suite results;
- break down model usage by multi-agent role;
- compare cost, tokens, steps, duration, and success;
- launch and stop local jobs and sweeps.

The API is bound to localhost by default. Cross-origin POST requests are
rejected unless `CONSOLE_ALLOWED_ORIGINS` is explicitly configured.

## Testing and Static Checks

```bash
python -m pytest
ruff check src eval tests
mypy src eval
```

Benchmark task tests under `repositories/tasks/**/tests` are not the project's
unit-test suite and are executed inside their task-specific sandboxes.

## Project Layout

```text
src/agents/                 agent loops, roles, actions, state, executor
src/benchmark/              task runner, sweep runner, evaluation, imports
src/metrics/                metrics, pricing, reports, quality scoring
src/console/                local console API and job management
repositories/collection.csv verified task registry
repositories/tasks/        repository snapshots, task text, hidden tests
eval/task_sets/             frozen versioned benchmark manifests
experiments/results/        run records, metrics, patches, sessions, figures
web-console/                React and TypeScript benchmark console
docs/                       specification, architecture notes, final report
tests/                      harness and architecture regression tests
```

## Reproducibility Notes

- Final task membership is frozen in
  [`eval/task_sets/final_v1.json`](eval/task_sets/final_v1.json).
- Final sessions record task-set, collection, pricing, harness, and settings
  fingerprints.
- Existing benchmark tests and hidden tests are read-only to agents.
- Attempted shell-side test-oracle changes remain recorded after automatic
  rollback and invalidate evaluator success.
- Infrastructure failures are stored separately and do not silently become
  task failures.
- Cost uses complete provider-reported billing when available; partial billing
  is combined with the full versioned static estimate as an explicit upper
  bound, while missing usage makes cost unknown rather than zero.
- New built-in runs fingerprint the exact prompts, action schemas, policy
  sources/configuration, model routes, and task text.
- Exploratory sessions and ablations are not mixed into final aggregates.

## Limitations

- one final attempt per task and architecture;
- one primary model family;
- mixed step caps outside the core-small pilot;
- 20 paired tasks had exploratory exposure before the final architecture
  freeze;
- the selected SWE-bench Pro tasks are all feature tasks;
- guarded orchestration was not evaluated on SWE-bench Pro;
- the local-large paired extension was not completed;
- final patches did not receive a new blinded maintainability and robustness
  review.

These limitations are discussed in detail in the
[final results report](docs/final_benchmark_results.md#threats-to-validity).

## Documentation

- [Final benchmark results](docs/final_benchmark_results.md)
- [Final benchmark task set](docs/final_benchmark_task_set.md)
- [Full project specification](docs/spec.md)
- [Short project specification](docs/spec_short.md)
- [Orchestrator ablation results](docs/orchestrator_ablation_results.md)
- [Role-specific and adaptive model routing](docs/role_model_routing.md)
- [Task benchmark observations](docs/task_benchmark_observations.md)
- [SWE-agent integration](docs/swe_agent.md)
- [Pseudo-SWE-agent kernel v2 and comparison protocol](docs/agent_kernel_v2.md)

## References

- Xu et al.,
  ["Rethinking the Value of Multi-Agent Workflow: A Strong Single Agent Baseline"](https://arxiv.org/abs/2601.12307),
  arXiv:2601.12307.
- Tran and Kiela,
  ["Single-Agent LLMs Outperform Multi-Agent Systems on Multi-Hop Reasoning Under Equal Thinking Token Budgets"](https://arxiv.org/abs/2604.02460),
  arXiv:2604.02460.
- Xia et al.,
  ["Agentless: Demystifying LLM-based Software Engineering Agents"](https://arxiv.org/abs/2407.01489),
  arXiv:2407.01489.

## Author

Roman Avanesov
