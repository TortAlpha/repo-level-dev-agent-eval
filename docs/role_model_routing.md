# Role Model Routing and Cost Experiments

Built-in multi-agent modes use one shared model by default. This preserves all
legacy commands and makes old sessions reproducible:

```bash
python -m src.benchmark.runner \
  --task-id sqlparse_pr_768 \
  --agent multi-orch-guarded \
  --model z-ai/glm-5.2
```

The run record stores a `model_policy` with `type: shared` and the resolved
model for every role.

## Reasoning Policy

Reasoning is homogeneous by default too. `--reasoning-effort` accepts
`none`, `minimal`, `low`, `medium`, `high`, `xhigh`, or `max` and is applied to
the primary model and every role route:

```bash
python -m src.benchmark.runner \
  --task-id sqlparse_pr_768 \
  --agent multi-graph \
  --model openai/gpt-5.6-terra-pro \
  --reasoning-effort high
```

`--reasoning-max-tokens N` is a gated alternative for a route whose frozen
profile explicitly verifies that the provider preserves an exact hidden-
reasoning budget end to end. Merely accepting or translating the field is not
enough. It is mutually exclusive with effort and may not exceed the configured
completion budget. The bundled v1 profiles do not currently assert this
guarantee, so comparable runs should use effort.

Known routes are checked against the dated model-capability registry. Unknown
routes remain usable with effort controls and are marked capability-unknown in
`model_routes`, but exact token budgets fail closed for unknown and local
routes. Provider defaults, requested controls, and the registry version are
recorded in the run fingerprint.

Role-specific effort is an explicit heterogeneous ablation:

```bash
--role-reasoning-effort planner=low \
--role-reasoning-effort developer=high \
--developer-escalation-reasoning-effort xhigh
```

The escalation effort requires an escalation model. Do not use role-specific
reasoning in a primary homogeneous architecture comparison; if it is used,
report it as a separate model-policy condition.

## Fixed Role Models

Use repeatable `--role-model ROLE=MODEL` flags. Unspecified roles continue to
use the primary `--model`:

```bash
python -m src.benchmark.runner \
  --task-id sqlparse_pr_768 \
  --agent multi-orch-guarded \
  --model z-ai/glm-5.2 \
  --role-model developer=anthropic/claude-sonnet-4 \
  --role-model reviewer=openai/gpt-5.1-codex-mini
```

Supported roles are `orchestrator`, `planner`, `developer`, `tester`, and
`reviewer`. Role overrides apply only to built-in multi-agent modes.

## Adaptive Developer Escalation

An optional escalation model replaces the configured developer model after a
no-edit developer episode or after the configured failed-test threshold:

```bash
python -m src.benchmark.runner \
  --task-id sqlparse_pr_768 \
  --agent multi-orch-guarded \
  --model z-ai/glm-5.2 \
  --developer-escalation-model anthropic/claude-sonnet-4 \
  --developer-escalate-after-no-edit-episodes 1 \
  --developer-escalate-after-failed-tests 1
```

The escalation is recorded as a `model_escalation` action and in
`developer_escalations`. A no-edit episode routes directly from the cheap
developer to the escalation developer without paying for an intermediate test
run on an unchanged workspace.

## Cost Guard

`--max-cost-usd` stops a single agent after a model step, or a multi-agent run
between role episodes, when cumulative known token cost reaches the limit:

```bash
--max-cost-usd 0.50
```

The runner refuses a cost-capped run if any configured model is missing from
`static/pricing.csv`; silently treating an unknown price as zero would make
comparisons invalid.

## Recorded Metadata

Every new run records:

- `model_policy`: base model, resolved role models, escalation model, and
  thresholds;
- `role_usage`: calls, input/output/total tokens, and cost per role and model;
- `role_transports`: actual resolved action transport per role and model;
- `developer_escalations` and `max_cost_usd`;
- aggregate `cost_usd`, summed across all unique model instances.

The metrics report includes `Cost per successful run`, defined as total spend
divided by successful evaluator outcomes when every run has known cost.

## Sweep Example

Role flags are forwarded only to multi-agent combinations, so one sweep can
compare single and adaptive guarded configurations without breaking the single
run:

```bash
python -m src.benchmark.sweep \
  --tasks sqlparse_pr_768,sqlglot_pr_ed58b9bd \
  --models z-ai/glm-5.2 \
  --agents single,multi-orch-guarded \
  --developer-escalation-model anthropic/claude-sonnet-4 \
  --max-cost-usd 0.50 \
  --session adaptive_cost_comparison_v1
```

For system-level comparisons, report both task success and cost per successful
run. Keep the same task set, evaluator, cost ceiling, and repetition count.
