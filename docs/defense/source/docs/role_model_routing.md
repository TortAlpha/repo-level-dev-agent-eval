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
