# Orchestrator Ablation Results

This living note records matched experiments on the dynamic multi-agent
orchestrator. The primary outcome is evaluator `task_success`, not the agent's
internal status.

## 2026-07-11: Phase Guidance vs Enforced Invariants

Task and controls:

- task: `sqlparse_pr_768`;
- model: `z-ai/glm-5.2` through OpenRouter;
- action transport: `text_json`;
- global limit: 75 steps;
- identical visible, hidden semantic, compatibility, and regression checks.

Variants:

- `multi-orch`: free dynamic supervisor, apart from rejecting redundant
  planner calls after a shared plan exists;
- `multi-orch-guided`: adds the evidence-derived current phase and recommended
  next role to the supervisor context, but does not enforce the recommendation;
- `multi-orch-guarded`: enforces `planner -> developer -> tester -> reviewer ->
  finish`, routes failed tests back to development, and caps late role episodes
  so that verification and review retain a step reserve.

| Variant | Runs passed | Steps | Mean tokens | Mean cost | Regressions |
| --- | ---: | ---: | ---: | ---: | ---: |
| `multi-orch` repeat baseline | 0/2 | 75, 75 | 283,222 | $0.2978 | 0, 5 |
| `multi-orch-guided` | 0/1 | 75 | 300,982 | $0.3188 | 5 |
| `multi-orch-guarded` | 2/2 | 54, 57 | 189,435 | $0.1973 | 0, 0 |

The guided run still allocated 50 role steps to the developer, six to the
tester, and zero to the reviewer. It exhausted the global budget with five
visible regressions. A prompt-level recommendation therefore did not correct
the supervisor's macro-routing failure.

The guarded runs allocated 15/16 developer steps, 16/18 tester steps, and 6/4
reviewer steps. Both repaired an initially incorrect patch, passed the full
visible suite and required hidden checks, and produced no regressions.

This is preliminary evidence from one task, not a general pass-rate estimate.
It does show that the improvement came from enforced workflow and budget
invariants rather than from merely describing the desired workflow more
clearly to the supervisor model.

Result sessions:

- `hardening_sqlparse_768_orch_fullcmdfix_20260711`;
- `orch_variants_sqlparse_768_glm_textjson_75_20260711`;
- `orch_guarded_repeat_sqlparse_768_glm_textjson_75_20260711`.

## 2026-07-11: First Large-Repository Probe

The guarded architecture was run on `sqlglot_pr_ed58b9bd`, a validated large
task (about 74k source LOC and 178 source files) that requires parsing and
generating ClickHouse parametric `groupConcat(separator)(value)` syntax.

Controls:

- model: `z-ai/glm-5.2` through OpenRouter;
- action transport: `text_json`;
- global limit: 100 steps and six test iterations;
- session: `guarded_large_sqlglot_groupconcat_glm_textjson_100_20260711`.

Outcome: unsuccessful (`task_success=false`). The visible ClickHouse suite and
compatibility checks passed with zero regressions, but no patch was produced and
the hidden semantic test failed. The run ended in human handoff after 99 steps,
474,519 tokens, 552.52 seconds, and $0.4797.

Role allocation exposed a new phase-starvation failure:

| Role | Steps |
| --- | ---: |
| planner | 70 |
| developer | 9 |
| tester | 4 |
| reviewer | 0 |

The guarded policy prevents moving to development until a shared plan exists,
but it does not cap cumulative planner retries. The planner repeatedly exhausted
its ten-step episode without reporting or recording a plan. Once a plan finally
existed, the adaptive reserve correctly shortened late worker episodes, but only
nine developer steps remained and no edit was made.

This distinguishes two budget invariants:

1. reserve verification/review after development (already enforced);
2. reserve implementation budget before allowing repeated planning (missing).

A next guarded version should cap cumulative planning, accept a useful planner
auto-report as sufficient handoff even without `set_plan`, and force transition
to development after one or two planner episodes. The setup command also failed
to install the checkout editable because setuptools-scm could not infer a
version without Git metadata, although source-tree imports and all scoped tests
still ran; future sqlglot runs should provide a pretend version during setup.

## 2026-07-11: Guarded v2 on the Same Large Task

Guarded v2 implemented the proposed phase-budget changes and repeated the same
`sqlglot_pr_ed58b9bd` experiment with identical model, transport, 100-step
limit, and evaluator. It also supplied a setuptools-scm pretend version, so the
editable setup completed without the earlier version error.

The planner auto-report was promoted into the shared plan after its first
episode. The new allocation was:

| Role | v1 steps | v2 steps |
| --- | ---: | ---: |
| planner | 70 | 10 |
| developer | 9 | 43 |
| tester | 4 | 27 |
| reviewer | 0 | 6 |

The phase-starvation fix therefore worked as intended. The task still failed:
the agent used all 100 steps, passed visible and compatibility checks, failed
the required hidden semantic test, and produced no production-code edit. The
only changed file was a new focused test. Developer episodes spent their budget
on search and inspection rather than `edit_file`; late instructions became
increasingly prescriptive but the worker still did not execute an edit.

v2 used 863,220 tokens, 106 model calls, 1,637.19 seconds, and $0.9642. This is
evidence that increasing available worker budget cannot by itself fix weak
action execution. The next ablation should target the developer interface:

- give the developer a compact structured code map from planner findings;
- require an edit-or-blocked report within the first few developer steps;
- detect inspection-only developer episodes and stop/retry them early;
- consider native tools or a stronger code-editing model for the developer
  while keeping the same guarded supervisor policy.

Result session:
`guarded_v2_large_sqlglot_groupconcat_glm_textjson_100_20260711`.
