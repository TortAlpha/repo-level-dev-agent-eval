# Final Benchmark Figures

Generated from the frozen scored sessions in `experiments/results/runs.jsonl`.
The primary outcome is evaluator `task_success`.

## Main result

- Core small: all three architectures resolved 11/12 tasks. Single cost
  $0.3765; multi-graph cost
  $0.7642; guarded cost
  $0.9707.
- Core medium: single resolved 11/12, while both multi variants resolved 10/12.
- SWE-bench Pro: single and multi-graph both resolved 9/15,
  with identical per-task outcomes. Single cost $3.3077;
  multi-graph cost $3.4412.
- Bug fixes: single resolved 18/18;
  multi-graph resolved 16/18.
- Features: single resolved 13/21;
  multi-graph resolved 14/21.
  Their feature cost per success was nearly identical at
  $0.270 and
  $0.269.
- The evaluated multi-agent configurations did not improve aggregate resolve
  rate or cost efficiency over the strengthened single loop.

## Figures

1. `figures/01-task-success-rate` — resolve rate with 95% Wilson intervals.
2. `figures/02-cost-efficiency` — total cost and cost per successful task.
3. `figures/03-resource-use` — mean agent steps and tokens per task.
4. `figures/04-swe-task-outcomes` — paired SWE Pro outcomes and cost deltas.
5. `figures/05-task-type-split` — bug-fix versus feature outcomes and cost efficiency.

Each figure is available as PNG and SVG. Exact plotted values are in
`summary.csv` and `task_type_summary.csv`.

## Interpretation constraint

Core small used a common 50-step cap. Core medium and SWE-bench Pro used 50
steps for single and 75 for multi architectures. Blocks must be reported
separately; a pooled 39-task aggregate would mix different architecture
coverage and step policies.
