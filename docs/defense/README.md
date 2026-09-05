# Defense materials — September 5, 2026

The defense covers the frozen master study at commit
`8da3f3f890340caac306af062e2adc21acfdb943` and a separately labeled kernel/v2
extension. The presentation and notes are in English.

## Presentation

- [Current presentation, v9](presentation/output/Repository_Level_Agent_Evaluation_Defense_EN_v9.pptx): 26 slides.
- [Current speaker notes, v9](presentation/output/Speaker_Notes_EN_v9.md).
- [Editable presentation builder](presentation/build_deck.mjs).
- [Previous delivery, v7](presentation/output/Repository_Level_Agent_Evaluation_Defense_EN_v7.pptx): retained as the builder's typography reference.

Slides 19–20 compare the project with Xu et al. and identify differences in
methodology and cost accounting. Slide 21 states the hypothesis outcomes with
numerical evidence. Slide 25 provides visible, clickable paper references and
scope caveats for Xu et al., Tran and Kiela, and Agentless. Full citations and
explanations are also included in the speaker notes.

## New results

Both additional DeepSeek repetition sessions completed: 48 new scored
observations and no new infrastructure failures. The final worker finished at
01:05 on September 5 in Europe/Belgrade.

| Attempt | Single success | Graph success | Single cost/success | Graph cost/success |
| --- | ---: | ---: | ---: | ---: |
| Original | 9/12 | 11/12 | $0.0409 | $0.0386 |
| Repeat 1 | 10/12 | 10/12 | $0.0187 | $0.0306 |
| Repeat 2 | 10/12 | 11/12 | $0.0229 | $0.0285 |
| All attempts | 29/36 | 32/36 | $0.0271 | $0.0326 |

Graph solved 10/12 tasks in all three attempts, versus 7/12 for single. Both
architectures solved the same 11 distinct tasks at least once. Graph cost 33.0%
more overall and 20.5% more per successful outcome. These are three attempts on
12 previously used tasks, not 36 independent tasks or a held-out benchmark.

H1, that single agents are more cost-effective on simple tasks, is supported
descriptively by the master core-small study; the DeepSeek extension has mixed
cost-efficiency outcomes across attempts. H2, that multi-agent benefits grow
with task complexity, is not supported by the master results and is not tested
by these small-task repetitions. Neither finding establishes a universal
architecture ranking.

- [Repetition audit and metric definitions](../../experiments/plans/defense_repeats_20260904/analysis/audit_repetitions.md).
- [Machine-readable results](../../experiments/plans/defense_repeats_20260904/analysis/audit_repetitions.json).
- [Recorded input and dependency comparison](../../experiments/plans/defense_repeats_20260904/analysis/provenance_review.md).
- [Execution protocol](../../experiments/plans/defense_repeats_20260904/PROTOCOL.md).
- [New-task preparation history](../../experiments/plans/defense_new_tasks_20260904/PREPARATION_STATUS.json): this proposed campaign did not produce a new-task model comparison.

Automated correctness and AI/agent proxies are included. No matching validated
quality-rubric scores or quantitative local-LLM pilot are available; the slides
state these limitations explicitly.

The same archive update also retains the GPT-5.1 Codex Mini observations: the
50-step comparison has 36 scored runs, while the later 75-step multi-agent block
has 23 of 24 scored observations and is missing `croniter_pr_235` for graph.
The latter is an incomplete extension and is not used as a complete comparison
in the presentation. Infrastructure failures remain in their separate journal.

## Frozen master evidence

[SOURCE.json](SOURCE.json) identifies the selected tracked master files in
`source/`. This is a source evidence snapshot, not a complete runnable checkout.
[DEFENSE_EN.md](DEFENSE_EN.md) is the earlier master-focused preparation guide;
use the v9 notes and repetition audit for the latest extension conclusions.
`kernel_v2_interim_snapshot.json` is an earlier snapshot, retained as historical
evidence, not the final repetition report.

Historical records and delivered speaker notes retain paths from the original
machine. Paths ending in `defense-master/...` map to this directory; paths
under `repo-level-dev-agent-eval/...` map to the repository root. These paths
describe provenance rather than portable hyperlinks.

Reproduce the read-only audits from the repository root:

```bash
python3 docs/defense/audit_master.py
python3 experiments/plans/defense_repeats_20260904/analysis/audit_repetitions.py
```

The scripts read archived records and write their audit outputs; they do not
run benchmark tests or call an LLM. The repetition script also consumes the
saved `provenance_review.json` comparison.

Git stores files rather than empty directories. Eighteen setup-artifact trees
in the 72-observation selection were empty when audited. A fresh checkout may
therefore report their directory existence as false; the saved provenance
review retains their empty-tree fingerprints. Do not place `.gitkeep` files
inside those trees, because that would change the evidence being hashed.

## Rebuilding slides

The builder requires the Codex Presentations skill, `@oai/artifact-tool`, Node.js
and Python. It is not part of the benchmark runtime dependencies. Expose the
artifact-tool package through normal Node module resolution, set
`PRESENTATIONS_SKILL_DIR`, `RUNTIME_NODE_MODULES` and optionally
`PYTHON_EXECUTABLE`, then run:

```bash
DECK_REVISION=v10 node docs/defense/presentation/build_deck.mjs
```

Choose a fresh revision for each delivery; the finalizer preserves existing
outputs. The builder uses the included frozen master evidence and current
repetition audit. Runtime paths are configurable; local build caches and
dependencies are ignored by Git.
