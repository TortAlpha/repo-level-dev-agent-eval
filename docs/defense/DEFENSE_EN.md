**Defense Preparation: Repository-Level Dev Agent Evaluation**

Review date: September 4, 2026. Version covered: `origin/master`, commit `8da3f3f890340caac306af062e2adc21acfdb943`. The local `master` branch still points to `5d9f0af` (`init`); the working branch is `kernel/v2`. The ongoing experiments belong to a different version. Preparing this guide did not change the working branch, project code, or experiments.

Selected source files from the reviewed version are saved in `source`, alongside an independent [results audit](/Users/tortalpha/.codex/visualizations/2026/09/04/01a06dd1-979e-7b72-a300-c6b646057748/defense-master/audit.json) and its [recalculation script](/Users/tortalpha/.codex/visualizations/2026/09/04/01a06dd1-979e-7b72-a300-c6b646057748/defense-master/audit_master.py). This is a source snapshot for reference, not a complete runnable checkout. The figures below were verified against saved records; model experiments and benchmark task tests were not rerun.

**1. What the project contributes**

The main contribution is a working experimental system for comparing how LLM agents are organized when modifying an existing repository. It includes task preparation, several agent architectures, file tools restricted to the working repository, command and test execution in Docker, final evaluation with hidden tests, resource accounting, and result analysis.

The project does not train a new neural network. Its AI component is the control of an existing model: action selection, context allocation, role specialization, use of test feedback, error repair, and stopping decisions.

The main hypothesis is that a single agent is more cost-effective on simple tasks, while a multi-agent workflow may become more useful as complexity increases. A hypothesis is a testable assumption; failing to confirm it does not make the project unsuccessful.

The defensible conclusion is: **in the recorded GLM-5.2 experiment, the multi-agent systems showed no convincing success-rate advantage and cost more.** This does not establish that single-agent systems are always better or that multi-agent systems are useless. The experiment compares specific implementations with different control policies.

Connection to prior work: Xu et al. investigate a strong single-agent baseline for reproducing homogeneous multi-agent workflows across several task categories, not only programming. This project's focus is the development loop in an existing repository, using tools and tests. It extends the research question rather than directly reproducing their OneFlow algorithm. [Xu et al., 2026](https://arxiv.org/abs/2601.12307).

Tran and Kiela provide useful context for controlling test-time computation and context use. Their results concern multi-hop reasoning under matched reasoning-token budgets, so they cannot be transferred directly to software development. [Tran and Kiela, 2026](https://arxiv.org/abs/2604.02460).

**2. A two-minute explanation of the system**

“The input consists of an initial repository state, a task description, and an available test command. The agent reads files, searches for relevant code, plans a change, edits the code, and tests the result. It can repeat this cycle within its budget. After it finishes, a separate evaluator checks the resulting solution with visible and hidden tests and checks for regressions. We compare success and resource use when this process is managed by a single agent or by several specialized roles.”

```text
Repository + task description + visible tests
                       ↓
     single / multi-graph / guarded orchestrator
                       ↓
     model action → validation → tool → observation
            ↑_________________________________|
                       ↓
     patch → external evaluator → tests and regressions
                       ↓
     outcome + cost + tokens + steps + traces
```

A model and an agent are different concepts. The model generates a response; the agent is the program that uses model responses to act in an environment. Several agents can use the same model backend while differing in prompts, contexts, permissions, and workflow responsibilities.

The multi-agent roles in this project execute **sequentially**. The experiment does not measure development speedups from parallel workers editing independent checkouts.

**3. The architectures to understand**

| Variant | How it works | What it tests |
|---|---|---|
| `single` | One loop and context handle research, implementation, testing, and repair | A strong baseline system |
| `multi-graph` | Fixed transitions through planner → developer → tester → reviewer, returning to the developer when necessary | Role separation without an LLM supervisor |
| `multi-orch-guarded` | A supervisor chooses delegations; code constrains transitions and reserves a verification budget | Adaptive coordination with safeguards against poor step allocation |

The single-agent baseline limits prolonged research without progress and requires an additional compatibility check before completion. This is why it is described as “strengthened”: the baseline was developed into a capable system rather than left weak to favor multi-agent execution.

The multi-agent roles have different permissions. The planner reads and plans; the developer changes production code; the tester runs checks and can add new tests; the reviewer inspects changes and reports concerns. The graph's finalizer is an ordinary function, not a separate LLM role. Roles share a repository and a global budget but receive separate contexts. Their reports are returned to the shared state.

Additional variants—`single-decomposed`, unguarded and guided orchestration, SWE-agent adapters, and role-specific models—exist as extensions or ablations. Their presence in the code does not mean every variant participated in every primary comparison.

Understand the possible mechanisms: roles can reduce the burden on an individual context and separate responsibilities, but they require information transfer and additional calls. When roles use the same model, their errors may be correlated. These are possible explanations of behavior, not proven causes of each outcome.

Source references: [single loop](/Users/tortalpha/.codex/visualizations/2026/09/04/01a06dd1-979e-7b72-a300-c6b646057748/defense-master/source/src/agents/single_agent.py:106), [graph and transitions](/Users/tortalpha/.codex/visualizations/2026/09/04/01a06dd1-979e-7b72-a300-c6b646057748/defense-master/source/src/agents/multi_agent.py:465), [roles and contexts](/Users/tortalpha/.codex/visualizations/2026/09/04/01a06dd1-979e-7b72-a300-c6b646057748/defense-master/source/src/agents/roles.py:80), [action validation](/Users/tortalpha/.codex/visualizations/2026/09/04/01a06dd1-979e-7b72-a300-c6b646057748/defense-master/source/src/agents/executor.py:427).

**4. Data: the proposal and the implemented benchmark**

| Original proposal | Evidence in master | How to explain it |
|---|---|---|
| Python repositories | The core uses Python repositories | Language and primary testing tools are controlled |
| At least 24 tasks, targeting 30 | 24 core tasks + 15 additional SWE-bench Pro tasks = 39 in the primary comparison | Distinguish the core dataset from the external extension |
| At least 12 small and 12 medium repositories | 12 tasks per group, from 11 small and 6 medium distinct upstream repositories | The literal commitment about repository counts was not met |
| Strict LOC and source-file thresholds | The later specification prioritizes LOC; exceptions and LOC ambiguity remain | Do not claim every task satisfies both original thresholds |
| Semi-automated collection | Instructions, a registry, a validator, and a SWE Pro importer exist; no general local PR collection script was found in tracked files | Validation, SWE Pro import, and execution are automated; not all local task discovery and preparation |
| PR-derived and, if needed, manual tasks | The final core uses real upstream changes, mainly PRs; no manually invented tasks are included | Manual task creation was a fallback option |
| Local LLM first | According to the author, the local model was too slow and insufficiently capable for the selected tasks; final runs use hosted GLM-5.2 | Describe local trials as a qualitative pilot; quantitative pilot results were not found in master |
| AI metrics | Outcome and behavioral metrics exist, including several rough proxies | Give the implemented definition of each metric |
| Solution-quality rubric | An optional LLM judge exists, but there was no new blinded review of the final patches | Do not claim demonstrated superiority in maintainability or safety |

The final manifest selects 41 tasks. Two local-large SQLGlot tasks did not produce a completed paired block: one failed during environment setup, and the other has only a single unsuccessful single-agent run. The primary comparison therefore contains 39 tasks. Not every architecture ran on every task: `24 × 3 + 15 × 2 = 102` scored runs.

The core contains 18 bug fixes and 6 feature tasks. All 15 tasks in the selected SWE Pro block are labeled as features. That block comes from three repositories: Ansible, OpenLibrary, and qutebrowser. It is an **adapted subset** of SWE-bench Pro: the importer limits test IDs and visible test files. The result of 9/15 must not be presented as an official score on the complete SWE-bench Pro benchmark.

The registry contains 302 candidate records, not 302 tasks in the final comparison. Multiple tasks from one repository are not fully independent examples: five medium tasks come from sqlparse.

There is a concrete size-definition issue. For small tasks, 9/12 satisfy the original combined LOC and file-count thresholds. For medium tasks, only 2/12 satisfy both thresholds when physical source lines are used. For example, parse has one source file and more-itertools has three. boltons and pydash are labeled medium despite `source_loc` values of 17,455 and 18,051; their nonblank LOC values are below 15,000. This could explain the labels, but using nonblank rather than physical lines is not explicitly established as the classification rule. The definition should not be silently changed after inspecting results.

Repository size is also different from change complexity: 19 of the 24 core reference patches modify one source file. Median patch size is 12 lines for small tasks and 5.5 lines for medium tasks. Consequently, the core provides limited evidence about difficult cross-module changes. Quantitative complexity fields are empty for imported SWE Pro tasks; the importer assigns the `large` label.

Sources: [manifest](/Users/tortalpha/.codex/visualizations/2026/09/04/01a06dd1-979e-7b72-a300-c6b646057748/defense-master/source/eval/task_sets/final_v1.json:7), [registry](/Users/tortalpha/.codex/visualizations/2026/09/04/01a06dd1-979e-7b72-a300-c6b646057748/defense-master/source/repositories/collection.csv:1), [size definition](/Users/tortalpha/.codex/visualizations/2026/09/04/01a06dd1-979e-7b72-a300-c6b646057748/defense-master/source/docs/spec.md:90), [collection instructions](/Users/tortalpha/.codex/visualizations/2026/09/04/01a06dd1-979e-7b72-a300-c6b646057748/defense-master/source/repositories/COLLECT_INSTRUCTION.md:210), [SWE Pro importer](/Users/tortalpha/.codex/visualizations/2026/09/04/01a06dd1-979e-7b72-a300-c6b646057748/defense-master/source/src/benchmark/import_swebench_pro.py:159).

**5. Why hidden tests matter more than an agent's completion claim**

Visible tests are available during task execution. They provide feedback and check existing behavior, but they do not necessarily cover the new requirement completely. Hidden tests are withheld from the agent during execution and are used by the external evaluator afterward.

Distinguish two groups:

- **Semantic / fail-to-pass:** tests of the required new behavior; they should fail on the base state and pass on the reference solution.
- **Compatibility / pass-to-pass:** tests of preserved behavior; they should pass on both the base state and the reference solution.

Therefore, “all hidden tests must fail on the base state” is incorrect for compatibility tests. For older local tasks without a separate compatibility command, the system can reuse the visible-suite result as `hidden_compat`; this is not an additional independent secret test suite. PR-parity checks can be diagnostic because a valid solution need not reproduce upstream details that the task description does not require.

The validator supports base/reference checks, but the existence of a validator does not replace archived validation logs. This review did not rerun every test oracle.

Under the standard evaluation configuration, success is defined as:

`task_success = visible_passed AND required_hidden_passed AND no_regressions AND NOT oracle_tampered`

`status="solved"` is the agent's internal execution status. It does not replace `task_success`; for example, the agent may reach a stopping limit while leaving a correct patch. In the 102 saved primary runs, these two indicators disagree in 30 cases, further illustrating the need for an external outcome measure.

A regression is detected when a test ID disappears from the set of previously passing tests. This gives a concrete check rather than relying on the agent's own assessment.

A useful example is `h11_pr_181`: pathological Content-Length parsing, a starting state before the upstream fix, visible tests, and a separate semantic check. The registry entry and run record can illustrate the process without requiring a detailed explanation of HTTP.

Sources: [success formula](/Users/tortalpha/.codex/visualizations/2026/09/04/01a06dd1-979e-7b72-a300-c6b646057748/defense-master/source/src/benchmark/evaluation.py:43), [task validation](/Users/tortalpha/.codex/visualizations/2026/09/04/01a06dd1-979e-7b72-a300-c6b646057748/defense-master/source/src/benchmark/validate.py:175), [hidden-suite types](/Users/tortalpha/.codex/visualizations/2026/09/04/01a06dd1-979e-7b72-a300-c6b646057748/defense-master/source/src/benchmark/collection.py:72).

**6. Metrics: definitions to know**

| Metric | Implemented meaning in master | Limitation |
|---|---|---|
| Task success rate | Evaluator-successful runs / evaluated runs | Primary measure of the final outcome |
| Resolved@1 | Outcome of the first run of a task within the selected records | One final run per configuration; not a repeated-sampling pass@k estimate |
| Visible / hidden pass rate | Fraction of runs in which the entire corresponding suite passed | Not the percentage of individual test cases passed |
| Regression rate | Fraction of runs that lose baseline passing tests | Depends on the completeness of the baseline suite and its report |
| Cost per solved task | Total cost of all selected attempts / number of solved tasks | Not average cost over successful attempts only |
| Steps / iterations / LLM calls | Actions / usually test runs / model calls | Different units of work |
| Patch validity rate | Fraction of runs with nonempty `changed_files` | A patch-production proxy, not evidence that the diff is valid |
| Tool-use validity rate | Valid action events / (valid + invalid events), using predefined event lists | Not every event type enters the denominator |
| Hallucinated references | Mean number of invalid, malformed, and no-action events | Does not directly check invented APIs or files |
| Repair success rate | Visible-pass fraction among runs with iterations > 1 | Does not require evidence of an earlier failure |
| Test-overfitting rate | Final-failure fraction among visible-pass runs | A gap between visible and final outcomes, not proof of overfitting |
| Quality score | Optional single LLM-judge score from 0 to 5 based on task and diff | Five criteria in the prompt, not five independently verified scores |

These proxies can support diagnosis when their limits are stated. A variable name alone does not establish that the intended scientific concept was measured.

In master, `duration_s` measures `agent.run`, including its setup, rather than the entire pipeline with baseline checks and subsequent independent evaluation. Total token usage includes repeatedly submitted contexts and is not the number of unique tokens in the task.

According to saved counters, provider billing covers all recorded LLM calls in each of the 102 primary runs; those values are used in the results table. This does not prove complete accounting of failed HTTP attempts or the entire provider invoice. The general fallback in the code is a token-based estimate when billing information is incomplete. Across the full history, $20.9396 is the sum of **available** cost records: 170 of the 291 historical records contain neither `cost_usd` nor provider-reported cost. It should not be presented as the verified total cost of development or an exact account invoice.

Sources: [metric formulas](/Users/tortalpha/.codex/visualizations/2026/09/04/01a06dd1-979e-7b72-a300-c6b646057748/defense-master/source/src/metrics/compute.py:114), [cost selection](/Users/tortalpha/.codex/visualizations/2026/09/04/01a06dd1-979e-7b72-a300-c6b646057748/defense-master/source/src/metrics/records.py:125), [quality judge](/Users/tortalpha/.codex/visualizations/2026/09/04/01a06dd1-979e-7b72-a300-c6b646057748/defense-master/source/src/metrics/quality.py:36).

**7. Results to present**

| Block | Single | Multi-graph | Multi-orch-guarded | Step cap |
|---|---|---|---|---|
| Core small, 12 tasks | 11/12; $0.3765 | 11/12; $0.7642 | 11/12; $0.9707 | 50 for all |
| Core medium, 12 tasks | 11/12; $0.4764 | 10/12; $0.7274 | 10/12; $0.9331 | Single 50, multi 75 |
| SWE Pro subset, 15 tasks | 9/15; $3.3077 | 9/15; $3.4412 | Not run | Single 50, graph 75 |

Across the 39 shared tasks, single resolves 31/39 (79.5%) and graph resolves 30/39 (76.9%). Their costs are $4.1605 and $4.9328. Graph costs 18.6% more; cost per solved task is approximately $0.1342 for single and $0.1644 for graph. The pooled table is secondary because it combines different budget policies.

Paired outcomes are especially informative: both systems solved 29 tasks, neither solved 7, only single solved 2, and only graph solved 1. There are just three disagreements, so this does not establish a statistically convincing success-rate advantage. All 15 paired outcomes in SWE Pro were identical.

Within the feature subset, graph resolves 14/21 versus 13/21 for single. The difference comes from one task, `sqlparse_pr_749`; this is not an independent confirmation that multi-agent systems perform better on complex tasks. Task type, repository size, and benchmark source are mixed in this comparison.

In the matched dataset, 20 tasks had been used in exploratory runs and 19 were held out from system development. This does not mean those 19 tasks were absent from the model's pretraining data. Development exposure and possible pretraining contamination from public code are separate issues.

The figure shows 95% Wilson intervals for success proportions. These are not a paired comparison test and do not account for dependencies among tasks from the same repository. Overlapping intervals alone do not prove the absence of a difference.

The primary comparison cost $10.9971; the campaign including the single local-large run cost $11.1070. All eight summary rows were independently checked against the raw records.

![Saved master benchmark results](/Users/tortalpha/.codex/visualizations/2026/09/04/01a06dd1-979e-7b72-a300-c6b646057748/defense-master/source/experiments/results/final_report/figures/01-task-success-rate.png)

Sources: [original report](/Users/tortalpha/.codex/visualizations/2026/09/04/01a06dd1-979e-7b72-a300-c6b646057748/defense-master/source/docs/final_benchmark_results.md:118), [CSV](/Users/tortalpha/.codex/visualizations/2026/09/04/01a06dd1-979e-7b72-a300-c6b646057748/defense-master/source/experiments/results/final_report/summary.csv), [independent audit](/Users/tortalpha/.codex/visualizations/2026/09/04/01a06dd1-979e-7b72-a300-c6b646057748/defense-master/audit.json).

**8. Key limitations and how to answer questions about them**

**Were the budgets equal?** Only core-small used the same maximum step cap for every architecture. On medium/SWE tasks, multi had 75 steps versus 50 for single. Even equal caps do not imply equal token usage, calls, or cost: a step may include a retry, and context compression makes separate model calls. The context budget limits the history retained, rather than the total computation used in an experiment. Results should therefore be compared alongside the resources actually consumed.

**Is the number of agents the only difference?** No. A shared executor and evaluator help control the experiment, but single has its own research and compatibility guards, while multi uses roles, permissions, a graph, or a supervisor. The experiment studies the selected workflow systems; it does not isolate agent count as a single variable.

**Why use only one model?** This keeps one important factor fixed in the main comparison. However, the conclusions are limited to that model and its settings. Supporting multiple backends in the code does not replace experiments across multiple models.

**Why not use a local model?** According to the author, the local model was too slow and produced solutions of insufficient quality. Suggested presentation wording: “In preliminary local trials, the model was too slow and did not produce sufficiently good solutions for the selected repository-level tasks. We therefore conducted the main architecture comparison using hosted GLM-5.2 through OpenRouter.” This is the author's qualitative observation from specific trials, rather than a quantitative result of the main benchmark or a conclusion about all local models. No measurements were found in master that establish the speed, success rate, or exact configuration of the local pilot. Do not guess the model name, quantization, or numerical results. Questions about the model and hardware should be answered from the author's actual experience. The instructor's comment that “a local LLM is sufficient” does not, by itself, prohibit a hosted model.

**Is there enough data?** The project meets the minimum task count, but this is not a statistical power calculation. With one run per configuration and small groups, the experiment cannot reliably distinguish small effects or estimate the stability of a stochastic agent. A lack of demonstrated advantage does not establish equivalence.

**Can everything be reproduced from Git?** The tables can be reproduced from the saved raw records. The full experiment cannot be reproduced from a clean clone alone: `repositories/tasks` and `repositories/repos` are excluded from Git, although task descriptions, hidden tests, and snapshots are present locally. Reproduction requires a separate frozen dataset archive with checksums and restoration instructions. A hash can verify that data matches, but it cannot restore missing content.

**Is hidden-test protection guaranteed?** The implementation separates hidden tests, restricts tools, disables networking, and restores the version-controlled test oracle. These measures do not prove complete isolation against every possible way of influencing the evaluator. In master, hidden-suite outcomes rely on exit codes without strictly checking the number of test cases actually executed; shell-based changes are not fully reflected in the internal revision counter. These are implementation limitations, rather than detected instances of result manipulation.

**What does kernel/v2 represent?** It is subsequent development of the runtime and evaluator. Its results belong to a different system version and cannot be added to the master tables as additional independent repetitions. A changed result after a harness update does not, by itself, identify the cause or prove that earlier passes were false. Begin the defense by explicitly identifying the version being discussed.

**Does overhead explain multi-agent's weaker results?** Higher recorded step counts and costs are observable. A claim that information loss during a handoff caused a particular failure requires trace analysis and controlled ablations. Present such causal explanations as hypotheses, rather than measured facts.

**What is the contribution?** The contribution lies in implementing and investigating workflows for autonomous repository-level development, preparing executable tasks, evaluating solutions with tests, accounting for resources, and identifying the limits of specialized roles. There is no need to claim a new LLM or a new fundamental algorithm.

**9. A 10–12-minute presentation outline**

| Time | Content | Key point |
|---|---|---|
| 0:00–1:00 | Problem and hypothesis | A more complex agent architecture needs to justify its additional cost through better results |
| 1:00–2:30 | Data | 24 core + 15 SWE Pro tasks; descriptions, base/reference states, visible/hidden tests |
| 2:30–4:30 | Three architectures | One loop, a fixed role graph, and a guarded supervisor |
| 4:30–6:00 | Evaluation and budgets | External `task_success`; 50/50 and 50/75 step-cap groups; cost and tokens |
| 6:00–8:00 | Results table and paired outcomes | 31/39 versus 30/39; three disagreements; graph costs 18.6% more |
| 8:00–10:00 | Limitations | One model, one run, task composition, proxy metrics, and dataset packaging |
| 10:00–12:00 | Conclusion and a short demonstration | The hypothesis was not supported under these conditions; show a result that can be checked |

Suggested opening:

“I investigated whether splitting an autonomous coding agent into specialized roles justifies the additional cost. I implemented a shared tool executor and a test-based evaluator, then built a single-agent loop, a fixed graph of roles, and guarded orchestration on top of them. The main experiment includes 102 runs across 39 tasks. In this configuration, multi-agent systems did not show a convincing improvement in success rate and were more expensive.”

Suggested closing:

“The main outcome is an executable comparison framework and a limited empirical finding: under the selected protocol, the additional architectural complexity did not pay off. Testing whether this finding generalizes requires repeated runs, other models, more demanding cross-module tasks, and tighter control of the computational budget.”

**10. What to show in the demonstration**

Open one completed task and its saved artifacts in advance. A lengthy live run is unnecessary for explaining how the system works.

1. The task's registry entry: repository, commit, description, and test commands.
2. A trace showing “read → edit → test,” followed by the final patch.
3. The agent's `status` alongside the external `task_success`, hidden-suite results, and regressions.
4. The final table broken down by task group, including cost per success.

One suitable example is `h11_pr_181`, single, run ID `b6bdb5f5-a144-4e58-b30b-608fd21491e4`. To illustrate why visible tests are insufficient, show an unsuccessful SWE Pro run: visible tests passed in every run in that group, while the final success rate was only 60%. Inspect the specific patch beforehand and explain only what it actually shows.

Suggested code-reading path: [runner](/Users/tortalpha/.codex/visualizations/2026/09/04/01a06dd1-979e-7b72-a300-c6b646057748/defense-master/source/src/benchmark/runner.py:501) → [single](/Users/tortalpha/.codex/visualizations/2026/09/04/01a06dd1-979e-7b72-a300-c6b646057748/defense-master/source/src/agents/single_agent.py:106) → [multi](/Users/tortalpha/.codex/visualizations/2026/09/04/01a06dd1-979e-7b72-a300-c6b646057748/defense-master/source/src/agents/multi_agent.py:465) → [executor](/Users/tortalpha/.codex/visualizations/2026/09/04/01a06dd1-979e-7b72-a300-c6b646057748/defense-master/source/src/agents/executor.py:427) → [evaluator](/Users/tortalpha/.codex/visualizations/2026/09/04/01a06dd1-979e-7b72-a300-c6b646057748/defense-master/source/src/benchmark/evaluation.py:43) → [metrics](/Users/tortalpha/.codex/visualizations/2026/09/04/01a06dd1-979e-7b72-a300-c6b646057748/defense-master/source/src/metrics/compute.py:114). You do not need to memorize the entire codebase; you need to be able to follow this path and explain each module's responsibilities.

**11. What to prepare before the defense**

Four things take priority. Identify the exact commit and use only its result tables. Explain the change from the local-first plan through the observed low speed and insufficient solution quality; prepare the model name, configuration, and actual pilot logs if they were retained. Prepare a frozen dataset package and restoration instructions, because a clone alone is insufficient. Make the presentation consistent about task and repository counts, the definition of LOC, budget settings, and the names of proxy metrics.

Also prepare one patch and its trace for a detailed walkthrough. If there is time for further experiments, repeated runs on held-out tasks under a matched budget are more useful than adding another role without a controlled experiment. Publish any new results as a separate version instead of silently changing the historical master table.

Do not present unverified promises as completed achievements. A strong defense rests on explaining clearly what was implemented, how it was measured, and how far the conclusions can reasonably extend.
