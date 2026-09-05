**Slide 1: Title**

My project evaluates single-agent and multi-agent LLM architectures for autonomous software development at repository level. The defense primarily covers origin/master at commit 8da3f3f. I will also show a separately labeled extension from kernel/v2. The question is whether specialized roles improve the final solution enough to justify their resource cost. The contribution is an executable evaluation framework and an empirical comparison, rather than training a new language model.

Sources:
/Users/tortalpha/Code/repo-level-dev-agent-eval/docs/defense/source/README.md
/Users/tortalpha/Code/repo-level-dev-agent-eval/docs/defense/source/docs/final_benchmark_results.md

**Slide 2: Research question**

The initial hypothesis was that a single agent would be more cost-effective on simple tasks, while specialized roles would become more useful as complexity increased. Existing research argues that a strong single-agent baseline can match homogeneous multi-agent workflows when computation and context are considered. This project examines that question in an existing repository, where the agent must find files, modify code, run tests, and repair failures. These are hypotheses and motivations, not conclusions established in advance.

Sources:
https://arxiv.org/abs/2601.12307
https://arxiv.org/abs/2604.02460
/Users/tortalpha/Code/repo-level-dev-agent-eval/docs/defense/source/docs/spec.md

**Slide 3: Repository development loop**

Each run starts from a task description and an initial repository state. The model proposes structured actions. File tools operate within the working repository, while shell commands and tests run in Docker. The agent receives observations and revises its plan or code. After the run, an external evaluator checks the final workspace and records the result. The agent runtime, executor, and evaluator are separate modules. This separation lets the experiment compare workflow policies while reusing much of the execution infrastructure.

Sources:
/Users/tortalpha/Code/repo-level-dev-agent-eval/docs/defense/source/src/benchmark/runner.py
/Users/tortalpha/Code/repo-level-dev-agent-eval/docs/defense/source/src/agents/executor.py
/Users/tortalpha/Code/repo-level-dev-agent-eval/docs/defense/source/src/benchmark/evaluation.py

**Slide 4: Benchmark composition**

The primary comparison contains 39 tasks and 102 scored runs. The core has 24 tasks from 17 Python repositories, split into 12 small and 12 medium tasks. These are tasks, not 24 distinct repositories. The extension contains 15 selected tasks from an adapted SWE-bench Pro subset. Two selected local-large SQLGlot tasks did not produce a completed paired comparison and are excluded. The task count met the project minimum, but it was not a statistical power calculation.

Sources:
/Users/tortalpha/Code/repo-level-dev-agent-eval/docs/defense/source/eval/task_sets/final_v1.json
/Users/tortalpha/Code/repo-level-dev-agent-eval/docs/defense/source/repositories/collection.csv
/Users/tortalpha/Code/repo-level-dev-agent-eval/docs/defense/source/docs/final_benchmark_results.md

**Slide 5: Task construction and validation**

The core reconstructs real upstream changes, mainly merged pull requests. The starting commit precedes the change. The task description explains the expected behavior without providing the solution diff. Visible tests should pass before the task. Semantic hidden tests should fail before the change and pass with the reference solution. Compatibility checks should pass in both states. Local task selection and descriptions involve manual preparation, while validation and execution are automated. An importer automates the SWE Pro adaptation.

Sources:
/Users/tortalpha/Code/repo-level-dev-agent-eval/docs/defense/source/repositories/COLLECT_INSTRUCTION.md
/Users/tortalpha/Code/repo-level-dev-agent-eval/docs/defense/source/src/benchmark/validate.py
/Users/tortalpha/Code/repo-level-dev-agent-eval/docs/defense/source/src/benchmark/import_swebench_pro.py

**Slide 6: Three architectures**

The single agent handles the complete cycle and includes guards against prolonged research without progress and a compatibility check before finishing. Multi-graph executes planner, developer, tester, and reviewer roles with deterministic routing. The guarded orchestrator delegates through an LLM supervisor, while code enforces phase constraints and reserves verification steps. The multi-agent roles execute sequentially and share the repository and global budget, with separate role contexts. These are concrete systems with different control rules, not an isolated test of agent count.

Sources:
/Users/tortalpha/Code/repo-level-dev-agent-eval/docs/defense/source/src/agents/single_agent.py
/Users/tortalpha/Code/repo-level-dev-agent-eval/docs/defense/source/src/agents/multi_agent.py
/Users/tortalpha/Code/repo-level-dev-agent-eval/docs/defense/source/src/agents/roles.py

**Slide 7: Evaluation and agent metrics**

The main outcome is task_success from the evaluator, not the agent saying it is solved. Success requires visible tests, every required hidden suite, no measured regressions, and no detected test-oracle tampering. Resolved at one summarizes the first final attempt. The system also records tool-use validity, repair behavior, tokens, calls, duration, and cost. Some named AI metrics are operational proxies. For example, patch validity measures whether files changed, and hallucinated references counts malformed or invalid actions. Those limitations are documented explicitly.

Sources:
/Users/tortalpha/Code/repo-level-dev-agent-eval/docs/defense/source/src/benchmark/evaluation.py
/Users/tortalpha/Code/repo-level-dev-agent-eval/docs/defense/source/src/metrics/compute.py
/Users/tortalpha/Code/repo-level-dev-agent-eval/docs/defense/source/src/metrics/records.py

**Slide 8: How final task PASS is decided**

An external evaluator checks the final repository state after the agent stops. In the reported experiments, PASS requires all four conditions on this slide. First, the visible suite is rerun and must pass. Second, every required hidden or withheld suite must pass; optional suites do not determine the combined hidden verdict. Third, the evaluator compares passing visible-test identifiers before and after the run. A baseline test that is no longer recorded as passing counts as a regression, including a removed, renamed, or skipped test. Fourth, detected test-oracle tampering invalidates the result. Kernel v2 additionally rejects protected setup-artifact tampering. These checks are independent of the agent reporting that it is done or requesting a handoff. A nonempty patch and an LLM quality-rubric score are not additional conditions in task_success. The code permits an unset regression count when regression checking is explicitly disabled; the reported comparisons use measured regression counts. Hidden compatibility results may reuse the visible suite, as they do in the DeepSeek repetition block, so they are not always independent test evidence. Infrastructure failures are recorded separately from ordinary task failures.

Sources:
/Users/tortalpha/Code/repo-level-dev-agent-eval/docs/defense/source/src/benchmark/evaluation.py
/Users/tortalpha/Code/repo-level-dev-agent-eval/src/benchmark/evaluation.py

**Slide 9: Model choice and execution budgets**

The author reports that preliminary local trials were too slow and produced insufficient solution quality for the selected repository tasks. This was a qualitative feasibility observation, without a quantified pilot table in the saved master artifacts. The primary comparison therefore uses hosted GLM-5.2 through OpenRouter. Core-small uses the same 50-step cap for all architectures. Medium and SWE Pro allow multi-agent systems 75 steps. The SWE Pro initial max_tokens setting is also unequal: 16,384 for single and 4,096 for graph. A truncation retry can increase the initial setting, so these are not absolute per-call limits. Equal steps still do not mean equal token usage or compute. Report the recorded system policies by block and actual resource use.

Sources:
/Users/tortalpha/Code/repo-level-dev-agent-eval/docs/defense/source/docs/final_benchmark_results.md
/Users/tortalpha/Code/repo-level-dev-agent-eval/docs/defense/source/docs/final_benchmark_task_set.md
/Users/tortalpha/Code/repo-level-dev-agent-eval/docs/defense/source/experiments/results/sessions/final_v1_swepro_single50/sweep.json
/Users/tortalpha/Code/repo-level-dev-agent-eval/docs/defense/source/experiments/results/sessions/final_v1_swepro_graph75/sweep.json
Author-provided explanation of preliminary local trials, September 4, 2026.

**Slide 10: Model configurations and temperature**

Every displayed experiment configures temperature at 0.0, consistently across its compared architectures. The primary master study uses GLM-5.2 without an explicit reasoning-effort override. In the kernel v2 initial comparisons, GLM-5.2 requests high reasoning effort, DeepSeek V4 Flash requests low, and GPT-5.1 Codex Mini has no explicit effort override. Both additional DeepSeek repetitions retain temperature 0.0 and low reasoning effort, matching the original scored comparison including the tenacity recovery. Unset means that the harness does not request a particular reasoning effort; it does not mean that model reasoning is disabled. Temperature controls sampling randomness, while reasoning effort is a separate provider-specific request. These are recorded/requested configuration values, not independent measurements of provider-side enforcement. The local client passes temperature 0.0 for the displayed OpenRouter model identifiers, but the saved artifacts do not establish how each provider applies it internally. The exact identifiers are z-ai/glm-5.2, deepseek/deepseek-v4-flash-0731:nitro, and openai/gpt-5.1-codex-mini. The experiment seed controls architecture execution order, not LLM sampling. A zero temperature does not guarantee identical outputs or task outcomes, as the repeated task results demonstrate. Reasoning effort and compaction settings differ between kernel model configurations, so the experiment does not isolate model identity. No temperature ablation was performed.

Sources:
/Users/tortalpha/Code/repo-level-dev-agent-eval/docs/defense/source/experiments/results/sessions/final_v1_core_small/sweep.json
/Users/tortalpha/Code/repo-level-dev-agent-eval/docs/defense/source/experiments/results/sessions/final_v1_core_medium_single50/sweep.json
/Users/tortalpha/Code/repo-level-dev-agent-eval/docs/defense/source/experiments/results/sessions/final_v1_core_medium_multi75/sweep.json
/Users/tortalpha/Code/repo-level-dev-agent-eval/docs/defense/source/experiments/results/sessions/final_v1_swepro_single50/sweep.json
/Users/tortalpha/Code/repo-level-dev-agent-eval/docs/defense/source/experiments/results/sessions/final_v1_swepro_graph75/sweep.json
/Users/tortalpha/Code/repo-level-dev-agent-eval/src/run/models.py
/Users/tortalpha/Code/repo-level-dev-agent-eval/src/benchmark/sweep.py
/Users/tortalpha/Code/repo-level-dev-agent-eval/experiments/results/sessions/kernel_v2_glm_5_2_core_small_control/sweep.json
/Users/tortalpha/Code/repo-level-dev-agent-eval/experiments/results/sessions/kernel_v2_ds_v4_flash_0731_nitro_low_core_small_v4/sweep.json
/Users/tortalpha/Code/repo-level-dev-agent-eval/experiments/results/sessions/kernel_v2_codex_mini_core_small_steps50/sweep.json
/Users/tortalpha/Code/repo-level-dev-agent-eval/experiments/results/sessions/kernel_v2_ds_defense_repeat_20260904_r1/sweep.json
/Users/tortalpha/Code/repo-level-dev-agent-eval/experiments/results/sessions/kernel_v2_ds_defense_repeat_20260904_r2/sweep.json

**Slide 11: Success rates by benchmark block**

All three architectures resolve 11 of the 12 small tasks. On medium tasks, single resolves 11 while graph and guarded resolve 10 each. On the selected SWE Pro subset, single and graph both resolve 9 of 15, with identical paired outcomes. Guarded was not run on that extension. The figure presents recorded outcomes, not a statistically established ranking. There is one final attempt per task and architecture. The medium and SWE blocks use different step caps across architectures. SWE Pro also uses different initial completion-token settings (16,384 single and 4,096 graph, before any truncation retry). Core-small is the equal-step comparison.

Sources:
/Users/tortalpha/Code/repo-level-dev-agent-eval/docs/defense/source/experiments/results/final_report/summary.csv
/Users/tortalpha/Code/repo-level-dev-agent-eval/docs/defense/source/docs/final_benchmark_results.md

**Slide 12: Functional quality results**

These figures reaggregate the 72 primary core runs: 24 tasks for each of the three architectures. Resolved at one is 22 of 24 for single and 21 of 24 for graph and guarded. Visible suites pass in 24, 22, and 23 runs respectively. Required hidden suites pass in 22, 21, and 21 runs. These rates count complete suites per run, not individual test cases. One graph run on sqlparse_pr_746 records three regressed tests, so its regression-run rate is 1 of 24. No handoffs occurred. The aggregate combines equal 50-step caps on small tasks and unequal 50/75-step caps on medium tasks. The rubric quality score is unavailable for all 102 primary final runs. The existing reviewer produces one overall 0-to-5 LLM-judge score from the task and diff, guided by correctness, minimality, maintainability, robustness, and safety. None of the historical review run IDs matches a primary final run. Functional test evidence therefore does not establish superior maintainability, robustness, or safety.

Sources:
/Users/tortalpha/Code/repo-level-dev-agent-eval/docs/defense/source/experiments/results/runs.jsonl
/Users/tortalpha/Code/repo-level-dev-agent-eval/docs/defense/source/src/metrics/compute.py
/Users/tortalpha/Code/repo-level-dev-agent-eval/docs/defense/source/src/metrics/quality.py
/Users/tortalpha/Code/repo-level-dev-agent-eval/docs/defense/source/docs/final_benchmark_results.md
/Users/tortalpha/Code/repo-level-dev-agent-eval/docs/defense/quality_metrics_audit.json

**Slide 13: AI and agent metric results**

These figures use the same 24 core tasks per architecture. Tool-use validity pools categorized actions: single 462 of 481, graph 729 of 765, guarded 692 of 719. It measures whether the action is categorized as valid, not whether every tool call achieves its goal. The remaining starred metrics are proxies whose definitions matter. Patch validity means a nonempty changed-files list and is 100 percent for all three systems. Hallucinated references means invalid, malformed, or missing actions per run: 19/24, 32/24, and 25/24, rounded to 0.79, 1.33, and 1.04. It does not count verified nonexistent APIs. Test overfitting is the final-failure fraction among visible-pass runs: 2/24, 1/22, and 2/23. A smaller visible/final gap does not establish better overall performance because denominators differ. Repair success means visible-pass among runs with more than one test iteration: 21/21, 16/17, and 18/19. The metric does not establish that a failure preceded the final pass. Mean test iterations are 3.00, 2.29, and 2.38, and mean LLM calls are 20.71, 37.00, and 40.00. These diagnostics show behavior within this implementation, not a general code-quality ranking.

Sources:
/Users/tortalpha/Code/repo-level-dev-agent-eval/docs/defense/source/src/metrics/compute.py
/Users/tortalpha/Code/repo-level-dev-agent-eval/docs/defense/source/src/metrics/records.py
/Users/tortalpha/Code/repo-level-dev-agent-eval/docs/defense/quality_metrics_audit.json

**Slide 14: Graph cost more on the 39 shared tasks**

Across the 39 tasks evaluated by both architectures, single resolves 31 and graph resolves 30. Graph costs 18.6 percent more overall. Its cost per solved task is about 0.1644 dollars, compared with 0.1342 for single. The denominator includes solved tasks, while the numerator includes the cost of both successful and unsuccessful attempts. This pooled comparison is secondary because it combines different step policies and unequal initial completion-token settings in the SWE Pro block. It supports an empirical cost-efficiency finding for this configuration, without establishing universal superiority.

Sources:
/Users/tortalpha/Code/repo-level-dev-agent-eval/docs/defense/source/experiments/results/final_report/summary.csv
/Users/tortalpha/Code/repo-level-dev-agent-eval/docs/defense/audit.json

**Slide 15: Only three tasks changed the ranking**

The paired comparison is more informative than looking at two percentages in isolation. Both systems solve 29 tasks, and neither solves 7. Single uniquely solves 2, while graph uniquely solves 1. That leaves only three disagreements out of 39. The feature subset favors graph by one task, but it is the same graph-only success rather than an independent confirmation. The experiment does not establish a statistically convincing difference or equivalence. Repeated runs and a larger independent task sample would be needed.

Sources:
/Users/tortalpha/Code/repo-level-dev-agent-eval/docs/defense/source/docs/final_benchmark_results.md
/Users/tortalpha/Code/repo-level-dev-agent-eval/docs/defense/audit.json

**Slide 16: Limits of the primary experiment**

The primary experiment uses one model and one final attempt per configuration. Twenty of the 39 shared tasks had exploratory exposure during system development. Most core reference changes are local: 19 of 24 touch one source file, so larger repositories do not necessarily mean harder cross-module changes. Size labels have ambiguities, and the literal repository-count commitment changed to task counts. Dataset files exist locally but need separate packaging for reproduction. The final patches did not receive a new blinded maintainability review.

Sources:
/Users/tortalpha/Code/repo-level-dev-agent-eval/docs/defense/source/docs/final_benchmark_results.md
/Users/tortalpha/Code/repo-level-dev-agent-eval/docs/defense/source/repositories/collection.csv
/Users/tortalpha/Code/repo-level-dev-agent-eval/docs/defense/DEFENSE_EN.md

**Slide 17: Kernel v2: changes from the first version**

This slide compares the frozen master implementation at commit 8da3f3f with the kernel v2 implementation. Both versions use a model-driven repository action loop, and the built-in single and multi-agent systems share the runtime controls. First, context budgeting now includes retained provider messages, tool-call envelopes, and reasoning metadata. It remains a conservative character-based token estimate, not exact tokenizer accounting. Kernel v2 adds deterministic checkpoint compaction. The displayed GLM kernel configuration retained LLM summarization, while DeepSeek and Codex Mini used checkpoints. Second, before-and-after shell snapshots register indirect file changes, update the workspace revision, invalidate stale test evidence and file observations, and retain detected test-oracle tampering attempts. Third, final scoring reconstructs a clean checkout from the archived patch and uses frozen dependencies and setup artifacts. Visible and hidden scoring validate JUnit evidence, with stricter rejection of missing, empty, skipped, or incomplete required hidden tests. Master already used visible JUnit, regression checks, and final test-oracle protection, but its hidden verdict relied on command success. Fourth, persistent Docker and offline agent execution already existed in master. Kernel v2 adds read-only filesystem and test mounts, resource limits, and process cleanup. Fifth, fingerprints identify more of the actual source, prompts, action schemas, policy, model routes, image, and evaluator inputs. Billing records distinguish complete and partial provider reports and record the cost source. Master already logged token estimates and some provider costs. Reasoning controls remain provider requests and do not train model weights. These are implementation changes intended to strengthen measurement and execution reliability. Each historical run retains its own exact configuration and kernel fingerprint, so this slide does not assert that every earlier kernel run used every latest refinement. The version comparison is not a controlled ablation proving that kernel v2 improves task success, and its results remain separate from the master totals.

Sources:
/Users/tortalpha/Code/repo-level-dev-agent-eval/docs/agent_kernel_v2.md
/Users/tortalpha/Code/repo-level-dev-agent-eval/docs/defense/source/src/agents/state/budget.py
/Users/tortalpha/Code/repo-level-dev-agent-eval/docs/defense/source/src/agents/single_agent.py
/Users/tortalpha/Code/repo-level-dev-agent-eval/docs/defense/source/src/agents/executor.py
/Users/tortalpha/Code/repo-level-dev-agent-eval/docs/defense/source/src/agents/sandbox.py
/Users/tortalpha/Code/repo-level-dev-agent-eval/docs/defense/source/src/benchmark/evaluation.py
/Users/tortalpha/Code/repo-level-dev-agent-eval/docs/defense/source/src/benchmark/sweep.py
/Users/tortalpha/Code/repo-level-dev-agent-eval/docs/defense/source/src/run/results.py
/Users/tortalpha/Code/repo-level-dev-agent-eval/src/agents/state/budget.py
/Users/tortalpha/Code/repo-level-dev-agent-eval/src/agents/state/entry.py
/Users/tortalpha/Code/repo-level-dev-agent-eval/src/agents/executor.py
/Users/tortalpha/Code/repo-level-dev-agent-eval/src/agents/sandbox.py
/Users/tortalpha/Code/repo-level-dev-agent-eval/src/benchmark/evaluation.py
/Users/tortalpha/Code/repo-level-dev-agent-eval/src/run/reproducibility.py
/Users/tortalpha/Code/repo-level-dev-agent-eval/src/agents/model.py
/Users/tortalpha/Code/repo-level-dev-agent-eval/src/run/results.py
/Users/tortalpha/Code/repo-level-dev-agent-eval/experiments/results/sessions/kernel_v2_glm_5_2_core_small_control/sweep.json
/Users/tortalpha/Code/repo-level-dev-agent-eval/experiments/results/sessions/kernel_v2_ds_v4_flash_0731_nitro_low_core_small_v4/sweep.json
/Users/tortalpha/Code/repo-level-dev-agent-eval/experiments/results/sessions/kernel_v2_codex_mini_core_small_steps50/sweep.json

**Slide 18: Kernel v2: initial comparisons**

Kernel v2 strengthens context accounting, test-evidence validation, execution isolation, and run fingerprints. Its completed core-small blocks show different architecture rankings across configurations. GLM gives 10, 9, and 10 successes. DeepSeek gives 9, 11, and 11. Codex Mini gives 9, 7, and 8. In the original DeepSeek attempt, graph has a slightly lower cost per solved task than single. The following slides report two additional repetitions and show how that relationship changes. However, reasoning and compaction settings differ between model configurations. These observations do not isolate model identity, and they do not establish a complexity effect. The 75-step extension remains separate.

Sources:
/Users/tortalpha/Code/repo-level-dev-agent-eval/docs/defense/kernel_v2_interim_snapshot.json
/Users/tortalpha/Code/repo-level-dev-agent-eval/docs/agent_kernel_v2.md
/Users/tortalpha/Code/repo-level-dev-agent-eval/experiments/results/kernel_v2_report/summary.csv

**Slide 19: Repeated runs: a smaller graph advantage**

The DeepSeek V4 Flash extension now contains the original comparison plus two additional repetitions. Each attempt uses the same 12 core-small tasks, the same 50-step limit, low reasoning effort, temperature zero, a 48,000-token context budget, and checkpoint compaction. All 48 new observations completed, with no new infrastructure failures. The historical comparison includes the scored tenacity recovery, which used the same recorded setup as the repeats. Graph solves two more tasks in the original attempt, ties single in repeat 1, and solves one more task in repeat 2. Across all 72 observations, single succeeds 29/36 times and graph 32/36. These are repeated outcomes on 12 exposed tasks. They do not provide 36 independent tasks per architecture, a complexity experiment, or a held-out generalization result. Interpret the advantage descriptively.

Sources:
/Users/tortalpha/Code/repo-level-dev-agent-eval/experiments/plans/defense_repeats_20260904/analysis/audit_repetitions.json
/Users/tortalpha/Code/repo-level-dev-agent-eval/experiments/plans/defense_repeats_20260904/analysis/provenance_review.md
/Users/tortalpha/Code/repo-level-dev-agent-eval/experiments/plans/defense_repeats_20260904/PROTOCOL.md

**Slide 20: Single cost less in both new repetitions**

Every selected observation has complete provider-reported billing. The table divides all billed LLM cost in a block, including failed task attempts, by the number of successful outcomes in that block. It does not include host electricity or hardware cost. Graph has a slightly lower cost per successful outcome in the original comparison, but a higher value in both new repetitions. Across all three attempts, single costs $0.78458366 for 29 successful outcomes, and graph costs $1.04365076 for 32. Thus graph costs about 33.0 percent more overall and 20.5 percent more per successful outcome. The two new repetitions alone cost $1.03556538 in total. Differences in token usage and cache hits contribute to cost variability. Both systems solve the same 11 distinct tasks at least once. The parse_pr_165 task fails in all three attempts for both. This supports a cost versus consistency tradeoff within the tested configurations, not a universal ranking.

Sources:
/Users/tortalpha/Code/repo-level-dev-agent-eval/experiments/plans/defense_repeats_20260904/analysis/audit_repetitions.json
/Users/tortalpha/Code/repo-level-dev-agent-eval/experiments/plans/defense_repeats_20260904/analysis/provenance_review.md
/Users/tortalpha/Code/repo-level-dev-agent-eval/experiments/plans/defense_repeats_20260904/PROTOCOL.md

**Slide 21: Repeated-run quality and agent metrics**

The denominator is 36 scored attempts per architecture on the same 12 tasks, comprising the original comparison and two repetitions. All visible suites pass. Required hidden-suite success is 29/36 for single and 32/36 for graph, and no run records a visible-test regression. The hidden compatibility result reuses the visible suite on every task. Patch presence measures a nonempty changed-files list: 31/36 and 36/36. Tool-use validity pools recognized action categories, including finish and other non-tool events, rather than measuring semantic usefulness or runtime success. Counts are 1002/1033 and 1342/1384. Hallucinated-reference proxies count invalid, malformed, or missing actions: 30/36 and 31/36 events per run. The visible/final gap is 7/36 versus 4/36 and does not prove intentional overfitting. Repair success is visible-pass among attempts with more than one test iteration, 23/23 and 29/29, without requiring a recorded failure-to-pass transition. Single records handoff status in 7/36 attempts and graph in 0/36, which is an agent status rather than evidence that a human intervened. All 72 rubric quality scores remain unavailable. No validated ranking of readability, maintainability, robustness, or safety follows from these diagnostics.

Sources:
/Users/tortalpha/Code/repo-level-dev-agent-eval/experiments/plans/defense_repeats_20260904/analysis/audit_repetitions.json
/Users/tortalpha/Code/repo-level-dev-agent-eval/experiments/plans/defense_repeats_20260904/analysis/provenance_review.md
/Users/tortalpha/Code/repo-level-dev-agent-eval/experiments/plans/defense_repeats_20260904/PROTOCOL.md
/Users/tortalpha/Code/repo-level-dev-agent-eval/src/metrics/compute.py

**Slide 22: Comparison with Xu et al.**

The table reproduces mean pass@1 and standard deviations from Table 1. Our master core-small comparison supports a similar cost-efficiency finding in a different task setting. The DeepSeek repetition result adds an observed cost-versus-consistency tradeoff. The numerical scores belong to separate benchmarks and should not be pooled or directly ranked across studies.

Sources:
https://arxiv.org/html/2601.12307v1#S4.T1
https://arxiv.org/html/2601.12307v1#S6
/Users/tortalpha/Code/repo-level-dev-agent-eval/docs/defense/source/experiments/results/final_report/summary.csv
/Users/tortalpha/Code/repo-level-dev-agent-eval/experiments/plans/defense_repeats_20260904/analysis/audit_repetitions.json
/Users/tortalpha/Code/repo-level-dev-agent-eval/experiments/plans/defense_repeats_20260904/analysis/provenance_review.md
/Users/tortalpha/Code/repo-level-dev-agent-eval/experiments/plans/defense_repeats_20260904/PROTOCOL.md

**Slide 23: Limits of the comparison**

Homogeneous means that the roles share the same base model. Xu et al. assess workflow simulation and optimize workflows with OneFlow. This project evaluates separately designed agent systems. The API single-execution cost calculation in Xu et al. assumes ideal KV-cache reuse. Our repeated DeepSeek observations use complete provider billing, but we did not experimentally isolate cache effects. Equal step limits do not ensure equal tokens or compute. Larger master blocks also use unequal limits. These design differences support a comparison of findings, without a claim of direct replication.

Sources:
https://arxiv.org/html/2601.12307v1#S3.SS2
https://arxiv.org/html/2601.12307v1#S4.SS1
/Users/tortalpha/Code/repo-level-dev-agent-eval/docs/defense/source/experiments/results/sessions/final_v1_swepro_single50/sweep.json
/Users/tortalpha/Code/repo-level-dev-agent-eval/docs/defense/source/experiments/results/sessions/final_v1_swepro_graph75/sweep.json
/Users/tortalpha/Code/repo-level-dev-agent-eval/experiments/plans/defense_repeats_20260904/analysis/audit_repetitions.json
/Users/tortalpha/Code/repo-level-dev-agent-eval/experiments/plans/defense_repeats_20260904/analysis/provenance_review.md
/Users/tortalpha/Code/repo-level-dev-agent-eval/experiments/plans/defense_repeats_20260904/PROTOCOL.md

**Slide 24: Hypotheses and conclusions**

H1 predicted that single agents would be more cost-effective on simple tasks. Master core-small supports it: all three systems solve 11/12 with a 50-step cap, while graph costs 2.03 times and guarded 2.58 times as much as single. H2 predicted that the multi-agent advantage would increase with complexity. That pattern did not appear: single resolves 11/12 medium tasks versus 10/12 for both multi variants, and single and graph tie at 9/15 on SWE Pro. H2 is not supported by this benchmark. Unequal step and initial output-token settings in larger blocks and limited cross-module changes prevent a universal rejection. The additional DeepSeek experiment repeats 12 familiar tasks. Graph succeeds in 32/36 observations versus 29/36 and solves 10 tasks in every attempt versus 7 for single. Both solve the same 11 distinct tasks at least once. Graph costs 20.5 percent more per successful outcome across all three attempts. Thus the observed benefit is greater consistency on this sample, without wider observed task coverage. It does not test complexity or establish better maintainability. Rubric quality scores remain unavailable.

Sources:
/Users/tortalpha/Code/repo-level-dev-agent-eval/docs/defense/source/docs/spec.md
/Users/tortalpha/Code/repo-level-dev-agent-eval/docs/defense/source/experiments/results/final_report/summary.csv
/Users/tortalpha/Code/repo-level-dev-agent-eval/experiments/plans/defense_repeats_20260904/analysis/audit_repetitions.json
/Users/tortalpha/Code/repo-level-dev-agent-eval/experiments/plans/defense_repeats_20260904/analysis/provenance_review.md
/Users/tortalpha/Code/repo-level-dev-agent-eval/experiments/plans/defense_repeats_20260904/PROTOCOL.md

**Slide 25: Dataset definitions and scope changes**

The original proposal defined small repositories as 500 to 3,000 Python source lines and 5 to 30 source files, and medium repositories as 3,001 to 15,000 lines and 31 to 120 files. The later specification prioritizes LOC when measures disagree. In the core, the original combined thresholds hold for 9 of 12 small tasks and 2 of 12 medium tasks when physical lines are used. boltons and pydash fit the medium range only under nonblank counts, a rule not clearly fixed in the metadata. Multiple tasks share repositories. These are limitations to disclose.

Sources:
/Users/tortalpha/Code/repo-level-dev-agent-eval/docs/defense/source/docs/spec.md
/Users/tortalpha/Code/repo-level-dev-agent-eval/docs/defense/source/repositories/README.md
/Users/tortalpha/Code/repo-level-dev-agent-eval/docs/defense/source/repositories/collection.csv

**Slide 26: Operational definitions of AI metrics**

These names should be interpreted through the actual implementation. Patch validity is the fraction of runs with changed files. Hallucinated references counts invalid, malformed, and missing actions rather than checking whether APIs exist. Repair success is the visible-pass fraction among runs with more than one test iteration, without requiring a preceding failure. Test overfitting is the final-failure fraction among visible-pass runs. They are useful diagnostics with limited construct validity. Functional test outcomes and cost provide the clearest primary evidence.

Sources:
/Users/tortalpha/Code/repo-level-dev-agent-eval/docs/defense/source/src/metrics/compute.py
/Users/tortalpha/Code/repo-level-dev-agent-eval/docs/defense/source/src/metrics/records.py

**Slide 27: SWE Pro: quality and agent metrics**

This appendix reports the 15 primary SWE Pro tasks evaluated by single and graph. Guarded was not run on this subset. Both systems pass all visible suites, yet only 9 of 15 pass the required hidden suites and resolve the task. Thus each has a visible/final gap of 6 of 15, or 40 percent. Neither records a visible-test regression. Tool-use validity is 390/408 for single and 639/661 for graph. Hallucinated-reference proxies are 17/15 and 18/15 per run. The repair proxy is 14/14 for single and 15/15 for graph. Patch validity is 15/15 and human handoff is 0/15 for both, while rubric quality scores remain unavailable. Mean test iterations are 3.33 and 3.47, and mean LLM calls are 31.53 and 49.20. Single uses a 50-step cap and graph 75. The initial completion-token settings are 16,384 and 4,096 respectively, with possible increases after truncation. This is a descriptive comparison under the recorded policies.

Sources:
/Users/tortalpha/Code/repo-level-dev-agent-eval/docs/defense/source/src/metrics/compute.py
/Users/tortalpha/Code/repo-level-dev-agent-eval/docs/defense/quality_metrics_audit.json

**Slide 28: Related work and scope**

Xu et al., Rethinking the Value of Multi-Agent Workflow: A Strong Single Agent Baseline (2026), motivates the primary research question. Dat Tran and Douwe Kiela, Single-Agent LLMs Outperform Multi-Agent Systems on Multi-Hop Reasoning Under Equal Thinking Token Budgets (2026), studies FRAMES and MuSiQue. Requested intermediate thinking caps exclude prompts and final answers, and actual consumption may differ. Gemini accounting is approximate. Our step caps do not reproduce this resource control. Chunqiu Steven Xia, Yinlin Deng, Soren Dunn and Lingming Zhang, Agentless: Demystifying LLM-based Software Engineering Agents (2024), uses a fixed localization, patching and test-selection pipeline. The LLM does not autonomously choose the next tool action. Its comparison does not isolate our single-versus-multi orchestration question.

Sources:
https://arxiv.org/html/2601.12307v1#S6
https://arxiv.org/html/2604.02460v2#S4
https://arxiv.org/html/2604.02460v2#A3
https://arxiv.org/html/2407.01489v2#S3

**Slide 29: Task outcomes across three attempts**

Each cell lists the original scored attempt, repeat 1, and repeat 2 in that order. P means final task_success is true and F means it is false. The original comparison includes the successful tenacity recovery session. Single has seven tasks solved in all three attempts, four with mixed results, and one with three failures. Graph has ten tasks solved in all three, one mixed result, and one with three failures. Both solve the same 11 tasks at least once. Across 36 paired task-by-attempt observations, both succeed in 28, both fail in 3, graph alone succeeds in 4, and single alone succeeds in 1. Repetitions within a task are dependent observations, so this table does not establish a statistically general ranking.

Sources:
/Users/tortalpha/Code/repo-level-dev-agent-eval/experiments/plans/defense_repeats_20260904/analysis/audit_repetitions.json
/Users/tortalpha/Code/repo-level-dev-agent-eval/experiments/plans/defense_repeats_20260904/analysis/provenance_review.md
/Users/tortalpha/Code/repo-level-dev-agent-eval/experiments/plans/defense_repeats_20260904/PROTOCOL.md
