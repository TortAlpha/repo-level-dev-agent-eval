# Repository-Level Dev Agent Evaluation

This repository contains a course project specification for evaluating
single-agent and multi-agent LLM architectures on autonomous software
development tasks at repository level.

## Task

The project investigates whether a multi-agent workflow gives a practical
advantage over a strong single-agent loop when both systems work under the same
constraints. Each system must solve realistic Python repository tasks by
understanding an issue, inspecting the codebase, finding relevant files, editing
code, running tests, interpreting failures, and iteratively repairing the
solution.

The central research hypothesis is that a single-agent loop may be more
cost-effective for simple localized tasks, while the advantage of a multi-agent
architecture should increase as repository and task complexity grow.

## Research Context

The project extends the evaluation domain of Xu et al.,
["Rethinking the Value of Multi-Agent Workflow: A Strong Single Agent Baseline"](https://arxiv.org/abs/2601.12307).
That work studies whether multi-agent workflows remain useful when compared
against a strong single-agent baseline. This repository applies the same core
question to a more demanding software engineering setting: full Python
repositories instead of simpler coding benchmarks such as HumanEval and MBPP.

Related work also motivates the comparison against simpler baselines in
software engineering agents, especially Agentless and studies that normalize
reasoning budgets across single-agent and multi-agent systems.

## Benchmark Scope

The benchmark is limited to Python repositories to keep language, dependency
management, test execution, Docker setup, and evaluation comparable across
tasks.

Each benchmark task is expected to include:

- an initial Git repository state;
- a natural language issue or task description;
- a visible test command available to the agent;
- a reproducible Docker execution environment;
- hidden or withheld evaluation tests;
- repository size and complexity metadata;
- the expected final status.

The planned benchmark contains at least 24 tasks, with a target of 30 tasks. It
will include at least 12 tasks from small repositories and at least 12 tasks
from medium repositories.

Repository sizes are defined as:

- small: 500-3,000 lines of Python code, excluding tests, and 5-30 source files;
- medium: 3,001-15,000 lines of Python code and 31-120 source files.

Tasks may be reconstructed from merged GitHub pull requests or created
manually. Hidden tests must fail on the initial commit and pass on the reference
solution, while visible tests must pass on the initial repository state.

## Compared Systems

The baseline is a single-agent loop where one LLM agent performs planning,
coding, testing, and revision.

The proposed system is a LangGraph-based multi-agent workflow with specialized
roles:

```text
planner -> developer -> tester -> reviewer -> finalizer
```

The role decomposition is treated as an experimental hypothesis. The project
will also compare ablation variants to determine which roles improve results and
which only add coordination cost.

Role-specific models, adaptive developer escalation, and cost-capped comparison
commands are documented in [docs/role_model_routing.md](docs/role_model_routing.md).

Both approaches are evaluated under the same constraints:

- same benchmark tasks;
- same iteration limits;
- same visible tests;
- same timeout limits;
- comparable model configuration;
- same Docker execution rules;
- same final evaluation procedure.

## Evaluation

The comparison uses outcome, efficiency, behavior, and solution-quality metrics:

- pass@1 / resolved@1;
- task success rate;
- hidden or withheld test pass rate;
- patch validity rate;
- tool-use validity rate;
- hallucinated reference count;
- test-overfitting rate;
- repair success rate;
- regression rate;
- iteration count;
- execution time;
- number of LLM calls and approximate token usage;
- solution quality score.

Solution quality is scored with a rubric covering functional correctness,
minimality, maintainability, robustness, and safety.

## Local Model Setting

The initial experiments are designed for local LLM execution to make cost,
privacy, and reproducibility easier to control. The target hardware described in
the specification is 32 GB RAM, an Intel i5-12500 CPU, and an NVIDIA RTX 3070
with about 8 GB VRAM. Quantized 7B/8B models are the realistic initial
candidates; larger models may require reducing the benchmark scope or using
hosted model access for secondary comparison.

## Documentation

- [Full project specification](docs/spec.md)
- [Short project specification](docs/spec_short.md)

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
