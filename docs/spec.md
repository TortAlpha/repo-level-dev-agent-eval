# Project Specification: Repository-Level Evaluation of Single-Agent and Multi-Agent LLM Architectures for Autonomous Software Development Tasks

## Course Project Application

### Team

- Student: Avanesov Roman
- Index number: SV 88/2024

## Problem Description

Large language models can already generate code, but a single prompt-response
interaction is often not enough for realistic software development tasks. A
useful autonomous coding system has to understand an issue, inspect the
repository, edit code, run tests, interpret failures, revise the solution, and
decide when the result is ready for review.

The problem addressed in this project is the evaluation of two architectures for
autonomous software development:

1. A **single-agent loop**, where one LLM agent performs planning, coding,
   testing, and revision by itself.
2. A **multi-agent system**, where the workflow is split between specialized
   agents: planner, developer, tester, reviewer, and finalizer.

The goal is to determine whether a multi-agent architecture produces better
software development outcomes than a simpler single-agent loop under comparable
constraints on repository-level tasks.

Unlike Xu et al., "Rethinking the Value of Multi-Agent Workflow: A Strong
Single Agent Baseline", this evaluation is not limited to simpler coding
benchmarks such as HumanEval and MBPP. It is a more complex extension because
each task is grounded in an existing repository and requires issue
understanding, repository navigation, relevant-file discovery, code changes,
test execution, and iterative repair.

## Research Motivation

Recent research questions whether complex multi-agent LLM workflows are always
worth their additional coordination and inference cost. Some studies argue that
stronger single-agent systems can match or outperform multi-agent systems when
the comparison controls for test-time computation, context reuse, and token
budget. In software engineering, simpler approaches have also shown competitive
results against more complex autonomous agents on benchmarks such as SWE-bench.

This creates a practical research question for autonomous software development:
if modern LLMs become more capable, is a single-agent loop usually the better
engineering choice, or do multi-agent systems still provide an advantage on
larger codebases and more complex tasks?

This project is motivated by that uncertainty. It will not assume that
multi-agent systems are always better. Instead, it will empirically compare
single-agent and multi-agent architectures under controlled conditions and
analyze when the additional structure of multiple roles is useful.

## Research Hypothesis

The main hypothesis of the project is that the advantage of a multi-agent
architecture increases with task and codebase complexity.

For simple, localized tasks, a single-agent loop may be sufficient and more
cost-efficient. However, as tasks require understanding more files, larger
repositories, longer test feedback, or multiple development decisions, a
multi-agent system is expected to perform better because planning, testing,
review, and finalization are separated into specialized roles.

The project will therefore evaluate not only the overall performance difference
between the two approaches, but also how this difference changes across tasks
of different complexity levels.

## Data

The project will use a benchmark of software development tasks. Each task will
contain:

- a small, medium, or large Python Git repository;
- a natural language issue or task description;
- a visible test command available to the agent;
- a reproducible Docker execution environment;
- hidden or withheld evaluation tests used to verify the final solution;
- repository size and complexity metadata;
- the final expected status: solved or not solved.

The benchmark will be limited to Python repositories. This constraint is used
to control the programming language, dependency management, test commands,
Docker setup, and comparison between agents. It also reduces confounding factors
that would appear if different language ecosystems required substantially
different build tools and execution environments.

A **small** repository is defined as a repository with approximately 500-3,000
lines of Python code, excluding tests, and 5-30 source files. A **medium**
repository is defined as a repository with 3,001-15,000 lines of Python code and
31-120 source files. A **large** repository is defined as a repository with more
than 15,000 lines of Python code or more than 120 source files. When the LOC
count and the file count disagree, the LOC bucket wins and the disagreement is
noted in the collection metadata. Repositories with fewer than 500 lines of
Python code will not be included because they are likely to be too small for a
meaningful repository-level evaluation.

The final benchmark will contain at least 24 tasks, with a target of 30 tasks.
It will include at least 12 tasks from small repositories and at least 12 tasks
from medium repositories. Large-repository tasks are an optional extension
bucket used to probe how the compared architectures scale with codebase size;
they are reported separately and are not required for the core comparison.

The dataset will be more complex than homework datasets because it is not a
single tabular dataset. Each example consists of a full repository state,
source code, tests, issue text, execution environment, and final evaluation
result.

### Data Collection Strategy

Data collection will be semi-automated. A collection script will:

- find candidate Python repositories;
- clone selected repositories;
- measure repository size and number of Python source files;
- detect dependency files such as `requirements.txt`, `pyproject.toml`,
  `setup.py`, or lock files;
- identify or verify a visible test command;
- check whether the repository can be executed inside Docker;
- record repository metadata needed for complexity grouping.

After this automated filtering step, tasks will be manually validated. Manual
validation is needed because a repository can be technically runnable while
still being unsuitable for a controlled benchmark, for example because the task
description is ambiguous, the tests are flaky, the project requires external
services, or the expected behavioral change cannot be isolated.

The benchmark will be considered sufficient only if each task has:

- a reproducible Docker environment;
- a visible test command that can be run by the agent;
- hidden or withheld tests for final evaluation;
- a clear natural language task description that does not reveal the solution;
- repository size and complexity metrics;
- a known reference state where the expected behavior is implemented.

### Task Formation

Part of the benchmark will be reconstructed from merged GitHub pull requests.
For these tasks:

- the commit before the merge is used as the initial repository state;
- the linked issue or pull request description is used as the task description;
- tests added in the pull request are used as hidden or withheld evaluation
  tests;
- the merged state, or a manually prepared equivalent solution, is used as the
  reference solution.

Hidden or withheld tests must fail on the initial commit because they verify the
behavior requested by the task. They must pass on the reference solution. Visible
tests must pass on the initial commit, so the benchmark can verify that the
repository is not already broken before the agent starts. The agent's final
solution will be evaluated using hidden tests and regression checks. During
evaluation, the agent will not have access to the pull request diff, the final
solution, or the hidden tests.

If no suitable issue or pull request exists, a task will be created manually.
For manually created tasks, the expected behavior and hidden tests will be
defined first. The natural language description will then be written using the
same style as the reconstructed tasks. A local LLM may help formulate the
description, but the final text will be manually checked to ensure that it does
not include implementation details or reveal the solution.

The benchmark will include bug-fixing and feature-implementation tasks. Tasks
will be grouped or annotated by approximate complexity, for example using
repository size, number of source files, number of relevant files, amount of
code that must be understood, number of tests, and whether the task requires a
local fix or a cross-module change. This will allow the evaluation to test
whether the multi-agent system provides larger benefits on more complex tasks.

### Threats to Validity

Because the repositories are public, it is not possible to fully guarantee that
an LLM has not seen part of the code during pretraining. This risk will be
treated as a threat to validity rather than a fully removable problem. To reduce
the risk, the benchmark will prefer newer and less popular repositories, and
some tasks will be manually created or adapted to reduce the chance of a
memorized solution.

## Methodology

The project will implement and compare two autonomous development approaches on
the same benchmark.

### Baseline: Single-Agent Loop

The baseline system will use one LLM agent that receives the task description,
has access to constrained repository tools, edits files, runs tests in a
sandbox, and iterates until either the tests pass or a predefined iteration
limit is reached.

The single agent is responsible for all decisions:

- understanding the task;
- choosing files to inspect;
- modifying code;
- running tests;
- interpreting failures;
- deciding whether the solution is complete.

### Proposed System: Multi-Agent Workflow

The proposed system will use a LangGraph-based multi-agent workflow:

```text
planner -> developer -> tester -> reviewer -> finalizer
```

This exact role decomposition is an initial design hypothesis, not a fixed
assumption of the project. During development, different multi-agent
configurations will be implemented and compared in order to identify which
division of responsibilities works best for autonomous software development
tasks.

The roles are separated as follows:

- **Planner:** inspects the repository and prepares a concise implementation
  plan.
- **Developer:** applies the code changes using constrained file, shell, and
  Git tools.
- **Tester:** runs the configured test command in a Docker sandbox and
  summarizes the result.
- **Reviewer:** inspects the diff and decides whether the solution is ready or
  should be sent back to the developer.
- **Finalizer:** commits the accepted solution or marks the task as requiring
  human intervention.

The graph routing is deterministic. The LLM agents do not decide the next graph
state directly. Instead, code-level routing functions decide whether the system
continues to testing, returns to development, finalizes the task, or stops at a
human handoff state.

Both systems will be evaluated under the same constraints:

- same benchmark tasks;
- same maximum number of iterations;
- same Docker execution rules;
- same visible tests;
- same timeout limits;
- comparable LLM model configuration;
- same final evaluation procedure.

The comparison will also include ablation experiments over the multi-agent
workflow, for example:

- single developer agent without separate planning or review;
- developer and tester loop without a reviewer;
- planner and developer loop without a separate reviewer;
- full planner, developer, tester, reviewer, and finalizer workflow;
- alternative role assignments discovered during implementation.

These variants will make it possible to determine whether each role improves
the final result or only adds cost and complexity.

### LLM Model Selection

The experiments will first be conducted with a local LLM. This is important
because local execution makes cost, privacy, and reproducibility easier to
control during benchmark development.

The available hardware configuration is:

- 32 GB RAM;
- Intel i5-12500 CPU;
- NVIDIA RTX 3070 GPU with about 8 GB VRAM.

On this configuration, quantized 7B/8B models are realistic candidates for the
initial experiments. Quantized 13B/14B models may also be tested, but they are
expected to be slower and more constrained in context length or GPU memory use.
Larger models are not practical as the primary choice for a larger
repository-level benchmark because they would require significantly slower
execution and heavier reliance on CPU/RAM.

For this reason, the project will start with a pilot evaluation. If the local
model is not sufficient for repository-level tasks, the benchmark scope will be
adjusted or the need for a stronger model will be explicitly justified. If
additional model access becomes available, stronger commercial or hosted models
may be used for a secondary comparison, but the initial experimental design is
based on the local-model setting.

## Evaluation

The evaluation will compare the baseline single-agent loop and the multi-agent
system using standard software engineering metrics, AI-agent behavior metrics,
efficiency metrics, and solution-quality metrics.

### Outcome Metrics

- **Pass@1 / resolved@1:** percentage of tasks solved by the first final patch
  produced by the system.
- **Task success rate:** percentage of tasks where the final solution passes
  visible tests and the required hidden suites. Required hidden suites include
  semantic tests derived from the task description and compatibility tests for
  preserved behavior.
- **Visible test pass rate:** percentage of visible tests passed by the final
  solution.
- **Hidden test pass rate:** percentage of required hidden or withheld tests
  passed by the final solution. PR-parity hidden tests that check upstream
  implementation details not stated in the task are reported separately.
- **Patch validity rate:** percentage of generated patches that apply cleanly,
  keep the repository runnable, and do not introduce invalid file states.
- **Human handoff rate:** percentage of tasks where the system reaches an
  iteration limit, error limit, or no-progress condition.
- **Regression rate:** number of previously passing tests that fail after the
  generated change.
- **Repair success rate:** percentage of initially failing attempts that are
  successfully repaired in later iterations.

### Efficiency Metrics

- **Iteration count:** number of development-test-review cycles used per task.
- **Execution time:** total wall-clock time needed to complete or abandon a
  task.
- **LLM usage:** number of LLM calls and approximate token usage per task.
- **Cost estimate:** estimated cost per attempted task and per solved task.

### Agent Behavior Metrics

- **Tool-use validity rate:** percentage of tool calls that are syntactically
  valid, allowed by the sandbox, and relevant to the task.
- **Hallucinated reference count:** number of references to files, functions,
  commands, tests, or APIs that do not exist in the repository or environment.
- **Test-overfitting rate:** percentage of solutions that appear to satisfy
  visible tests while failing hidden or withheld tests that check the requested
  behavior.

### Solution Quality Metrics

Passing tests alone is not always enough to measure the quality of a software
change. Therefore, each final solution will also receive a quality score based
on a fixed rubric.

The proposed quality rubric is:

| Criterion | Description |
| --- | --- |
| Functional correctness | The solution satisfies the task requirements and passes evaluation tests. |
| Minimality | The solution changes only the necessary files and avoids unrelated edits. |
| Maintainability | The code is readable, idiomatic, and consistent with the existing project style. |
| Robustness | The solution handles relevant edge cases and does not overfit visible tests. |
| Safety | The solution avoids unsafe commands, hardcoded test hacks, generated artifacts, and unrelated side effects. |

Each solution can be scored from 0 to 5:

- **0:** incorrect or unusable solution;
- **1:** mostly incorrect, with only minor useful parts;
- **2:** partially correct but incomplete or fragile;
- **3:** functionally acceptable but with maintainability or robustness issues;
- **4:** correct and mostly clean, with only minor issues;
- **5:** correct, minimal, maintainable, robust, and safe.

To reduce bias, the quality score should be assigned without revealing whether
the solution was produced by the single-agent or multi-agent system. The scoring
can be performed manually using the rubric or by an independent reviewer model,
with a subset manually checked for consistency.

### Comparative Analysis

The final report will compare the two systems by answering the following
questions:

- Does the multi-agent system solve more tasks than the single-agent baseline?
- Are strong single-agent loops more cost-effective on simple tasks?
- Does the multi-agent advantage increase as task and codebase complexity
  increase?
- Does the reviewer reduce incorrect or low-quality solutions?
- Does planning reduce unnecessary exploration and repeated edits?
- What is the cost of the multi-agent architecture in time and token usage?
- On which types of tasks does the multi-agent system perform better or worse?
- Does the difference between the two approaches change between small and
  medium Python repositories?
- Is the local LLM setting sufficient for repository-level tasks, or does it
  require a reduced benchmark scope or a stronger model?

## References

- Xu et al., "Rethinking the Value of Multi-Agent Workflow: A Strong Single
  Agent Baseline", arXiv:2601.12307.
- Tran and Kiela, "Single-Agent LLMs Outperform Multi-Agent Systems on
  Multi-Hop Reasoning Under Equal Thinking Token Budgets", arXiv:2604.02460.
- Xia et al., "Agentless: Demystifying LLM-based Software Engineering Agents",
  arXiv:2407.01489.
