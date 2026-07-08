# Repository-Level Evaluation of Single-Agent and Multi-Agent LLM Architectures for Autonomous Software Development

## Team

- Avanesov Roman, SV 88/2024

## Problem Description

The project compares single-agent and multi-agent LLM approaches on
repository-level software engineering tasks. The goal is to evaluate whether a
multi-agent workflow provides a practical advantage over a strong single-agent
loop when both systems must understand an issue, inspect an existing repository,
find relevant files, edit code, run tests, interpret failures, and iteratively
repair the solution.

Unlike Xu et al., "Rethinking the Value of Multi-Agent Workflow: A Strong
Single Agent Baseline", this evaluation is not based on simpler coding
benchmarks such as HumanEval or MBPP. It is a more complex extension because
each task is grounded in an existing repository and requires repository-level
navigation, dependency handling, test execution, and regression checking.

The main hypothesis is that single-agent systems may be more cost-effective on
simple tasks, while multi-agent systems should provide more benefit as task and
codebase complexity increase.

## Data

The benchmark will be limited to Python repositories. This constraint is used
to control the programming language, dependency management, test commands,
Docker environment, and comparison between agents.

Each task will include:

- an initial Git repository state;
- a natural language issue or task description;
- a visible test command available to the agent;
- a reproducible Docker execution environment;
- hidden or withheld evaluation tests;
- repository size and complexity metadata;
- the expected final status, solved or not solved.

A small repository is defined as a repository with approximately 500-3,000
lines of Python code, excluding tests, and 5-30 source files. A medium
repository is defined as a repository with 3,001-15,000 lines of Python code
and 31-120 source files. A large repository has more than 15,000 lines of
Python code or more than 120 source files; large tasks form an optional
extension bucket reported separately from the core small/medium comparison.
When LOC and file count disagree, the LOC bucket wins. Repositories below 500
lines of Python code will not be included.

Data collection will be semi-automated. A script will find and clone candidate
repositories, measure repository size, check dependency files, identify or
verify a test command, and test whether the project can run in Docker. The
final decision about task suitability will be made manually.

The final benchmark will contain at least 24 tasks, with a target of 30 tasks.
It will include at least 12 small-repository tasks and 12 medium-repository
tasks. The dataset will be considered sufficient only if every task has a
reproducible Docker environment, a visible test command, hidden or withheld
tests, a clear task description that does not reveal the solution, and recorded
repository complexity metrics.

Some tasks will be reconstructed from merged GitHub pull requests. The commit
before the merge will be used as the initial state, the linked issue or pull
request description will be used as the task description, and tests added in
the pull request will be used as hidden or withheld evaluation tests. Hidden
tests must fail on the initial commit and pass on the reference solution. Visible
tests must pass on the initial commit to confirm that the repository is not
already broken. During evaluation, the agent will not receive the pull request
diff, the final solution, or the hidden tests.

If a suitable issue or pull request is not available, a task will be created
manually. In that case, the expected behavior and hidden tests will be defined
first, and then the natural language task description will be written using the
same template. A local LLM may help formulate the description, but the final
version will be manually checked to avoid implementation details or solution
leakage.

Because public repositories may have appeared in model pretraining data, this
risk will be treated as a threat to validity rather than a fully removable
problem. To reduce the risk, preference will be given to newer and less popular
repositories, and some tasks will be manually created or adapted.

## Methodology

Two approaches will be implemented and compared on the same benchmark:

1. **Single-agent loop:** one LLM agent performs planning, coding, testing, and
   revision.
2. **Multi-agent workflow:** the task is split between specialized roles such
   as planner, developer, tester, reviewer, and finalizer.

The exact multi-agent role decomposition is treated as an initial hypothesis.
During development, different multi-agent configurations and ablation variants
will be compared to determine which roles improve performance and which only
add cost.

Both systems will be evaluated under the same constraints: same tasks, same
iteration limits, same visible tests, same timeout limits, comparable model
configuration, same final evaluation procedure, and same Docker execution
rules.

The experiments will first be run with a local LLM. On the available hardware,
32 GB RAM, an i5-12500 CPU, and an RTX 3070 with about 8 GB VRAM, quantized
7B/8B models are realistic, while 13B/14B models may be usable with limits on
speed and context length. Larger models are not practical as the primary choice
for a larger repository-level benchmark because they would require slow
CPU/RAM-heavy execution. A pilot evaluation will be run first; if the local
model is not sufficient, the benchmark scope will be adjusted or the need for a
stronger model will be justified.

## Evaluation

The evaluation will include standard software engineering metrics and
AI-agent-specific metrics:

- pass@1 / resolved@1;
- task success rate (visible + required hidden semantic/compat suites);
- hidden or withheld test pass rate, with PR-parity checks reported separately;
- patch validity rate;
- tool-use validity rate;
- number of hallucinated references;
- test-overfitting rate;
- repair success rate;
- regression rate;
- iteration count;
- execution time;
- number of LLM calls and tokens;
- solution quality score.

Solution quality will be scored using a rubric covering functional correctness,
minimality, maintainability, robustness, and safety.

The final comparison will analyze whether the multi-agent system solves more
tasks than the single-agent baseline, whether its advantage increases with task
complexity, and whether the improvement is worth the additional cost in time,
tokens, and implementation complexity.
