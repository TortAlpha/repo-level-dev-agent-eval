# Task Benchmark Observations

This document collects working terminology and observations about benchmark
task descriptions. It is meant as a living notes file for comparing local
PR-derived tasks with imported SWE-bench Pro tasks.

## Terms

- **Task specification**: the text shown to the agent as `task.md`.
- **Issue-style task**: a compact task written like a real GitHub issue or PR
  request. It describes the problem and expected behavior, but usually leaves
  repository navigation to the agent.
- **Behavior contract**: a task statement that defines externally visible
  behavior, compatibility constraints, and public API expectations without
  naming the implementation.
- **Augmented task spec**: a task statement extended with explicit
  requirements, acceptance criteria, and interface details.
- **Requirements**: human-written constraints or expected behaviors that reduce
  ambiguity and make the task self-contained.
- **Interface**: explicit names, signatures, files, classes, methods, or
  outputs expected by the evaluator.
- **Localization hint**: any task text that points the agent toward a concrete
  file, class, method, module, route, or API surface.
- **Underspecification**: missing context in the task statement that a human
  might resolve through discussion or exploration, but that can make automated
  grading brittle.
- **Resolvability**: whether a competent solver can infer a valid patch from
  the task statement and repository state without hidden information.
- **False negative**: a correct or reasonable solution marked wrong because the
  task or tests required an unstated name, interface, edge case, or exact
  behavior.
- **Fail-to-pass tests**: tests that fail on the base repository and pass after
  the intended fix.
- **Pass-to-pass tests**: tests that pass before and after the fix, used to
  check that existing behavior was preserved.
- **Hidden semantic tests**: evaluator-only tests that directly follow from the
  task description and count toward task success.
- **Hidden compatibility tests**: evaluator-only tests that check preserved
  behavior and also count toward task success.
- **Hidden PR-parity tests**: evaluator-only tests that check exact upstream PR
  behavior but may not be fully implied by the task description.
- **Task realism**: how closely the task resembles the way software work is
  assigned in practice.
- **Benchmark controllability**: how easy it is to reproduce, score, and compare
  results across agents.
- **Contamination resistance**: how well the benchmark avoids using tasks or
  solutions that models likely saw during training.
- **Execution capability under detailed spec**: the ability to implement,
  test, and repair a repository-level change when the desired behavior is
  spelled out.
- **Ambiguity handling**: the ability to make progress when the task is
  incomplete, implicit, or issue-like.
- **Codebase localization**: the ability to find the relevant files and symbols
  from a task description and repository inspection.
- **Agentic realism**: the degree to which the benchmark requires realistic
  agent behavior: exploration, localization, testing, iteration, and judgment
  under incomplete information.

## Current Observations

The local benchmark tasks and imported SWE-bench Pro tasks differ most strongly
in how the task is specified.

Local tasks are usually compact, PR-derived behavior contracts. In the current
collection, every readable local task starts with `# Task:`. The text usually
states what is wrong or missing, what behavior should happen instead, and what
public behavior must be preserved. It rarely names implementation files or exact
internal symbols.

SWE-bench Pro tasks are augmented specs. In this repository, the importer builds
each `task.md` as:

```text
problem_statement + "## Requirements" + "## Interface"
```

This makes the tasks more self-contained and reduces ambiguity, but also makes
them less like a raw real-world issue. Many SWE-bench Pro tasks explicitly name
files, classes, methods, inputs, and outputs.

Measured on the current `repositories/collection.csv` readable task files:

- Local tasks: 35 readable `task.md` files.
- SWE-bench Pro tasks: 265 readable `task.md` files.
- Local median task length: 117 words.
- SWE-bench Pro median task length: 430 words.
- Local tasks with `## Requirements` or `## Interface`: 0 of 35.
- SWE-bench Pro tasks with `## Requirements` and `## Interface`: 265 of 265.
- Local tasks mentioning `.py` paths: 0 of 35.
- SWE-bench Pro tasks mentioning `.py` paths: 165 of 265.
- SWE-bench Pro tasks with quoted or literal escaped-newline formatting: 104 of
  265.

## Interpretation

SWE-bench Pro is a good standard yardstick for repository-level execution under
a detailed specification. It tests whether an agent can operate inside a real
codebase, edit multiple files, run tests, preserve compatibility, and produce a
valid patch in a reproducible environment.

It is weaker as a proxy for the task-intake part of real software work. The
agent often receives more explicit requirements and interface details than a
developer would get from an ordinary issue or PR request. This lowers ambiguity
and reduces false negatives, but it also gives extra localization and API
signals.

The local tasks are closer to issue-style work. They put more pressure on
codebase localization, ambiguity handling, and inferring the required behavior
from a compact description. They are less standardized and may be easier to make
underspecified, so they need careful validation against hidden semantic tests.

## Useful Framing

One useful axis is:

```text
raw issue -> behavior contract -> augmented spec -> implementation-localized spec
```

Another useful axis is:

```text
task realism <-> benchmark controllability
```

The local benchmark sits closer to task realism. SWE-bench Pro sits closer to
benchmark controllability and large-repository execution.

For reporting, avoid saying that one is simply "more realistic" or "better".
They measure different parts of agent performance:

- Local PR-derived tasks: issue understanding, ambiguity handling, and codebase
  localization.
- SWE-bench Pro tasks: execution capability under detailed spec, scalability to
  larger repositories, and standardized evaluation.

