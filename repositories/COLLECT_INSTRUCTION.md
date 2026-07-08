# Repository Task Collection Instructions

This document describes how to collect repositories and PR-derived tasks for
the repository-level agent evaluation benchmark.

The benchmark row is a task, not just a repository. A single upstream
repository may contribute multiple tasks if each task is based on a different
merged pull request and can be validated independently.

## Dataset Targets

The final benchmark should contain at least 24 tasks, with a target of 30, and
at least 12 small-repository tasks and 12 medium-repository tasks. A single
repository may supply several tasks, but do not over-represent one repository.
Every accepted task must pass the Final Acceptance Checklist below.

## Target Task Shape

Each benchmark task must contain:

- an initial Python repository checkout before the target change;
- a natural-language task description in `task.md`;
- a visible test command that the agent is allowed to run;
- hidden tests copied from the reference state and withheld from the agent;
- metadata in `repositories/collection.csv`;
- enough commit information to reproduce the initial and reference states.

For PR-derived tasks, use code before the merged PR as the initial state. The
merged PR, or an equivalent manually prepared solution, is the reference state.

Do not use a PR that was merely closed without being merged unless it is being
converted into a manually designed task with a separate reference solution.

## Repository Selection Criteria

Use Python repositories only. Prefer repositories that:

- have a clear test suite using `pytest` or another simple Python test runner;
- can be installed and tested without external services;
- have deterministic tests and no mandatory network access during tests;
- have ordinary dependency files such as `pyproject.toml`, `setup.py`,
  `setup.cfg`, `requirements.txt`, `tox.ini`, or lock files;
- are small or medium according to the project specification;
- contain real library or application logic, not only scripts or examples;
- have permissive enough licensing for local evaluation use.

Avoid repositories that:

- require databases, cloud credentials, browsers, GPUs, or paid services for
  normal tests;
- are mostly generated code, notebooks, examples, docs, or vendored code;
- have very slow or flaky test suites;
- require unusual system packages that make Docker reproduction difficult;
- are too popular if a lower-exposure alternative with similar quality exists.

Repository size buckets (both LOC and source-file count should fall in the
bucket; if they disagree, prefer the LOC bucket and note it):

- `small`: 500-3000 Python source LOC and 5-30 source files, excluding tests.
- `medium`: 3001-15000 Python source LOC and 31-120 source files, excluding
  tests.
- `large`: more than 15000 Python source LOC or more than 120 source files,
  excluding tests.

Repositories below 500 source LOC are too small for this benchmark. `large`
tasks are an optional extension bucket for probing how architectures scale
with codebase size; they are reported separately and do not count toward the
12-small/12-medium core targets.

## Pull Request Selection Criteria

Use merged pull requests that introduce a bug fix or a small feature. A good PR
candidate has:

- a clear issue, PR title, or PR description that can be rewritten as a task;
- tests added or modified in the PR that directly check the requested behavior;
- a reasonably small implementation diff;
- a reference commit that can be checked out locally;
- an initial commit where existing visible tests pass;
- hidden tests that fail on the initial commit and pass on the reference state.

Reject PRs that:

- are only formatting, typing, packaging, CI, documentation, or dependency
  updates;
- are broad refactors with no isolated behavioral objective;
- require external services or environment-specific setup;
- add tests that are too coupled to the exact implementation;
- expose the solution in the only available task text;
- cannot be reproduced from available commits.

## Directory Layout

Use this layout:

```text
repositories/
  collection.csv
  repos/
    <upstream_repo_name>/
  tasks/
    <task_id>/
      repo/
      task.md
      hidden_tests/
```

`repositories/repos/<name>` is a source checkout used for exploration. It may
have full history if that is useful for PR analysis.

`repositories/tasks/<task_id>/repo` is the actual initial repository state that
will be given to the agent. It must be checked out at `base_commit`.

`repositories/tasks/<task_id>/hidden_tests` contains evaluator-only tests copied
from the reference state. Preserve the original relative paths under this
directory when possible.

## Task ID Format

Use a stable task id:

```text
<repo_name>_pr_<number>
```

Examples:

```text
h11_pr_181
humanize_pr_329
pluggy_pr_646
```

If a task is manually created instead of PR-derived, use:

```text
<repo_name>_manual_<short_slug>
```

## Collection Workflow

1. Screen candidate repositories.

   Check language, size, dependency files, test suite, license, and whether
   tests can plausibly run in Docker without external services.

2. Find candidate merged PRs.

   Search for merged PRs with tests. Prefer PRs with labels such as bug,
   regression, feature, behavior, or tests. Inspect the PR title, description,
   linked issue, changed files, and test diff.

3. Determine commits.

   Record:

   - `base_commit`: the commit used as the initial state before the PR change.
   - `merge_commit`: the merged/reference state that contains the expected
     behavior and the hidden tests.

   For a normal merge commit on the default branch, `base_commit` is usually
   the first parent of the merge commit and `merge_commit` is the merge commit.

   For squash-merged or rebase-merged PRs, only use the PR if the exact
   pre-change state and reference commit can be identified. If that cannot be
   done reliably, reject the candidate.

4. Create the task checkout.

   Clone or copy the repository into:

   ```text
   repositories/tasks/<task_id>/repo
   ```

   Then check out `base_commit`. The agent must start from this state, not from
   the merged PR state.

5. Extract hidden tests.

   Inspect the diff between `base_commit` and `merge_commit`. Copy relevant
   added or modified test files from the reference state into:

   ```text
   repositories/tasks/<task_id>/hidden_tests
   ```

   Mirror the repository's own test layout exactly. The evaluator overlays this
   tree onto the task repo before running the hidden tests, which is what lets
   in-package relative imports (e.g. `from .._events import ...`) resolve.
   Examples:

   ```text
   reference tests/test_url.py         -> hidden_tests/tests/test_url.py
   reference h11/tests/test_headers.py -> hidden_tests/h11/tests/test_headers.py
   ```

   Do not place hidden tests at a path that only runs from a sibling directory:
   once the package is no longer their parent, their imports fail on collection.

   Copy only the tests the PR adds or modifies for the target behavior. If the
   reference test file also contains unrelated tests (for example ones added by
   a different PR), they can fail on this PR's `merge_commit` and make the task
   look invalid. In that case scope the hidden test command to the specific
   test(s) rather than the whole file (see step 8).

   Hidden tests must not be copied into the task repository in a way that the
   agent can inspect during solving.

6. Write `task.md`.

   Create a concise natural-language task description from the issue or PR. The
   text should describe the expected behavior, constraints, and public API
   expectations. It must not include the solution diff, exact implementation
   steps, or hidden test code.

   A good task description answers:

   - what behavior is currently wrong or missing;
   - what behavior should happen instead;
   - what compatibility or public API constraints must be preserved.

7. Measure repository size.

   Count Python source files and LOC in the initial checkout, excluding tests,
   docs, examples, benchmarks, fuzzing, scripts, build artifacts, generated
   files, and vendored code. Record:

   - `source_files`;
   - `source_loc`;
   - `source_loc_nonblank`;
   - `test_files`;
   - `size`.

8. Identify test commands.

   Record a visible test command that the agent may run and that MUST pass on
   `base_commit` (verified in step 9). Prefer the normal project command if it
   is fast and green on base:

   ```text
   python -m pytest
   ```

   If the full suite is slow, flaky, or not fully green on `base_commit` —
   including tests that fail for reasons unrelated to the task, such as missing
   data files or optional benchmark tests — record a narrower but conventional
   subset that is green on base and still covers the regression. Point it at the
   relevant package or module and exclude unrelated folders such as `bench/` or
   `examples/` (for example `python -m pytest toolz` instead of bare
   `python -m pytest`).

   Record a hidden test command for the evaluator only. Write it against the
   path the hidden test has once overlaid into the repo. By convention the CSV
   stores it prefixed with `../hidden_tests/`; the evaluator strips that prefix,
   overlays `hidden_tests/` onto the repo, and runs it from the repo root:

   ```text
   python -m pytest ../hidden_tests/h11/tests/test_headers.py
   ```

   When the overlaid file contains tests beyond the PR's scope, target the exact
   test(s) the PR added with `::test_name`:

   ```text
   python -m pytest ../hidden_tests/testing/test_pluginmanager.py::test_unregister_plugin_with_multi_hookimpls
   ```

9. Validate the task.

   Before marking a task as verified, run these checks in a clean environment:

   - visible tests pass on `base_commit`;
   - hidden tests fail on `base_commit`;
   - hidden tests pass on `merge_commit` or an equivalent reference solution;
   - the task description does not reveal the solution;
   - the repository can be installed and tested reproducibly.

   Installation must work in the clean container, not only on the collector's
   machine. Two common gotchas seen in practice:

   - version-from-VCS build tools (e.g. `setuptools-scm`) need a reachable git
     tag or `SETUPTOOLS_SCM_PRETEND_VERSION=<x>` set at install time, or
     `pip install -e .` fails with a metadata-generation error;
   - install the project's test extras (e.g. `pip install -e .[tests]` or the
     packages the suite imports, such as `freezegun` or `pytest-benchmark`),
     not just `pytest`.

   Record the working install and test steps in `notes` so the evaluator can
   reproduce them.

10. Add one row to `collection.csv`.

    Use one row per task. Keep paths relative to the project root.

## CSV Format

The current CSV header is:

```csv
task_id,repo_path,task_file_path,size,task_type,repo_url,pr_number,pr_url,base_commit,merge_commit,hidden_tests_path,source_files,source_loc,source_loc_nonblank,patch_files,patch_loc,test_files,visible_test_command,hidden_test_command,hidden_semantic_test_command,hidden_compat_test_command,hidden_pr_parity_test_command,task_status,notes
```

Column meanings:

- `task_id`: stable identifier, usually `<repo_name>_pr_<number>`.
- `repo_path`: path to the initial task checkout at `base_commit`.
- `task_file_path`: path to the task description.
- `size`: `small`, `medium`, or `large`.
- `task_type`: `bugfix` or `feature` — the kind of change the PR makes, so
  results can be compared within a task type (features are typically harder and
  more open-ended than localized bug fixes).
- `repo_url`: upstream Git repository URL.
- `pr_number`: source PR number, if PR-derived.
- `pr_url`: source PR URL, if PR-derived.
- `base_commit`: initial state before the target change.
- `merge_commit`: reference state containing the expected behavior.
- `hidden_tests_path`: path to evaluator-only tests.
- `source_files`: Python source file count, excluding tests and non-source
  folders.
- `source_loc`: Python source LOC, excluding tests and non-source folders.
- `source_loc_nonblank`: non-blank Python source LOC with the same exclusions.
- `test_files`: Python test file count in the initial checkout.
- `visible_test_command`: test command visible to the agent.
- `hidden_test_command`: legacy/all evaluator-only hidden test command.
  References `../hidden_tests/<repo-relative path>`; the evaluator overlays
  `hidden_tests/` onto the repo and runs it from the repo root (the prefix is
  stripped).
- `hidden_semantic_test_command`: hidden tests that directly follow from
  `task.md`; these count toward task success.
- `hidden_compat_test_command`: hidden tests for preserving existing behavior;
  these count toward task success.
- `hidden_pr_parity_test_command`: hidden tests for exact upstream PR parity
  when the PR added behavior not stated in `task.md`; these are reported
  separately and do not count toward task success.
- `task_status`: validation status.
- `notes`: short screening and validation notes.
- `setup_commands`: optional `;`-separated container setup commands for this
  task (e.g. `SETUPTOOLS_SCM_PRETEND_VERSION=<x> python -m pip install -q -e .`
  or extra test deps). Empty means the evaluator defaults
  (`pip install -e .` + `pip install pytest`). Used by the runner, the agent
  sandbox, and `src.benchmark.validate` alike; a `--setup-command` CLI flag
  still overrides it.

Recommended `task_status` values:

- `candidate`: repository or PR is being screened.
- `pr_task_unverified`: task files exist, but dependency/test validation is not
  complete.
- `pr_task_verified`: visible/base and hidden/reference validation passed.
- `manual_task_unverified`: manually created task is not fully validated yet.
- `manual_task_verified`: manually created task passed validation.
- `rejected`: candidate was rejected; explain why in `notes`.

## Leakage Rules

During agent evaluation, expose only:

- the initial repository checkout;
- `task.md`;
- the visible test command;
- normal repository files present at `base_commit`.

Do not expose:

- `collection.csv`;
- `pr_url`, `pr_number`, `base_commit`, or `merge_commit`;
- hidden tests;
- the PR diff;
- the reference solution;
- notes that identify the implementation.

It is fine for `collection.csv` to contain reproducibility metadata because it
is evaluator-only metadata.

## Final Acceptance Checklist

A task can be used in the final benchmark only if all of the following are true:

- the task repository is checked out at `base_commit`;
- `task.md` is clear and does not reveal the solution;
- the hidden tests are outside the agent-visible repository path;
- visible tests pass on the initial checkout;
- hidden tests fail on the initial checkout;
- hidden tests pass on the reference state;
- dependencies can be installed in the benchmark environment;
- size metadata is measured and recorded;
- the CSV row is complete and paths exist;
- the task does not require network access, credentials, or external services.
