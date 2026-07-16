# Repository Collection

This directory stores PR-derived Python repository tasks for repository-level
agent evaluation.

## Layout

- `repos/`: shallow local checkouts of upstream repositories used as source
  pools.
- `tasks/<task_id>/repo`: the initial repository state for one benchmark task,
  checked out at the PR base commit.
- `tasks/<task_id>/task.md`: natural-language task text derived from the merged
  PR description, without the solution diff.
- `tasks/<task_id>/hidden_tests`: test files extracted from the merged PR state.
- `collection.csv`: one row per candidate repository.

## CSV Format

The first three columns are the minimal fields needed by the evaluator:

- `repo_path`: local path to the repository checkout.
- `task_file_path`: local path to the task description file.
- `size`: LOC bucket, using the project specification thresholds:
  `small` for 500-3000 Python source LOC, `medium` for 3001-15000 Python
  source LOC, and `large` above 15000 Python source LOC (extension bucket,
  reported separately from the core small/medium comparison).

Additional columns make the collection reproducible and easier to filter:

- `task_id`: stable task identifier.
- `repo_url`: upstream repository URL.
- `pr_number`, `pr_url`: source merged PR.
- `base_commit`: commit before the PR; `repo_path` is checked out here.
- `merge_commit`: merged state used as the reference source for hidden tests.
- `hidden_tests_path`: local directory with tests withheld from the agent.
- `source_files`: Python source file count, excluding tests, docs, examples,
  benchmarks, fuzzing, scripts, and downstream tooling.
- `source_loc`: Python source LOC with the same exclusions.
- `source_loc_nonblank`: non-blank Python source LOC with the same exclusions.
- `test_files`: Python test file count under conventional test directories.
- `visible_test_command`: suggested visible test command after dependencies are
  installed.
- `hidden_test_command`: legacy/all evaluator-only hidden test command.
- `hidden_semantic_test_command`: hidden tests that directly follow from the
  task description and count toward task success.
- `hidden_compat_test_command`: hidden tests that check preserved behavior and
  count toward task success. For an older local PR row with this column empty,
  the evaluator records the post-run `visible_test_command` as the required
  compatibility result; this makes the split explicit without inventing
  unreviewed hidden requirements. SWE-bench Pro rows are never given this
  fallback because they already carry their imported F2P/P2P split.
- `hidden_pr_parity_test_command`: hidden tests that check exact upstream PR
  parity but do not count toward task success.
- `task_status`: current validation status.
- `notes`: screening notes for later task creation.
- `setup_commands`: optional `;`-separated per-task container setup commands;
  empty means the evaluator defaults.

Before a row becomes a final benchmark task, verify that visible tests pass on
`base_commit`, hidden tests fail on `base_commit`, and hidden tests pass on
`merge_commit` or an equivalent reference solution.
