from __future__ import annotations

import re
import shlex
from pathlib import Path, PurePosixPath

from pydantic import BaseModel, Field

from .actions import (
    AgentAction,
    CompleteSubtaskAction,
    EditFileAction,
    FinishAction,
    HandoffAction,
    InspectFileAction,
    ListDirectoryAction,
    ReopenSubtaskAction,
    RunShellAction,
    RunTestsAction,
    SearchAction,
    SetPlanAction,
    SetSubtasksAction,
    WriteFileAction,
)
from .sandbox import DockerSandbox
from .state import State
from .tracing import trace_action_execution
from .workspace import (
    TEST_ORACLE_DIR_NAMES,
    Workspace,
    WorktreeSnapshot,
    is_test_oracle_path,
)

MAX_OBSERVATION_CHARS = 12000
EDIT_SNIPPET_CHARS = 600
DENIED_SHELL_COMMANDS = (
    "git reset --hard",
    "git clean",
    "rm -rf /",
    "rm -rf .",
    "rm -rf /workspace",
)

_TEST_LAUNCHERS = (
    "pytest",
    "py.test",
    "python -m pytest",
    "python3 -m pytest",
    "python -m unittest",
    "python3 -m unittest",
)
_SHELL_OPERATORS = ("&&", "||", ";", "|", ">", "<")
_ENV_ASSIGNMENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")


def _is_test_invocation(command: str) -> bool:
    """A bare test-runner call (no shell composition), e.g. ``pytest -x tests``.

    Agents often run pytest through run_shell instead of run_tests. Left
    unrecorded, that never sets ``test_passed`` (so finish is always rejected)
    and gives unlimited test feedback outside the iteration budget. Compound
    commands are skipped: their exit code can't be attributed to the tests.
    """
    if any(op in command for op in _SHELL_OPERATORS):
        return False
    parts = command.strip().split()
    while parts and _ENV_ASSIGNMENT.match(parts[0]):
        parts = parts[1:]
    normalized = " ".join(parts)
    return any(
        normalized == launcher or normalized.startswith(f"{launcher} ")
        for launcher in _TEST_LAUNCHERS
    )


def _is_test_path(path: str) -> bool:
    """Conservative benchmark-test path detection.

    Existing test files are part of the evaluator oracle and must never be
    rewritten by an agent. New regression tests remain allowed for the single
    agent and tester role.
    """
    return is_test_oracle_path(path)


_SAFE_FULL_SUITE_SUFFIXES = frozenset(
    {
        "-v",
        "-vv",
        "-q",
        "-qq",
        "-s",
        "-x",
        "--disable-warnings",
        "--strict-config",
        "--strict-markers",
    }
)
_SAFE_FULL_SUITE_PREFIXES = (
    "--color=",
    "--durations=",
    "--maxfail=",
    "--tb=",
)


def _is_full_test_command(command: str | None, configured: str | None) -> bool:
    """Whether a test invocation covers the configured visible suite.

    Exact matches are accepted, as are suffix-only reporting/execution flags
    such as ``-v``. Extra paths and selectors (``-k``, ``-m``, ``--deselect``)
    remain focused runs and cannot unlock finish.
    """
    if not command or not configured:
        return False
    try:
        actual = shlex.split(command)
        baseline = shlex.split(configured)
    except ValueError:
        return False
    if actual[: len(baseline)] != baseline:
        return False
    extras = actual[len(baseline) :]
    return all(
        extra in _SAFE_FULL_SUITE_SUFFIXES
        or extra.startswith(_SAFE_FULL_SUITE_PREFIXES)
        for extra in extras
    )


class ActionExecutor(BaseModel):
    """Applies one parsed agent action to the task state.

    File actions go through the workspace, shell and test actions through the
    Docker sandbox. The executor is agent-agnostic, so the single-agent loop
    and multi-agent roles can share the same action semantics.
    """

    workspace: Workspace
    sandbox: DockerSandbox
    denied_shell_commands: tuple[str, ...] = DENIED_SHELL_COMMANDS
    max_observation_chars: int = Field(default=MAX_OBSERVATION_CHARS, gt=0)

    def model_post_init(self, __context: object) -> None:
        """Install the immutable test-oracle boundary before Docker starts."""
        protected_files: list[str] = []
        protected_directories: set[str] = set()
        for path in self.workspace.baseline_tracked_files():
            if not is_test_oracle_path(path):
                continue
            parts = PurePosixPath(path).parts[:-1]
            test_root: str | None = None
            for index, part in enumerate(parts, start=1):
                if part.lower() in TEST_ORACLE_DIR_NAMES:
                    test_root = PurePosixPath(*parts[:index]).as_posix()
                    break
            if test_root is None:
                protected_files.append(path)
            else:
                protected_directories.add(test_root)
        self.sandbox.configure_protected_test_paths(
            protected_files,
            readonly_directories=sorted(protected_directories),
        )

    @trace_action_execution
    def execute(self, state: State, action: AgentAction) -> State:
        """Execute an action, representing policy/input rejection in state.

        Syntactically valid actions can still violate role/sandbox policy.
        Keeping that rejection inside this traced boundary prevents LangSmith
        from recording a noisy exception span while still feeding the exact
        policy error back to the agent on its next step.
        """
        try:
            return self._execute(state, action)
        except ValueError as exc:
            return (
                state.with_error(str(exc))
                .with_context(kind="policy_rejection", text=f"error: {exc}")
                .advance_step()
            )

    def _execute(self, state: State, action: AgentAction) -> State:
        if state.decomposition_required and not state.subtasks:
            allowed_initial_list = (
                isinstance(action, ListDirectoryAction)
                and state.action_counts.get("list_dir", 0) == 0
            )
            if not (
                allowed_initial_list
                or isinstance(action, (SetSubtasksAction, HandoffAction))
            ):
                raise ValueError(
                    "Decomposed mode requires set_subtasks immediately after "
                    "the initial list_dir. Create the structured queue before "
                    "inspection, search, edits, or tests."
                )
        match action:
            case SetSubtasksAction():
                from .decomposition import initialize_decomposition

                return initialize_decomposition(
                    state, action, max_subtasks=state.decomposition_max_subtasks
                )
            case CompleteSubtaskAction():
                from .decomposition import complete_subtask

                return complete_subtask(state, action)
            case ReopenSubtaskAction():
                from .decomposition import reopen_subtask

                return reopen_subtask(state, action)
            case SetPlanAction():
                return self._set_plan(state, action)
            case InspectFileAction():
                return self._inspect_file(state, action)
            case ListDirectoryAction():
                return self._list_dir(state, action)
            case SearchAction():
                return self._search(state, action)
            case WriteFileAction():
                return self._write_file(state, action)
            case EditFileAction():
                return self._edit_file(state, action)
            case RunShellAction():
                return self._run_shell(state, action)
            case RunTestsAction():
                return self._run_tests(state, action)
            case FinishAction():
                return self._finish(state, action)
            case HandoffAction():
                return state.mark_handoff(action.reason).with_context(
                    kind="handoff", text=action.reason
                )
        raise ValueError(f"Unsupported action type: {type(action).__name__}")

    def _set_plan(self, state: State, action: SetPlanAction) -> State:
        return (
            state.with_plan(action.plan)
            .with_context(kind="set_plan", text=action.plan)
            .advance_step()
        )

    def _inspect_file(self, state: State, action: InspectFileAction) -> State:
        rel_path = self.workspace.to_relative(self.workspace.resolve(action.path))
        lines = self.workspace.read_file(rel_path).splitlines()
        total = len(lines)
        start = min(max(action.offset, 1), total or 1)
        window = lines[start - 1 : start - 1 + action.limit]
        end = start - 1 + len(window)
        header = f"File: {rel_path} (lines {start}-{end} of {total})"
        if end < total:
            header += (
                f"  [+{total - end} more lines below — inspect_file with "
                f"offset={end + 1} to continue]"
            )
        rendered = self._truncate(f"{header}\n\n" + "\n".join(window))
        return (
            state.with_relevant_files([rel_path])
            .with_observation(rendered)
            .with_context(kind="inspect_file", path=rel_path, text=rendered)
            .record_research_step()
            .advance_step()
        )

    def _list_dir(self, state: State, action: ListDirectoryAction) -> State:
        rel_path = self.workspace.to_relative(self.workspace.resolve(action.path))
        output = self._truncate(self.workspace.list_files(rel_path))
        return (
            state.with_observation(output)
            .with_context(kind="list_dir", path=rel_path, text=output)
            .record_research_step()
            .advance_step()
        )

    def _search(self, state: State, action: SearchAction) -> State:
        rel_path = self.workspace.to_relative(self.workspace.resolve(action.path))
        key = self.workspace.search_key(
            action.query, rel_path, action.max_results, action.mode
        )
        cached = state.search_results.get(key)
        if cached is not None:
            output = self._truncate(
                "Cached result from the same search before compaction. Reuse "
                "it and move to planning or implementation; do not broaden "
                f"the same query again.\n{cached}"
            )
            return (
                state.with_observation(output)
                .with_context(
                    kind="search_reused",
                    path=rel_path,
                    text=f"query={action.query!r}\n{output}",
                )
                .record_research_step()
                .advance_step()
            )

        output = self._truncate(
            self.workspace.search(
                action.query, rel_path, action.max_results, action.mode
            )
        )
        return (
            state.record_search_result(key, output)
            .with_observation(output)
            .with_context(
                kind="search",
                path=rel_path,
                text=f"query={action.query!r}\n{output}",
            )
            .record_research_step()
            .advance_step()
        )

    def _write_file(self, state: State, action: WriteFileAction) -> State:
        self._validate_file_change(state, action.path, is_write=True)
        rel_path = self.workspace.write_file(action.path, action.content)
        observation = f"Wrote {rel_path} ({len(action.content)} chars)"
        entry_text = f"{observation}\n{_snippet(action.content)}"
        return (
            state.with_changed_files([rel_path])
            .invalidate_path(rel_path)
            .with_observation(observation)
            .with_context(kind="write_file", path=rel_path, text=entry_text)
            .advance_step()
        )

    def _edit_file(self, state: State, action: EditFileAction) -> State:
        self._validate_file_change(state, action.path, is_write=False)
        rel_path, count = self.workspace.edit_file(
            action.path,
            action.old_string,
            action.new_string,
            action.replace_all,
        )
        replacements = "replacement" if count == 1 else "replacements"
        observation = f"Edited {rel_path} ({count} {replacements})"
        entry_text = (
            f"{observation}\n"
            f"--- removed\n{_snippet(action.old_string)}\n"
            f"+++ inserted\n{_snippet(action.new_string)}"
        )
        return (
            state.with_changed_files([rel_path])
            .invalidate_path(rel_path)
            .with_observation(observation)
            .with_context(kind="edit_file", path=rel_path, text=entry_text)
            .advance_step()
        )

    def _run_shell(self, state: State, action: RunShellAction) -> State:
        self._validate_shell_command(action.command)
        if state.active_role == "reviewer":
            self._validate_reviewer_command(action.command)
        before = self._snapshot_before_shell()
        result = self.sandbox.run_shell(action.command, action.timeout_seconds)
        state, tampered, mutation_error = self._reconcile_shell_mutations(
            state, before
        )
        output = self._truncate(result.output or "Command passed.", keep_tail=True)
        entry_text = f"$ {action.command}\n{output}"
        if tampered or mutation_error:
            reason = mutation_error or (
                "Protected benchmark tests were modified through the shell "
                f"and restored: {', '.join(tampered)}."
            )
            return (
                state.with_error(reason)
                .with_context(
                    kind="policy_rejection",
                    text=f"{entry_text}\nerror: {reason}",
                )
                .advance_step()
            )
        if _is_test_invocation(action.command):
            # Same bookkeeping as run_tests: sets test_passed (unblocks
            # finish) and consumes one test iteration from the budget.
            status = "PASSED" if result.success else "FAILED"
            note = "(counted as a test run — prefer the run_tests action)"
            return state.with_context(
                kind="run_shell",
                text=f"$ {action.command}\n{status}\n{output}\n{note}",
            ).record_test_result(
                passed=result.success,
                output=output,
                command=action.command,
                full_suite=_is_full_test_command(action.command, state.test_command),
            )
        if result.success:
            return (
                state.with_status("running")
                .with_observation(output)
                .with_context(kind="run_shell", text=entry_text)
                .advance_step()
            )
        return (
            state.with_error(output)
            .with_context(kind="run_shell", text=f"{entry_text}\n-> command failed")
            .advance_step()
        )

    def _run_tests(self, state: State, action: RunTestsAction) -> State:
        command = action.command or state.test_command
        if not command:
            return (
                state.with_error("No test command configured.")
                .with_context(
                    kind="run_tests",
                    text="error: no test command configured",
                )
                .advance_step()
            )

        compatibility_check = action.purpose == "compatibility"
        if compatibility_check:
            if not action.command:
                raise ValueError(
                    "A compatibility check needs an explicit focused command "
                    "that exercises pre-existing behavior."
                )
            if _is_full_test_command(command, state.test_command):
                raise ValueError(
                    "The configured full suite is verification, not a focused "
                    "compatibility check. Probe a concrete legacy behavior."
                )
            changed_tests = [
                str(path) for path in state.changed_files if _is_test_path(str(path))
            ]
            if any(path in command for path in changed_tests):
                raise ValueError(
                    "Compatibility checks cannot target a newly created or "
                    "changed regression test. Exercise pre-existing behavior "
                    "directly or target an unchanged test."
                )

        before = self._snapshot_before_shell()
        result = self.sandbox.run_tests(command, action.timeout_seconds)
        state, tampered, mutation_error = self._reconcile_shell_mutations(
            state, before
        )
        output = self._truncate(result.output or "Tests passed.", keep_tail=True)
        if tampered or mutation_error:
            reason = mutation_error or (
                "Protected benchmark tests were modified during the test run "
                f"and restored: {', '.join(tampered)}."
            )
            return (
                state.with_error(reason)
                .with_context(
                    kind="policy_rejection",
                    text=f"$ {command}\n{output}\nerror: {reason}",
                )
                .advance_step()
            )
        status = "PASSED" if result.success else "FAILED"
        return state.with_context(
            kind=("compatibility_check" if compatibility_check else "run_tests"),
            text=f"$ {command}\n{status}\n{output}",
        ).record_test_result(
            passed=result.success,
            output=output,
            command=command,
            full_suite=_is_full_test_command(command, state.test_command),
            compatibility_check=compatibility_check,
            # Required successful verification is overhead, not a repair
            # attempt. A failed compatibility probe still consumes budget.
            consume_iteration=not (compatibility_check and result.success),
        )

    def _finish(self, state: State, action: FinishAction) -> State:
        if state.decomposition_required and not state.decomposition_complete:
            pending = ", ".join(
                item.id for item in state.subtasks if item.status != "completed"
            ) or "decomposition not created"
            reason = (
                "Finish rejected: all structured subtasks must be completed. "
                f"Remaining: {pending}."
            )
            return (
                state.with_error(reason)
                .with_context(kind="finish", text=f"rejected: {reason}")
                .advance_step()
            )
        if (
            state.compatibility_check_required
            and (
                not state.compatibility_check_passed
                or state.compatibility_verified_revision != state.workspace_revision
            )
        ):
            reason = (
                "Finish rejected: run a focused passing compatibility check "
                "after the latest edit. Use run_tests with "
                "purpose=compatibility to exercise concrete pre-existing "
                "behavior near the changed code; the full suite and a newly "
                "created regression test do not satisfy this gate."
            )
            return (
                state.with_error(reason)
                .with_context(kind="finish", text=f"rejected: {reason}")
                .advance_step()
            )
        if (
            not state.full_test_passed
            or state.full_suite_verified_revision != state.workspace_revision
        ):
            reason = (
                "Finish rejected: run the configured full visible test command "
                "after the latest edit and make sure it passes. A focused test "
                "does not certify the final workspace."
            )
            return (
                state.with_error(reason)
                .with_context(kind="finish", text=f"rejected: {reason}")
                .advance_step()
            )
        if not state.changed_files:
            reason = (
                "Finish rejected: no files were changed. The visible tests "
                "may pass even before the task is done; you must implement "
                "the change the task describes before finishing."
            )
            return (
                state.with_error(reason)
                .with_context(kind="finish", text=f"rejected: {reason}")
                .advance_step()
            )
        return (
            state.with_status("solved")
            .with_observation(action.summary)
            .with_context(kind="finish", text=action.summary)
            .advance_step()
        )

    def _validate_shell_command(self, command: str) -> None:
        normalized = " ".join(command.lower().split())
        for denied in self.denied_shell_commands:
            if denied in normalized:
                raise ValueError(f"Denied shell command: {command}")

    def _snapshot_before_shell(self) -> WorktreeSnapshot | None:
        """Take a bounded snapshot only for real Git benchmark workspaces."""
        if not self.workspace.is_git_checkout():
            return None
        return self.workspace.worktree_snapshot()

    def _reconcile_shell_mutations(
        self,
        state: State,
        before: WorktreeSnapshot | None,
    ) -> tuple[State, list[str], str | None]:
        """Track shell writes and restore any tracked test-oracle changes."""
        if before is None:
            return state, [], None
        try:
            after = self.workspace.worktree_snapshot()
        except ValueError as exc:
            # The command crossed the workspace trust boundary in a way we
            # cannot audit. Invalidate prior verification and fail closed.
            return (
                state.with_changed_files([]),
                [],
                f"Could not verify shell workspace mutations: {exc}",
            )

        head_changed = before.head != after.head
        raw_changed = before.changed_paths(after)
        protected = [
            path
            for path in raw_changed
            if _is_test_path(path) and self.workspace.is_tracked(path)
        ]
        if protected:
            state = state.record_test_oracle_tamper()
        try:
            if head_changed:
                self.workspace.restore_git_head()
            self.workspace.restore_tracked_files(protected)
            final = self.workspace.worktree_snapshot()
        except ValueError as exc:
            return (
                state.with_changed_files([]),
                protected,
                f"Could not restore protected workspace state: {exc}",
            )

        changed = before.changed_paths(final)
        if changed or head_changed:
            changed_files: list[Path | str] = list(changed)
            state = state.with_changed_files(changed_files)
            # Shell actions can rewrite files without going through the
            # structured edit/write handlers.  Invalidate any earlier file
            # snapshots here as well so context compaction cannot rescue a
            # byte-for-byte view of content that is no longer current.
            for path in changed_files:
                state = state.invalidate_path(str(path))
        return state, protected, None

    def _validate_file_change(self, state: State, path: str, *, is_write: bool) -> None:
        is_test = _is_test_path(path)
        tracked = self.workspace.is_tracked(path)

        if is_test and tracked:
            raise ValueError(
                f"Existing benchmark tests are read-only: {path}. "
                "Fix production code or create a new regression test file."
            )
        if state.active_role == "developer" and is_test:
            raise ValueError(
                f"Developer role cannot change tests: {path}. "
                "Hand verification to the tester role."
            )
        if state.active_role == "tester" and not is_test:
            raise ValueError(
                f"Tester role can only create new test files, not production "
                f"code: {path}."
            )
        if not is_write and is_test:
            # Kept explicit even though an edit target necessarily exists.
            raise ValueError(f"Existing benchmark tests are read-only: {path}.")

    @staticmethod
    def _validate_reviewer_command(command: str) -> None:
        """Reviewer shell access is read-only by construction, not by prompt."""
        if any(op in command for op in _SHELL_OPERATORS):
            raise ValueError("Reviewer commands cannot use shell operators.")
        try:
            parts = shlex.split(command)
        except ValueError as exc:
            raise ValueError(f"Invalid reviewer command: {exc}") from exc
        allowed = parts[:2] == ["git", "diff"] or parts == ["git", "status", "--short"]
        unsafe_options = {"--output", "--ext-diff"}
        if not allowed or any(
            part in unsafe_options or part.startswith("--output=") for part in parts
        ):
            raise ValueError(
                "Reviewer role is read-only; allowed commands are `git diff` "
                "and `git status --short`."
            )

    def _truncate(self, text: str, *, keep_tail: bool = False) -> str:
        if len(text) <= self.max_observation_chars:
            return text
        if not keep_tail:
            return (
                f"{text[: self.max_observation_chars]}\n"
                f"... truncated after {self.max_observation_chars} characters ..."
            )
        return (
            f"... truncated to last {self.max_observation_chars} characters ...\n"
            f"{text[-self.max_observation_chars :]}"
        )


def _snippet(text: str, max_chars: int = EDIT_SNIPPET_CHARS) -> str:
    """Head of an edit payload, capped so diffs never flood the history."""
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars]}\n... truncated ({len(text)} chars total) ..."
