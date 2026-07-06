from __future__ import annotations

from pydantic import BaseModel, Field

from .actions import (
    AgentAction,
    EditFileAction,
    FinishAction,
    HandoffAction,
    InspectFileAction,
    ListDirectoryAction,
    RunShellAction,
    RunTestsAction,
    SearchAction,
    SetPlanAction,
    WriteFileAction,
)
from .sandbox import DockerSandbox
from .state import State
from .tracing import trace_action_execution
from .workspace import Workspace

MAX_OBSERVATION_CHARS = 12000
EDIT_SNIPPET_CHARS = 600
DENIED_SHELL_COMMANDS = (
    "git reset --hard",
    "git clean",
    "rm -rf /",
    "rm -rf .",
    "rm -rf /workspace",
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

    @trace_action_execution
    def execute(self, state: State, action: AgentAction) -> State:
        match action:
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
        lines = self.workspace.read_file(action.path).splitlines()
        total = len(lines)
        start = min(max(action.offset, 1), total or 1)
        window = lines[start - 1 : start - 1 + action.limit]
        end = start - 1 + len(window)
        header = f"File: {action.path} (lines {start}-{end} of {total})"
        if end < total:
            header += (
                f"  [+{total - end} more lines below — inspect_file with "
                f"offset={end + 1} to continue]"
            )
        rendered = self._truncate(f"{header}\n\n" + "\n".join(window))
        return (
            state.with_relevant_files([action.path])
            .with_observation(rendered)
            .with_context(kind="inspect_file", path=action.path, text=rendered)
            .advance_step()
        )

    def _list_dir(self, state: State, action: ListDirectoryAction) -> State:
        output = self._truncate(self.workspace.list_files(action.path))
        return (
            state.with_observation(output)
            .with_context(kind="list_dir", path=action.path, text=output)
            .advance_step()
        )

    def _search(self, state: State, action: SearchAction) -> State:
        output = self._truncate(
            self.workspace.search(action.query, action.path, action.max_results)
        )
        return (
            state.with_observation(output)
            .with_context(
                kind="search",
                path=action.path,
                text=f"query={action.query!r}\n{output}",
            )
            .advance_step()
        )

    def _write_file(self, state: State, action: WriteFileAction) -> State:
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
        result = self.sandbox.run_shell(action.command, action.timeout_seconds)
        output = self._truncate(result.output or "Command passed.")
        entry_text = f"$ {action.command}\n{output}"
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

        result = self.sandbox.run_tests(command, action.timeout_seconds)
        output = self._truncate(result.output or "Tests passed.")
        status = "PASSED" if result.success else "FAILED"
        return state.with_context(
            kind="run_tests", text=f"$ {command}\n{status}\n{output}"
        ).record_test_result(passed=result.success, output=output)

    def _finish(self, state: State, action: FinishAction) -> State:
        if not state.test_passed:
            reason = (
                "Finish rejected: run the visible tests first and make sure "
                "they pass before finishing."
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

    def _truncate(self, text: str) -> str:
        if len(text) <= self.max_observation_chars:
            return text
        return (
            f"... truncated to last {self.max_observation_chars} characters ...\n"
            f"{text[-self.max_observation_chars:]}"
        )


def _snippet(text: str, max_chars: int = EDIT_SNIPPET_CHARS) -> str:
    """Head of an edit payload, capped so diffs never flood the history."""
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars]}\n... truncated ({len(text)} chars total) ..."
