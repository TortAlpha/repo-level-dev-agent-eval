from __future__ import annotations

from pydantic import BaseModel, Field

from .actions import AgentAction, parse_action
from .context import CompactionMode, ContextCompactor
from .executor import ActionExecutor
from .model import LangChainModel
from .prompts import SINGLE_AGENT_PROMPT
from .sandbox import DockerSandbox
from .state import ContextBudget, State
from .tracing import trace_agent_run, trace_agent_step
from .workspace import Workspace


class SingleAgent(BaseModel):
    """Baseline loop where one model plans, edits, tests, and decides."""

    model: LangChainModel
    docker_image: str = "python:3.11-slim"
    docker_network_disabled: bool = True
    shell_timeout_seconds: int = Field(default=300, gt=0)
    test_timeout_seconds: int = Field(default=600, gt=0)
    stop_container: bool = True
    context_window_tokens: int = Field(default=8192, gt=0)
    max_response_tokens: int = Field(default=4096, gt=0)
    compaction_mode: CompactionMode = "summarize"
    max_repeated_actions: int = Field(default=3, gt=1)
    setup_commands: list[str] = Field(default_factory=list)

    @trace_agent_run
    def run(self, state: State) -> State:
        workspace = Workspace(root=state.workdir)
        sandbox = DockerSandbox(
            workdir=state.workdir,
            image=self.docker_image,
            network_disabled=self.docker_network_disabled,
            shell_timeout_seconds=self.shell_timeout_seconds,
            test_timeout_seconds=self.test_timeout_seconds,
        )
        executor = ActionExecutor(workspace=workspace, sandbox=sandbox)
        compactor = ContextCompactor(
            budget=ContextBudget(
                window_tokens=self.context_window_tokens,
                reserved_output_tokens=self.max_response_tokens,
                overhead_chars=len(SINGLE_AGENT_PROMPT),
            ),
            mode=self.compaction_mode,
            model=self.model,
        )

        try:
            state = state.with_container(sandbox.start())
            state = self._run_setup(state, sandbox)

            previous_action: AgentAction | None = None
            while state.can_continue:
                state, previous_action = self._step(
                    state, executor, previous_action
                )
                state = compactor.apply(state)

            return self._finalize_if_needed(state)
        finally:
            if self.stop_container:
                sandbox.stop()

    def _run_setup(self, state: State, sandbox: DockerSandbox) -> State:
        """Run one-off environment setup (e.g. installing dependencies) before
        the agent loop, recording each command in the history so the model
        knows the environment is ready. Requires network access in the sandbox.
        """
        for command in self.setup_commands:
            result = sandbox.run_shell(command, self.shell_timeout_seconds)
            status = "ok" if result.success else "FAILED"
            text = f"$ {command}\n[{status}]"
            if not result.success:
                text += f"\n{result.output[-1000:]}"
            state = state.with_context(kind="setup", text=text)
        return state

    @trace_agent_step
    def _step(
        self,
        state: State,
        executor: ActionExecutor,
        previous_action: AgentAction | None,
    ) -> tuple[State, AgentAction | None]:
        try:
            response = self.model.generate(SINGLE_AGENT_PROMPT, state)
        except Exception as exc:  # noqa: BLE001 - a dead endpoint ends the run, not the process
            reason = f"Model request failed: {exc}"
            return (
                state.mark_handoff(reason).with_context(
                    kind="model_error", text=reason
                ),
                previous_action,
            )

        try:
            action = parse_action(response)
        except ValueError as exc:
            return (
                state.with_error(str(exc))
                .with_context(kind="invalid_action", text=f"error: {exc}")
                .with_last_action_json(response)
                .advance_step(),
                previous_action,
            )

        action_json = action.model_dump_json()
        if action == previous_action:
            return (
                self._reject_repeat(state, action).with_last_action_json(
                    action_json
                ),
                action,
            )

        try:
            return (
                executor.execute(state, action).with_last_action_json(
                    action_json
                ),
                action,
            )
        except ValueError as exc:
            return (
                state.with_error(str(exc))
                .with_context(kind="invalid_action", text=f"error: {exc}")
                .with_last_action_json(action_json)
                .advance_step(),
                action,
            )

    def _reject_repeat(self, state: State, action: AgentAction) -> State:
        """Push back on a verbatim-repeated action, escalating each time.

        Every rejection is worded differently: a wall of identical entries
        in the history is itself a repetition pattern that locks greedy
        decoding into the loop. After max_repeated_actions attempts the run
        ends as a no-progress handoff instead of burning the step budget.
        """
        attempt = self._repeat_streak(state) + 1
        kind = type(action).__name__

        if attempt >= self.max_repeated_actions:
            reason = (
                f"No progress: {kind} was repeated and rejected "
                f"{attempt} times in a row."
            )
            return state.mark_handoff(reason).with_context(
                kind="repeated_action", text=reason
            )

        if attempt == 1:
            reason = (
                f"Repeated action rejected: this {kind} is identical to the "
                "previous step and its result is already shown above. Move "
                "forward instead: for example, inspect_file one of the paths "
                "already listed in the history, or search for code related "
                "to the task."
            )
        else:
            reason = (
                f"Repeated action rejected again ({attempt} times). FINAL "
                "WARNING: if you send this same action once more, the task "
                "will be handed off as unsolved. You must choose a different "
                "action now: inspect_file a specific path, search, edit_file, "
                "run_tests, or set_plan."
            )
        return (
            state.with_error(reason)
            .with_context(kind="repeated_action", text=reason)
            .advance_step()
        )

    def _repeat_streak(self, state: State) -> int:
        streak = 0
        for entry in reversed(state.context):
            if entry.kind != "repeated_action":
                break
            streak += 1
        return streak

    def _finalize_if_needed(self, state: State) -> State:
        if state.is_terminal:
            return state
        if state.step >= state.max_steps:
            return state.with_error("Maximum action count reached.").with_status(
                "failed"
            )
        if state.iteration >= state.max_iterations:
            return (
                state.with_error("Maximum test iteration count reached.")
                .with_status("failed")
            )
        return state.mark_handoff(
            "Agent loop stopped before reaching a terminal state."
        )
