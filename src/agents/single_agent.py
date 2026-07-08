from __future__ import annotations

from pydantic import BaseModel, Field, PrivateAttr

from .actions import ActionParseError, AgentAction, same_action
from .context import CompactionMode, ContextCompactor
from .executor import ActionExecutor
from .model import LangChainModel
from .prompts import SINGLE_AGENT_PROMPT
from .sandbox import DockerSandbox
from .state import ContextBudget, State
from .tracing import trace_agent_run, trace_agent_step
from .transport import ActionTransport, resolve_action_transport
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
    # Compactor working budget; None = the full window (see Config).
    context_budget_tokens: int | None = Field(default=None, gt=0)
    max_response_tokens: int = Field(default=4096, gt=0)
    compaction_mode: CompactionMode = "summarize"
    max_repeated_actions: int = Field(default=3, gt=1)
    max_parse_failures: int = Field(default=5, gt=0)
    action_transport: ActionTransport = "text_json"
    setup_commands: list[str] = Field(default_factory=list)

    # Cached probe result: resolving auto/tools costs one live request.
    _transport_cache: tuple[str, bool] | None = PrivateAttr(default=None)

    def _resolve_transport(self) -> tuple[str, bool]:
        if self._transport_cache is None:
            self._transport_cache = resolve_action_transport(
                self.action_transport, self.model
            )
        return self._transport_cache

    @property
    def resolved_action_transport(self) -> str:
        return self._resolve_transport()[0]

    @property
    def transport_downgraded(self) -> bool:
        """True when an explicit ``tools`` request failed the live probe and
        the run fell back to text JSON."""
        return self._resolve_transport()[1]

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
                window_tokens=min(
                    self.context_budget_tokens or self.context_window_tokens,
                    self.context_window_tokens,
                ),
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
            generated = self.model.generate_action(
                SINGLE_AGENT_PROMPT,
                state,
                self.resolved_action_transport,
            )
        except ActionParseError as exc:
            return (
                self._reject_parse_failure(state, exc, exc.raw_response),
                previous_action,
            )
        # A dead endpoint ends the run, not the whole benchmark process.
        except ValueError as exc:
            return (
                state.with_error(str(exc))
                .with_context(kind="invalid_action", text=f"error: {exc}")
                .with_last_action_json("")
                .advance_step(),
                previous_action,
            )
        except Exception as exc:  # noqa: BLE001
            reason = f"Model request failed: {exc}"
            return (
                state.mark_handoff(reason).with_context(
                    kind="model_error", text=reason
                ),
                previous_action,
            )

        action = generated.action
        action_json = generated.action_json
        tool_call_id = generated.tool_call_id or None
        tool_name = generated.tool_name or None
        if same_action(action, previous_action):
            return (
                self._reject_repeat(state, action).with_last_action_json(
                    action_json, tool_call_id, tool_name
                ),
                action,
            )

        try:
            return (
                executor.execute(state, action).with_last_action_json(
                    action_json, tool_call_id, tool_name
                ),
                action,
            )
        except ValueError as exc:
            return (
                state.with_error(str(exc))
                .with_context(kind="invalid_action", text=f"error: {exc}")
                .with_last_action_json(action_json, tool_call_id, tool_name)
                .advance_step(),
                action,
            )

    def _reject_parse_failure(
        self, state: State, error: ActionParseError, response: str
    ) -> State:
        """Handle model responses that are not executable JSON actions.

        Reasoning-capable models can produce reasoning-only turns. Retrying a
        few times is useful; retrying until max_steps only burns budget and
        hides the failure mode behind a pile of identical invalid actions.
        """
        attempt = self._parse_failure_streak(state) + 1
        kind = error.kind

        if attempt >= self.max_parse_failures:
            reason = (
                "No executable action returned after "
                f"{attempt} consecutive model responses. Last failure "
                f"({kind}): {error}"
            )
            return (
                state.mark_handoff(reason)
                .with_context(kind=kind, text=reason)
                .with_last_action_json(response)
            )

        if self.resolved_action_transport == "tools":
            expected = "Call exactly one of the provided action tools."
        else:
            expected = "Return exactly one JSON action object and no reasoning text."
        guidance = (
            f"error: {error}\n"
            f"Attempt {attempt}/{self.max_parse_failures}. {expected}"
        )
        return (
            state.with_error(str(error))
            .with_context(kind=kind, text=guidance)
            .with_last_action_json(response)
            .advance_step()
        )

    def _parse_failure_streak(self, state: State) -> int:
        parse_failure_kinds = {"no_action", "malformed_action", "invalid_action"}
        streak = 0
        for entry in reversed(state.context):
            if entry.kind not in parse_failure_kinds:
                break
            streak += 1
        return streak

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
