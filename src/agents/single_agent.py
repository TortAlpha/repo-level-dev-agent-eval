from __future__ import annotations

from pydantic import BaseModel, Field, PrivateAttr

from .actions import (
    ActionParseError,
    AgentAction,
    InspectFileAction,
    ListDirectoryAction,
    SearchAction,
    same_action,
)
from .context import CompactionMode, ContextCompactor
from .executor import ActionExecutor
from .model import ActionGenerationFailure, LangChainModel
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
    # Strong-baseline progress guard. Multi-agent roles disable this because
    # their own short role quotas already bound exploration; the single loop
    # otherwise has no structural transition from research to implementation.
    research_guard_enabled: bool = True
    research_warning_steps: int = Field(default=12, gt=0)
    research_hard_limit: int = Field(default=20, gt=0)
    post_plan_research_warning_steps: int = Field(default=5, gt=0)
    post_plan_research_hard_limit: int = Field(default=8, gt=0)
    max_research_guard_violations: int = Field(default=2, gt=0)
    compatibility_guard_enabled: bool = True
    decomposition_enabled: bool = False
    max_subtasks: int = Field(default=8, ge=4, le=12)
    decomposition_implementation_warning_steps: int = Field(default=8, gt=0)
    decomposition_implementation_hard_limit: int = Field(default=14, gt=0)
    decomposition_verification_step_reserve: int = Field(default=20, ge=0)
    max_repair_cycles: int = Field(default=2, ge=0)
    action_transport: ActionTransport = "text_json"
    setup_commands: list[str] = Field(default_factory=list)
    # Optional system-level dollar guard. It is exact for single-agent runs
    # and checked between role episodes for multi-model agents.
    max_cost_usd: float | None = Field(default=None, gt=0)
    # Multi-agent parametrization: a role is this same loop with its own
    # prompt, a restricted action vocabulary, and role-terminal actions that
    # are captured by the caller instead of reaching the executor.
    system_prompt: str = SINGLE_AGENT_PROMPT
    action_space: str | None = None
    intercept_actions: frozenset[str] = frozenset()

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

    def usage_models(self) -> list[LangChainModel]:
        return [self.model]

    def estimated_cost_usd(self) -> float | None:
        costs = [model.estimated_cost_usd for model in self.usage_models()]
        if any(cost is None for cost in costs):
            return None
        return sum(cost for cost in costs if cost is not None)

    def _enforce_cost_limit(self, state: State) -> State:
        if self.max_cost_usd is None or state.is_terminal:
            return state
        cost = self.estimated_cost_usd()
        if cost is None or cost < self.max_cost_usd:
            return state
        reason = f"Cost limit reached: ${cost:.4f} >= ${self.max_cost_usd:.4f}."
        return state.mark_handoff(reason).with_context(kind="cost_limit", text=reason)

    @trace_agent_run
    def run(self, state: State) -> State:
        workspace = Workspace(root=state.workdir)
        # Setup may download dependencies. Start temporarily online, then
        # disconnect before the first model-controlled action.
        sandbox = DockerSandbox(
            workdir=state.workdir,
            image=self.docker_image,
            network_disabled=(
                self.docker_network_disabled and not self.setup_commands
            ),
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
                overhead_chars=len(self.system_prompt),
            ),
            mode=self.compaction_mode,
            model=self.model,
        )

        try:
            state = state.with_compatibility_requirement(
                self.compatibility_guard_enabled
            )
            state = state.with_decomposition_requirement(
                self.decomposition_enabled,
                self.max_subtasks,
                self.max_repair_cycles,
            )
            state = state.with_container(sandbox.start())
            state = self._run_setup(state, sandbox)
            if self.docker_network_disabled:
                sandbox.disable_network()
                state = state.with_context(
                    kind="setup", text="Benchmark network isolated after setup."
                )

            previous_action: AgentAction | None = None
            while state.can_continue:
                state, previous_action = self._step(state, executor, previous_action)
                state = self._enforce_cost_limit(state)
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
                raise RuntimeError(
                    f"Benchmark environment setup failed ({command}):\n"
                    f"{result.output[-1500:]}"
                )
            state = state.with_context(kind="setup", text=text)
        return state

    def _research_limits(self, state: State) -> tuple[str, int, int]:
        if state.active_subtask and state.active_subtask.kind == "investigate":
            return (
                "subtask investigation",
                self.research_warning_steps,
                self.research_hard_limit,
            )
        if state.plan.strip():
            return (
                "implementation",
                self.post_plan_research_warning_steps,
                self.post_plan_research_hard_limit,
            )
        return "exploration", self.research_warning_steps, self.research_hard_limit

    def _action_space_for_state(self, state: State) -> str | None:
        """Hook for architectures with phase-specific action vocabularies."""
        return self.action_space

    def _with_research_warning(self, state: State) -> State:
        if not self.research_guard_enabled:
            return state
        phase, warning, hard = self._research_limits(state)
        already_warned = bool(
            state.research_warning_phase
            and state.research_warning_phase.startswith(phase)
        )
        if state.research_streak < warning or already_warned:
            return state
        text = (
            f"Read-only {phase} budget is nearly exhausted "
            f"({state.research_streak}/{hard}). Consolidate the evidence now. "
        )
        if phase == "exploration":
            text += "The next useful action should be set_plan."
        elif phase == "subtask investigation":
            text += (
                "If the objective is understood, use complete_subtask now "
                "with concrete findings. Otherwise inspect only the exact "
                "remaining evidence needed for that subtask."
            )
        else:
            text += (
                "Re-inspect only an exact edit target if necessary, then use "
                "edit_file/write_file, run tests, or hand off with a blocker."
            )
        return state.with_context(
            kind="research_warning", text=text
        ).record_research_warning(phase)

    def _reject_excess_research(self, state: State) -> State:
        phase, _, hard = self._research_limits(state)
        state = state.record_research_guard_violation()
        reason = (
            f"Read-only {phase} limit reached after {state.research_streak} "
            f"steps (limit {hard}). Repeated search/inspection is no longer "
            "allowed. "
        )
        if phase == "exploration":
            reason += "Use set_plan next, or hand off with a concrete blocker."
        elif phase == "subtask investigation":
            reason += (
                "Use complete_subtask now with the findings already gathered, "
                "or hand off with a concrete blocker."
            )
        else:
            reason += (
                "Use edit_file/write_file, run tests, update the plan, or hand "
                "off with a concrete blocker."
            )
        if state.research_guard_streak >= self.max_research_guard_violations:
            return state.mark_handoff(reason).with_context(
                kind="research_guard", text=reason
            )
        return (
            state.with_error(reason)
            .with_context(kind="research_guard", text=reason)
            .advance_step()
        )

    @trace_agent_step
    def _step(
        self,
        state: State,
        executor: ActionExecutor,
        previous_action: AgentAction | None,
    ) -> tuple[State, AgentAction | None]:
        state = self._with_research_warning(state)
        try:
            generated = self.model.generate_action(
                self.system_prompt,
                state,
                self.resolved_action_transport,
                self._action_space_for_state(state),
            )
            if isinstance(generated, ActionGenerationFailure):
                return (
                    self._reject_parse_failure(
                        state,
                        generated.error,
                        generated.action_json,
                        tool_call_id=generated.tool_call_id,
                        tool_name=generated.tool_name,
                    ),
                    previous_action,
                )
        except ActionParseError as exc:
            # Compatibility guard for a custom model implementation that has
            # not yet adopted ActionGenerationFailure.
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
        if self.research_guard_enabled and isinstance(
            action, (InspectFileAction, ListDirectoryAction, SearchAction)
        ):
            _, _, hard = self._research_limits(state)
            if state.research_streak >= hard:
                return self._reject_excess_research(state), action
        if same_action(action, previous_action):
            return (
                self._reject_repeat(state, action).with_last_action_json(
                    action_json, tool_call_id, tool_name
                ),
                action,
            )
        if action.action in self.intercept_actions:
            # Role-terminal / orchestration actions never reach the executor:
            # the caller reads them off the history (kind == action name).
            text = (
                getattr(action, "summary", None)
                or getattr(action, "instruction", None)
                or action.model_dump_json(exclude_none=True)
            )
            return (
                state.with_context(kind=action.action, text=text)
                .with_last_action_json(action_json, tool_call_id, tool_name)
                .advance_step(),
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
        self,
        state: State,
        error: ActionParseError,
        response: str,
        *,
        tool_call_id: str | None = None,
        tool_name: str | None = None,
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
                .with_last_action_json(response, tool_call_id, tool_name)
            )

        if self.resolved_action_transport == "tools":
            expected = "Call exactly one of the provided action tools."
        else:
            expected = "Return exactly one JSON action object and no reasoning text."
        guidance = (
            f"error: {error}\nAttempt {attempt}/{self.max_parse_failures}. {expected}"
        )
        return (
            state.with_error(str(error))
            .with_context(kind=kind, text=guidance)
            .with_last_action_json(response, tool_call_id, tool_name)
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
            return state.with_error(
                "Maximum test iteration count reached."
            ).with_status("failed")
        return state.mark_handoff(
            "Agent loop stopped before reaching a terminal state."
        )
