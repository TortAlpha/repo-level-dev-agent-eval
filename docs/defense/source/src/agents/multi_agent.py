"""Multi-agent architectures over the shared role toolkit (see roles.py).

Five variants, sharing the single-agent step engine, executor, budgets and
evaluation so that *only the orchestration differs*:

- ``MultiGraphAgent``: fixed LangGraph pipeline
  planner -> developer <-> tester -> reviewer -> finish.
- ``MultiOrchestratorAgent``: a dynamic orchestrator loop whose action
  vocabulary is {delegate, finish, handoff} — calling a role is a tool call.
- ``GuidedOrchestratorAgent``: the same dynamic loop with an explicit current
  phase and recommended next action rendered into the supervisor context.
- ``GuardedOrchestratorAgent``: guided orchestration plus code-enforced phase
  transitions and budget reservation for verification/review.
- ``MultiSweAgent``: the fixed pipeline with the developer role backed by a
  SWE-agent episode (modernized open-source architecture).

All variants spend the same global step/iteration budgets as the single
agent, so comparisons measure orchestration, not extra compute.
"""

from __future__ import annotations

from dataclasses import replace

from pydantic import Field, PrivateAttr

from .actions import DelegateAction, FinishAction
from .context import ContextCompactor
from .executor import ActionExecutor
from .model import LangChainModel  # noqa: F401 - re-exported agent surface
from .prompts import load_prompt
from .roles import ROLES, RoleSpec, build_role_agent, run_role
from .sandbox import DockerSandbox
from .single_agent import SingleAgent
from .state import ContextBudget, ContextEntry, State
from .swe_agent import SweAgentAdapter
from .tracing import trace_agent_run
from .workspace import Workspace

ORCHESTRATOR_PROMPT = load_prompt("multi/orchestrator")

GUARDED_PLANNER_MAX_STEPS = 20


def developer_instruction(planner_report: str, feedback: str) -> str:
    """Build the explicit cross-role handoff for a developer invocation."""
    parts = ["Implement the shared plan (see the progress line)."]
    if planner_report:
        parts.append("Planner handoff:\n" + planner_report)
    if feedback:
        parts.append("Feedback to address:\n" + feedback)
    return "\n\n".join(parts)


def swe_developer_call_limit(configured_limit: int, remaining_steps: int) -> int:
    """Keep a SWE-agent developer episode inside the shared role/run budget."""
    return min(
        configured_limit,
        ROLES["developer"].step_quota,
        max(remaining_steps, 1),
    )


def orchestration_next_action(
    state: State,
    last_role: str | None,
    last_report: str,
    *,
    planner_step_cap: int | None = None,
) -> tuple[str, str]:
    """Return the evidence-based phase and safest next orchestration action.

    This is advisory for ``orch-guided`` and an invariant for
    ``orch-guarded``.  Keeping it pure makes the policy independently testable
    and prevents the supervisor from routing on a stale prose summary.
    """
    planner_exhausted = (
        planner_step_cap is not None
        and state.role_steps.get("planner", 0) >= planner_step_cap
    )
    if not state.plan.strip() and not planner_exhausted:
        return "planning", "planner"
    if last_role == "planner":
        return "implementation", "developer"
    if last_role == "developer":
        return "verification", "tester"
    if last_role == "tester":
        if state.full_test_passed:
            return "review", "reviewer"
        return "repair", "developer"
    if last_role == "reviewer":
        approved = last_report.strip().upper().startswith("APPROVE")
        if approved and state.full_test_passed and state.changed_files:
            return "ready", "finish"
        return "revision", "developer"
    if not state.changed_files:
        return "implementation", "developer"
    if state.full_test_passed:
        return "review", "reviewer"
    return "verification", "tester"


def guarded_phase_budgets(max_steps: int) -> dict[str, int]:
    """Allocate guarded capacity across phases for a given global budget."""
    planner = max(5, min(GUARDED_PLANNER_MAX_STEPS, round(max_steps * 0.20)))
    implementation = max(10, min(40, round(max_steps * 0.40)))
    verification = max(6, min(15, round(max_steps * 0.15)))
    orchestration = max(2, min(5, round(max_steps * 0.05)))
    return {
        "planner": planner,
        "implementation": implementation,
        "verification": verification,
        "orchestration": orchestration,
    }


def planner_handoff_plan(
    state: State,
    report: str,
    *,
    force: bool = False,
) -> str:
    """Promote useful planner findings, or synthesize a bounded fallback plan."""
    clean = report.strip()
    useful = bool(clean) and (
        not clean.startswith("planner stopped at its step budget")
        or bool(state.relevant_files)
    )
    if useful:
        return "Planner handoff (promoted automatically):\n" + clean
    if not force:
        return ""

    files = ", ".join(str(path) for path in state.relevant_files[-8:])
    localization = files or "the files named or implied by the task"
    return (
        "Planner budget exhausted. Continue directly with implementation. "
        f"Start from {localization}; inspect only what is needed, make the "
        "smallest task-aligned change, and hand the result to the tester."
    )


def guarded_role_step_limit(
    role: str,
    remaining_steps: int,
    *,
    max_steps: int = 75,
    role_steps: int = 0,
) -> int:
    """Cap a role episode while reserving a path to verification and finish."""
    budgets = guarded_phase_budgets(max_steps)
    if role == "planner":
        planner_left = max(budgets["planner"] - role_steps, 0)
        downstream = (
            budgets["implementation"]
            + budgets["verification"]
            + budgets["orchestration"]
        )
        available = max(remaining_steps - downstream, 0)
        return min(ROLES[role].step_quota, planner_left, available)

    # A late repair may use the remaining implementation capacity, but cannot
    # consume the test/review/finish tail of the run.
    reserve_after = {
        "developer": budgets["verification"],
        "tester": 7,
        "reviewer": 1,
    }
    available = max(remaining_steps - reserve_after.get(role, 1), 1)
    return min(ROLES[role].step_quota, available)


class MultiAgentBase(SingleAgent):
    """Shared lifecycle for the multi-agent variants.

    Subclasses SingleAgent so it accepts the exact same construction kwargs
    the runner passes, and reuses the transport probe, step engine and
    budgets. Only ``run``'s inner orchestration differs per variant.
    """

    # Multi-agent roles already have explicit quotas and transitions; keep the
    # new progress guard isolated to the single-agent baseline.
    research_guard_enabled: bool = False
    compatibility_guard_enabled: bool = False
    enable_reviewer: bool = True
    role_models: dict[str, LangChainModel] = Field(default_factory=dict)
    developer_escalation_model: LangChainModel | None = None
    developer_escalate_after_no_edit_episodes: int = Field(default=1, gt=0)
    developer_escalate_after_failed_tests: int = Field(default=1, gt=0)

    _role_agents: dict = PrivateAttr(default_factory=dict)
    _role_usage: dict = PrivateAttr(default_factory=dict)
    _role_transports: dict = PrivateAttr(default_factory=dict)
    _developer_episodes: int = PrivateAttr(default=0)
    _developer_no_edit_episodes: int = PrivateAttr(default=0)
    _developer_escalated: bool = PrivateAttr(default=False)
    _developer_escalations: int = PrivateAttr(default=0)

    @trace_agent_run
    def run(self, state: State) -> State:
        workspace = Workspace(root=state.workdir)
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
                overhead_chars=len(ORCHESTRATOR_PROMPT),
            ),
            mode=self.compaction_mode,
            model=self.model,
        )
        try:
            state = state.with_container(sandbox.start())
            state = self._run_setup(state, sandbox)
            if self.docker_network_disabled:
                sandbox.disable_network()
                state = state.with_context(
                    kind="setup", text="Benchmark network isolated after setup."
                )
            state = self._orchestrate(state, executor, compactor)
            return self._finalize_if_needed(state)
        finally:
            if self.stop_container:
                sandbox.stop()

    # -- shared helpers ---------------------------------------------------- #

    def _orchestrate(
        self, state: State, executor: ActionExecutor, compactor: ContextCompactor
    ) -> State:
        raise NotImplementedError

    def usage_models(self) -> list[LangChainModel]:
        models = [self.model, *self.role_models.values()]
        if self.developer_escalation_model is not None:
            models.append(self.developer_escalation_model)
        unique: list[LangChainModel] = []
        seen: set[int] = set()
        for model in models:
            if id(model) not in seen:
                unique.append(model)
                seen.add(id(model))
        return unique

    def model_policy_metadata(self) -> dict:
        roles = {
            name: self.role_models.get(name, self.model).model
            for name in ("orchestrator", "planner", "developer", "tester", "reviewer")
        }
        return {
            "type": (
                "adaptive"
                if self.developer_escalation_model is not None
                else ("role_routed" if self.role_models else "shared")
            ),
            "base_model": self.model.model,
            "roles": roles,
            "developer_escalation_model": (
                self.developer_escalation_model.model
                if self.developer_escalation_model is not None
                else None
            ),
            "developer_escalate_after_no_edit_episodes": (
                self.developer_escalate_after_no_edit_episodes
            ),
            "developer_escalate_after_failed_tests": (
                self.developer_escalate_after_failed_tests
            ),
        }

    def role_usage(self) -> dict:
        return self._role_usage

    def role_transports(self) -> dict:
        return self._role_transports

    @property
    def developer_escalations(self) -> int:
        return self._developer_escalations

    def developer_escalation_pending(self) -> bool:
        return (
            self.developer_escalation_model is not None
            and self._developer_escalated
            and self._developer_escalations == 0
        )

    def _base_role_model(self, name: str) -> LangChainModel:
        return self.role_models.get(name, self.model)

    def _select_role_model(self, name: str, state: State) -> LangChainModel:
        if name != "developer" or self.developer_escalation_model is None:
            return self._base_role_model(name)

        failed_test_escalation = (
            self._developer_episodes > 0
            and state.failed_test_runs >= self.developer_escalate_after_failed_tests
        )
        no_edit_escalation = (
            self._developer_no_edit_episodes
            >= self.developer_escalate_after_no_edit_episodes
        )
        if failed_test_escalation or no_edit_escalation:
            self._developer_escalated = True
        if self._developer_escalated:
            return self.developer_escalation_model
        return self._base_role_model(name)

    def _role(self, name: str, model: LangChainModel) -> SingleAgent:
        key = (name, id(model))
        if key not in self._role_agents:
            legacy_shared = model is self.model
            transport = (
                self.resolved_action_transport
                if legacy_shared
                else self.action_transport
            )
            self._role_agents[key] = build_role_agent(
                ROLES[name],
                model,
                transport,
                transport_resolved=legacy_shared,
            )
        return self._role_agents[key]

    @staticmethod
    def _usage_snapshot(model: LangChainModel) -> tuple[int, int, int]:
        return model.calls, model.input_tokens, model.output_tokens

    def _record_role_usage(
        self,
        role: str,
        model: LangChainModel,
        before: tuple[int, int, int],
    ) -> None:
        calls = model.calls - before[0]
        input_tokens = model.input_tokens - before[1]
        output_tokens = model.output_tokens - before[2]
        role_models = self._role_usage.setdefault(role, {})
        usage = role_models.setdefault(
            model.model,
            {"calls": 0, "input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
        )
        usage["calls"] += calls
        usage["input_tokens"] += input_tokens
        usage["output_tokens"] += output_tokens
        usage["total_tokens"] += input_tokens + output_tokens

    def _run_role(
        self,
        name: str,
        state: State,
        executor: ActionExecutor,
        compactor: ContextCompactor,
        instruction: str,
        *,
        step_quota: int | None = None,
    ) -> tuple[State, str]:
        spec: RoleSpec = ROLES[name]
        if step_quota is not None:
            spec = replace(spec, step_quota=max(step_quota, 1))
        model = self._select_role_model(name, state)
        if (
            name == "developer"
            and model is self.developer_escalation_model
            and self._developer_escalations == 0
        ):
            self._developer_escalations = 1
            state = state.with_context(
                kind="model_escalation",
                text=f"Developer escalated to model {model.model}.",
            )

        before_usage = self._usage_snapshot(model)
        before_edits = state.action_counts.get(
            "edit_file", 0
        ) + state.action_counts.get("write_file", 0)
        role_agent = self._role(name, model)
        result, report = run_role(
            spec,
            agent=role_agent,
            executor=executor,
            compactor=compactor,
            state=state,
            instruction=instruction,
        )
        self._role_transports.setdefault(name, {})[model.model] = (
            role_agent.resolved_action_transport
        )
        self._record_role_usage(name, model, before_usage)

        if name == "developer":
            self._developer_episodes += 1
            after_edits = result.action_counts.get(
                "edit_file", 0
            ) + result.action_counts.get("write_file", 0)
            if after_edits <= before_edits:
                self._developer_no_edit_episodes += 1
                if (
                    self.developer_escalation_model is not None
                    and self._developer_no_edit_episodes
                    >= self.developer_escalate_after_no_edit_episodes
                ):
                    self._developer_escalated = True
            else:
                self._developer_no_edit_episodes = 0
        return self._enforce_cost_limit(result), report

    def _try_finish(
        self, state: State, executor: ActionExecutor, summary: str
    ) -> State:
        """Terminal decision through the real executor gate (tests + diff)."""
        return executor.execute(state, FinishAction(action="finish", summary=summary))


class MultiGraphAgent(MultiAgentBase):
    """Fixed pipeline planner -> developer <-> tester -> reviewer -> finish,
    wired as a LangGraph state machine (nodes are thin role calls)."""

    def _orchestrate(
        self, state: State, executor: ActionExecutor, compactor: ContextCompactor
    ) -> State:
        graph = self._build_graph(executor, compactor)
        result = graph.invoke(
            {
                "state": state,
                "planner_report": "",
                "feedback": "",
                "dev_report": "",
                "verdict": "",
            },
            {"recursion_limit": 200},
        )
        return result["state"]

    def _build_graph(self, executor: ActionExecutor, compactor: ContextCompactor):
        try:
            from langgraph.graph import END, StateGraph
        except ImportError as exc:  # pragma: no cover - env dependent
            raise SystemExit(
                "multi-graph requires langgraph. Install it with: "
                "python -m pip install -e .[graph]"
            ) from exc
        from typing import TypedDict

        class PipelineState(TypedDict):
            state: State
            planner_report: str
            feedback: str
            dev_report: str
            verdict: str

        def planner(ps: PipelineState) -> PipelineState:
            state, report = self._run_role(
                "planner",
                ps["state"],
                executor,
                compactor,
                "Study the task and the codebase, then record an implementation "
                "plan with set_plan and report a handoff note for the developer.",
            )
            return {**ps, "state": state, "planner_report": report}

        def developer(ps: PipelineState) -> PipelineState:
            instruction = developer_instruction(ps["planner_report"], ps["feedback"])
            state, report = self._develop(ps["state"], executor, compactor, instruction)
            return {**ps, "state": state, "dev_report": report, "feedback": ""}

        def tester(ps: PipelineState) -> PipelineState:
            state, report = self._run_role(
                "tester",
                ps["state"],
                executor,
                compactor,
                "Verify the developer's change:\n" + ps["dev_report"],
            )
            return {**ps, "state": state, "feedback": report}

        def reviewer(ps: PipelineState) -> PipelineState:
            state, report = self._run_role(
                "reviewer",
                ps["state"],
                executor,
                compactor,
                "Review the final diff against the task.",
            )
            # If routing returns to development, this is the feedback the
            # developer must see (not the stale tester report).
            return {**ps, "state": state, "verdict": report, "feedback": report}

        def finalize(ps: PipelineState) -> PipelineState:
            if ps["state"].is_terminal:
                return ps
            state = self._try_finish(
                ps["state"],
                executor,
                f"Pipeline finished. Developer: {ps['dev_report'][:400]}",
            )
            return {**ps, "state": state}

        # NOTE: branch callables are deliberately unannotated — langgraph runs
        # get_type_hints() on them, which cannot resolve a method-local class.
        def after_tester(ps):
            state = ps["state"]
            if state.full_test_passed:
                return "reviewer" if self.enable_reviewer else "finalize"
            return "developer" if state.can_continue else "finalize"

        def after_reviewer(ps):
            approved = ps["verdict"].strip().upper().startswith("APPROVE")
            if not approved and ps["state"].can_continue:
                return "developer"
            return "finalize"

        def after_developer(ps):
            if self.developer_escalation_pending() and ps["state"].can_continue:
                return "developer"
            return "tester" if ps["state"].can_continue else "finalize"

        routes = {
            "developer": "developer",
            "tester": "tester",
            "reviewer": "reviewer",
            "finalize": "finalize",
        }
        graph = StateGraph(PipelineState)
        graph.add_node("planner", planner)
        graph.add_node("developer", developer)
        graph.add_node("tester", tester)
        graph.add_node("reviewer", reviewer)
        graph.add_node("finalize", finalize)
        graph.set_entry_point("planner")
        graph.add_edge("planner", "developer")
        graph.add_conditional_edges("developer", after_developer, routes)
        graph.add_conditional_edges("tester", after_tester, routes)
        graph.add_conditional_edges("reviewer", after_reviewer, routes)
        graph.add_edge("finalize", END)
        return graph.compile()

    def _develop(
        self,
        state: State,
        executor: ActionExecutor,
        compactor: ContextCompactor,
        instruction: str,
    ) -> tuple[State, str]:
        """Developer backend: our role loop (overridden in MultiSweAgent).

        The reviewer feedback loop re-enters here, so failed review rounds
        become sharper developer instructions."""
        return self._run_role("developer", state, executor, compactor, instruction)


class MultiOrchestratorAgent(MultiAgentBase):
    """Dynamic orchestration: calling a role is a tool of the orchestrator."""

    def _orchestrate(
        self, state: State, executor: ActionExecutor, compactor: ContextCompactor
    ) -> State:
        return self._run_orchestrator(
            state, executor, compactor, guided=False, guarded=False
        )

    def _run_orchestrator(
        self,
        state: State,
        executor: ActionExecutor,
        compactor: ContextCompactor,
        *,
        guided: bool,
        guarded: bool,
    ) -> State:
        orchestrator_model = self._base_role_model("orchestrator")
        legacy_shared = orchestrator_model is self.model
        orchestrator = SingleAgent(
            model=orchestrator_model,
            system_prompt=ORCHESTRATOR_PROMPT,
            action_space="orchestrator",
            intercept_actions=(
                frozenset({"delegate", "finish"})
                if guarded
                else frozenset({"delegate"})
            ),
            action_transport=(
                self.resolved_action_transport
                if legacy_shared
                else self.action_transport
            ),
        )
        if legacy_shared:
            orchestrator._transport_cache = (self.resolved_action_transport, False)

        previous = None
        last_role: str | None = None
        last_report = ""
        while state.can_continue:
            planner_cap = guarded_phase_budgets(state.max_steps)["planner"]
            if guarded and not state.plan.strip():
                planner_exhausted = state.role_steps.get("planner", 0) >= planner_cap
                promoted = planner_handoff_plan(
                    state, last_report, force=planner_exhausted
                )
                if promoted:
                    state = state.with_plan(promoted).with_context(
                        kind="planner_handoff_promoted",
                        text=promoted,
                    )

            phase, expected = orchestration_next_action(
                state,
                last_role,
                last_report,
                planner_step_cap=planner_cap if guarded else None,
            )
            if (
                guarded
                and last_role == "developer"
                and self.developer_escalation_pending()
            ):
                phase, expected = "developer escalation", "developer"
            if guided:
                strength = "ONLY VALID" if guarded else "RECOMMENDED"
                guidance = ContextEntry(
                    step=state.step,
                    kind="orchestration_guidance",
                    role="orchestrator",
                    text=(
                        f"Orchestration phase: {phase}. {strength} next action: "
                        f"{expected}. Base this transition on shared state, not "
                        "on an older test report."
                    ),
                )
                context = [
                    entry
                    for entry in state.context
                    if entry.kind != "orchestration_guidance"
                ]
                state = state.model_copy(update={"context": [*context, guidance]})

            before_usage = self._usage_snapshot(orchestrator_model)
            state, previous = orchestrator._step(state, executor, previous)
            self._role_transports.setdefault("orchestrator", {})[
                orchestrator_model.model
            ] = orchestrator.resolved_action_transport
            self._record_role_usage("orchestrator", orchestrator_model, before_usage)
            state = self._enforce_cost_limit(state)
            if (
                isinstance(previous, DelegateAction)
                and state.context
                and state.context[-1].kind == "delegate"
            ):
                if guarded and previous.role != expected:
                    state = state.with_context(
                        kind="delegate_rejected",
                        text=(
                            f"Delegation to {previous.role} rejected during "
                            f"{phase}; required next role is {expected}."
                        ),
                    )
                elif previous.role == "planner" and state.plan.strip():
                    # Prompt-only guidance was not enough: real runs spent
                    # most of the global budget re-planning. Enforce the
                    # phase invariant once a usable plan exists.
                    state = state.with_context(
                        kind="delegate_rejected",
                        text=(
                            "Planner delegation rejected: a shared plan already "
                            "exists. Delegate the developer or tester next."
                        ),
                    )
                else:
                    quota = None
                    if guarded:
                        quota = guarded_role_step_limit(
                            previous.role,
                            state.max_steps - state.step,
                            max_steps=state.max_steps,
                            role_steps=state.role_steps.get(previous.role, 0),
                        )
                    state, last_report = self._run_role(
                        previous.role,
                        state,
                        executor,
                        compactor,
                        previous.instruction,
                        step_quota=quota,
                    )
                    last_role = previous.role
                # run_role appended instruction+report entries; the report is
                # the orchestrator's observation for its next decision.
            elif guarded and isinstance(previous, FinishAction):
                if (
                    expected == "finish"
                    and state.full_test_passed
                    and state.changed_files
                ):
                    state = state.with_status("solved").with_observation(
                        previous.summary
                    )
                else:
                    state = state.with_status("running").with_context(
                        kind="finish_rejected",
                        text=(
                            f"Finish rejected during {phase}; required next "
                            f"action is {expected}."
                        ),
                    )
            state = compactor.apply(state)
        return state


class GuidedOrchestratorAgent(MultiOrchestratorAgent):
    """Dynamic supervisor with an explicit evidence-based phase hint."""

    def _orchestrate(
        self, state: State, executor: ActionExecutor, compactor: ContextCompactor
    ) -> State:
        return self._run_orchestrator(
            state, executor, compactor, guided=True, guarded=False
        )


class GuardedOrchestratorAgent(MultiOrchestratorAgent):
    """Dynamic supervisor constrained by phase and verification invariants."""

    def _orchestrate(
        self, state: State, executor: ActionExecutor, compactor: ContextCompactor
    ) -> State:
        return self._run_orchestrator(
            state, executor, compactor, guided=True, guarded=True
        )


class MultiSweAgent(MultiGraphAgent):
    """The fixed pipeline with a SWE-agent episode as the developer backend:
    plan and tester feedback are folded into the problem statement, the
    produced patch lands on the shared workspace and is verified by our
    tester role."""

    developer_adapter: SweAgentAdapter | None = None

    def _develop(
        self,
        state: State,
        executor: ActionExecutor,
        compactor: ContextCompactor,
        instruction: str,
    ) -> tuple[State, str]:
        if self.developer_adapter is None:
            raise SystemExit("multi-swe requires a configured SWE-agent adapter.")
        problem = (
            f"{state.task}\n\n## Implementation plan\n{state.plan or '(none)'}"
            f"\n\n## Additional instructions\n{instruction}"
        )
        episode = state.model_copy(
            update={"task": problem, "context": [], "active_role": "developer"}
        )
        episode_call_limit = swe_developer_call_limit(
            self.developer_adapter.call_limit,
            state.max_steps - state.step,
        )
        adapter = self.developer_adapter.model_copy(
            update={"call_limit": episode_call_limit}
        )
        result = adapter.run(episode)

        produced = result.status != "handoff" and bool(result.changed_files)
        changed = ", ".join(str(p) for p in result.changed_files) or "none"
        report = (
            f"SWE-agent episode {'produced a patch' if produced else 'failed'}. "
            f"Changed files: {changed}."
        )
        steps_spent = max(result.step - state.step, 1)
        merged = state.model_copy(
            update={
                "step": min(state.step + steps_spent, state.max_steps),
                "changed_files": result.changed_files,
                "active_role": None,
                "role_steps": {
                    **state.role_steps,
                    "developer": state.role_steps.get("developer", 0) + steps_spent,
                },
            }
        ).with_context(kind="report", text=f"[developer report]\n{report}")
        return merged, report
