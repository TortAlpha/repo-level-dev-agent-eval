"""Multi-agent roles: restricted slices of the single-agent loop.

A role is the proven single-agent step engine parametrized with its own
prompt, a restricted action vocabulary, and a soft step quota. Roles run over
the *shared* workspace/sandbox/executor and spend the *shared* global step
budget, so single-agent and multi-agent runs stay comparable. Each role works
in a fresh context seeded with its instruction; its entries are merged back
into the shared history (tagged with the role name) and its findings travel
forward only through the plan and its ``report``.
"""

from __future__ import annotations

from dataclasses import dataclass

from .actions import ActionSpace, ReportAction
from .context import ContextCompactor
from .executor import ActionExecutor
from .model import LangChainModel
from .prompts import load_prompt
from .single_agent import SingleAgent
from .state import ContextEntry, State
from .transport import ActionTransport

# Role vocabularies (registered as named action spaces).
ActionSpace("planner", ["list_dir", "inspect_file", "search", "set_plan", "report"])
ActionSpace(
    "developer",
    ["inspect_file", "search", "list_dir", "edit_file", "write_file", "report"],
)
ActionSpace("tester", ["run_tests", "inspect_file", "search", "write_file", "report"])
ActionSpace("reviewer", ["run_shell", "inspect_file", "search", "report"])
ActionSpace("orchestrator", ["delegate", "finish", "handoff"])


@dataclass(frozen=True)
class RoleSpec:
    name: str
    step_quota: int

    @property
    def prompt(self) -> str:
        return load_prompt(f"multi/{self.name}")


ROLES: dict[str, RoleSpec] = {
    "planner": RoleSpec("planner", step_quota=10),
    "developer": RoleSpec("developer", step_quota=20),
    "tester": RoleSpec("tester", step_quota=10),
    "reviewer": RoleSpec("reviewer", step_quota=6),
}


def build_role_agent(
    spec: RoleSpec,
    model: LangChainModel,
    transport: ActionTransport,
    *,
    transport_resolved: bool = True,
) -> SingleAgent:
    """A step engine for one role. The transport is pre-resolved by the
    multi-agent (one probe per run, not one per delegation)."""
    agent = SingleAgent(
        model=model,
        system_prompt=spec.prompt,
        action_space=spec.name,
        intercept_actions=frozenset({"report"}),
        action_transport=transport,  # already resolved: "text_json" | "tools"
        # Role quotas already enforce planner/developer phase boundaries. Keep
        # the new research guard specific to the strong single-agent baseline
        # so the architecture comparison does not silently alter role loops.
        research_guard_enabled=False,
        compatibility_guard_enabled=False,
    )
    if transport_resolved:
        agent._transport_cache = (transport, False)
    return agent


def run_role(
    spec: RoleSpec,
    *,
    agent: SingleAgent,
    executor: ActionExecutor,
    compactor: ContextCompactor,
    state: State,
    instruction: str,
) -> tuple[State, str]:
    """Run one role over the shared state; return (merged state, report).

    The role sees a fresh context (instruction seed + the shared plan, which
    renders in every progress message), spends global steps, and ends by
    ``report``, quota exhaustion, or a stall. A stalled role never kills the
    run — it is converted into a report for the orchestration layer.
    """
    seed_lines = [instruction.strip() or f"Act as the {spec.name}."]
    if state.relevant_files:
        known = ", ".join(str(p) for p in state.relevant_files[-8:])
        seed_lines.append(
            f"Files already identified as relevant by the team: {known}. "
            "Start from these instead of re-surveying the repository."
        )
    seed_lines.append(
        f"Your role budget is {spec.step_quota} steps; report before it runs out."
    )
    seed = ContextEntry(
        step=state.step,
        kind="instruction",
        text="\n".join(seed_lines),
        role=spec.name,
    )
    role_state = state.model_copy(
        update={
            "context": [seed],
            "active_role": spec.name,
            "last_error": None,
            "last_observation": "",
        }
    )

    steps_used = 0
    previous_action = None
    report: str | None = None
    warned = False
    while role_state.can_continue and steps_used < spec.step_quota:
        before = role_state.step
        role_state, previous_action = agent._step(role_state, executor, previous_action)
        steps_used += max(role_state.step - before, 0)
        role_state = compactor.apply(role_state)
        if isinstance(previous_action, ReportAction):
            report = previous_action.summary
            break
        if not warned and spec.step_quota - steps_used <= 2:
            # Roles that silently hit their quota hand over nothing useful;
            # an explicit warning converts most of them into real reports.
            warned = True
            role_state = role_state.with_context(
                kind="quota_warning",
                text="Role budget nearly exhausted (2 steps left). Send your "
                "report action NOW with what you have.",
            )

    if report is None:
        report = _auto_report(spec, role_state)

    stalled = role_state.status == "handoff"
    if stalled:
        report = f"{report}\n[role stalled: {role_state.last_error or 'no progress'}]"

    # Only the instruction and the report survive into the shared history:
    # that keeps the orchestration context compact (the whole point of role
    # isolation). Role internals stay in traces, action_counts and role_steps.
    report_entry = ContextEntry(
        step=role_state.step,
        kind="report",
        text=f"[{spec.name} report]\n{report}",
        role=spec.name,
    )
    role_steps = {
        **state.role_steps,
        spec.name: state.role_steps.get(spec.name, 0) + steps_used,
    }
    merged = state.model_copy(
        update={
            "context": [*state.context, seed, report_entry],
            "active_role": None,
            "step": role_state.step,
            "iteration": role_state.iteration,
            "action_counts": role_state.action_counts,
            "compaction_count": role_state.compaction_count,
            "role_steps": role_steps,
            "plan": role_state.plan,
            "test_passed": role_state.test_passed,
            "full_test_passed": role_state.full_test_passed,
            "last_test_command": role_state.last_test_command,
            "last_test_output": role_state.last_test_output,
            "changed_files": role_state.changed_files,
            "changed_file_revisions": role_state.changed_file_revisions,
            "workspace_revision": role_state.workspace_revision,
            "last_test_revision": role_state.last_test_revision,
            "full_suite_verified_revision": role_state.full_suite_verified_revision,
            "compatibility_verified_revision": (
                role_state.compatibility_verified_revision
            ),
            "relevant_files": role_state.relevant_files,
            # A stalled role is the orchestration layer's problem, not a
            # terminal state for the whole run.
            "status": "running" if stalled else role_state.status,
            "last_error": None,
        }
    )
    return merged, report


def _auto_report(spec: RoleSpec, state: State) -> str:
    """Fallback report when a role ran out of budget without reporting."""
    recent = [
        f"{entry.kind}{f' {entry.path}' if entry.path else ''}"
        for entry in state.context[-5:]
    ]
    lines = [
        f"{spec.name} stopped at its step budget without an explicit report.",
        f"Recent actions: {', '.join(recent) or 'none'}.",
    ]
    if spec.name == "planner" and state.plan.strip():
        # A recorded plan makes the stall harmless: surface it so the
        # orchestrator moves on to the developer instead of re-delegating.
        lines.append(f"A plan WAS recorded and is shared:\n{state.plan.strip()}")
    elif spec.name == "planner":
        if state.relevant_files:
            lines.append(
                "Relevant files found: "
                + ", ".join(str(path) for path in state.relevant_files[-8:])
            )
        if state.last_observation:
            lines.append("Latest planner finding:\n" + state.last_observation[-1000:])
    if state.changed_files:
        lines.append(
            "Changed files so far: "
            + ", ".join(str(path) for path in state.changed_files)
        )
    if spec.name == "tester":
        if state.full_test_passed:
            verdict = "PASS (configured full visible suite)"
        elif state.test_passed:
            verdict = "FOCUSED PASS ONLY (full suite still required)"
        else:
            verdict = "FAIL or not run"
        lines.append(f"Test verification: {verdict}.")
        if state.last_test_output:
            lines.append(f"Last test output tail:\n{state.last_test_output[-800:]}")
    return "\n".join(lines)
