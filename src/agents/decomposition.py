"""Structured task decomposition and checkpointing for a single model.

This is deliberately a separate architecture from ``SingleAgent``. It keeps
one model/provider but turns a free-form plan into a verified dependency-aware
queue. Completing a work unit collapses its verbose episode into a durable
checkpoint, giving the next unit a small context without losing outcomes.
"""

from __future__ import annotations

from collections import Counter

from pydantic import Field

from .actions import (
    ActionSpace,
    CompleteSubtaskAction,
    ReopenSubtaskAction,
    SetSubtasksAction,
)
from .prompts import DECOMPOSED_SINGLE_AGENT_PROMPT
from .single_agent import SingleAgent
from .state import ContextEntry, State, Subtask

ActionSpace("decomposed_init", ["list_dir", "set_subtasks", "handoff"])
ActionSpace(
    "decomposed_investigate",
    ["inspect_file", "search", "complete_subtask", "handoff"],
)
ActionSpace(
    "decomposed_force_investigate", ["complete_subtask", "handoff"]
)
ActionSpace(
    "decomposed_implement",
    [
        "inspect_file",
        "search",
        "edit_file",
        "write_file",
        "run_shell",
        "run_tests",
        "complete_subtask",
        "handoff",
    ],
)
ActionSpace(
    "decomposed_force_implement",
    [
        "edit_file",
        "write_file",
        "run_shell",
        "run_tests",
        "complete_subtask",
        "handoff",
    ],
)
ActionSpace(
    "decomposed_compatibility",
    ["run_tests", "reopen_subtask", "complete_subtask", "handoff"],
)
ActionSpace(
    "decomposed_verify",
    ["run_tests", "reopen_subtask", "complete_subtask", "handoff"],
)
ActionSpace("decomposed_finish", ["finish", "handoff"])


class DecomposedSingleAgent(SingleAgent):
    """One model executing a structured, verified sequence of subtasks."""

    system_prompt: str = DECOMPOSED_SINGLE_AGENT_PROMPT
    action_space: str | None = "decomposed_init"
    decomposition_enabled: bool = True
    max_subtasks: int = Field(default=8, ge=4, le=12)

    def _research_limits(self, state: State) -> tuple[str, int, int]:
        active = state.active_subtask
        if active and active.kind == "implement":
            return (
                "implementation subtask",
                self.decomposition_implementation_warning_steps,
                self.decomposition_implementation_hard_limit,
            )
        return super()._research_limits(state)

    def _with_research_warning(self, state: State) -> State:
        state = super()._with_research_warning(state)
        phase, _, hard = self._research_limits(state)
        forced_phase = f"{phase}:forced"
        if (
            state.research_streak >= hard
            and state.research_warning_phase != forced_phase
            and state.active_subtask
            and state.active_subtask.kind in {"investigate", "implement"}
        ):
            if state.active_subtask.kind == "investigate":
                guidance = "Only complete_subtask or handoff is available now."
            else:
                guidance = (
                    "Read-only tools are now disabled. Edit using the exact "
                    "targets already shown, run tests, complete the subtask, "
                    "or hand off with a concrete blocker."
                )
            state = state.with_context(
                kind="forced_transition", text=guidance
            ).record_research_warning(forced_phase)
        return state

    def _action_space_for_state(self, state: State) -> str | None:
        if not state.subtasks:
            return "decomposed_init"
        active = state.active_subtask
        if active is None:
            return "decomposed_finish"
        _, _, hard = self._research_limits(state)
        remaining = state.max_steps - state.step
        if active.kind == "investigate":
            return (
                "decomposed_force_investigate"
                if state.research_streak >= hard
                else "decomposed_investigate"
            )
        if active.kind == "implement":
            forced = (
                state.research_streak >= hard
                or remaining <= self.decomposition_verification_step_reserve
            )
            return (
                "decomposed_force_implement"
                if forced
                else "decomposed_implement"
            )
        if active.kind == "compatibility":
            return "decomposed_compatibility"
        return "decomposed_verify"

    def model_policy_metadata(self) -> dict:
        return {
            "type": "single_decomposed",
            "base_model": self.model.model,
            "max_subtasks": self.max_subtasks,
            "implementation_research": {
                "warning": self.decomposition_implementation_warning_steps,
                "hard": self.decomposition_implementation_hard_limit,
            },
            "verification_step_reserve": self.decomposition_verification_step_reserve,
            "max_repair_cycles": self.max_repair_cycles,
        }


def initialize_decomposition(
    state: State, action: SetSubtasksAction, *, max_subtasks: int = 8
) -> State:
    if state.subtasks:
        raise ValueError("Task decomposition is already set and cannot be replaced.")
    if not state.test_command:
        raise ValueError("Decomposed mode requires a configured full test command.")
    drafts = action.subtasks
    system_tail_count = 1 + int(state.compatibility_check_required)
    max_model_subtasks = max_subtasks - system_tail_count
    if not 2 <= len(drafts) <= max_model_subtasks:
        raise ValueError(
            f"Model decomposition must contain 2-{max_model_subtasks} subtasks; "
            f"the system reserves {system_tail_count} verification slots; "
            f"received {len(drafts)}."
        )

    unsupported = sorted(
        item.id for item in drafts if item.kind not in {"investigate", "implement"}
    )
    if unsupported:
        raise ValueError(
            "The system owns compatibility and verification subtasks. Model "
            "subtasks may only be investigate or implement: "
            + ", ".join(unsupported)
        )

    ids = [item.id for item in drafts]
    reserved = sorted({"system_compat", "system_verify"} & set(ids))
    if reserved:
        raise ValueError("Reserved system subtask ids: " + ", ".join(reserved))
    duplicates = sorted(key for key, count in Counter(ids).items() if count > 1)
    if duplicates:
        raise ValueError("Duplicate subtask ids: " + ", ".join(duplicates))
    known = set(ids)
    for item in drafts:
        missing = sorted(set(item.dependencies) - known)
        if missing:
            raise ValueError(
                f"Subtask {item.id!r} has unknown dependencies: "
                + ", ".join(missing)
            )
        if item.id in item.dependencies:
            raise ValueError(f"Subtask {item.id!r} cannot depend on itself.")

    _validate_acyclic(drafts)
    if not any(item.kind == "implement" for item in drafts):
        raise ValueError("Decomposition needs at least one implementation subtask.")

    subtasks = [Subtask(**item.model_dump()) for item in drafts]
    depended_on = {dep for item in subtasks for dep in item.dependencies}
    leaves = [item.id for item in subtasks if item.id not in depended_on]
    tail_dependency = leaves
    if state.compatibility_check_required:
        subtasks.append(
            Subtask(
                id="system_compat",
                objective="Verify concrete pre-existing behavior near the change",
                kind="compatibility",
                dependencies=tail_dependency,
            )
        )
        tail_dependency = ["system_compat"]
    subtasks.append(
        Subtask(
            id="system_verify",
            objective="Run the configured full visible test suite",
            kind="verify",
            dependencies=tail_dependency,
            acceptance_command=state.test_command,
        )
    )
    first = _next_ready(subtasks)
    if first is None:
        raise ValueError("Decomposition has no dependency-free starting subtask.")
    subtasks = [
        item.model_copy(
            update={"status": "active", "started_revision": state.workspace_revision}
        )
        if item.id == first
        else item
        for item in subtasks
    ]
    plan = "\n".join(
        f"{index}. [{item.kind}] {item.id}: {item.objective}"
        for index, item in enumerate(subtasks, start=1)
    )
    rendered = _render_queue(subtasks)
    return (
        state.model_copy(
            update={
                "subtasks": subtasks,
                "active_subtask_id": first,
                "plan": plan,
                "status": "planning",
                "last_error": None,
                "research_streak": 0,
                "research_guard_streak": 0,
                "research_warning_phase": None,
            }
        )
        .with_observation(f"Decomposition accepted; active subtask: {first}")
        .with_context(kind="set_subtasks", text=rendered)
        .advance_step()
    )


def complete_subtask(state: State, action: CompleteSubtaskAction) -> State:
    active = state.active_subtask
    if active is None:
        raise ValueError("There is no active subtask to complete.")
    if action.subtask_id != active.id:
        raise ValueError(
            f"Only active subtask {active.id!r} can be completed; "
            f"received {action.subtask_id!r}."
        )
    _validate_acceptance(state, active)

    completed = [
        item.model_copy(
            update={"status": "completed", "evidence": action.evidence.strip()}
        )
        if item.id == active.id
        else item
        for item in state.subtasks
    ]
    next_id = _next_ready(completed)
    if next_id:
        completed = [
            item.model_copy(
                update={
                    "status": "active",
                    "started_revision": state.workspace_revision,
                }
            )
            if item.id == next_id
            else item
            for item in completed
        ]

    checkpoint_text = (
        f"Completed {active.id} [{active.kind}]: {active.objective}\n"
        f"Evidence: {action.evidence.strip()}\n"
        f"Changed files: {', '.join(map(str, state.changed_files)) or 'none'}\n"
        f"Last test: {state.last_test_command or 'none'} "
        f"({'passed' if state.test_passed else 'not passed'})\n"
        f"Next subtask: {next_id or 'none; all subtasks completed'}"
    )
    checkpoint = ContextEntry(
        step=state.step,
        kind="subtask_checkpoint",
        text=checkpoint_text,
        role=state.active_role,
    )
    # Keep setup, the queue, compact checkpoints, and the freshest exact edit
    # targets discovered by investigation. This avoids paying to rediscover
    # the same source immediately after the subtask boundary.
    durable = [
        entry
        for entry in state.context
        if entry.kind in {"setup", "set_subtasks", "subtask_checkpoint"}
    ]
    rescued = _latest_live_inspections(state.context, limit=4) if active.kind == "investigate" else []
    counts = {
        **state.action_counts,
        "complete_subtask": state.action_counts.get("complete_subtask", 0) + 1,
    }
    return state.model_copy(
        update={
            "subtasks": completed,
            "active_subtask_id": next_id,
            "context": [*durable, *rescued, checkpoint],
            "action_counts": counts,
            "last_observation": checkpoint_text,
            "last_error": None,
            "status": "running",
            "research_streak": 0,
            "research_guard_streak": 0,
            "research_warning_phase": None,
        }
    ).advance_step()


def _validate_acceptance(state: State, subtask: Subtask) -> None:
    if subtask.kind == "implement":
        production_edit = any(
            revision > subtask.started_revision and not _is_test_path(path)
            for path, revision in state.changed_file_revisions.items()
        )
        if not production_edit:
            raise ValueError(
                f"Implementation subtask {subtask.id!r} needs a production "
                "edit made during its current episode."
            )
        if not state.test_passed or state.last_test_revision != state.workspace_revision:
            raise ValueError(
                f"Implementation subtask {subtask.id!r} needs a passing "
                "focused test or executable reproduction on the current revision."
            )
    if subtask.kind == "verify":
        if not state.test_passed:
            raise ValueError(
                f"Verification subtask {subtask.id!r} needs a passing test."
            )
        if state.last_test_command.strip() != (subtask.acceptance_command or "").strip():
            raise ValueError(
                f"Verification subtask {subtask.id!r} requires this exact "
                f"command: {subtask.acceptance_command}"
            )
        if state.full_suite_verified_revision != state.workspace_revision:
            raise ValueError("Full-suite verification belongs to a stale revision.")
    if subtask.kind == "compatibility" and not state.compatibility_check_passed:
        raise ValueError(
            f"Compatibility subtask {subtask.id!r} needs a passing "
            "run_tests action with purpose=compatibility."
        )
    if (
        subtask.kind == "compatibility"
        and state.compatibility_verified_revision != state.workspace_revision
    ):
        raise ValueError("Compatibility verification belongs to a stale revision.")
    if (
        subtask.kind == "compatibility"
        and subtask.acceptance_command
        and state.last_compatibility_command.strip()
        != subtask.acceptance_command.strip()
    ):
        raise ValueError(
            f"Compatibility subtask {subtask.id!r} requires this exact "
            f"command: {subtask.acceptance_command}"
        )


def _next_ready(subtasks: list[Subtask]) -> str | None:
    completed = {item.id for item in subtasks if item.status == "completed"}
    return next(
        (
            item.id
            for item in subtasks
            if item.status == "pending" and set(item.dependencies) <= completed
        ),
        None,
    )


def _validate_acyclic(drafts: list) -> None:
    dependencies = {item.id: set(item.dependencies) for item in drafts}
    remaining = set(dependencies)
    while remaining:
        ready = {item for item in remaining if not (dependencies[item] & remaining)}
        if not ready:
            raise ValueError("Subtask dependency graph contains a cycle.")
        remaining -= ready


def _render_queue(subtasks: list[Subtask]) -> str:
    lines = ["Structured decomposition accepted:"]
    for item in subtasks:
        deps = ", ".join(item.dependencies) or "none"
        acceptance = item.acceptance_command or "evidence"
        lines.append(
            f"- {item.id} [{item.kind}/{item.status}] deps={deps}; "
            f"acceptance={acceptance}; objective={item.objective}"
        )
    return "\n".join(lines)


def reopen_subtask(state: State, action: ReopenSubtaskAction) -> State:
    active = state.active_subtask
    target = next((item for item in state.subtasks if item.id == action.subtask_id), None)
    if active is None or active.kind not in {"compatibility", "verify"}:
        raise ValueError("Only compatibility or verification can reopen implementation.")
    if target is None or target.kind != "implement" or target.status != "completed":
        raise ValueError("reopen_subtask must target a completed implementation subtask.")
    repair_cycles = state.repair_cycles + 1
    if repair_cycles > state.max_repair_cycles:
        reason = (
            f"Repair cycle limit reached ({state.max_repair_cycles}). "
            f"Last reason: {action.reason.strip()}"
        )
        return state.mark_handoff(reason).with_context(kind="repair_limit", text=reason)

    target_id = target.id
    dependents = _transitive_dependents(state.subtasks, target_id)
    updated: list[Subtask] = []
    for item in state.subtasks:
        if item.id == target_id:
            updated.append(
                item.model_copy(
                    update={
                        "status": "active",
                        "evidence": "",
                        "started_revision": state.workspace_revision,
                    }
                )
            )
        elif item.id in dependents:
            updated.append(item.model_copy(update={"status": "pending", "evidence": ""}))
        else:
            updated.append(item)

    text = (
        f"Reopened {target_id} for repair cycle {repair_cycles}/"
        f"{state.max_repair_cycles}.\nReason: {action.reason.strip()}\n"
        f"Last failing test: {state.last_test_command or 'none'}\n"
        f"Output tail:\n{state.last_test_output[-1200:]}"
    )
    counts = {
        **state.action_counts,
        "reopen_subtask": state.action_counts.get("reopen_subtask", 0) + 1,
    }
    checkpoint = ContextEntry(
        step=state.step,
        kind="repair_checkpoint",
        text=text,
        role=state.active_role,
    )
    durable = [
        entry
        for entry in state.context
        if entry.kind in {"setup", "set_subtasks", "subtask_checkpoint", "repair_checkpoint"}
    ]
    return state.model_copy(
        update={
            "subtasks": updated,
            "active_subtask_id": target_id,
            "repair_cycles": repair_cycles,
            "context": [*durable, checkpoint],
            "action_counts": counts,
            "status": "running",
            "last_error": None,
            "research_streak": 0,
            "research_guard_streak": 0,
            "research_warning_phase": None,
        }
    ).advance_step()


def _latest_live_inspections(
    entries: list[ContextEntry], *, limit: int
) -> list[ContextEntry]:
    rescued: list[ContextEntry] = []
    seen: set[str] = set()
    for entry in reversed(entries):
        if entry.kind != "inspect_file" or not entry.path or entry.path in seen:
            continue
        if entry.text.startswith("[stale:"):
            continue
        seen.add(entry.path)
        rescued.append(entry)
        if len(rescued) >= limit:
            break
    return list(reversed(rescued))


def _transitive_dependents(subtasks: list[Subtask], target_id: str) -> set[str]:
    result: set[str] = set()
    changed = True
    while changed:
        changed = False
        for item in subtasks:
            if item.id in result or item.id == target_id:
                continue
            if target_id in item.dependencies or set(item.dependencies) & result:
                result.add(item.id)
                changed = True
    return result


def _is_test_path(path: str) -> bool:
    parts = {part.lower() for part in path.split("/")[:-1]}
    name = path.rsplit("/", 1)[-1].lower()
    return bool(parts & {"test", "tests", "testing"}) or name.startswith("test_")
