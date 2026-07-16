"""The agent's task ``State``: an immutable snapshot updated via ``with_*`` copies."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from . import rendering
from .entry import ContextEntry
from .status import AgentStatus, STALE_FILE_SNAPSHOT_KINDS, TERMINAL_STATUSES
from .subtask import Subtask


class State(BaseModel):
    model_config = ConfigDict(validate_assignment=True)

    task: str
    workdir: Path

    status: AgentStatus = "running"
    plan: str = ""
    # Multi-agent: role currently driving the loop; stamped onto new context
    # entries so metrics/traces can attribute steps per role.
    active_role: str | None = None
    # Cumulative steps spent per role (survives compaction, lands in metrics).
    role_steps: dict[str, int] = Field(default_factory=dict)
    iteration: int = Field(default=0, ge=0)
    max_iterations: int = Field(default=5, gt=0)
    step: int = Field(default=0, ge=0)
    max_steps: int = Field(default=25, gt=0)

    test_command: str | None = None
    container_id: str | None = None
    # ``test_passed`` is the result of the most recent test invocation, which
    # may be a focused subset. ``full_test_passed`` is the finish gate: it is
    # true only when the configured visible suite passed after the latest edit.
    test_passed: bool = False
    full_test_passed: bool = False
    # Strong single baseline: a distinct passing probe of pre-existing nearby
    # behavior is required after the latest edit. Multi-agent states keep the
    # requirement disabled unless their orchestration explicitly opts in.
    compatibility_check_required: bool = False
    compatibility_check_passed: bool = False
    last_compatibility_command: str = ""
    last_compatibility_output: str = ""
    # Cumulative failed test invocations. Unlike ``iteration`` this does not
    # grow on successful focused/full-suite checks, so adaptive model routing
    # can react to actual failures rather than ordinary verification work.
    failed_test_runs: int = Field(default=0, ge=0)
    last_test_command: str = ""
    last_test_output: str = ""
    workspace_revision: int = Field(default=0, ge=0)
    last_test_revision: int | None = Field(default=None, ge=0)
    full_suite_verified_revision: int | None = Field(default=None, ge=0)
    compatibility_verified_revision: int | None = Field(default=None, ge=0)

    relevant_files: list[Path] = Field(default_factory=list)
    changed_files: list[Path] = Field(default_factory=list)
    changed_file_revisions: dict[str, int] = Field(default_factory=dict)

    last_observation: str = ""
    last_error: str | None = None
    context: list[ContextEntry] = Field(default_factory=list)
    # Cumulative count of every action kind ever added, so behavior metrics
    # survive compaction (which trims ``context`` but not this).
    action_counts: dict[str, int] = Field(default_factory=dict)
    # Durable single-agent progress signals. They intentionally live outside
    # ``context`` so rolling summaries cannot erase evidence that the loop has
    # spent many consecutive steps only reading the repository.
    research_streak: int = Field(default=0, ge=0)
    # Cumulative metric plus a resettable consecutive-violation streak used
    # for the terminal guard decision.
    research_guard_violations: int = Field(default=0, ge=0)
    research_guard_streak: int = Field(default=0, ge=0)
    research_warning_phase: str | None = None
    # Search results are stable until an edit. Keeping the cache outside the
    # rendered prompt lets a repeated post-compaction search recover the exact
    # earlier result without paying another repository scan or losing it to a
    # summary-of-a-summary.
    search_results: dict[str, str] = Field(default_factory=dict)
    compaction_count: int = Field(default=0, ge=0)
    decomposition_required: bool = False
    decomposition_max_subtasks: int = Field(default=8, ge=4, le=12)
    subtasks: list[Subtask] = Field(default_factory=list)
    active_subtask_id: str | None = None
    repair_cycles: int = Field(default=0, ge=0)
    max_repair_cycles: int = Field(default=2, ge=0)

    @property
    def is_terminal(self) -> bool:
        return self.status in TERMINAL_STATUSES

    @property
    def can_continue(self) -> bool:
        return (
            not self.is_terminal
            and self.iteration < self.max_iterations
            and self.step < self.max_steps
        )

    @property
    def context_chars(self) -> int:
        return sum(len(entry.text) for entry in self.context)

    @property
    def active_subtask(self) -> Subtask | None:
        return next(
            (item for item in self.subtasks if item.id == self.active_subtask_id),
            None,
        )

    @property
    def decomposition_complete(self) -> bool:
        return bool(self.subtasks) and all(
            item.status == "completed" for item in self.subtasks
        )

    def with_status(self, status: AgentStatus) -> State:
        return self.model_copy(update={"status": status})

    def advance_step(self) -> State:
        step = self.step + 1
        status = self.status
        if status not in TERMINAL_STATUSES and step >= self.max_steps:
            status = "failed"
        return self.model_copy(update={"step": step, "status": status})

    def with_plan(self, plan: str) -> State:
        return self.model_copy(
            update={
                "plan": plan.strip(),
                "status": "planning",
                "last_error": None,
                "research_streak": 0,
                "research_guard_streak": 0,
                "research_warning_phase": None,
            }
        )

    def with_test_command(self, command: str | None) -> State:
        changed = command != self.test_command
        return self.model_copy(
            update={
                "test_command": command,
                "full_test_passed": (
                    False if changed else self.full_test_passed
                ),
                "full_suite_verified_revision": (
                    None if changed else self.full_suite_verified_revision
                ),
            }
        )

    def with_container(self, container_id: str | None) -> State:
        return self.model_copy(update={"container_id": container_id})

    def with_context(self, kind: str, text: str, path: str | None = None) -> State:
        entry = ContextEntry(
            step=self.step,
            kind=kind,
            path=path,
            text=text.strip(),
            role=self.active_role,
        )
        counts = {**self.action_counts, kind: self.action_counts.get(kind, 0) + 1}
        return self.model_copy(
            update={"context": [*self.context, entry], "action_counts": counts}
        )

    def with_last_action_json(
        self,
        action_json: str,
        tool_call_id: str | None = None,
        tool_name: str | None = None,
    ) -> State:
        """Attach the raw action to the newest history entry, so the history
        can be replayed to the model as an assistant/user dialogue. When the
        action came in as a native tool call, its id/name are kept so the
        replay can use the provider's tool-message protocol instead of text."""
        if not self.context:
            return self
        last = self.context[-1].model_copy(update={
            "action_json": action_json,
            "tool_call_id": tool_call_id,
            "tool_name": tool_name,
        })
        return self.model_copy(update={"context": [*self.context[:-1], last]})

    def invalidate_path(self, path: str) -> State:
        updated = [
            entry.model_copy(update={"text": "[stale: file changed by a later edit]"})
            if entry.kind in STALE_FILE_SNAPSHOT_KINDS and entry.path == path
            else entry
            for entry in self.context
        ]
        return self.model_copy(update={"context": updated})

    def split_context(
        self, target_chars: int
    ) -> tuple[list[ContextEntry], list[ContextEntry]]:
        """Split history into (dropped, kept) so the kept part fits the target.

        Drops oldest entries first and always keeps at least one entry.
        """
        kept = list(self.context)
        dropped: list[ContextEntry] = []
        total = sum(len(entry.text) for entry in kept)

        while total > target_chars and len(kept) > 1:
            removed = kept.pop(0)
            total -= len(removed.text)
            dropped.append(removed)

        return dropped, kept

    def with_observation(self, observation: str) -> State:
        return self.model_copy(
            update={"last_observation": observation.strip(), "last_error": None}
        )

    def with_error(self, error: str) -> State:
        status: AgentStatus = (
            "failed" if self.iteration >= self.max_iterations else "needs_revision"
        )
        return self.model_copy(update={"status": status, "last_error": error.strip()})

    def with_relevant_files(self, files: list[Path | str]) -> State:
        return self.model_copy(
            update={"relevant_files": _merge_paths(self.relevant_files, files)}
        )

    def with_changed_files(self, files: list[Path | str]) -> State:
        revision = self.workspace_revision + 1
        return self.model_copy(
            update={
                "changed_files": _merge_paths(self.changed_files, files),
                "changed_file_revisions": {
                    **self.changed_file_revisions,
                    **{str(Path(path)): revision for path in files},
                },
                "status": "editing",
                # A green result belongs to the workspace revision it tested.
                # Any later write must be verified again before finish.
                "test_passed": False,
                "full_test_passed": False,
                "compatibility_check_passed": False,
                "workspace_revision": revision,
                "last_test_revision": None,
                "full_suite_verified_revision": None,
                "compatibility_verified_revision": None,
                "research_streak": 0,
                "research_guard_streak": 0,
                "research_warning_phase": None,
                "search_results": {},
            }
        )

    def record_research_step(self) -> State:
        return self.model_copy(
            update={"research_streak": self.research_streak + 1}
        )

    def record_search_result(self, key: str, result: str) -> State:
        return self.model_copy(
            update={"search_results": {**self.search_results, key: result}}
        )

    def record_research_guard_violation(self) -> State:
        return self.model_copy(
            update={
                "research_guard_violations": self.research_guard_violations + 1,
                "research_guard_streak": self.research_guard_streak + 1,
            }
        )

    def record_research_warning(self, phase: str) -> State:
        return self.model_copy(update={"research_warning_phase": phase})

    def record_compaction(self) -> State:
        return self.model_copy(
            update={"compaction_count": self.compaction_count + 1}
        )

    def with_compatibility_requirement(self, required: bool) -> State:
        return self.model_copy(
            update={
                "compatibility_check_required": required,
                "compatibility_check_passed": (
                    self.compatibility_check_passed if required else False
                ),
            }
        )

    def with_decomposition_requirement(
        self,
        required: bool,
        max_subtasks: int = 8,
        max_repair_cycles: int = 2,
    ) -> State:
        return self.model_copy(
            update={
                "decomposition_required": required,
                "decomposition_max_subtasks": max_subtasks,
                "subtasks": self.subtasks if required else [],
                "active_subtask_id": (
                    self.active_subtask_id if required else None
                ),
                "max_repair_cycles": max_repair_cycles,
            }
        )

    def record_test_result(
        self,
        passed: bool,
        output: str,
        *,
        command: str = "",
        full_suite: bool = False,
        compatibility_check: bool = False,
        consume_iteration: bool = True,
    ) -> State:
        iteration = self.iteration + (1 if consume_iteration else 0)
        step = self.step + 1
        full_test_passed = self.full_test_passed
        if not passed:
            full_test_passed = False
        elif full_suite:
            full_test_passed = True
        compatibility_check_passed = self.compatibility_check_passed
        compatibility_verified_revision = self.compatibility_verified_revision
        if compatibility_check:
            compatibility_check_passed = passed
            compatibility_verified_revision = (
                self.workspace_revision if passed else None
            )
        full_suite_verified_revision = self.full_suite_verified_revision
        if not passed:
            full_suite_verified_revision = None
        elif full_suite:
            full_suite_verified_revision = self.workspace_revision

        # Passing visible tests is never terminal: benchmark tasks start with
        # green visible tests, so success requires an explicit finish action.
        if passed:
            status: AgentStatus = "running"
        elif iteration >= self.max_iterations or step >= self.max_steps:
            status = "failed"
        else:
            status = "needs_revision"

        return self.model_copy(
            update={
                "iteration": iteration,
                "step": step,
                "status": status,
                "test_passed": passed,
                # Focused tests are useful feedback but never certify finish.
                # A failed full-suite run also clears an older green result.
                "full_test_passed": full_test_passed,
                "compatibility_check_passed": compatibility_check_passed,
                "last_test_revision": self.workspace_revision,
                "full_suite_verified_revision": full_suite_verified_revision,
                "compatibility_verified_revision": compatibility_verified_revision,
                "last_compatibility_command": (
                    command.strip()
                    if compatibility_check
                    else self.last_compatibility_command
                ),
                "last_compatibility_output": (
                    output.strip()
                    if compatibility_check
                    else self.last_compatibility_output
                ),
                "failed_test_runs": self.failed_test_runs + (0 if passed else 1),
                "research_streak": 0,
                "research_guard_streak": 0,
                "research_warning_phase": None,
                "last_test_command": command.strip(),
                "last_test_output": output.strip(),
                "last_error": None,
            }
        )

    def mark_handoff(self, reason: str) -> State:
        return self.model_copy(
            update={"status": "handoff", "last_error": reason.strip()}
        )

    def to_messages(self) -> list[tuple[str, str]]:
        return rendering.render_messages(self)

    def to_string(self) -> str:
        return rendering.render_state(self)


def _merge_paths(existing: list[Path], new_files: list[Path | str]) -> list[Path]:
    merged: list[Path] = []
    seen: set[str] = set()

    for path in [*existing, *new_files]:
        normalized = Path(path)
        key = str(normalized)
        if key in seen:
            continue
        seen.add(key)
        merged.append(normalized)

    return merged
