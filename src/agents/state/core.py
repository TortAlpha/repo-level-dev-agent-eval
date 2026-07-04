"""The agent's task ``State``: an immutable snapshot updated via ``with_*`` copies."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from . import rendering
from .entry import ContextEntry
from .status import AgentStatus, STALE_FILE_SNAPSHOT_KINDS, TERMINAL_STATUSES


class State(BaseModel):
    model_config = ConfigDict(validate_assignment=True)

    task: str
    workdir: Path

    status: AgentStatus = "running"
    plan: str = ""
    iteration: int = Field(default=0, ge=0)
    max_iterations: int = Field(default=5, gt=0)
    step: int = Field(default=0, ge=0)
    max_steps: int = Field(default=25, gt=0)

    test_command: str | None = None
    container_id: str | None = None
    test_passed: bool = False
    last_test_output: str = ""

    relevant_files: list[Path] = Field(default_factory=list)
    changed_files: list[Path] = Field(default_factory=list)

    last_observation: str = ""
    last_error: str | None = None
    context: list[ContextEntry] = Field(default_factory=list)
    # Cumulative count of every action kind ever added, so behavior metrics
    # survive compaction (which trims ``context`` but not this).
    action_counts: dict[str, int] = Field(default_factory=dict)

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
            update={"plan": plan.strip(), "status": "planning", "last_error": None}
        )

    def with_test_command(self, command: str | None) -> State:
        return self.model_copy(update={"test_command": command})

    def with_container(self, container_id: str | None) -> State:
        return self.model_copy(update={"container_id": container_id})

    def with_context(self, kind: str, text: str, path: str | None = None) -> State:
        entry = ContextEntry(step=self.step, kind=kind, path=path, text=text.strip())
        counts = {**self.action_counts, kind: self.action_counts.get(kind, 0) + 1}
        return self.model_copy(
            update={"context": [*self.context, entry], "action_counts": counts}
        )

    def with_last_action_json(self, action_json: str) -> State:
        """Attach the raw action to the newest history entry, so the history
        can be replayed to the model as an assistant/user dialogue."""
        if not self.context:
            return self
        last = self.context[-1].model_copy(update={"action_json": action_json})
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
        return self.model_copy(
            update={
                "changed_files": _merge_paths(self.changed_files, files),
                "status": "editing",
            }
        )

    def record_test_result(self, passed: bool, output: str) -> State:
        iteration = self.iteration + 1
        step = self.step + 1

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
