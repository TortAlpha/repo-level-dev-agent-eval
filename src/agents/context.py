from __future__ import annotations

import logging
from typing import Literal

from pydantic import BaseModel, Field

from .model import LangChainModel
from .prompts import CONTEXT_SUMMARY_PROMPT
from .state import ContextBudget, ContextEntry, State
from .tracing import trace_compaction

CompactionMode = Literal["drop", "summarize"]

logger = logging.getLogger(__name__)

_TEST_STATUSES = ("PASSED", "FAILED")


def _hard_facts(dropped: list[ContextEntry]) -> list[str]:
    """Facts extracted mechanically from dropped entries.

    These survive compaction verbatim, so a weak summarizer can never lose
    which files were changed or how the compacted test runs ended.
    """
    changed: list[str] = []
    test_results: list[str] = []

    for entry in dropped:
        if entry.kind in ("edit_file", "write_file") and entry.path:
            if entry.path not in changed:
                changed.append(entry.path)
        elif entry.kind == "run_tests":
            lines = entry.text.splitlines()
            command = lines[0] if lines else "?"
            status = next(
                (line for line in lines[1:2] if line in _TEST_STATUSES),
                "UNKNOWN",
            )
            test_results.append(f"{status} <- {command}")

    facts: list[str] = []
    if changed:
        facts.append("Files changed in compacted steps: " + ", ".join(changed))
    if test_results:
        facts.append(
            f"Test runs compacted: {len(test_results)}; "
            f"most recent: {test_results[-1]}"
        )
    return facts


def _inspected_paths(entries: list[ContextEntry]) -> list[str]:
    """Distinct file paths inspected among the given entries, in order."""
    paths: list[str] = []
    for entry in entries:
        if entry.kind == "inspect_file" and entry.path and entry.path not in paths:
            paths.append(entry.path)
    return paths


class ContextCompactor(BaseModel):
    """Shrinks the action history when the prompt outgrows the model window.

    In ``summarize`` mode the dropped entries are compressed by the model
    into a factual digest that stays in the history, so earlier findings
    survive compaction. Because the digest becomes the oldest entry, the
    next compaction folds it into the next digest, producing a rolling
    summary of the whole run. In ``drop`` mode (ablation baseline) dropped
    entries are replaced with a one-line omission marker.
    """

    budget: ContextBudget
    mode: CompactionMode = "summarize"
    model: LangChainModel | None = None
    summary_max_chars: int = Field(default=2000, gt=0)

    def apply(self, state: State) -> State:
        if not self.budget.needs_compaction(state):
            return state
        return self._compact(state)

    @trace_compaction
    def _compact(self, state: State) -> State:
        reserve = self.summary_max_chars if self._can_summarize else 0
        target = self.budget.history_target_chars(state, reserve_chars=reserve)
        dropped, kept = state.split_context(target)
        if not dropped:
            return state

        # (A) Keep the freshest not-yet-edited inspection of each file verbatim,
        # so edit_file still has exact source; only summarize the rest.
        rescued, remaining = self._rescue_live_inspections(dropped, kept, state)
        if not remaining:
            return state.model_copy(update={"context": [*rescued, *kept]})

        digest = self._digest_entry(remaining)
        return state.model_copy(update={"context": [digest, *rescued, *kept]})

    def _rescue_live_inspections(
        self,
        dropped: list[ContextEntry],
        kept: list[ContextEntry],
        state: State,
    ) -> tuple[list[ContextEntry], list[ContextEntry]]:
        """Split ``dropped`` into (rescued, remaining).

        Rescued = the latest inspection of each file the agent may still need
        verbatim: not yet edited, not already re-inspected in ``kept``, not a
        stale marker. Freshest files win a bounded char budget (the gap between
        the low and high watermark) so the prompt stays under the compaction
        threshold and does not immediately re-fire.
        """
        edited = {str(path) for path in state.changed_files}
        kept_inspected = {
            entry.path for entry in kept if entry.kind == "inspect_file" and entry.path
        }
        latest: dict[str, ContextEntry] = {}
        for entry in dropped:
            if (
                entry.kind == "inspect_file"
                and entry.path
                and entry.path not in edited
                and entry.path not in kept_inspected
                and not entry.text.startswith("[stale")
            ):
                latest[entry.path] = entry  # dropped is oldest-first -> keeps latest

        cap = int(
            self.budget.input_budget_chars
            * (self.budget.high_watermark - self.budget.low_watermark)
        )
        rescued_ids: set[int] = set()
        used = 0
        for entry in sorted(latest.values(), key=lambda e: e.step, reverse=True):
            if used + len(entry.text) > cap:
                continue
            rescued_ids.add(id(entry))
            used += len(entry.text)

        rescued = [entry for entry in dropped if id(entry) in rescued_ids]
        remaining = [entry for entry in dropped if id(entry) not in rescued_ids]
        return rescued, remaining

    @property
    def _can_summarize(self) -> bool:
        return self.mode == "summarize" and self.model is not None

    def _digest_entry(self, dropped: list[ContextEntry]) -> ContextEntry:
        # Oldest summarized step, so the digest sorts before rescued entries.
        step = dropped[0].step

        if self._can_summarize:
            facts = _hard_facts(dropped)
            try:
                return ContextEntry(
                    step=step,
                    kind="context_summary",
                    text=self._summarize(dropped, facts),
                )
            except Exception:  # noqa: BLE001 - a failed digest must not kill the run
                logger.exception(
                    "Context summarization failed; falling back to drop mode."
                )
                return ContextEntry(
                    step=step,
                    kind="context_clean",
                    text="\n".join(
                        [
                            f"[{len(dropped)} earlier action(s) omitted "
                            "to stay within the context budget]",
                            *facts,
                        ]
                    ),
                )

        return ContextEntry(
            step=step,
            kind="context_clean",
            text=(
                f"[{len(dropped)} earlier action(s) omitted "
                "to stay within the context budget]"
            ),
        )

    def _summarize(self, dropped: list[ContextEntry], facts: list[str]) -> str:
        assert self.model is not None
        content = "\n\n".join(
            entry.render(index) for index, entry in enumerate(dropped, start=1)
        )
        instructions = CONTEXT_SUMMARY_PROMPT.format(
            max_chars=self.summary_max_chars
        )
        summary = self.model.summarize(instructions, content).strip()
        if not summary:
            raise ValueError("Model returned an empty digest.")
        if len(summary) > self.summary_max_chars:
            summary = summary[: self.summary_max_chars].rstrip() + " ..."

        header = [f"[digest of {len(dropped)} earlier action(s)]", *facts]
        # (B) The summary is not verbatim; edit_file needs exact text, so tell
        # the agent to re-read any summarized file before editing it.
        summarized_files = _inspected_paths(dropped)
        if summarized_files:
            header.append(
                "Summarized below (NOT verbatim) — re-inspect with inspect_file "
                "before editing: " + ", ".join(summarized_files)
            )
        return "\n".join([*header, summary])
