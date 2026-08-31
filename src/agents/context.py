from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, Field, PrivateAttr

from .model import LangChainModel
from .prompts import CONTEXT_SUMMARY_PROMPT
from .state import ContextBudget, ContextEntry, State, normalize_repo_path
from .tracing import trace_compaction

CompactionMode = Literal["drop", "checkpoint", "summarize"]

logger = logging.getLogger(__name__)

_CHECKPOINT_PREFIX = "[deterministic context checkpoint:"
_MODEL_DIGEST_MARKER = "[model-generated digest follows]"
_EDIT_LABEL = "Files changed in compacted steps: "
_TEST_LABEL = "Test runs compacted: "
_SEARCH_LABEL = "Recent searches before compaction: "
_INSPECT_LABEL = "Files inspected before compaction: "
_TEST_STATUSES = ("PASSED", "FAILED")
_HEADER_RE = re.compile(
    r"^(?:\[deterministic context checkpoint: (\d+) action\(s\)\]"
    r"|\[digest of (\d+) earlier action\(s\)\])$"
)

_MAX_EDITS = 12
_MAX_TESTS = 8
_MAX_SEARCHES = 8
_MAX_INSPECTIONS = 12
_FACT_CHARS = 220


@dataclass
class _Ledger:
    """Small, deterministic state carried through rolling compactions."""

    action_count: int = 0
    edits: list[tuple[str, str]] = field(default_factory=list)
    tests: list[tuple[str, str]] = field(default_factory=list)
    searches: list[tuple[str, str]] = field(default_factory=list)
    inspected: list[str] = field(default_factory=list)
    prior_semantic: str = ""


def _limit_text(text: str, max_chars: int) -> str:
    """Return text within an exact character ceiling."""
    if len(text) <= max_chars:
        return text
    if max_chars <= 3:
        return text[:max_chars]
    return text[: max_chars - 3].rstrip() + "..."


def _one_line(text: str, max_chars: int = _FACT_CHARS) -> str:
    return _limit_text(" ".join(text.split()) or "No recorded result.", max_chars)


def _metadata(text: str, max_chars: int = _FACT_CHARS) -> str:
    """Escape control characters so tool-controlled metadata cannot mint rows."""
    escaped: list[str] = []
    for char in text:
        if char == "\n":
            escaped.append(r"\n")
        elif char == "\r":
            escaped.append(r"\r")
        elif char == "\t":
            escaped.append(r"\t")
        elif char.isprintable():
            escaped.append(char)
        else:
            escaped.append(f"\\u{ord(char):04x}")
    return _limit_text("".join(escaped) or "?", max_chars)


def _sample(text: str, max_chars: int) -> str:
    """Bound semantic prose while retaining its beginning, middle, and end."""
    compact = " ".join(text.split())
    if len(compact) <= max_chars:
        return compact
    marker = " ... "
    available = max_chars - 2 * len(marker)
    if available < 3:
        return _limit_text(compact, max_chars)
    part = available // 3
    middle = max((len(compact) - part) // 2, part)
    return "".join(
        (
            compact[:part].rstrip(),
            marker,
            compact[middle : middle + part].strip(),
            marker,
            compact[-part:].lstrip(),
        )
    )[:max_chars]


def _upsert(
    values: list[tuple[str, str]],
    key: str,
    value: str,
    *,
    limit: int,
) -> None:
    values[:] = [item for item in values if item[0] != key]
    values.append((key, value))
    del values[:-limit]


def _remember(values: list[str], value: str, *, limit: int) -> None:
    if value in values:
        values.remove(value)
    values.append(value)
    del values[:-limit]


def _summary_count(text: str) -> int | None:
    first_line = text.splitlines()[0] if text else ""
    match = _HEADER_RE.fullmatch(first_line)
    return int(match.group(1) or match.group(2)) if match else None


def _semantic_part(text: str) -> str:
    marker = f"\n{_MODEL_DIGEST_MARKER}\n"
    _machine, separator, semantic = text.partition(marker)
    return semantic.strip() if separator else ""


def _merge_checkpoint(ledger: _Ledger, entry: ContextEntry) -> bool:
    """Merge only our parseable machine prefix, never model/tool prose."""
    count = _summary_count(entry.text)
    if count is None or entry.kind not in {"context_summary", "context_clean"}:
        return False

    machine = entry.text.partition(f"\n{_MODEL_DIGEST_MARKER}\n")[0]
    edits: list[tuple[str, str]] = []
    tests: list[tuple[str, str]] = []
    searches: list[tuple[str, str]] = []
    inspected: list[str] = []
    for line in machine.splitlines()[1:]:
        if line.startswith(_EDIT_LABEL):
            path, separator, result = line.removeprefix(_EDIT_LABEL).partition(" -> ")
            if separator:
                edits.append((path, result))
        elif line.startswith(_TEST_LABEL):
            value = line.removeprefix(_TEST_LABEL)
            _status, separator, rest = value.partition(" <- ")
            command, detail_separator, _detail = rest.partition(" => ")
            if separator:
                tests.append((command if detail_separator else rest, value))
        elif line.startswith(_SEARCH_LABEL):
            query, separator, result = line.removeprefix(_SEARCH_LABEL).partition(
                " -> "
            )
            if separator:
                searches.append((query, result))
        elif line.startswith(_INSPECT_LABEL):
            inspected.append(
                normalize_repo_path(line.removeprefix(_INSPECT_LABEL))
            )

    # Rows are encoded newest-first so the exact-size fitter keeps recent facts.
    for path, result in reversed(edits):
        _upsert(ledger.edits, path, result, limit=_MAX_EDITS)
    for command, value in reversed(tests):
        _upsert(ledger.tests, command, value, limit=_MAX_TESTS)
    for query, result in reversed(searches):
        _upsert(ledger.searches, query, result, limit=_MAX_SEARCHES)
    for path in reversed(inspected):
        _remember(ledger.inspected, path, limit=_MAX_INSPECTIONS)

    ledger.action_count += count
    semantic = _semantic_part(entry.text)
    if semantic:
        ledger.prior_semantic = _sample(semantic, 1200)
    return True


def _collect_facts(entries: list[ContextEntry]) -> _Ledger:
    ledger = _Ledger()
    for entry in entries:
        if _merge_checkpoint(ledger, entry):
            continue

        ledger.action_count += 1
        if entry.kind in {"edit_file", "write_file"} and entry.path:
            path = _metadata(entry.path)
            _upsert(
                ledger.edits,
                path,
                _one_line(entry.text),
                limit=_MAX_EDITS,
            )
        elif entry.kind == "run_tests":
            lines = [line.strip() for line in entry.text.splitlines()]
            command = _metadata((lines[0] if lines else "?").removeprefix("$ "))
            status = next(
                (line for line in lines[1:] if line in _TEST_STATUSES),
                "UNKNOWN",
            )
            detail = next(
                (
                    line
                    for line in lines[1:]
                    if line and line not in _TEST_STATUSES
                ),
                "No recorded result.",
            )
            value = f"{status} <- {command} => {_one_line(detail)}"
            _upsert(ledger.tests, command, value, limit=_MAX_TESTS)
        elif entry.kind in {"search", "search_reused"}:
            lines = entry.text.splitlines()
            query = _metadata(lines[0] if lines else "query=?")
            result = next(
                (_one_line(line) for line in lines[1:] if line.strip()),
                "No recorded result.",
            )
            _upsert(ledger.searches, query, result, limit=_MAX_SEARCHES)
        elif entry.kind == "inspect_file" and entry.path:
            _remember(
                ledger.inspected,
                _metadata(normalize_repo_path(entry.path)),
                limit=_MAX_INSPECTIONS,
            )
    return ledger


def _fact_lines(ledger: _Ledger) -> list[str]:
    """Return bounded rows, newest fact from every category first."""
    groups = [
        [
            f"{_EDIT_LABEL}{path} -> {result}"
            for path, result in reversed(ledger.edits)
        ],
        [
            f"{_TEST_LABEL}{value}"
            for _command, value in reversed(ledger.tests)
        ],
        [
            f"{_SEARCH_LABEL}{query} -> {result}"
            for query, result in reversed(ledger.searches)
        ],
        [
            f"{_INSPECT_LABEL}{path}"
            for path in reversed(ledger.inspected)
        ],
    ]
    rows: list[str] = []
    for index in range(max((len(group) for group in groups), default=0)):
        rows.extend(group[index] for group in groups if index < len(group))
    return rows


def _fit_lines(header: str, rows: list[str], max_chars: int) -> str:
    if len(header) > max_chars:
        raise ValueError("checkpoint ceiling cannot fit a parseable header")
    selected = [header]
    used = len(header)
    for row in rows:
        cost = 1 + len(row)
        if used + cost <= max_chars:
            selected.append(row)
            used += cost
    return "\n".join(selected)


def _compose(
    header: str,
    ledger: _Ledger,
    semantic: str,
    *,
    max_chars: int,
) -> str:
    rows = _fact_lines(ledger)
    semantic = " ".join(semantic.split())
    if not semantic:
        return _fit_lines(header, rows, max_chars)

    boundary_chars = len(_MODEL_DIGEST_MARKER) + 2
    desired = min(len(semantic), max(max_chars // 3, 48))
    machine_ceiling = max_chars - boundary_chars - desired
    if machine_ceiling < len(header):
        return _fit_lines(header, rows, max_chars)

    machine = _fit_lines(header, rows, machine_ceiling)
    semantic_room = max_chars - len(machine) - boundary_chars
    if semantic_room <= 0:
        return machine
    return "\n".join(
        (machine, _MODEL_DIGEST_MARKER, _sample(semantic, semantic_room))
    )


def _minimum_digest_chars(entries: list[ContextEntry]) -> int:
    count = max(_collect_facts(entries).action_count, 1)
    checkpoint = f"{_CHECKPOINT_PREFIX} {count} action(s)]"
    summary = f"[digest of {count} earlier action(s)]"
    return max(len(checkpoint), len(summary))


class ContextCompactor(BaseModel):
    """Bounded rolling compactor for the pseudo-SWE-agent action history."""

    budget: ContextBudget
    mode: CompactionMode = "checkpoint"
    model: LangChainModel | None = None
    summary_max_chars: int = Field(default=4000, gt=0)
    _summarizer_disabled: bool = PrivateAttr(default=False)

    def apply(self, state: State) -> State:
        if not self.budget.needs_compaction(state):
            return state
        return self._compact(state)

    @trace_compaction
    def _compact(self, state: State) -> State:
        fixed_chars = self.budget.prompt_chars(state) - state.context_chars
        low_limit = int(self.budget.input_budget_chars * self.budget.low_watermark)
        reserve = (
            min(self.summary_max_chars, max(low_limit - fixed_chars, 0))
            if self.mode != "drop"
            else 0
        )
        dropped, kept = state.split_context(
            self.budget.history_target_chars(state, reserve_chars=reserve)
        )
        if not dropped:
            if len(state.context) != 1:
                return state
            dropped, kept = list(state.context), []

        rescued, remaining = self._rescue_live_inspections(dropped, kept)
        compact_context = [*rescued, *kept]
        if not remaining:
            candidate = state.model_copy(update={"context": compact_context})
            if not compact_context or not self.budget.needs_compaction(candidate):
                return candidate.record_compaction()
            remaining, compact_context = list(state.context), []

        if self.mode == "drop":
            return self._compact_drop(state, remaining, compact_context)

        minimum = _minimum_digest_chars(remaining)
        ceiling = self._effective_digest_ceiling(
            state,
            compact_context,
            step=remaining[0].step,
            minimum_chars=minimum,
            cap=self.summary_max_chars,
        )
        without_digest = state.model_copy(update={"context": compact_context})
        if (
            ceiling < minimum
            and compact_context
            and self.budget.needs_compaction(without_digest)
        ):
            remaining, compact_context = list(state.context), []
            minimum = _minimum_digest_chars(remaining)
            ceiling = self._effective_digest_ceiling(
                state,
                compact_context,
                step=remaining[0].step,
                minimum_chars=minimum,
                cap=self.summary_max_chars,
            )
        if ceiling < minimum:
            return state.model_copy(
                update={"context": compact_context}
            ).record_compaction()

        digest = self._digest_entry(remaining, max_chars=ceiling)
        return state.model_copy(
            update={"context": [digest, *compact_context]}
        ).record_compaction()

    def _compact_drop(
        self,
        state: State,
        dropped: list[ContextEntry],
        compact_context: list[ContextEntry],
    ) -> State:
        marker = (
            f"[{len(dropped)} earlier action(s) omitted "
            "to stay within the context budget]"
        )
        ceiling = self._effective_digest_ceiling(
            state,
            compact_context,
            step=dropped[0].step,
            minimum_chars=1,
            cap=len(marker),
        )
        if ceiling == 0 and compact_context:
            dropped, compact_context = list(state.context), []
            marker = (
                f"[{len(dropped)} earlier action(s) omitted "
                "to stay within the context budget]"
            )
            ceiling = self._effective_digest_ceiling(
                state,
                compact_context,
                step=dropped[0].step,
                minimum_chars=1,
                cap=len(marker),
            )
        if ceiling == 0:
            return state.model_copy(
                update={"context": compact_context}
            ).record_compaction()
        digest = ContextEntry(
            step=dropped[0].step,
            kind="context_clean",
            text=_limit_text(marker, ceiling),
        )
        return state.model_copy(
            update={"context": [digest, *compact_context]}
        ).record_compaction()

    def _effective_digest_ceiling(
        self,
        state: State,
        compact_context: list[ContextEntry],
        *,
        step: int,
        minimum_chars: int,
        cap: int,
    ) -> int:
        placeholder = ContextEntry(step=step, kind="context_summary", text="")
        projected = state.model_copy(
            update={"context": [placeholder, *compact_context]}
        )
        used = self.budget.prompt_chars(projected)
        low_limit = int(self.budget.input_budget_chars * self.budget.low_watermark)
        high_limit = int(self.budget.input_budget_chars * self.budget.high_watermark)
        low_room = low_limit - used
        high_room = high_limit - used
        room = low_room if low_room >= minimum_chars else high_room
        return min(cap, max(room, 0))

    def _rescue_live_inspections(
        self,
        dropped: list[ContextEntry],
        kept: list[ContextEntry],
    ) -> tuple[list[ContextEntry], list[ContextEntry]]:
        """Keep latest non-stale file snapshots byte-for-byte when they fit."""
        kept_paths = {
            normalize_repo_path(entry.path)
            for entry in kept
            if entry.kind == "inspect_file" and entry.path
        }
        latest: dict[str, ContextEntry] = {}
        for entry in dropped:
            normalized_path = (
                normalize_repo_path(entry.path) if entry.path else None
            )
            if (
                entry.kind == "inspect_file"
                and normalized_path
                and normalized_path not in kept_paths
                and not entry.text.startswith("[stale")
            ):
                latest[normalized_path] = entry

        cap = int(
            self.budget.input_budget_chars
            * (self.budget.high_watermark - self.budget.low_watermark)
        )
        rescued_ids: set[int] = set()
        used = 0
        for entry in sorted(latest.values(), key=lambda item: item.step, reverse=True):
            if used + entry.context_chars <= cap:
                rescued_ids.add(id(entry))
                used += entry.context_chars

        return (
            [entry for entry in dropped if id(entry) in rescued_ids],
            [entry for entry in dropped if id(entry) not in rescued_ids],
        )

    @property
    def _can_summarize(self) -> bool:
        return (
            self.mode == "summarize"
            and self.model is not None
            and not self._summarizer_disabled
        )

    def _digest_entry(
        self,
        dropped: list[ContextEntry],
        *,
        max_chars: int,
    ) -> ContextEntry:
        step = dropped[0].step
        if self.mode == "drop":
            marker = (
                f"[{len(dropped)} earlier action(s) omitted "
                "to stay within the context budget]"
            )
            return ContextEntry(
                step=step,
                kind="context_clean",
                text=_limit_text(marker, max_chars),
            )

        minimum = _minimum_digest_chars(dropped)
        if max_chars < minimum:
            raise ValueError("digest ceiling cannot fit a parseable rolling header")

        ledger = _collect_facts(dropped)
        count = max(ledger.action_count, 1)
        if self._can_summarize:
            try:
                return ContextEntry(
                    step=step,
                    kind="context_summary",
                    text=self._summarize(
                        dropped,
                        ledger,
                        action_count=count,
                        max_chars=max_chars,
                    ),
                )
            except Exception:  # noqa: BLE001 - compaction must not kill the run
                self._summarizer_disabled = True
                logger.exception(
                    "Context summarization failed; using deterministic checkpoints "
                    "for the rest of this run."
                )

        header = f"{_CHECKPOINT_PREFIX} {count} action(s)]"
        return ContextEntry(
            step=step,
            kind="context_clean",
            text=_compose(
                header,
                ledger,
                ledger.prior_semantic,
                max_chars=max_chars,
            ),
        )

    def _summarize(
        self,
        dropped: list[ContextEntry],
        ledger: _Ledger,
        *,
        action_count: int,
        max_chars: int,
    ) -> str:
        assert self.model is not None
        content = "\n\n".join(
            entry.render(index)
            for index, entry in enumerate(dropped, start=1)
        )
        instructions = CONTEXT_SUMMARY_PROMPT.format(max_chars=max_chars)
        semantic = self.model.summarize(instructions, content).strip()
        if not semantic:
            raise ValueError("Model returned an empty digest.")
        header = f"[digest of {action_count} earlier action(s)]"
        return _compose(header, ledger, semantic, max_chars=max_chars)
