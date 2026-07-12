"""Agent task state, split by concern.

- ``status``    — status values and rendering constants.
- ``entry``     — one action-history entry.
- ``core``      — the ``State`` snapshot and its ``with_*`` updates.
- ``rendering`` — turning a state into the model prompt / message list.
- ``budget``    — context-window sizing math for compaction.
"""

from .budget import ContextBudget
from .core import State
from .entry import ContextEntry
from .subtask import Subtask, SubtaskKind, SubtaskStatus
from .status import (
    AgentStatus,
    MAX_RENDERED_TEST_OUTPUT_CHARS,
    STALE_FILE_SNAPSHOT_KINDS,
    TERMINAL_STATUSES,
)

__all__ = [
    "State",
    "ContextEntry",
    "ContextBudget",
    "AgentStatus",
    "TERMINAL_STATUSES",
    "STALE_FILE_SNAPSHOT_KINDS",
    "MAX_RENDERED_TEST_OUTPUT_CHARS",
    "Subtask",
    "SubtaskKind",
    "SubtaskStatus",
]
