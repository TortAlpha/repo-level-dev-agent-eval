"""Agent status values and history-rendering constants."""

from typing import Literal

AgentStatus = Literal[
    "running",
    "planning",
    "editing",
    "testing",
    "needs_revision",
    "reviewing",
    "submitted",
    "solved",
    "failed",
    "handoff",
]

TERMINAL_STATUSES: set[AgentStatus] = {
    "submitted", "solved", "failed", "handoff"
}

STALE_FILE_SNAPSHOT_KINDS = frozenset({"inspect_file"})

MAX_RENDERED_TEST_OUTPUT_CHARS = 4000
