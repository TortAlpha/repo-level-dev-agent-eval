"""LangSmith tracing for the agents.

Split by concern:
- ``decorator``  — the ``make_traced`` mechanism (dynamic span naming).
- ``payloads``   — how each span's inputs/outputs are shaped for the trace.
- ``describers`` — how each span renames/tags itself from what happened.
- ``spans``      — the concrete ``trace_*`` decorators wiring the above.
- ``session``    — LangSmith env setup, run metadata, post-run feedback.

The public names below are unchanged from the old ``tracing.py`` module, so
callers keep importing them from ``.tracing``.
"""

from .decorator import LANGSMITH_PROJECT_NAME, make_traced
from .session import attach_run_feedback, configure_langsmith, run_trace_extra
from .spans import (
    trace_action_execution,
    trace_agent_run,
    trace_agent_step,
    trace_compaction,
    trace_model_call,
    trace_shell_command,
    trace_summarize_call,
    trace_workspace_op,
)

__all__ = [
    "LANGSMITH_PROJECT_NAME",
    "make_traced",
    "attach_run_feedback",
    "configure_langsmith",
    "run_trace_extra",
    "trace_action_execution",
    "trace_agent_run",
    "trace_agent_step",
    "trace_compaction",
    "trace_model_call",
    "trace_shell_command",
    "trace_summarize_call",
    "trace_workspace_op",
]
