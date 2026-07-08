"""Render a ``State`` into the model prompt (message list or one dump).

Kept separate from the state's mutation logic: this is all pure formatting,
and the section/message ordering here is load-bearing for prefix caching.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from .status import MAX_RENDERED_TEST_OUTPUT_CHARS

if TYPE_CHECKING:
    from .core import State


def render_messages(state: State) -> list[tuple[str, str]]:
    """Render the state as an assistant/user dialogue.

    Local instruction-tuned models follow the agent loop far more reliably
    when the history is replayed in the chat format they were trained on (their
    own action as an assistant turn, the observation as a user turn) than when
    everything is packed into one state dump. The message list is append-only,
    so server-side prefix caching stays effective; every per-step volatile
    field lives in the final user turn.
    """
    messages: list[tuple[str, str]] = [("user", _header(state))]
    for entry in state.context:
        if entry.action_json:
            messages.append(("assistant", entry.action_json))
        messages.append(("user", entry.text))
    messages.append(("user", _progress_message(state)))
    return _merge_consecutive_users(messages)


def render_header(state: State) -> str:
    """Stable opening message (task, workdir, test command)."""
    return _header(state)


def render_progress(state: State) -> str:
    """Volatile per-step tail message (plan, iteration, budget)."""
    return _progress_message(state)


def render_state(state: State) -> str:
    """Render the whole state as one text dump (used for tracing and by the
    context budget). Section order matters for llama.cpp/LM Studio prefix
    caching: stable fields first, then the append-only history, and every
    per-step volatile field last, so each step re-evaluates only the newest
    history entry and the short tail instead of the whole prompt.
    """
    plan = state.plan.strip() or "No plan yet."
    container_id = state.container_id or "No container assigned."
    tests_status = "passed" if state.test_passed else "not passed"
    test_command = state.test_command or "No test command configured."
    last_test_output = _tail(
        state.last_test_output.strip() or "No tests run yet.",
        MAX_RENDERED_TEST_OUTPUT_CHARS,
    )
    relevant_files = _format_paths(state.relevant_files)
    changed_files = _format_paths(state.changed_files)
    context = (
        "\n\n".join(
            entry.render(i) for i, entry in enumerate(state.context, start=1)
        )
        or "No actions taken yet."
    )

    return f"""
Current repository task state:

Task:
{state.task}

Working directory:
{state.workdir}

Test command:
{test_command}

Docker container:
{container_id}

Plan:
{plan}

Action history (oldest first):
{context}

Status:
{state.status}

Iteration:
{state.iteration} / {state.max_iterations}

Step:
{state.step} / {state.max_steps}

Relevant files:
{relevant_files}

Changed files:
{changed_files}

Visible tests:
{tests_status}

Last test output:
{last_test_output}
""".strip()


def _header(state: State) -> str:
    test_command = state.test_command or "No test command configured."
    return (
        f"{state.task.strip()}\n\n"
        f"Test command: {test_command}\n"
        "All paths are relative to the repository root.\n\n"
        "Choose your first action."
    )


def _progress_message(state: State) -> str:
    last_test_output = _tail(
        state.last_test_output.strip() or "No tests run yet.",
        MAX_RENDERED_TEST_OUTPUT_CHARS,
    )
    plan = state.plan.strip() or "No plan yet."
    return (
        f"Current progress: step {state.step}/{state.max_steps}, "
        f"test iteration {state.iteration}/{state.max_iterations}.\n"
        f"Plan: {plan}\n"
        f"Changed files: {_format_paths_inline(state.changed_files)}\n"
        f"Visible tests: {'passed' if state.test_passed else 'not passed'}\n"
        f"Last test output:\n{last_test_output}\n\n"
        "Reply with exactly one JSON object for your next action."
    )


def _tail(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return f"... truncated to last {max_chars} characters ...\n{text[-max_chars:]}"


def _merge_consecutive_users(
    messages: list[tuple[str, str]],
) -> list[tuple[str, str]]:
    merged: list[tuple[str, str]] = []
    for role, text in messages:
        if merged and role == "user" and merged[-1][0] == "user":
            merged[-1] = ("user", f"{merged[-1][1]}\n\n{text}")
        else:
            merged.append((role, text))
    return merged


def _format_paths(paths: list[Path]) -> str:
    if not paths:
        return "None."
    return "\n".join(f"- {path}" for path in paths)


def _format_paths_inline(paths: list[Path]) -> str:
    if not paths:
        return "none"
    return ", ".join(str(path) for path in paths)
