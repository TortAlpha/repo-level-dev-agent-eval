"""How each span renames and tags itself from what actually happened.

``enter_*`` hooks name a span from its call arguments (so failures are named
after what they attempted); ``exit_*`` hooks refine it from the result.
"""

from __future__ import annotations

from typing import Any

from ..state import State
from .decorator import RunHook, annotate


def _action_label(action: Any) -> str:
    """The action's own discriminator (``edit_file``), not its class name."""
    return getattr(action, "action", None) or type(action).__name__


def enter_generate(run: Any, arguments: dict) -> None:
    state = arguments.get("state")
    if isinstance(state, State):
        annotate(
            run,
            name=f"generate (step {state.step})",
            tags=[f"status:{state.status}"],
        )


def exit_step(run: Any, output: Any) -> None:
    if not (isinstance(output, tuple) and len(output) == 2):
        return
    state, action = output
    label = _action_label(action) if action is not None else "no-op"
    step = getattr(state, "step", "?")
    tags = [f"action:{label}"]
    if getattr(state, "status", None):
        tags.append(f"status:{state.status}")
    annotate(run, name=f"step {step}: {label}", tags=tags)


def enter_execute(run: Any, arguments: dict) -> None:
    action = arguments.get("action")
    label = _action_label(action)
    path = getattr(action, "path", None)
    tags = [f"action:{label}"]
    if path:
        tags.append(f"path:{path}")
    annotate(
        run,
        name=f"execute {label}: {path}" if path else f"execute {label}",
        tags=tags,
        metadata={"action": label},
    )


def enter_shell(run: Any, arguments: dict) -> None:
    command = (arguments.get("command") or "").strip()
    head = command.split("\n", 1)[0]
    verb = head.split()[0] if head.split() else "shell"
    annotate(run, name=f"$ {head[:60]}" if head else "$ (empty)", tags=[f"cmd:{verb}"])


def exit_shell(run: Any, result: Any) -> None:
    code = getattr(result, "returncode", None)
    if code is None:
        return
    outcome = "ok" if code == 0 else "FAILED"
    annotate(
        run,
        name=f"{run.name or 'shell'} [{outcome}]",
        tags=[f"exit:{code}", f"outcome:{outcome}"],
    )


def enter_compaction(run: Any, arguments: dict) -> None:
    compactor = arguments.get("self")
    state = arguments.get("state")
    entries = len(state.context) if isinstance(state, State) else "?"
    mode = getattr(compactor, "mode", None)
    annotate(run, name=f"compact {entries} entries ({mode})", tags=[f"mode:{mode}"])


def describe_workspace(verb: str) -> RunHook:
    """A workspace span names itself after the file/query it touched."""

    def on_enter(run: Any, arguments: dict) -> None:
        if verb == "search":
            query = arguments.get("query")
            path = arguments.get("path")
            scope = f" in {path}" if path and path != "." else ""
            annotate(run, name=f"search {query!r}{scope}", tags=["op:search"])
            return
        path = arguments.get("path")
        tags = [f"op:{verb}"]
        if path:
            tags.append(f"path:{path}")
        annotate(run, name=f"{verb} {path}" if path else verb, tags=tags)

    return on_enter
