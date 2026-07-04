"""How each span's inputs and outputs are shaped for the trace.

These ``process_inputs`` / ``process_outputs`` callbacks decide what LangSmith
stores for a run: enough to debug from, with long text clipped by ``_head``.
"""

from __future__ import annotations

from typing import Any

from ..state import State

TRACE_TEXT_HEAD_CHARS = 2000


def _head(text: str, limit: int = TRACE_TEXT_HEAD_CHARS) -> str:
    if len(text) <= limit:
        return text
    return f"{text[:limit]}... [{len(text)} chars total]"


def generate_inputs(inputs: dict) -> dict:
    model = getattr(inputs.get("self"), "model", "unknown")
    state = inputs.get("state")

    return {
        "model": model,
        "prompt": inputs.get("prompt"),
        "state": state.to_string() if isinstance(state, State) else state,
    }


def generate_outputs(output: str) -> dict:
    return {"response": output}


def run_inputs(inputs: dict) -> dict:
    agent = inputs.get("self")
    state = inputs.get("state")

    return {
        "docker_image": getattr(agent, "docker_image", None),
        "task": state.task if isinstance(state, State) else None,
        "workdir": str(state.workdir) if isinstance(state, State) else None,
        "test_command": state.test_command if isinstance(state, State) else None,
        "max_iterations": state.max_iterations if isinstance(state, State) else None,
        "max_steps": state.max_steps if isinstance(state, State) else None,
    }


def run_outputs(state: State) -> dict:
    if not isinstance(state, State):
        return {"status": "run_crashed", "output": repr(state)}
    return {
        "status": state.status,
        "iteration": f"{state.iteration}/{state.max_iterations}",
        "step": f"{state.step}/{state.max_steps}",
        "test_passed": state.test_passed,
        "changed_files": [str(path) for path in state.changed_files],
        "last_error": state.last_error,
        "action_counts": dict(state.action_counts),
        "context_chars": state.context_chars,
    }


def summarize_inputs(inputs: dict) -> dict:
    return {
        "model": getattr(inputs.get("self"), "model", "unknown"),
        "instructions": inputs.get("instructions"),
        "content": inputs.get("content"),
    }


def step_inputs(inputs: dict) -> dict:
    state = inputs.get("state")
    previous = inputs.get("previous_action")
    if not isinstance(state, State):
        return {"state": repr(state)}
    return {
        "step": state.step,
        "iteration": state.iteration,
        "status": state.status,
        "context_chars": state.context_chars,
        "previous_action": type(previous).__name__ if previous else None,
    }


def step_outputs(output: Any) -> dict:
    if not (isinstance(output, tuple) and len(output) == 2):
        return {"output": repr(output)}
    state, action = output
    return {
        "action": type(action).__name__ if action is not None else None,
        "status": getattr(state, "status", None),
        "step": getattr(state, "step", None),
        "iteration": getattr(state, "iteration", None),
        "last_error": getattr(state, "last_error", None),
    }


def action_inputs(inputs: dict) -> dict:
    action = inputs.get("action")
    state = inputs.get("state")
    payload: dict[str, Any] = {}
    if hasattr(action, "model_dump"):
        payload = {
            key: _head(value) if isinstance(value, str) else value
            for key, value in action.model_dump().items()
        }
    return {"action": payload, "step": getattr(state, "step", None)}


def action_outputs(state: Any) -> dict:
    if not isinstance(state, State):
        return {"output": repr(state)}
    return {
        "status": state.status,
        "observation": _head(state.last_observation),
        "error": state.last_error,
    }


def shell_inputs(inputs: dict) -> dict:
    return {
        "command": inputs.get("command"),
        "timeout_seconds": inputs.get("timeout_seconds"),
    }


def shell_outputs(result: Any) -> dict:
    return {
        "returncode": getattr(result, "returncode", None),
        "output": _head(getattr(result, "output", "") or ""),
    }


def compaction_inputs(inputs: dict) -> dict:
    compactor = inputs.get("self")
    state = inputs.get("state")
    payload: dict[str, Any] = {"mode": getattr(compactor, "mode", None)}
    if isinstance(state, State):
        payload["context_chars"] = state.context_chars
        payload["entries"] = len(state.context)
        budget = getattr(compactor, "budget", None)
        if budget is not None:
            payload["prompt_chars"] = budget.prompt_chars(state)
            payload["input_budget_chars"] = budget.input_budget_chars
    return payload


def compaction_outputs(state: Any) -> dict:
    if not isinstance(state, State):
        return {"output": repr(state)}
    first = state.context[0] if state.context else None
    return {
        "context_chars": state.context_chars,
        "entries": len(state.context),
        "digest_kind": getattr(first, "kind", None),
        "digest": _head(getattr(first, "text", "") or "", 1000),
    }


def tool_inputs(inputs: dict) -> dict:
    return {
        key: _head(value) if isinstance(value, str) else value
        for key, value in inputs.items()
        if key != "self"
    }


def tool_outputs(output: Any) -> dict:
    if isinstance(output, str):
        return {"output": _head(output)}
    if isinstance(output, tuple):
        return {
            "output": [
                _head(item) if isinstance(item, str) else item for item in output
            ]
        }
    return {"output": output}
