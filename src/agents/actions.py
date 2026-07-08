from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Annotated, Any, Literal, TypeAlias

from pydantic import BaseModel, Field, TypeAdapter, ValidationError


class ActionBase(BaseModel):
    """Shared optional fields for every action.

    ``thought`` is a visible ReAct-style reasoning channel: 1-3 sentences the
    model may emit *before* the action fields. It gives reasoning models a
    legal place to think (instead of burning hidden reasoning budget and
    returning action-less turns), is ignored by the executor, and is replayed
    in the history/traces. Declared first so it precedes the action fields in
    schemas and generation order.
    """

    thought: str | None = None


class SetPlanAction(ActionBase):
    action: Literal["set_plan"]
    plan: str


class InspectFileAction(ActionBase):
    action: Literal["inspect_file"]
    path: str
    # Windowed view (SWE-agent-style): show `limit` lines starting at 1-based
    # `offset`, so large files don't flood the context. Scroll by re-inspecting
    # with a later offset.
    offset: int = Field(default=1, ge=1)
    limit: int = Field(default=400, ge=1, le=2000)


class ListDirectoryAction(ActionBase):
    action: Literal["list_dir"]
    path: str = "."


class SearchAction(ActionBase):
    action: Literal["search"]
    query: str
    path: str = "."
    max_results: int = Field(default=80, ge=1, le=500)


class WriteFileAction(ActionBase):
    action: Literal["write_file"]
    path: str
    content: str


class EditFileAction(ActionBase):
    action: Literal["edit_file"]
    path: str
    old_string: str
    new_string: str
    replace_all: bool = False


class RunShellAction(ActionBase):
    action: Literal["run_shell"]
    command: str
    timeout_seconds: int | None = Field(default=None, gt=0)


class RunTestsAction(ActionBase):
    action: Literal["run_tests"]
    command: str | None = None
    timeout_seconds: int | None = Field(default=None, gt=0)


class FinishAction(ActionBase):
    action: Literal["finish"]
    summary: str


class HandoffAction(ActionBase):
    action: Literal["handoff"]
    reason: str


AgentAction: TypeAlias = Annotated[
    SetPlanAction
    | InspectFileAction
    | ListDirectoryAction
    | SearchAction
    | WriteFileAction
    | EditFileAction
    | RunShellAction
    | RunTestsAction
    | FinishAction
    | HandoffAction,
    Field(discriminator="action"),
]

ACTION_ADAPTER: TypeAdapter[AgentAction] = TypeAdapter(AgentAction)
ACTION_MODELS: dict[str, type[BaseModel]] = {
    "set_plan": SetPlanAction,
    "inspect_file": InspectFileAction,
    "list_dir": ListDirectoryAction,
    "search": SearchAction,
    "write_file": WriteFileAction,
    "edit_file": EditFileAction,
    "run_shell": RunShellAction,
    "run_tests": RunTestsAction,
    "finish": FinishAction,
    "handoff": HandoffAction,
}
ACTION_DESCRIPTIONS = {
    "set_plan": "Set or update the implementation plan.",
    "inspect_file": "Read a line window from a repository file.",
    "list_dir": "List files below a repository-relative path.",
    "search": "Search repository files by text or regex.",
    "write_file": "Create or fully rewrite one repository file.",
    "edit_file": "Replace an exact string in one repository file.",
    "run_shell": "Run a shell command inside the repository sandbox.",
    "run_tests": "Run the visible tests or a focused test command.",
    "finish": "Finish after code changed and visible tests pass.",
    "handoff": "Stop and explain why the task cannot be completed.",
}


def same_action(a: AgentAction | None, b: AgentAction | None) -> bool:
    """Semantic equality for the repeated-action guard: two identical actions
    with different ``thought`` texts are still the same repeated action."""
    if a is None or b is None:
        return False
    return a.model_dump(exclude={"thought"}) == b.model_dump(exclude={"thought"})


class ActionParseError(ValueError):
    """Raised when a model response cannot be parsed into an action."""

    def __init__(self, kind: str, message: str, raw_response: str = "") -> None:
        super().__init__(message)
        self.kind = kind
        self.raw_response = raw_response


def parse_action(text: str) -> AgentAction:
    if not text or not text.strip():
        # Reasoning models sometimes return a reasoning-only turn (no answer),
        # which arrives here as empty text. Nudge for an actual action rather
        # than emitting an opaque JSON error.
        raise ActionParseError(
            "no_action",
            "No action returned — reply with exactly one JSON action object and "
            'nothing else, e.g. {"thought": "why", "action": "inspect_file", '
            '"path": "..."}. Put brief reasoning in the optional thought field '
            "instead of returning reasoning-only output.",
            raw_response=text,
        )
    try:
        # strict=False accepts raw newlines/tabs inside string values: local
        # models routinely emit multiline file content without \n escapes.
        raw_json = _extract_json_object(text)
        data = json.loads(raw_json, strict=False)
    except json.JSONDecodeError as exc:
        raise ActionParseError(
            "malformed_action", f"Malformed action JSON: {exc}", raw_response=text
        ) from exc
    except ValueError as exc:
        raise ActionParseError("malformed_action", str(exc), raw_response=text) from exc

    try:
        return ACTION_ADAPTER.validate_python(data)
    except ValidationError as exc:
        raise ActionParseError(
            "invalid_action", _validation_message(data, exc), raw_response=text
        ) from exc


def _validation_message(data: Any, exc: ValidationError) -> str:
    """Actionable validation feedback for the model.

    Raw pydantic errors (union_tag_not_found, docs URLs) are noise a model
    cannot act on; name the actual problem and the valid vocabulary instead.
    """
    names = ", ".join(ACTION_MODELS)
    if isinstance(data, dict):
        tag = data.get("action")
        if tag is None:
            keys = ", ".join(sorted(map(str, data.keys()))) or "none"
            return (
                f'Invalid agent action: missing the "action" field. Received '
                f"keys: {keys}. Reply with one JSON action object whose "
                f'"action" is one of: {names}.'
            )
        if tag not in ACTION_MODELS:
            return (
                f"Invalid agent action: unknown action {tag!r}. "
                f'Valid "action" values: {names}.'
            )
    problems = "; ".join(
        f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}"
        for err in exc.errors()[:5]
    )
    return f"Invalid agent action: {problems}"


def action_tool_schemas() -> list[dict[str, Any]]:
    """OpenAI-compatible tool definitions for one-action-at-a-time agents."""
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": ACTION_DESCRIPTIONS[name],
                "parameters": _parameters_without_action(model_type),
            },
        }
        for name, model_type in ACTION_MODELS.items()
    ]


def parse_tool_action(tool_calls: Sequence[Mapping[str, Any]]) -> AgentAction:
    """Convert a chat-model tool call into an ``AgentAction``."""
    if not tool_calls:
        raise ActionParseError(
            "no_action",
            "No tool action returned — call exactly one available action tool.",
        )
    # Some backends ignore single-call hints and return several calls; running
    # the first one keeps the one-action-per-step loop moving instead of
    # burning the step on a rejection.
    call = tool_calls[0]
    name = str(call.get("name") or "")
    if name not in ACTION_MODELS:
        raise ActionParseError("invalid_action", f"Unknown action tool: {name}")

    raw_args = call.get("args") or {}
    if isinstance(raw_args, str):
        try:
            args = json.loads(raw_args, strict=False)
        except json.JSONDecodeError as exc:
            raise ActionParseError(
                "malformed_action", f"Malformed tool arguments JSON: {exc}"
            ) from exc
    elif isinstance(raw_args, Mapping):
        args = dict(raw_args)
    else:
        raise ActionParseError(
            "invalid_action",
            f"Tool arguments must be an object, got {type(raw_args).__name__}.",
        )

    return _validate_action({"action": name, **args})


def _extract_json_object(text: str) -> str:
    stripped = text.strip()

    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()

    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped

    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("Model response does not contain a JSON object.")

    return stripped[start : end + 1]


def _validate_action(data: dict[str, Any]) -> AgentAction:
    try:
        return ACTION_ADAPTER.validate_python(data)
    except ValidationError as exc:
        raise ActionParseError(
            "invalid_action", _validation_message(data, exc)
        ) from exc


def _parameters_without_action(model_type: type[BaseModel]) -> dict[str, Any]:
    schema = model_type.model_json_schema()
    schema.pop("title", None)
    schema.pop("$defs", None)
    properties = dict(schema.get("properties") or {})
    properties.pop("action", None)
    required = [
        key for key in schema.get("required", [])
        if key != "action"
    ]
    schema["properties"] = properties
    schema["required"] = required
    return schema
