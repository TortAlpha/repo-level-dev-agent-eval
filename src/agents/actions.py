from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Annotated, Any, Literal, TypeAlias

from pydantic import BaseModel, Field, TypeAdapter, ValidationError


class SetPlanAction(BaseModel):
    action: Literal["set_plan"]
    plan: str


class InspectFileAction(BaseModel):
    action: Literal["inspect_file"]
    path: str
    # Windowed view (SWE-agent-style): show `limit` lines starting at 1-based
    # `offset`, so large files don't flood the context. Scroll by re-inspecting
    # with a later offset.
    offset: int = Field(default=1, ge=1)
    limit: int = Field(default=400, ge=1, le=2000)


class ListDirectoryAction(BaseModel):
    action: Literal["list_dir"]
    path: str = "."


class SearchAction(BaseModel):
    action: Literal["search"]
    query: str
    path: str = "."
    max_results: int = Field(default=80, ge=1, le=500)


class WriteFileAction(BaseModel):
    action: Literal["write_file"]
    path: str
    content: str


class EditFileAction(BaseModel):
    action: Literal["edit_file"]
    path: str
    old_string: str
    new_string: str
    replace_all: bool = False


class RunShellAction(BaseModel):
    action: Literal["run_shell"]
    command: str
    timeout_seconds: int | None = Field(default=None, gt=0)


class RunTestsAction(BaseModel):
    action: Literal["run_tests"]
    command: str | None = None
    timeout_seconds: int | None = Field(default=None, gt=0)


class FinishAction(BaseModel):
    action: Literal["finish"]
    summary: str


class HandoffAction(BaseModel):
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
            'nothing else, e.g. {"action": "inspect_file", "path": "..."}. Do not '
            "return reasoning-only output.",
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
            "invalid_action", f"Invalid agent action: {exc}", raw_response=text
        ) from exc


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
    if len(tool_calls) != 1:
        raise ActionParseError(
            "invalid_action",
            f"Expected exactly one action tool call, got {len(tool_calls)}.",
        )

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
            "invalid_action", f"Invalid agent action: {exc}"
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
