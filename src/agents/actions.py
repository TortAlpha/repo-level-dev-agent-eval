from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from copy import deepcopy
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
    purpose: Literal["verification", "compatibility"] = "verification"
    timeout_seconds: int | None = Field(default=None, gt=0)


class FinishAction(ActionBase):
    action: Literal["finish"]
    summary: str


class HandoffAction(ActionBase):
    action: Literal["handoff"]
    reason: str


class SubtaskDraft(BaseModel):
    id: str = Field(pattern=r"^[a-z][a-z0-9_-]{0,31}$")
    objective: str = Field(min_length=8, max_length=500)
    kind: Literal["investigate", "implement", "verify", "compatibility"]
    dependencies: list[str] = Field(default_factory=list, max_length=8)
    acceptance_command: str | None = Field(default=None, max_length=1000)


class SetSubtasksAction(ActionBase):
    action: Literal["set_subtasks"]
    subtasks: list[SubtaskDraft] = Field(min_length=2, max_length=12)


class CompleteSubtaskAction(ActionBase):
    action: Literal["complete_subtask"]
    subtask_id: str
    evidence: str = Field(min_length=8, max_length=2000)


class ReopenSubtaskAction(ActionBase):
    action: Literal["reopen_subtask"]
    subtask_id: str
    reason: str = Field(min_length=8, max_length=2000)


class ReportAction(ActionBase):
    """Role-local terminal action (multi-agent): hand results back to the
    orchestration layer. Unlike ``finish`` it carries no solved/failed
    semantics and never reaches the executor."""

    action: Literal["report"]
    summary: str


class DelegateAction(ActionBase):
    """Orchestrator-only: run one specialist role and observe its report."""

    action: Literal["delegate"]
    role: Literal["planner", "developer", "tester", "reviewer"]
    instruction: str


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
    | HandoffAction
    | SetSubtasksAction
    | CompleteSubtaskAction
    | ReopenSubtaskAction
    | ReportAction
    | DelegateAction,
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
    "run_tests": (
        "Run visible/focused tests, or an explicit legacy-behavior probe with "
        "purpose=compatibility."
    ),
    "finish": "Finish after code changed and visible tests pass.",
    "handoff": "Stop and explain why the task cannot be completed.",
    "set_subtasks": "Create the structured dependency-aware work queue.",
    "complete_subtask": "Complete the active subtask with concrete evidence.",
    "reopen_subtask": "Reopen implementation after failed verification.",
    "report": "Finish your role: report your results back to the orchestrator.",
    "delegate": "Delegate a focused instruction to one specialist role.",
}

_EXTRA_MODELS: dict[str, type[BaseModel]] = {
    "report": ReportAction,
    "delegate": DelegateAction,
    "set_subtasks": SetSubtasksAction,
    "complete_subtask": CompleteSubtaskAction,
    "reopen_subtask": ReopenSubtaskAction,
}


class ActionSpace:
    """A named subset of the action vocabulary (multi-agent roles restrict it).

    Registered by name so pydantic agents can reference a space as a plain
    string field. The default space is the full single-agent vocabulary.
    """

    def __init__(self, name: str, action_names: Sequence[str]) -> None:
        catalog = {**ACTION_MODELS, **_EXTRA_MODELS}
        unknown = [n for n in action_names if n not in catalog]
        if unknown:
            raise ValueError(f"unknown actions for space {name!r}: {unknown}")
        self.name = name
        self.models: dict[str, type[BaseModel]] = {
            n: catalog[n] for n in action_names
        }
        types = list(self.models.values())
        if len(types) == 1:
            self.adapter: TypeAdapter[Any] = TypeAdapter(types[0])
        else:
            union = types[0]
            for t in types[1:]:
                union = union | t
            self.adapter = TypeAdapter(
                Annotated[union, Field(discriminator="action")]
            )
        ACTION_SPACES[name] = self


ACTION_SPACES: dict[str, ActionSpace] = {}
DEFAULT_SPACE = ActionSpace("default", list(ACTION_MODELS))


def resolve_space(space: str | None) -> ActionSpace:
    if space is None:
        return DEFAULT_SPACE
    if space not in ACTION_SPACES:
        raise ValueError(f"unknown action space: {space!r}")
    return ACTION_SPACES[space]


def same_action(a: AgentAction | None, b: AgentAction | None) -> bool:
    """Semantic equality for the repeated-action guard: two identical actions
    with different ``thought`` texts are still the same repeated action."""
    if a is None or b is None:
        return False
    return a.model_dump(exclude={"thought"}) == b.model_dump(exclude={"thought"})


class ActionParseError(ValueError):
    """Raised when a model response cannot be parsed into an action."""

    def __init__(
        self,
        kind: str,
        message: str,
        raw_response: str = "",
        *,
        tool_call_id: str = "",
        tool_name: str = "",
        tool_arguments: Mapping[str, Any] | str | None = None,
    ) -> None:
        super().__init__(message)
        self.kind = kind
        self.raw_response = raw_response
        # A malformed native tool call still needs to be replayed as a tool
        # call plus a ToolMessage error.  Keeping this protocol metadata on
        # the parse failure lets the single-agent recovery preserve that
        # conversation shape instead of silently degrading into text history.
        self.tool_call_id = tool_call_id
        self.tool_name = tool_name
        self.tool_arguments = tool_arguments


def parse_action(text: str, space: str | None = None) -> AgentAction:
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
        return resolve_space(space).adapter.validate_python(data)
    except ValidationError as exc:
        raise ActionParseError(
            "invalid_action",
            _validation_message(data, exc, space),
            raw_response=text,
        ) from exc


def _validation_message(data: Any, exc: ValidationError, space: str | None = None) -> str:
    """Actionable validation feedback for the model.

    Raw pydantic errors (union_tag_not_found, docs URLs) are noise a model
    cannot act on; name the actual problem and the valid vocabulary instead.
    """
    vocabulary = resolve_space(space).models
    names = ", ".join(vocabulary)
    if isinstance(data, dict):
        tag = data.get("action")
        if tag is None:
            keys = ", ".join(sorted(map(str, data.keys()))) or "none"
            return (
                f'Invalid agent action: missing the "action" field. Received '
                f"keys: {keys}. Reply with one JSON action object whose "
                f'"action" is one of: {names}.'
            )
        if tag not in vocabulary:
            return (
                f"Invalid agent action: unknown action {tag!r}. "
                f'Valid "action" values: {names}.'
            )
    problems = "; ".join(
        f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}"
        for err in exc.errors()[:5]
    )
    return f"Invalid agent action: {problems}"


def action_tool_schemas(space: str | None = None) -> list[dict[str, Any]]:
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
        for name, model_type in resolve_space(space).models.items()
    ]


def parse_tool_action(
    tool_calls: Sequence[Mapping[str, Any]], space: str | None = None
) -> AgentAction:
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
    call_id = str(call.get("id") or "")
    vocabulary = resolve_space(space).models
    name = str(call.get("name") or "")
    raw_args = call.get("args")

    def tool_error(kind: str, message: str) -> ActionParseError:
        return ActionParseError(
            kind,
            message,
            tool_call_id=call_id,
            tool_name=name,
            tool_arguments=raw_args,
        )

    if name not in vocabulary:
        raise tool_error(
            "invalid_action",
            f"Unknown action tool: {name}. Available: {', '.join(vocabulary)}.",
        )

    raw_args = raw_args or {}
    if isinstance(raw_args, str):
        try:
            args = json.loads(raw_args, strict=False)
        except json.JSONDecodeError as exc:
            raise tool_error(
                "malformed_action", f"Malformed tool arguments JSON: {exc}"
            ) from exc
    elif isinstance(raw_args, Mapping):
        args = dict(raw_args)
    else:
        raise tool_error(
            "invalid_action",
            f"Tool arguments must be an object, got {type(raw_args).__name__}.",
        )

    try:
        return _validate_action({"action": name, **args}, space)
    except ActionParseError as exc:
        raise tool_error(exc.kind, str(exc)) from exc


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


def _validate_action(data: dict[str, Any], space: str | None = None) -> AgentAction:
    try:
        return resolve_space(space).adapter.validate_python(data)
    except ValidationError as exc:
        raise ActionParseError(
            "invalid_action", _validation_message(data, exc, space)
        ) from exc


def _parameters_without_action(model_type: type[BaseModel]) -> dict[str, Any]:
    schema = model_type.model_json_schema()
    schema.pop("title", None)
    definitions = dict(schema.pop("$defs", None) or {})
    schema = _inline_schema_refs(schema, definitions)
    properties = dict(schema.get("properties") or {})
    properties.pop("action", None)
    required = [
        key for key in schema.get("required", [])
        if key != "action"
    ]
    schema["properties"] = properties
    schema["required"] = required
    return schema


def _inline_schema_refs(value: Any, definitions: Mapping[str, Any]) -> Any:
    """Inline Pydantic-local refs so provider tool schemas are self-contained."""
    if isinstance(value, list):
        return [_inline_schema_refs(item, definitions) for item in value]
    if not isinstance(value, dict):
        return value
    reference = value.get("$ref")
    if isinstance(reference, str) and reference.startswith("#/$defs/"):
        name = reference.removeprefix("#/$defs/")
        if name not in definitions:
            raise ValueError(f"Unknown local schema reference: {reference}")
        resolved = _inline_schema_refs(deepcopy(definitions[name]), definitions)
        siblings = {
            key: _inline_schema_refs(item, definitions)
            for key, item in value.items()
            if key != "$ref"
        }
        return {**resolved, **siblings}
    return {
        key: _inline_schema_refs(item, definitions) for key, item in value.items()
    }
