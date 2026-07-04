from __future__ import annotations

import json
from typing import Annotated, Literal, TypeAlias

from pydantic import BaseModel, Field, TypeAdapter, ValidationError


class SetPlanAction(BaseModel):
    action: Literal["set_plan"]
    plan: str


class InspectFileAction(BaseModel):
    action: Literal["inspect_file"]
    path: str


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


def parse_action(text: str) -> AgentAction:
    try:
        # strict=False accepts raw newlines/tabs inside string values: local
        # models routinely emit multiline file content without \n escapes.
        data = json.loads(_extract_json_object(text), strict=False)
        return ACTION_ADAPTER.validate_python(data)
    except (json.JSONDecodeError, ValidationError) as exc:
        raise ValueError(f"Invalid agent action: {exc}") from exc


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
