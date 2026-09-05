"""Durable structured work units for decomposed single-agent runs."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

SubtaskKind = Literal["investigate", "implement", "verify", "compatibility"]
SubtaskStatus = Literal["pending", "active", "completed", "reopened"]


class Subtask(BaseModel):
    id: str
    objective: str
    kind: SubtaskKind
    dependencies: list[str] = Field(default_factory=list)
    acceptance_command: str | None = None
    status: SubtaskStatus = "pending"
    evidence: str = ""
    started_revision: int = Field(default=0, ge=0)
