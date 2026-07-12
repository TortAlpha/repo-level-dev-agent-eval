"""Versioned, machine-readable benchmark task sets."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path


TASK_SET_DIR = Path("eval/task_sets")
TASK_SET_ALIASES = {
    "final-v1": TASK_SET_DIR / "final_v1.json",
    "final_v1": TASK_SET_DIR / "final_v1.json",
}


@dataclass(frozen=True)
class TaskSet:
    task_set_id: str
    path: Path
    groups: dict[str, list[str]]
    calibration_excluded: list[str]
    matrices: dict[str, dict]
    previously_exercised: list[str]
    sha256: str

    def tasks(self, selected_groups: list[str] | None = None) -> list[str]:
        names = selected_groups or list(self.groups)
        unknown = [name for name in names if name not in self.groups]
        if unknown:
            raise ValueError(
                f"unknown task-set group(s): {', '.join(unknown)}; available: "
                f"{', '.join(self.groups)}"
            )
        result: list[str] = []
        seen: set[str] = set()
        for name in names:
            for task_id in self.groups[name]:
                if task_id not in seen:
                    result.append(task_id)
                    seen.add(task_id)
        return result

    def matrix(self, name: str) -> tuple[list[str], list[str]]:
        if name not in self.matrices:
            raise ValueError(
                f"unknown task-set matrix {name!r}; available: "
                f"{', '.join(self.matrices)}"
            )
        item = self.matrices[name]
        agents = [str(agent) for agent in item.get("agents") or []]
        if not agents:
            raise ValueError(f"task-set matrix {name!r} has no agents")
        if item.get("tasks"):
            tasks = [str(task_id) for task_id in item["tasks"]]
        else:
            tasks = self.tasks([str(group) for group in item.get("groups") or []])
        unknown = sorted(set(tasks) - set(self.tasks()))
        if unknown:
            raise ValueError(
                f"task-set matrix {name!r} references unknown tasks: "
                f"{', '.join(unknown)}"
            )
        return tasks, agents


def resolve_task_set_path(value: str | Path) -> Path:
    raw = str(value).strip()
    return TASK_SET_ALIASES.get(raw, Path(raw))


def load_task_set(value: str | Path) -> TaskSet:
    path = resolve_task_set_path(value)
    raw = path.read_bytes()
    payload = json.loads(raw)
    groups = {
        str(name): [str(task_id) for task_id in tasks]
        for name, tasks in dict(payload.get("groups") or {}).items()
    }
    if not groups or any(not tasks for tasks in groups.values()):
        raise ValueError(f"task set {path} must contain non-empty groups")

    flattened = [task_id for tasks in groups.values() for task_id in tasks]
    duplicates = sorted(
        {task_id for task_id in flattened if flattened.count(task_id) > 1}
    )
    if duplicates:
        raise ValueError(f"task set {path} contains duplicates: {', '.join(duplicates)}")

    excluded = [str(item) for item in payload.get("calibration_excluded") or []]
    matrices = {
        str(name): dict(item)
        for name, item in dict(payload.get("matrices") or {}).items()
    }
    previously_exercised = [
        str(item) for item in payload.get("previously_exercised") or []
    ]
    unknown_exposure = sorted(set(previously_exercised) - set(flattened))
    if unknown_exposure:
        raise ValueError(
            f"task set {path} marks unknown exercised tasks: "
            f"{', '.join(unknown_exposure)}"
        )
    overlap = sorted(set(flattened) & set(excluded))
    if overlap:
        raise ValueError(
            f"task set {path} includes excluded calibration tasks: {', '.join(overlap)}"
        )
    return TaskSet(
        task_set_id=str(payload.get("id") or path.stem),
        path=path,
        groups=groups,
        calibration_excluded=excluded,
        matrices=matrices,
        previously_exercised=previously_exercised,
        sha256=hashlib.sha256(raw).hexdigest(),
    )
