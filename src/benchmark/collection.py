"""Benchmark task specs, parsed from ``repositories/collection.csv``."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

DEFAULT_COLLECTION = Path("repositories/collection.csv")


@dataclass
class TaskSpec:
    """One benchmark task, parsed from a ``collection.csv`` row."""

    task_id: str
    repo_path: Path
    task_file_path: Path
    size: str
    visible_test_command: str
    hidden_test_command: str
    hidden_tests_path: Path
    base_commit: str

    @classmethod
    def from_row(cls, row: dict[str, str]) -> TaskSpec:
        return cls(
            task_id=row["task_id"],
            repo_path=Path(row["repo_path"]),
            task_file_path=Path(row["task_file_path"]),
            size=row.get("size", ""),
            visible_test_command=row["visible_test_command"],
            hidden_test_command=row.get("hidden_test_command", ""),
            hidden_tests_path=Path(row["hidden_tests_path"])
            if row.get("hidden_tests_path")
            else Path(),
            base_commit=row.get("base_commit", ""),
        )


def load_collection(csv_path: Path) -> dict[str, TaskSpec]:
    with csv_path.open(encoding="utf-8") as handle:
        return {
            row["task_id"]: TaskSpec.from_row(row)
            for row in csv.DictReader(handle)
        }
