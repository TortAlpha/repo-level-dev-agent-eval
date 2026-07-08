"""Benchmark task specs, parsed from ``repositories/collection.csv``."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

DEFAULT_COLLECTION = Path("repositories/collection.csv")


@dataclass(frozen=True)
class HiddenTestSuite:
    """Evaluator-only test suite declared in ``collection.csv``."""

    name: str
    command: str
    required: bool


@dataclass
class TaskSpec:
    """One benchmark task, parsed from a ``collection.csv`` row."""

    task_id: str
    repo_path: Path
    task_file_path: Path
    size: str
    task_type: str
    visible_test_command: str
    hidden_test_command: str
    hidden_semantic_test_command: str
    hidden_compat_test_command: str
    hidden_pr_parity_test_command: str
    hidden_tests_path: Path
    base_commit: str
    task_status: str
    setup_commands: list[str]

    @classmethod
    def from_row(cls, row: dict[str, str]) -> TaskSpec:
        return cls(
            task_id=row["task_id"],
            repo_path=Path(row["repo_path"]),
            task_file_path=Path(row["task_file_path"]),
            size=row.get("size", ""),
            task_type=row.get("task_type", ""),
            visible_test_command=row["visible_test_command"],
            hidden_test_command=row.get("hidden_test_command", ""),
            hidden_semantic_test_command=row.get("hidden_semantic_test_command", ""),
            hidden_compat_test_command=row.get("hidden_compat_test_command", ""),
            hidden_pr_parity_test_command=row.get("hidden_pr_parity_test_command", ""),
            hidden_tests_path=Path(row["hidden_tests_path"])
            if row.get("hidden_tests_path")
            else Path(),
            base_commit=row.get("base_commit", ""),
            task_status=row.get("task_status", ""),
            # Per-task container setup (e.g. SETUPTOOLS_SCM_PRETEND_VERSION or
            # extra test deps), ';'-separated. Empty -> evaluator defaults.
            setup_commands=[
                part.strip()
                for part in row.get("setup_commands", "").split(";")
                if part.strip()
            ],
        )

    def hidden_suites(self) -> list[HiddenTestSuite]:
        """Hidden suites to run for this task.

        New rows can split hidden tests into semantic/compat/pr_parity columns.
        Older rows keep using ``hidden_test_command`` as one required suite.
        """
        suites: list[HiddenTestSuite] = []
        if self.hidden_semantic_test_command:
            suites.append(
                HiddenTestSuite(
                    "hidden_semantic",
                    self.hidden_semantic_test_command,
                    required=True,
                )
            )
        if self.hidden_compat_test_command:
            suites.append(
                HiddenTestSuite(
                    "hidden_compat",
                    self.hidden_compat_test_command,
                    required=True,
                )
            )
        if self.hidden_pr_parity_test_command:
            suites.append(
                HiddenTestSuite(
                    "hidden_pr_parity",
                    self.hidden_pr_parity_test_command,
                    required=False,
                )
            )
        if suites:
            return suites
        if self.hidden_test_command:
            return [HiddenTestSuite("hidden", self.hidden_test_command, required=True)]
        return []

    def required_hidden_suites(self) -> list[HiddenTestSuite]:
        return [suite for suite in self.hidden_suites() if suite.required]


def load_collection(csv_path: Path) -> dict[str, TaskSpec]:
    with csv_path.open(encoding="utf-8") as handle:
        return {
            row["task_id"]: TaskSpec.from_row(row)
            for row in csv.DictReader(handle)
        }
