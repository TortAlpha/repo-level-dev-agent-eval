"""Post-run evaluation in a Docker sandbox: regressions and hidden tests.

Runs outside the agent's LangSmith trace (this is harness infrastructure, not
agent behavior), so setup/test spans do not pollute the project.
"""

from __future__ import annotations

import shutil
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

from langsmith import tracing_context

from ..agents.sandbox import DockerSandbox
from .collection import TaskSpec

# Enough to import the package under test and run its pytest suite. Override
# per-repo with --setup-command when a task needs extra dependencies.
DEFAULT_SETUP_COMMANDS = [
    "python -m pip install -q -e .",
    "python -m pip install -q pytest",
]

_JUNIT_FILE = ".pytest-report.xml"


@dataclass
class EvalResult:
    visible_passed: bool | None = None
    hidden_passed: bool | None = None
    hidden_semantic_passed: bool | None = None
    hidden_compat_passed: bool | None = None
    hidden_pr_parity_passed: bool | None = None
    task_success: bool | None = None
    regressions: int | None = None
    hidden_suite_results: dict[str, bool] = field(default_factory=dict)


def _sandbox(repo: Path, image: str, network_disabled: bool, timeout: int) -> DockerSandbox:
    return DockerSandbox(
        workdir=repo,
        image=image,
        network_disabled=network_disabled,
        shell_timeout_seconds=timeout,
        test_timeout_seconds=timeout,
    )


def _parse_junit_passing(path: Path) -> set[str]:
    """Passing test ids from a pytest JUnit-XML report (built-in, no plugin)."""
    if not path.is_file():
        return set()
    passing: set[str] = set()
    for case in ET.parse(path).getroot().iter("testcase"):
        test_id = f"{case.get('classname', '')}::{case.get('name', '')}"
        if not any(child.tag in ("failure", "error", "skipped") for child in case):
            passing.add(test_id)
    return passing


def _run_visible_junit(
    sandbox: DockerSandbox, repo: Path, command: str, timeout: int
) -> tuple[bool, set[str], str]:
    """Run the visible suite with a JUnit report and return the passing ids.
    The report is removed afterwards so it never pollutes the repo/diff."""
    run = sandbox.run_shell(f"{command} --junit-xml={_JUNIT_FILE}", timeout)
    report = repo / _JUNIT_FILE
    passing = _parse_junit_passing(report)
    report.unlink(missing_ok=True)
    return run.success, passing, run.output


def _repo_relative_hidden_command(command: str) -> str:
    # Hidden tests are copied into the repo before execution, so a command like
    # ``pytest ../hidden_tests/tests/x.py`` becomes ``pytest tests/x.py``.
    return command.replace("../hidden_tests/", "")


def _hidden_suite_label(name: str) -> str:
    return {
        "hidden": "Hidden tests",
        "hidden_semantic": "Hidden semantic tests",
        "hidden_compat": "Hidden compat tests",
        "hidden_pr_parity": "Hidden PR-parity tests",
    }.get(name, name.replace("_", " ").title())


def _set_hidden_result(result: EvalResult, name: str, passed: bool) -> None:
    result.hidden_suite_results[name] = passed
    if name == "hidden_semantic":
        result.hidden_semantic_passed = passed
    elif name == "hidden_compat":
        result.hidden_compat_passed = passed
    elif name == "hidden_pr_parity":
        result.hidden_pr_parity_passed = passed


def collect_visible_passing(
    repo: Path,
    command: str,
    *,
    docker_image: str,
    network_disabled: bool,
    setup_commands: list[str],
    shell_timeout: int,
    test_timeout: int,
) -> set[str]:
    """Baseline for the regression metric: visible tests passing on the
    pristine repo, before the agent touches it."""
    sandbox = _sandbox(repo, docker_image, network_disabled, test_timeout)
    with tracing_context(enabled=False):
        try:
            sandbox.start()
            for setup in setup_commands:
                sandbox.run_shell(setup, shell_timeout)
            _, passing, _ = _run_visible_junit(sandbox, repo, command, test_timeout)
            return passing
        finally:
            sandbox.stop()


def evaluate_solution(
    task: TaskSpec,
    task_ws: Path,
    visible_command: str,
    *,
    baseline_passing: set[str] | None,
    do_hidden: bool,
    docker_image: str,
    network_disabled: bool,
    setup_commands: list[str],
    shell_timeout: int,
    test_timeout: int,
) -> EvalResult:
    """Post-run evaluation in one sandbox: the after-state visible run (for
    regressions, before any overlay) and then the withheld hidden tests.

    Hidden tests are overlaid onto the repo package so their in-package
    relative imports resolve; the overlay happens only now, after the agent
    finished, so the tests never leak into its run.
    """
    repo = task_ws / "repo"
    hidden = task_ws / "hidden_tests"
    result = EvalResult()

    sandbox = _sandbox(repo, docker_image, network_disabled, test_timeout)
    with tracing_context(enabled=False):
        try:
            sandbox.start()
            for setup in setup_commands:
                sandbox.run_shell(setup, shell_timeout)

            visible_passed, after, visible_output = _run_visible_junit(
                sandbox, repo, visible_command, test_timeout
            )
            result.visible_passed = visible_passed
            verdict = "PASSED" if visible_passed else "FAILED"
            print(f"\n=== Visible tests: {verdict} ===")
            if not visible_passed:
                print(visible_output[-2000:])

            if baseline_passing is not None:
                regressed = sorted(baseline_passing - after)
                result.regressions = len(regressed)
                print(f"\n=== Regressions: {len(regressed)} ===")
                for test_id in regressed[:20]:
                    print(f"  - {test_id}")

            hidden_suites = task.hidden_suites() if do_hidden else []
            if hidden_suites:
                if hidden.exists():
                    shutil.copytree(hidden, repo, dirs_exist_ok=True)

                required: list[bool] = []
                for suite in hidden_suites:
                    command = _repo_relative_hidden_command(suite.command)
                    hidden_result = sandbox.run_shell(command, test_timeout)
                    passed = hidden_result.success
                    _set_hidden_result(result, suite.name, passed)
                    if suite.required:
                        required.append(passed)
                    verdict = "PASSED" if passed else "FAILED"
                    label = _hidden_suite_label(suite.name)
                    print(f"\n=== {label}: {verdict} ===")
                    print(hidden_result.output[-2000:])

                if required:
                    result.hidden_passed = all(required)
                    verdict = "PASSED" if result.hidden_passed else "FAILED"
                    print(f"\n=== Required hidden tests: {verdict} ===")

            if result.hidden_passed is not None:
                result.task_success = bool(result.visible_passed) and result.hidden_passed
        finally:
            sandbox.stop()
    return result
