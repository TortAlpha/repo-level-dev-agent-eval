"""Post-run evaluation in a Docker sandbox: regressions and hidden tests.

Runs outside the agent's LangSmith trace (this is harness infrastructure, not
agent behavior), so setup/test spans do not pollute the project.
"""

from __future__ import annotations

import shutil
import xml.etree.ElementTree as ET
from dataclasses import dataclass
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
    hidden_passed: bool | None = None
    regressions: int | None = None


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
) -> set[str]:
    """Run the visible suite with a JUnit report and return the passing ids.
    The report is removed afterwards so it never pollutes the repo/diff."""
    sandbox.run_shell(f"{command} --junit-xml={_JUNIT_FILE}", timeout)
    report = repo / _JUNIT_FILE
    passing = _parse_junit_passing(report)
    report.unlink(missing_ok=True)
    return passing


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
            return _run_visible_junit(sandbox, repo, command, test_timeout)
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

            if baseline_passing is not None:
                after = _run_visible_junit(sandbox, repo, visible_command, test_timeout)
                regressed = sorted(baseline_passing - after)
                result.regressions = len(regressed)
                print(f"\n=== Regressions: {len(regressed)} ===")
                for test_id in regressed[:20]:
                    print(f"  - {test_id}")

            if do_hidden and task.hidden_test_command:
                if hidden.exists():
                    shutil.copytree(hidden, repo, dirs_exist_ok=True)
                # "pytest ../hidden_tests/x.py" -> a repo-relative path, since
                # the overlay put those files at the same path in the repo.
                command = task.hidden_test_command.replace("../hidden_tests/", "")
                hidden_result = sandbox.run_shell(command, test_timeout)
                result.hidden_passed = hidden_result.success
                verdict = "PASSED" if result.hidden_passed else "FAILED"
                print(f"\n=== Hidden tests: {verdict} ===")
                print(hidden_result.output[-2000:])
        finally:
            sandbox.stop()
    return result
