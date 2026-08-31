"""Evaluator-owned launch normalization for Python test commands."""

from __future__ import annotations

import os
import re
import shlex
from pathlib import PurePosixPath

from ..agents.sandbox import (
    EVALUATOR_BIN_CONTAINER_PATH,
    EVALUATOR_HELPER_CONTAINER_PATH,
)

_ENV_ASSIGNMENT = re.compile(r"^(?P<name>[A-Za-z_][A-Za-z0-9_]*)=(?P<value>.*)$")
_PYTHON_EXECUTABLES = frozenset({"python", "python3"})
_PYTEST_EXECUTABLES = frozenset({"pytest", "py.test"})
_SHELL_EXECUTABLES = frozenset({"bash", "sh"})
_UNTRUSTED_PYTHON_ENV = frozenset(
    {
        "PATH",
        "PYTHONHOME",
        "PYTHONINSPECT",
        "PYTHONSAFEPATH",
        "PYTHONSTARTUP",
        "PYTHONUSERBASE",
        "PYTEST_ADDOPTS",
        "PYTEST_PLUGINS",
        "REPO_EVAL_JUNIT_PATH",
        "REPO_EVAL_PYTEST_MARKER",
        "REPO_EVAL_REAL_PYTHON",
    }
)
_PYTEST_OPTIONS_WITH_VALUE = frozenset(
    {
        "-c",
        "-k",
        "-m",
        "-o",
        "-p",
        "--basetemp",
        "--capture",
        "--confcutdir",
        "--deselect",
        "--ignore",
        "--ignore-glob",
        "--import-mode",
        "--junit-prefix",
        "--junit-xml",
        "--maxfail",
        "--override-ini",
        "--rootdir",
        "--tb",
    }
)


def _executable_name(token: str) -> str:
    return PurePosixPath(token).name


def _direct_pytest_arguments(tokens: list[str]) -> list[str] | None:
    index = 0
    while index < len(tokens) and _ENV_ASSIGNMENT.fullmatch(tokens[index]):
        index += 1
    remaining = tokens[index:]
    if not remaining:
        return None
    executable = _executable_name(remaining[0])
    if executable in _PYTEST_EXECUTABLES:
        return remaining[1:]
    if (
        executable in _PYTHON_EXECUTABLES
        or re.fullmatch(r"python3\.\d+", executable)
    ) and remaining[1:3] == ["-m", "pytest"]:
        return remaining[3:]
    return None


def _embedded_pytest_arguments(tokens: list[str]) -> list[str] | None:
    """Return pytest arguments from a direct call nested under xvfb/env."""
    for index in range(len(tokens)):
        arguments = _direct_pytest_arguments(tokens[index:])
        if arguments is not None:
            return arguments
    return None


def _shell_wrapper_arguments(tokens: list[str]) -> list[str] | None:
    """Return forwarded arguments for one explicit, relative shell runner."""
    index = 0
    while index < len(tokens) and _ENV_ASSIGNMENT.fullmatch(tokens[index]):
        index += 1
    remaining = tokens[index:]
    if not remaining:
        return None
    executable = _executable_name(remaining[0])
    if executable in _SHELL_EXECUTABLES:
        index = 1
        while index < len(remaining) and remaining[index].startswith("-"):
            index += 1
        if index >= len(remaining):
            return None
        script = PurePosixPath(remaining[index])
        if (
            script.is_absolute()
            or ".." in script.parts
            or script.suffix.lower() != ".sh"
        ):
            return None
        return remaining[index + 1 :]
    candidate = PurePosixPath(remaining[0])
    if (
        not candidate.is_absolute()
        and ".." not in candidate.parts
        and candidate.suffix.lower() == ".sh"
    ):
        return remaining[1:]
    return None


def _configured_pytest_arguments(tokens: list[str]) -> list[str] | None:
    direct = _direct_pytest_arguments(tokens)
    if direct is not None:
        return direct
    index = 0
    while index < len(tokens) and _ENV_ASSIGNMENT.fullmatch(tokens[index]):
        index += 1
    remaining = tokens[index:]
    if remaining and _executable_name(remaining[0]) == "xvfb-run":
        return _embedded_pytest_arguments(remaining[1:])
    return _shell_wrapper_arguments(tokens)


def pytest_selection_lower_bound(command: str) -> int | None:
    """Minimum cases implied by explicit selectors in a direct pytest call."""
    try:
        tokens = shlex.split(command)
    except ValueError:
        return None
    arguments = _configured_pytest_arguments(tokens)
    if arguments is None:
        return None
    selectors = 0
    skip_value = False
    options_finished = False
    for token in arguments:
        if skip_value:
            skip_value = False
            continue
        if not options_finished and token == "--":
            options_finished = True
            continue
        if not options_finished and token in _PYTEST_OPTIONS_WITH_VALUE:
            skip_value = True
            continue
        if not options_finished and token.startswith("-"):
            continue
        selectors += 1
    return max(selectors, 1)


def trusted_pytest_command(
    command: str,
    *,
    trusted_python: str | None = None,
) -> str:
    """Route a direct pytest command through the evaluator bootstrap.

    Non-pytest commands are returned unchanged.  The benchmark's explicit
    ``PYTHONPATH`` is passed as data and exposed only after immutable pytest
    has loaded; interpreter-control assignments are rejected fail-closed.
    """
    try:
        tokens = shlex.split(command)
    except ValueError as exc:
        raise RuntimeError(f"invalid configured test command: {exc}") from exc

    environment: list[str] = []
    pythonpath = ""
    index = 0
    while index < len(tokens):
        match = _ENV_ASSIGNMENT.fullmatch(tokens[index])
        if match is None:
            break
        name = match.group("name")
        if name == "PYTHONPATH":
            pythonpath = match.group("value")
        elif name in _UNTRUSTED_PYTHON_ENV:
            raise RuntimeError(
                "configured pytest command cannot override evaluator "
                f"interpreter control {name}"
            )
        else:
            environment.append(tokens[index])
        index += 1

    remaining = tokens[index:]
    if not remaining:
        return command

    executable = _executable_name(remaining[0])
    if executable in _PYTEST_EXECUTABLES:
        interpreter = trusted_python or "python"
        pytest_arguments = remaining[1:]
    elif (
        executable in _PYTHON_EXECUTABLES
        or re.fullmatch(r"python3\.\d+", executable)
    ) and remaining[1:3] == ["-m", "pytest"]:
        interpreter = trusted_python or remaining[0]
        pytest_arguments = remaining[3:]
    else:
        return command

    # ``-I -S`` ignores Python environment controls, excludes cwd at startup,
    # and skips .pth execution. The helper re-adds immutable package roots and
    # then the benchmark's project paths in that order.
    argv = [
        "/usr/bin/env",
        "-u",
        "PYTHONPATH",
        "-u",
        "PYTHONHOME",
        *environment,
        interpreter,
        "-I",
        "-S",
        EVALUATOR_HELPER_CONTAINER_PATH,
        "--repo-eval-pythonpath",
        pythonpath or os.pathsep.join(["/workspace"]),
        "--",
        *pytest_arguments,
    ]
    return shlex.join(argv)


def trusted_evaluator_test_command(
    command: str,
    *,
    trusted_python: str,
    junit_path: str,
    invocation_marker: str,
) -> str:
    """Secure direct and benchmark-wrapper pytest invocations.

    Direct calls are rewritten to the isolated bootstrap. Shell and xvfb
    wrappers remain byte-for-byte benchmark behavior, but receive a read-only
    PATH layer which intercepts every ``pytest`` or ``python -m pytest`` call.
    Unsupported launch shapes fail closed instead of producing forgeable
    evaluator evidence.
    """
    try:
        tokens = shlex.split(command)
    except ValueError as exc:
        raise RuntimeError(f"invalid configured test command: {exc}") from exc
    if not tokens:
        raise RuntimeError("configured test command is empty")
    if any(
        token == "--junit-xml"
        or token.startswith("--junit-xml=")
        or token == "--junitxml"
        or token.startswith("--junitxml=")
        for token in tokens
    ):
        raise RuntimeError(
            "configured test commands cannot choose the evaluator JUnit path"
        )

    for token in tokens:
        match = _ENV_ASSIGNMENT.fullmatch(token)
        if match is not None and match.group("name") in _UNTRUSTED_PYTHON_ENV:
            name = match.group("name")
            if name != "PYTHONPATH":
                raise RuntimeError(
                    "configured test command cannot override evaluator "
                    f"interpreter control {name}"
                )

    arguments = _configured_pytest_arguments(tokens)
    if arguments is None:
        raise RuntimeError(
            "configured test command is not a supported direct pytest, "
            "xvfb-run pytest, or relative shell-runner invocation"
        )
    executable_index = 0
    while executable_index < len(tokens) and _ENV_ASSIGNMENT.fullmatch(
        tokens[executable_index]
    ):
        executable_index += 1
    if _direct_pytest_arguments(tokens) is not None:
        executable = _executable_name(tokens[executable_index])
        tokens[executable_index] = (
            "pytest" if executable in _PYTEST_EXECUTABLES else "python"
        )
    trusted_path = os.pathsep.join(
        [
            EVALUATOR_BIN_CONTAINER_PATH,
            "/usr/local/bin",
            "/usr/bin",
            "/bin",
            "/home/agent/.local/bin",
        ]
    )
    argv = [
        "/usr/bin/env",
        "-u",
        "PYTHONHOME",
        "-u",
        "PYTHONSTARTUP",
        "-u",
        "PYTHONUSERBASE",
        "-u",
        "PYTEST_ADDOPTS",
        "-u",
        "PYTEST_PLUGINS",
        f"PATH={trusted_path}",
        f"REPO_EVAL_REAL_PYTHON={trusted_python}",
        f"REPO_EVAL_JUNIT_PATH={junit_path}",
        f"REPO_EVAL_PYTEST_MARKER={invocation_marker}",
        *tokens,
    ]
    return shlex.join(argv)
