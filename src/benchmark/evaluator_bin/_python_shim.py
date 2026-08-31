"""Intercept ``python ... -m pytest`` inside immutable benchmark wrappers."""

from __future__ import annotations

import os
import sys

_HELPER = "/tmp/repo-eval-trusted-pytest.py"
_UNTRUSTED_ENV = {
    "PYTHONHOME",
    "PYTHONINSPECT",
    "PYTHONPATH",
    "PYTHONSTARTUP",
    "PYTHONUSERBASE",
    "PYTEST_ADDOPTS",
    "PYTEST_PLUGINS",
    "REPO_EVAL_JUNIT_PATH",
    "REPO_EVAL_PYTEST_MARKER",
}


def _real_python() -> str:
    value = os.environ.get("REPO_EVAL_REAL_PYTHON", "")
    allowed_parents = {"/bin", "/usr/bin", "/usr/local/bin"}
    if not value.startswith("/") or os.path.dirname(value) not in allowed_parents:
        raise RuntimeError("invalid evaluator Python interpreter")
    return value


def _pytest_module_index(arguments: list[str]) -> int | None:
    """Find the interpreter-level ``-m pytest`` before any script/module."""
    index = 0
    while index < len(arguments):
        argument = arguments[index]
        if argument == "-m":
            if index + 1 < len(arguments) and arguments[index + 1] == "pytest":
                return index
            return None
        if argument in {"-c", "--"} or not argument.startswith("-"):
            return None
        # CPython options whose value is the following token. They may appear
        # before ``-m``; retaining them preserves wrapper semantics.
        index += 2 if argument in {"-W", "-X"} else 1
    return None


def _isolated_path(environment_name: str) -> str:
    value = os.environ.get(environment_name, "")
    if (
        os.path.dirname(value) != "/tmp"
        or not os.path.basename(value).startswith("repo-eval-")
        or ".." in value.split("/")
    ):
        raise RuntimeError(f"invalid evaluator path in {environment_name}")
    return value


def _claim_single_pytest_invocation(marker: str) -> None:
    try:
        descriptor = os.open(
            marker,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
    except FileExistsError:
        with open(marker, "ab", buffering=0) as handle:
            handle.write(b"duplicate\n")
        raise RuntimeError(
            "benchmark wrapper invoked pytest more than once"
        ) from None
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(b"single\n")


def main() -> None:
    real_python = _real_python()
    arguments = sys.argv[1:]
    module_index = _pytest_module_index(arguments)
    if module_index is None:
        os.execve(real_python, [real_python, *arguments], os.environ.copy())

    project_pythonpath = os.environ.get("PYTHONPATH", "/workspace")
    report_path = _isolated_path("REPO_EVAL_JUNIT_PATH")
    marker_path = _isolated_path("REPO_EVAL_PYTEST_MARKER")
    _claim_single_pytest_invocation(marker_path)
    environment = os.environ.copy()
    for name in _UNTRUSTED_ENV:
        environment.pop(name, None)
    environment["PYTHONSAFEPATH"] = "1"
    pytest_arguments = arguments[module_index + 2 :]
    interpreter_arguments = arguments[:module_index]
    os.execve(
        real_python,
        [
            real_python,
            *interpreter_arguments,
            "-I",
            "-S",
            _HELPER,
            "--repo-eval-pythonpath",
            project_pythonpath,
            "--",
            *pytest_arguments,
            f"--junit-xml={report_path}",
        ],
        environment,
    )


if __name__ == "__main__":
    main()
