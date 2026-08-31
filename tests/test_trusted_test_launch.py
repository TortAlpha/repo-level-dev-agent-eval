from __future__ import annotations

import shlex

import pytest

from src.agents.sandbox import EVALUATOR_HELPER_CONTAINER_PATH
from src.benchmark.test_launch import (
    pytest_selection_lower_bound,
    trusted_evaluator_test_command,
    trusted_pytest_command,
)


def test_direct_pytest_is_bootstrapped_before_project_pythonpath() -> None:
    command = trusted_pytest_command(
        "PYTHONPATH=/workspace/lib:/workspace/test/lib "
        "python -m pytest 'tests/test_core.py::test value' -q"
    )
    tokens = shlex.split(command)

    assert tokens[:5] == [
        "/usr/bin/env",
        "-u",
        "PYTHONPATH",
        "-u",
        "PYTHONHOME",
    ]
    assert tokens[5:9] == ["python", "-I", "-S", EVALUATOR_HELPER_CONTAINER_PATH]
    assert tokens[tokens.index("--repo-eval-pythonpath") + 1] == (
        "/workspace/lib:/workspace/test/lib"
    )
    assert tokens[-2:] == ["tests/test_core.py::test value", "-q"]


def test_bare_pytest_and_benign_environment_are_preserved() -> None:
    tokens = shlex.split(trusted_pytest_command("LANG=C pytest tests -q"))

    assert "LANG=C" in tokens
    assert ["python", "-I", "-S"] == tokens[6:9]
    assert tokens[-2:] == ["tests", "-q"]


@pytest.mark.parametrize(
    "assignment",
    [
        "PATH=/workspace/bin",
        "PYTHONHOME=/workspace/home",
        "PYTHONUSERBASE=/workspace/user",
        "PYTHONSTARTUP=/workspace/startup.py",
    ],
)
def test_interpreter_control_overrides_fail_closed(assignment: str) -> None:
    with pytest.raises(RuntimeError, match="interpreter control"):
        trusted_pytest_command(f"{assignment} python -m pytest tests")


def test_non_pytest_command_is_not_rewritten() -> None:
    command = "npm test -- --runInBand"
    assert trusted_pytest_command(command) == command


def test_shell_wrapper_gets_read_only_pytest_path_interception() -> None:
    tokens = shlex.split(
        trusted_evaluator_test_command(
            "bash ./run_tests.sh tests/test_core.py",
            trusted_python="/usr/local/bin/python",
            junit_path="/tmp/repo-eval-visible-test.xml",
            invocation_marker="/tmp/repo-eval-visible-test.invocation",
        )
    )

    expected_path = (
        "PATH=/tmp/repo-eval-bin:/usr/local/bin:/usr/bin:/bin:"
        "/home/agent/.local/bin"
    )
    assert expected_path in tokens
    assert "REPO_EVAL_REAL_PYTHON=/usr/local/bin/python" in tokens
    assert tokens[-3:] == ["bash", "./run_tests.sh", "tests/test_core.py"]


def test_unsupported_junit_launcher_fails_closed() -> None:
    with pytest.raises(RuntimeError, match="not a supported"):
        trusted_evaluator_test_command(
            "npm test",
            trusted_python="/usr/local/bin/python",
            junit_path="/tmp/repo-eval-visible-test.xml",
            invocation_marker="/tmp/repo-eval-visible-test.invocation",
        )


def test_explicit_pytest_selectors_provide_a_hidden_evidence_lower_bound() -> None:
    assert (
        pytest_selection_lower_bound(
            "PYTHONPATH=/workspace/lib python -m pytest "
            "tests/test_one.py::test_a tests/test_two.py::test_b -q"
        )
        == 2
    )
    assert pytest_selection_lower_bound("pytest -k expression tests") == 1
    assert pytest_selection_lower_bound("bash ./run_tests.sh") == 1
    assert (
        pytest_selection_lower_bound(
            "bash ./run_tests.sh one.py::test_a two.py::test_b"
        )
        == 2
    )
    assert (
        pytest_selection_lower_bound(
            "xvfb-run -a env QT_QPA_PLATFORM=offscreen "
            "python -m pytest -p no:xvfb tests/test_ui.py"
        )
        == 1
    )
