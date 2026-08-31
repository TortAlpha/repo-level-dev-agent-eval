from __future__ import annotations

import os
import shlex
import shutil
import subprocess
from pathlib import Path

import pytest

from src.agents.sandbox import CommandResult
from src.benchmark.evaluation import (
    EvalResult,
    _configured_test_control_paths,
    _exact_visible_oracle_mounts,
    _record_runtime_pytest_config_tampering,
    _run_hidden_junit,
    _run_visible_junit,
    _sandbox,
    copy_hidden_oracle,
    prepare_hidden_oracle_directories,
    prepare_scoring_dependencies,
    restore_test_oracle,
    scoring_oracle_manifest,
)


def _commit(repo: Path) -> None:
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "add", "-A"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "-c",
            "user.name=Harness Test",
            "-c",
            "user.email=harness@example.invalid",
            "commit",
            "-qm",
            "base",
        ],
        check=True,
    )


def _baseline(tmp_path: Path) -> tuple[Path, Path]:
    pristine = tmp_path / "pristine"
    (pristine / "src").mkdir(parents=True)
    (pristine / "src" / "core.py").write_text("VALUE = 1\n", encoding="utf-8")
    (pristine / "tests").mkdir()
    (pristine / "tests" / "test_original.py").write_text(
        "def test_original():\n    assert True\n", encoding="utf-8"
    )
    repo = tmp_path / "repo"
    shutil.copytree(pristine, repo)
    _commit(repo)
    return pristine, repo


def test_agent_added_regression_test_is_excluded_but_production_is_kept(
    tmp_path: Path,
) -> None:
    pristine, repo = _baseline(tmp_path)
    added_test = repo / "tests" / "test_added.py"
    added_test.write_text("def test_added():\n    assert True\n", encoding="utf-8")
    feature = repo / "src" / "feature.py"
    feature.write_text("FEATURE = True\n", encoding="utf-8")
    excluded: list[str] = []

    tampered = restore_test_oracle(
        repo,
        pristine_repo=pristine,
        test_commands=["python -m pytest tests"],
        excluded_agent_test_files=excluded,
    )

    assert tampered == []
    assert excluded == ["tests/test_added.py"]
    assert not added_test.exists()
    assert feature.read_text(encoding="utf-8") == "FEATURE = True\n"


def test_agent_added_conftest_is_excluded_and_is_a_policy_violation(
    tmp_path: Path,
) -> None:
    pristine, repo = _baseline(tmp_path)
    conftest = repo / "tests" / "conftest.py"
    conftest.write_text(
        "def pytest_collection_modifyitems(items):\n"
        "    for item in items:\n"
        "        item.add_marker('skip')\n",
        encoding="utf-8",
    )
    excluded: list[str] = []

    tampered = restore_test_oracle(
        repo,
        pristine_repo=pristine,
        test_commands=["python -m pytest tests"],
        excluded_agent_test_files=excluded,
    )

    assert tampered == ["tests/conftest.py"]
    assert excluded == ["tests/conftest.py"]
    assert not conftest.exists()


@pytest.mark.parametrize(
    "name",
    ["pyproject.toml", "setup.cfg", ".pytest.ini", "pytest.toml"],
)
def test_agent_added_pytest_configuration_is_removed(
    tmp_path: Path,
    name: str,
) -> None:
    pristine, repo = _baseline(tmp_path)
    config = repo / name
    config.write_text("[tool.pytest.ini_options]\n", encoding="utf-8")

    tampered = restore_test_oracle(
        repo,
        pristine_repo=pristine,
        test_commands=["python -m pytest tests"],
    )

    assert tampered == [name]
    assert not config.exists()


def test_non_pytest_project_configuration_change_is_preserved(
    tmp_path: Path,
) -> None:
    pristine, repo = _baseline(tmp_path)
    baseline = "[tool.ruff.lint]\nselect = ['E']\n"
    changed = "[tool.ruff.lint]\nselect = ['E', 'F']\n"
    (pristine / "pyproject.toml").write_text(baseline, encoding="utf-8")
    (repo / "pyproject.toml").write_text(baseline, encoding="utf-8")
    subprocess.run(
        ["git", "-C", str(repo), "add", "pyproject.toml"],
        check=True,
    )
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "-c",
            "user.name=Harness Test",
            "-c",
            "user.email=harness@example.invalid",
            "commit",
            "-qm",
            "add baseline config",
        ],
        check=True,
    )
    # Simulate a submitted non-pytest edit from that same baseline.
    (repo / "pyproject.toml").write_text(changed, encoding="utf-8")

    tampered = restore_test_oracle(
        repo,
        pristine_repo=pristine,
        test_commands=["python -m pytest tests"],
    )

    assert tampered == []
    assert (repo / "pyproject.toml").read_text(encoding="utf-8") == changed


def test_pytest_marker_declaration_change_is_preserved(tmp_path: Path) -> None:
    pristine, repo = _baseline(tmp_path)
    baseline = "[pytest]\nmarkers =\n    baseline: existing marker\naddopts = -q\n"
    changed = baseline.replace(
        "addopts = -q",
        "    feature: submitted marker\naddopts = -q",
    )
    for root in (pristine, repo):
        (root / "pytest.ini").write_text(baseline, encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "add", "pytest.ini"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "-c",
            "user.name=Harness Test",
            "-c",
            "user.email=harness@example.invalid",
            "commit",
            "-qm",
            "add pytest config",
        ],
        check=True,
    )
    (repo / "pytest.ini").write_text(changed, encoding="utf-8")

    tampered = restore_test_oracle(
        repo,
        pristine_repo=pristine,
        test_commands=["python -m pytest tests"],
    )

    assert tampered == []
    assert (repo / "pytest.ini").read_text(encoding="utf-8") == changed


def test_pytest_addopts_change_is_restored_and_rejected(tmp_path: Path) -> None:
    pristine, repo = _baseline(tmp_path)
    baseline = "[pytest]\naddopts = -q\n"
    for root in (pristine, repo):
        (root / "pytest.ini").write_text(baseline, encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "add", "pytest.ini"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "-c",
            "user.name=Harness Test",
            "-c",
            "user.email=harness@example.invalid",
            "commit",
            "-qm",
            "add pytest config",
        ],
        check=True,
    )
    (repo / "pytest.ini").write_text(
        "[pytest]\naddopts = -q -p agent_plugin\n",
        encoding="utf-8",
    )

    tampered = restore_test_oracle(
        repo,
        pristine_repo=pristine,
        test_commands=["python -m pytest tests"],
    )

    assert tampered == ["pytest.ini"]
    assert (repo / "pytest.ini").read_text(encoding="utf-8") == baseline


def test_runtime_pytest_config_change_marks_result_and_restores(
    tmp_path: Path,
) -> None:
    pristine, repo = _baseline(tmp_path)
    result = EvalResult()
    (repo / "pytest.ini").write_text(
        "[pytest]\naddopts = -p agent_plugin\n",
        encoding="utf-8",
    )

    _record_runtime_pytest_config_tampering(
        result,
        repo,
        pristine,
        phase="Visible-suite",
    )

    assert result.test_oracle_tampered
    assert not (repo / "pytest.ini").exists()


@pytest.mark.parametrize("mutation", ["modified", "deleted"])
def test_pristine_ignored_runner_is_restored(
    tmp_path: Path,
    mutation: str,
) -> None:
    pristine = tmp_path / "pristine"
    pristine.mkdir()
    (pristine / ".gitignore").write_text("run_tests.sh\n", encoding="utf-8")
    original = "#!/bin/sh\npython -m pytest tests\n"
    (pristine / "run_tests.sh").write_text(original, encoding="utf-8")
    repo = tmp_path / "repo"
    shutil.copytree(pristine, repo)
    _commit(repo)
    runner = repo / "run_tests.sh"
    if mutation == "modified":
        runner.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    else:
        runner.unlink()

    tampered = restore_test_oracle(
        repo,
        pristine_repo=pristine,
        test_commands=["./run_tests.sh"],
    )

    assert tampered == ["run_tests.sh"]
    assert runner.read_text(encoding="utf-8") == original


def test_symlink_test_parent_cannot_redirect_restore_outside_repo(
    tmp_path: Path,
) -> None:
    pristine, repo = _baseline(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    sentinel = outside / "test_original.py"
    sentinel.write_text("outside sentinel\n", encoding="utf-8")
    shutil.rmtree(repo / "tests")
    (repo / "tests").symlink_to(outside, target_is_directory=True)

    tampered = restore_test_oracle(repo, pristine_repo=pristine)

    assert "tests" in tampered
    assert "tests/test_original.py" in tampered
    assert not (repo / "tests").is_symlink()
    assert "assert True" in (repo / "tests" / "test_original.py").read_text(
        encoding="utf-8"
    )
    assert sentinel.read_text(encoding="utf-8") == "outside sentinel\n"


def test_git_inventory_failure_is_fail_closed(tmp_path: Path) -> None:
    repo = tmp_path / "not-a-git-repo"
    repo.mkdir()

    with pytest.raises(RuntimeError, match="failed to inventory scoring checkout"):
        restore_test_oracle(repo)


def test_scoring_manifest_protects_baseline_files_and_test_root(
    tmp_path: Path,
) -> None:
    pristine, repo = _baseline(tmp_path)

    files, directories = scoring_oracle_manifest(
        repo,
        pristine_repo=pristine,
        test_commands=["python -m pytest tests"],
    )

    assert files == ["tests/test_original.py"]
    assert directories == ["tests"]


def test_configured_controls_do_not_treat_selectors_as_launcher_files(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    (repo / "tests/test_syntax").mkdir(parents=True)
    (repo / "run_tests.sh").write_text("pytest \"$@\"\n", encoding="utf-8")

    controls = _configured_test_control_paths(
        repo,
        [
            "python -m pytest tests/test_syntax -k test_groupconcat",
            "bash ./run_tests.sh tests/x.py::test_y",
        ],
    )

    assert controls == {"run_tests.sh"}


def test_hidden_manifest_keeps_legitimate_target_support_files(
    tmp_path: Path,
) -> None:
    pristine, repo = _baseline(tmp_path)
    hidden = tmp_path / "hidden"
    supporting = hidden / "test/lib/ansible_test/_util/target/common/constants.py"
    supporting.parent.mkdir(parents=True)
    supporting.write_text("HIDDEN_CONSTANT = 1\n", encoding="utf-8")

    files, directories, tampered = prepare_hidden_oracle_directories(
        repo,
        hidden,
        pristine_repo=pristine,
    )
    copy_hidden_oracle(hidden, repo, files)

    assert files == ["test/lib/ansible_test/_util/target/common/constants.py"]
    assert directories == ["test"]
    assert tampered == []
    assert (repo / files[0]).read_text(encoding="utf-8") == "HIDDEN_CONSTANT = 1\n"


def test_hidden_overlap_preserves_visible_bytes_until_overlay(tmp_path: Path) -> None:
    pristine, repo = _baseline(tmp_path)
    visible = (repo / "tests/test_original.py").read_bytes()
    hidden = tmp_path / "hidden"
    hidden_test = hidden / "tests/test_original.py"
    hidden_test.parent.mkdir(parents=True)
    hidden_test.write_text(
        "def test_original():\n    assert 'hidden behavior'\n",
        encoding="utf-8",
    )

    files, directories, tampered = prepare_hidden_oracle_directories(
        repo,
        hidden,
        pristine_repo=pristine,
    )

    assert (repo / "tests/test_original.py").read_bytes() == visible
    assert files == ["tests/test_original.py"]
    assert directories == ["tests"]
    assert tampered == []
    assert _exact_visible_oracle_mounts(
        ["tests/test_original.py", "tests/test_other.py"], files
    ) == ["tests/test_other.py"]

    copy_hidden_oracle(hidden, repo, files)

    assert (repo / "tests/test_original.py").read_bytes() == hidden_test.read_bytes()


def test_hidden_overlay_fails_if_source_manifest_changes(tmp_path: Path) -> None:
    pristine, repo = _baseline(tmp_path)
    hidden = tmp_path / "hidden"
    fixture = hidden / "tests/hidden/test_feature.py"
    fixture.parent.mkdir(parents=True)
    fixture.write_text("def test_feature(): pass\n", encoding="utf-8")
    files, _directories, _tampered = prepare_hidden_oracle_directories(
        repo,
        hidden,
        pristine_repo=pristine,
    )
    (hidden / "tests/hidden/support.py").write_text("VALUE = 1\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="changed after inventory"):
        copy_hidden_oracle(hidden, repo, files)

    assert not (repo / "tests/hidden/test_feature.py").exists()
    assert not (repo / "tests/hidden/support.py").exists()


def test_hidden_overlay_rejects_symlinked_destination_parent(
    tmp_path: Path,
) -> None:
    pristine, repo = _baseline(tmp_path)
    hidden = tmp_path / "hidden"
    fixture = hidden / "tests/hidden/test_feature.py"
    fixture.parent.mkdir(parents=True)
    fixture.write_text("def test_feature(): pass\n", encoding="utf-8")
    files, _directories, _tampered = prepare_hidden_oracle_directories(
        repo,
        hidden,
        pristine_repo=pristine,
    )
    outside = tmp_path / "outside"
    outside.mkdir()
    shutil.rmtree(repo / "tests/hidden")
    (repo / "tests/hidden").symlink_to(outside, target_is_directory=True)

    with pytest.raises(RuntimeError, match="destination path traverses a symlink"):
        copy_hidden_oracle(hidden, repo, files)

    assert list(outside.iterdir()) == []


class _JUnitSandbox:
    def __init__(self, repo: Path, report: str | None) -> None:
        self.repo = repo
        self.report = report
        self.command = ""
        self.report_path = ""
        self.marker_path = ""

    def resolve_trusted_python(self) -> str:
        return "/usr/local/bin/python"

    def run_shell(self, command: str, timeout: int) -> CommandResult:
        del timeout
        self.command = command
        self.report_path = next(
            token.partition("=")[2]
            for token in shlex.split(command)
            if token.startswith("REPO_EVAL_JUNIT_PATH=")
        )
        self.marker_path = next(
            token.partition("=")[2]
            for token in shlex.split(command)
            if token.startswith("REPO_EVAL_PYTEST_MARKER=")
        )
        return CommandResult(returncode=0, output="configured command returned zero")

    def pop_isolated_file(self, path: str) -> bytes | None:
        if path == self.marker_path:
            return b"single\n"
        assert path == self.report_path
        return self.report.encode("utf-8") if self.report is not None else None


@pytest.mark.parametrize(
    ("report", "reason"),
    [
        (None, "missing or is not a regular file"),
        (
            '<testsuites><testsuite tests="0" failures="0" errors="0" '
            'skipped="0" /></testsuites>',
            "zero collected tests",
        ),
        (
            '<testsuites><testsuite tests="2" failures="0" errors="0" '
            'skipped="0"><testcase classname="hidden" name="test_one" />'
            "</testsuite></testsuites>",
            "declared 2 tests but contains 1 cases",
        ),
    ],
)
def test_hidden_junit_fails_closed_without_complete_evidence(
    tmp_path: Path,
    report: str | None,
    reason: str,
) -> None:
    sandbox = _JUnitSandbox(tmp_path, report)

    passed, output = _run_hidden_junit(
        sandbox,  # type: ignore[arg-type]
        tmp_path,
        "python -m pytest tests/hidden/test_feature.py",
        30,
    )

    assert not passed
    assert reason in output
    assert "REPO_EVAL_JUNIT_PATH=/tmp/repo-eval-hidden-" in sandbox.command
    assert not (tmp_path / ".pytest-hidden-report.xml").exists()


def test_hidden_junit_accepts_nonempty_complete_evidence(tmp_path: Path) -> None:
    report = (
        '<testsuites><testsuite tests="1" failures="0" errors="0" skipped="0">'
        '<testcase classname="hidden" name="test_feature" />'
        "</testsuite></testsuites>"
    )
    sandbox = _JUnitSandbox(tmp_path, report)

    passed, output = _run_hidden_junit(
        sandbox,  # type: ignore[arg-type]
        tmp_path,
        "python -m pytest tests/hidden/test_feature.py",
        30,
    )

    assert passed
    assert "Evaluator rejected" not in output


def test_wrapper_cannot_hide_failures_from_junit_evidence(tmp_path: Path) -> None:
    report = (
        '<testsuites><testsuite tests="1" failures="1" errors="0" skipped="0">'
        '<testcase classname="hidden" name="test_feature">'
        '<failure message="boom" /></testcase></testsuite></testsuites>'
    )
    sandbox = _JUnitSandbox(tmp_path, report)

    passed, output = _run_hidden_junit(
        sandbox,  # type: ignore[arg-type]
        tmp_path,
        "bash ./run_tests.sh tests/hidden/test_feature.py",
        30,
    )

    assert not passed
    assert "failures=1, errors=0" in output


def test_required_hidden_junit_rejects_partial_skip(tmp_path: Path) -> None:
    report = (
        '<testsuites><testsuite tests="2" failures="0" errors="0" skipped="1">'
        '<testcase classname="hidden" name="test_pass" />'
        '<testcase classname="hidden" name="test_skipped">'
        '<skipped message="submission skipped behavior" />'
        "</testcase></testsuite></testsuites>"
    )
    sandbox = _JUnitSandbox(tmp_path, report)

    passed, output = _run_hidden_junit(
        sandbox,  # type: ignore[arg-type]
        tmp_path,
        "python -m pytest hidden.py::test_pass hidden.py::test_skipped",
        30,
    )

    assert not passed
    assert "skipped outcomes (skipped=1)" in output


def test_hidden_junit_requires_every_explicit_selector_to_have_evidence(
    tmp_path: Path,
) -> None:
    report = (
        '<testsuites><testsuite tests="1" failures="0" errors="0" skipped="0">'
        '<testcase classname="hidden" name="test_one" />'
        "</testsuite></testsuites>"
    )
    sandbox = _JUnitSandbox(tmp_path, report)

    passed, output = _run_hidden_junit(
        sandbox,  # type: ignore[arg-type]
        tmp_path,
        "python -m pytest hidden.py::test_one hidden.py::test_two",
        30,
    )

    assert not passed
    assert "fewer executed cases than the configured selectors (1 < 2)" in output


@pytest.mark.skipif(
    os.environ.get("REPO_EVAL_RUN_DOCKER_TESTS") != "1",
    reason="set REPO_EVAL_RUN_DOCKER_TESTS=1 for Docker evaluator smoke tests",
)
def test_real_scoring_ignores_pytest_module_on_project_pythonpath(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    fake_pytest = repo / "lib/pytest"
    fake_pytest.mkdir(parents=True)
    (fake_pytest / "__init__.py").write_text("", encoding="utf-8")
    (fake_pytest / "__main__.py").write_text(
        "from pathlib import Path\n"
        "Path('/workspace/FORGED_PYTEST').write_text('ran')\n",
        encoding="utf-8",
    )
    package = repo / "src/example_package"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("VALUE = 4\n", encoding="utf-8")
    tests = repo / "tests"
    tests.mkdir()
    (tests / "test_real.py").write_text(
        "from example_package import VALUE\n\n"
        "def test_real():\n    assert 2 + 2 == VALUE\n",
        encoding="utf-8",
    )
    dependencies = tmp_path / "dependencies"
    prepare_scoring_dependencies(
        repo,
        docker_image="python:3.11-slim",
        network_disabled=True,
        setup_commands=["python -m pip install -q pytest"],
        shell_timeout=180,
        test_timeout=180,
        dependency_environment=dependencies,
    )
    sandbox = _sandbox(
        repo,
        "python:3.11-slim",
        True,
        180,
        dependency_environment=dependencies,
        dependency_environment_readonly=True,
    )
    sandbox.configure_protected_test_paths([], readonly_directories=["tests"])
    try:
        sandbox.start()
        passed, passing, output = _run_visible_junit(
            sandbox,
            repo,
            "PYTHONPATH=/workspace/lib python -m pytest tests",
            180,
        )
        (tests / "test_real.py").write_text(
            "def test_hidden_overlay():\n    assert 'hidden'\n",
            encoding="utf-8",
        )
        hidden_passed, hidden_output = _run_hidden_junit(
            sandbox,
            repo,
            "PYTHONPATH=/workspace/lib python -m pytest tests/test_real.py",
            180,
        )
    finally:
        sandbox.stop()

    assert passed, output
    assert passing
    assert hidden_passed, hidden_output
    assert not (repo / "FORGED_PYTEST").exists()


@pytest.mark.skipif(
    os.environ.get("REPO_EVAL_RUN_DOCKER_TESTS") != "1",
    reason="set REPO_EVAL_RUN_DOCKER_TESTS=1 for Docker evaluator smoke tests",
)
def test_real_shell_wrapper_cannot_import_agent_pytest_or_config_plugin(
    tmp_path: Path,
) -> None:
    pristine = tmp_path / "pristine"
    tests = pristine / "tests"
    tests.mkdir(parents=True)
    (tests / "test_real.py").write_text(
        "def test_real():\n    assert 2 + 2 == 4\n",
        encoding="utf-8",
    )
    runner = pristine / "run_tests.sh"
    runner.write_text(
        "#!/bin/sh\nexport PYTHONPATH=/workspace\npython -m pytest \"$@\"\n",
        encoding="utf-8",
    )
    loop_runner = pristine / "loop_tests.sh"
    loop_runner.write_text(
        "#!/bin/sh\n"
        "for target in \"$@\"; do\n"
        "  python -m pytest \"$target\" || true\n"
        "done\n",
        encoding="utf-8",
    )
    cases = pristine / "cases"
    cases.mkdir()
    (cases / "failing.py").write_text(
        "def test_failure():\n    assert False\n",
        encoding="utf-8",
    )
    repo = tmp_path / "repo"
    shutil.copytree(pristine, repo)
    _commit(repo)

    # Both common attacks are present in the submitted checkout. The config is
    # removed/restored as oracle tampering; the production-looking module can
    # remain, but the wrapper must never import it as pytest itself.
    (repo / "pytest.py").write_text(
        "from pathlib import Path\n"
        "Path('/workspace/FORGED_PYTEST').write_text('ran')\n",
        encoding="utf-8",
    )
    (repo / "pyproject.toml").write_text(
        "[tool.pytest.ini_options]\naddopts = ['-p', 'agent_plugin']\n",
        encoding="utf-8",
    )
    (repo / "agent_plugin.py").write_text(
        "from pathlib import Path\n"
        "Path('/workspace/FORGED_PLUGIN').write_text('ran')\n",
        encoding="utf-8",
    )
    tampered = restore_test_oracle(
        repo,
        pristine_repo=pristine,
        test_commands=["bash ./run_tests.sh tests"],
    )
    assert tampered == ["pyproject.toml"]

    dependencies = tmp_path / "dependencies"
    prepare_scoring_dependencies(
        pristine,
        docker_image="python:3.11-slim",
        network_disabled=True,
        setup_commands=["python -m pip install -q pytest"],
        shell_timeout=180,
        test_timeout=180,
        dependency_environment=dependencies,
    )
    sandbox = _sandbox(
        repo,
        "python:3.11-slim",
        True,
        180,
        dependency_environment=dependencies,
        dependency_environment_readonly=True,
    )
    sandbox.configure_protected_test_paths(
        [
            "cases/failing.py",
            "loop_tests.sh",
            "run_tests.sh",
            "tests/test_real.py",
        ],
        readonly_directories=["tests"],
    )
    try:
        sandbox.start()
        passed, passing, output = _run_visible_junit(
            sandbox,
            repo,
            "bash ./run_tests.sh tests",
            180,
        )
        loop_passed, loop_output = _run_hidden_junit(
            sandbox,
            repo,
            "bash ./loop_tests.sh cases/failing.py tests/test_real.py",
            180,
        )
    finally:
        sandbox.stop()

    assert passed, output
    assert passing
    assert not loop_passed
    assert "invoked pytest more than once" in loop_output
    assert not (repo / "FORGED_PYTEST").exists()
    assert not (repo / "FORGED_PLUGIN").exists()
