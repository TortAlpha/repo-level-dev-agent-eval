from __future__ import annotations

import base64
import hashlib
import json
import shlex
import sys
from pathlib import Path
from types import ModuleType

import pytest

from src.agents.sandbox import EVALUATOR_HELPER_CONTAINER_PATH
from src.benchmark import _trusted_pytest as trusted_pytest
from src.benchmark._trusted_pytest import (
    _editable_workspace_roots,
    _reject_explicit_plugins,
)
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


def _write_editable_metadata(
    site_packages: Path,
    workspace: Path,
    *,
    distribution: str = "pluggy",
    metadata_name: str = "pluggy",
    top_level: str = "pluggy",
    editable: bool = True,
    pth_text: str | None = None,
) -> Path:
    source_root = workspace / "src"
    source_root.mkdir(parents=True, exist_ok=True)
    metadata = site_packages / f"{distribution}-1.0.dist-info"
    metadata.mkdir(parents=True)
    (metadata / "direct_url.json").write_text(
        json.dumps(
            {
                "dir_info": {"editable": editable},
                "url": workspace.resolve().as_uri(),
            }
        ),
        encoding="utf-8",
    )
    (metadata / "top_level.txt").write_text(f"{top_level}\n", encoding="utf-8")
    (metadata / "METADATA").write_text(
        f"Metadata-Version: 2.1\nName: {metadata_name}\nVersion: 1.0\n",
        encoding="utf-8",
    )
    pth_name = f"__editable__.{distribution}-1.0.pth"
    pth_data = (pth_text or f"{source_root.resolve()}\n").encode()
    (site_packages / pth_name).write_bytes(pth_data)
    pth_hash = base64.urlsafe_b64encode(hashlib.sha256(pth_data).digest()).rstrip(
        b"="
    ).decode()
    (metadata / "RECORD").write_text(
        f"{pth_name},sha256={pth_hash},{len(pth_data)}\n",
        encoding="utf-8",
    )
    return metadata


def test_editable_pluggy_bootstrap_requires_frozen_pep610_metadata(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    site_packages = tmp_path / "site-packages"
    _write_editable_metadata(site_packages, workspace)

    assert _editable_workspace_roots(
        [site_packages], workspace=workspace
    ) == {"pluggy": (workspace / "src").resolve()}


def test_physical_pluggy_distribution_needs_no_bootstrap(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    site_packages = tmp_path / "site-packages"
    metadata = site_packages / "pluggy-1.6.0.dist-info"
    workspace.mkdir()
    metadata.mkdir(parents=True)
    (metadata / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: pluggy\nVersion: 1.6.0\n",
        encoding="utf-8",
    )

    assert not _editable_workspace_roots([site_packages], workspace=workspace)


@pytest.mark.parametrize(
    ("top_level", "editable"),
    [("arbitrary_project", True), ("pluggy", False)],
)
def test_editable_bootstrap_rejects_unapproved_or_noneditable_roots(
    tmp_path: Path,
    top_level: str,
    editable: bool,
) -> None:
    workspace = tmp_path / "workspace"
    site_packages = tmp_path / "site-packages"
    _write_editable_metadata(
        site_packages,
        workspace,
        top_level=top_level,
        editable=editable,
    )

    with pytest.raises(RuntimeError, match="declaration is inconsistent"):
        _editable_workspace_roots([site_packages], workspace=workspace)


def test_editable_bootstrap_rejects_mismatched_or_aliased_metadata(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    site_packages = tmp_path / "site-packages"
    metadata_source = tmp_path / "pluggy-1.0.dist-info"
    workspace.mkdir(parents=True)
    site_packages.mkdir()
    metadata_source.mkdir()
    (metadata_source / "direct_url.json").write_text(
        json.dumps(
            {
                "dir_info": {"editable": True},
                "url": workspace.resolve().as_uri(),
            }
        ),
        encoding="utf-8",
    )
    (metadata_source / "top_level.txt").write_text("pluggy\n", encoding="utf-8")
    (site_packages / "pluggy-1.0.dist-info").symlink_to(
        metadata_source, target_is_directory=True
    )

    with pytest.raises(RuntimeError, match="metadata is an alias"):
        _editable_workspace_roots([site_packages], workspace=workspace)


def test_self_hosted_plugin_rejects_only_actual_plugin_flags() -> None:
    _reject_explicit_plugins(["--pyargs", "--pdb", "--pastebin=failed"])

    for argument in ("-p", "-p=example", "-pexample"):
        with pytest.raises(RuntimeError, match="explicit pytest plugins"):
            _reject_explicit_plugins([argument])


def test_candidate_bootstrap_loads_only_the_frozen_exact_source_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    source_package = workspace / "src/pluggy"
    source_package.mkdir(parents=True)
    (source_package / "__init__.py").write_text(
        "MARKER = 'exact-source'\n",
        encoding="utf-8",
    )
    shadow_package = workspace / "pluggy"
    shadow_package.mkdir()
    (shadow_package / "__init__.py").write_text(
        "MARKER = 'root-shadow'\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(trusted_pytest, "_WORKSPACE", workspace)
    original = {
        name: module
        for name, module in sys.modules.items()
        if name == "pluggy" or name.startswith("pluggy.")
    }
    for name in original:
        sys.modules.pop(name, None)
    try:
        module, entry = trusted_pytest._load_candidate_root(
            "pluggy", workspace / "src"
        )
        assert module.MARKER == "exact-source"
        assert entry == (source_package / "__init__.py").resolve()
        trusted_pytest._attest_candidate_root("pluggy", module, entry)
    finally:
        for name in tuple(sys.modules):
            if name == "pluggy" or name.startswith("pluggy."):
                sys.modules.pop(name, None)
        sys.modules.update(original)


def test_candidate_bootstrap_does_not_fall_back_to_workspace_shadow(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    (workspace / "src").mkdir(parents=True)
    shadow_package = workspace / "pluggy"
    shadow_package.mkdir()
    (shadow_package / "__init__.py").write_text("SHADOW = True\n", encoding="utf-8")
    monkeypatch.setattr(trusted_pytest, "_WORKSPACE", workspace)

    with pytest.raises(RuntimeError, match="package is not a directory"):
        trusted_pytest._load_candidate_root("pluggy", workspace / "src")


def test_candidate_bootstrap_rejects_symlinked_submodule(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    package = workspace / "src/pluggy"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    outside = tmp_path / "outside.py"
    outside.write_text("VALUE = 'outside'\n", encoding="utf-8")
    (package / "_manager.py").symlink_to(outside)
    monkeypatch.setattr(trusted_pytest, "_WORKSPACE", workspace)

    with pytest.raises(RuntimeError, match="aliased or special file"):
        trusted_pytest._load_candidate_root("pluggy", workspace / "src")


def test_post_run_attestation_rejects_critical_module_replacement() -> None:
    snapshot = trusted_pytest._critical_runtime_snapshot()
    original = sys.modules["pytest"]
    sys.modules["pytest"] = ModuleType("pytest")
    try:
        with pytest.raises(RuntimeError, match="identity changed"):
            trusted_pytest._attest_post_run_runtime([], snapshot)
    finally:
        sys.modules["pytest"] = original


def test_frozen_bootstrap_requires_a_matching_editable_route(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    bootstrap_package = tmp_path / "bootstrap/src/pluggy"
    bootstrap_package.mkdir(parents=True)
    (bootstrap_package / "__init__.py").write_text("", encoding="utf-8")
    monkeypatch.setattr(trusted_pytest, "_WORKSPACE", workspace)
    monkeypatch.setattr(trusted_pytest, "_BOOTSTRAP", tmp_path / "bootstrap")

    with pytest.raises(RuntimeError, match="do not match editable routes"):
        trusted_pytest._validate_frozen_bootstrap_routes({})

    trusted_pytest._validate_frozen_bootstrap_routes(
        {"pluggy": workspace / "src"}
    )


def test_editable_bootstrap_rejects_distribution_name_mismatch(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    site_packages = tmp_path / "site-packages"
    _write_editable_metadata(
        site_packages,
        workspace,
        distribution="unrelated",
        metadata_name="unrelated",
        top_level="pluggy",
    )

    assert not _editable_workspace_roots([site_packages], workspace=workspace)


@pytest.mark.parametrize(
    "pth_text",
    ["import candidate\n", "/workspace/src\n/workspace/other\n"],
)
def test_editable_bootstrap_rejects_executable_or_multiline_pth(
    tmp_path: Path,
    pth_text: str,
) -> None:
    workspace = tmp_path / "workspace"
    site_packages = tmp_path / "site-packages"
    _write_editable_metadata(
        site_packages,
        workspace,
        pth_text=pth_text,
    )

    with pytest.raises(RuntimeError, match="one inert absolute"):
        _editable_workspace_roots([site_packages], workspace=workspace)


def test_editable_bootstrap_rejects_record_drift(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    site_packages = tmp_path / "site-packages"
    metadata = _write_editable_metadata(site_packages, workspace)
    (metadata / "RECORD").write_text(
        "__editable__.pluggy-1.0.pth,sha256=wrong,1\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="RECORD mismatch"):
        _editable_workspace_roots([site_packages], workspace=workspace)


def test_editable_bootstrap_rejects_malformed_additional_pth_record(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    site_packages = tmp_path / "site-packages"
    metadata = _write_editable_metadata(site_packages, workspace)
    with (metadata / "RECORD").open("a", encoding="utf-8") as handle:
        handle.write("unverified.pth,sha256=missing-size\n")

    with pytest.raises(RuntimeError, match="RECORD row is invalid"):
        _editable_workspace_roots([site_packages], workspace=workspace)


@pytest.mark.parametrize(
    "url",
    [
        "file:",
        "file://localhost",
        "file:///workspace?redirect=1",
        "relative/path",
        "dotdot",
    ],
)
def test_editable_bootstrap_rejects_non_exact_workspace_url(
    tmp_path: Path,
    url: str,
) -> None:
    workspace = tmp_path / "workspace"
    site_packages = tmp_path / "site-packages"
    metadata = _write_editable_metadata(site_packages, workspace)
    payload = json.loads((metadata / "direct_url.json").read_text(encoding="utf-8"))
    if url == "dotdot":
        url = f"file://{workspace.as_posix()}/../{workspace.name}"
    payload["url"] = url
    (metadata / "direct_url.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RuntimeError, match="declaration is inconsistent"):
        _editable_workspace_roots([site_packages], workspace=workspace)
