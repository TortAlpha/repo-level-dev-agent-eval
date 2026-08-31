from __future__ import annotations

import os
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from src.git_safety import safe_git_command, safe_git_env, validate_local_git_config


def _repository(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(
        ["git", "-C", str(repo), "config", "user.name", "Safety Test"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(repo), "config", "user.email", "safe@example.invalid"],
        check=True,
    )
    (repo / "tracked.txt").write_text("base\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "add", "tracked.txt"], check=True)
    subprocess.run(
        ["git", "-C", str(repo), "commit", "-qm", "base"], check=True
    )
    return repo


def test_safe_git_environment_discards_inherited_git_routing(tmp_path: Path) -> None:
    with patch.dict(
        os.environ,
        {
            "GIT_DIR": str(tmp_path / "attacker"),
            "GIT_CONFIG_PARAMETERS": "'core.hooksPath=/tmp/hooks'",
            "UNCHANGED_SENTINEL": "yes",
        },
    ):
        environment = safe_git_env()

    assert "GIT_DIR" not in environment
    assert "GIT_CONFIG_PARAMETERS" not in environment
    assert environment["GIT_CONFIG_GLOBAL"] == os.devnull
    assert environment["UNCHANGED_SENTINEL"] == "yes"


def test_dangerous_local_git_configuration_is_rejected(tmp_path: Path) -> None:
    repo = _repository(tmp_path)
    subprocess.run(
        ["git", "-C", str(repo), "config", "core.hooksPath", ".hooks"],
        check=True,
    )

    with pytest.raises(RuntimeError, match="core.hookspath"):
        validate_local_git_config(repo)


def test_safe_git_command_disables_repository_hooks(tmp_path: Path) -> None:
    repo = _repository(tmp_path)
    marker = tmp_path / "hook-ran"
    hook = repo / ".git" / "hooks" / "pre-commit"
    hook.write_text(
        f"#!/bin/sh\nprintf ran > {marker}\nexit 1\n",
        encoding="utf-8",
    )
    hook.chmod(0o755)
    (repo / "tracked.txt").write_text("changed\n", encoding="utf-8")
    subprocess.run(
        safe_git_command(repo, "add", "tracked.txt"),
        check=True,
        env=safe_git_env(),
    )

    committed = subprocess.run(
        safe_git_command(repo, "commit", "-m", "safe"),
        check=False,
        capture_output=True,
        text=True,
        env=safe_git_env(),
    )

    assert committed.returncode == 0, committed.stderr
    assert not marker.exists()
