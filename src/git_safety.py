"""Fail-closed host-side Git invocation helpers for untrusted worktrees."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

SAFE_GIT_OPTIONS = (
    "-c",
    "core.hooksPath=/dev/null",
    "-c",
    "core.fsmonitor=false",
    "-c",
    "core.untrackedCache=false",
    "-c",
    "diff.external=",
    "-c",
    "color.ui=false",
)


def safe_git_env() -> dict[str, str]:
    """Drop inherited Git routing/config and disable interactive execution."""
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("GIT_")
    }
    env.update(
        {
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_ATTR_NOSYSTEM": "1",
            "GIT_TERMINAL_PROMPT": "0",
            "GIT_PAGER": "cat",
        }
    )
    return env


def safe_git_command(repo: Path, *arguments: str) -> list[str]:
    return [
        "git",
        *SAFE_GIT_OPTIONS,
        "-C",
        str(repo.resolve()),
        *arguments,
    ]


def validate_local_git_config(repo: Path) -> None:
    """Reject local config keys that can execute or redirect host Git."""
    result = subprocess.run(
        safe_git_command(
            repo,
            "config",
            "--local",
            "--no-includes",
            "--name-only",
            "--list",
            "-z",
        ),
        capture_output=True,
        check=False,
        env=safe_git_env(),
        timeout=30,
    )
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace")[-1000:]
        raise RuntimeError(f"cannot validate local Git config:\n{detail}")
    keys = {
        item.decode("utf-8", errors="replace").lower()
        for item in result.stdout.split(b"\0")
        if item
    }
    exact = {
        "core.attributesfile",
        "core.excludesfile",
        "core.fsmonitor",
        "core.gitproxy",
        "core.hookspath",
        "core.sshcommand",
        "core.worktree",
        "diff.external",
    }
    dangerous = sorted(
        key
        for key in keys
        if key in exact
        or key.startswith(("filter.", "include.", "includeif."))
        or (key.startswith("diff.") and key.endswith((".command", ".textconv")))
        or (key.startswith("merge.") and key.endswith(".driver"))
    )
    if dangerous:
        raise RuntimeError(
            "unsafe executable/redirecting local Git configuration: "
            + ", ".join(dangerous)
        )


def verify_pristine_git_state(repo: Path, expected_head: str) -> None:
    """Verify base ref/index before any host command stages model output."""
    validate_local_git_config(repo)
    head = subprocess.run(
        safe_git_command(repo, "rev-parse", "--verify", "HEAD^{commit}"),
        capture_output=True,
        text=True,
        check=False,
        env=safe_git_env(),
        timeout=30,
    )
    if head.returncode != 0 or head.stdout.strip() != expected_head:
        raise RuntimeError(
            "Git HEAD changed before patch capture: "
            f"{head.stdout.strip()!r} != {expected_head!r}"
        )
    index = subprocess.run(
        safe_git_command(repo, "diff", "--cached", "--quiet", expected_head, "--"),
        capture_output=True,
        check=False,
        env=safe_git_env(),
        timeout=60,
    )
    if index.returncode != 0:
        raise RuntimeError("Git index changed before patch capture")
