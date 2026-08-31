"""Per-run workspace: copy the pristine repo and capture the agent's patch."""

from __future__ import annotations

import shutil
import stat
import subprocess
from collections.abc import Iterable
from pathlib import Path, PurePosixPath

from ..git_safety import (
    safe_git_command,
    safe_git_env,
    validate_local_git_config,
    verify_pristine_git_state,
)
from .collection import TaskSpec

# Build artifacts that setup (`pip install -e .`) or test runs drop into the
# workspace; never part of a real solution, so they are excluded from the
# captured patch.
# NOTE: the ``glob`` magic is required — without it ``**/`` does not match zero
# leading dirs, so root-level artifacts (e.g. ``parse.egg-info/``) slip through.
_ARTIFACT_EXCLUDES = (
    ":(exclude,glob)**/__pycache__/**",
    ":(exclude,glob)**/*.pyc",
    ":(exclude,glob)**/*.egg-info/**",
    ":(exclude,glob)**/*.egg-info",
    ":(exclude,glob)**/.pytest_cache/**",
    ":(exclude,glob)**/build/**",
    ":(exclude,glob)**/.eggs/**",
)


def prepare_workspace(task: TaskSpec, workspaces_dir: Path, run_id: str) -> Path:
    """Copy the pristine repo (and hidden tests) into a fresh per-run workspace.

    Each run gets a unique ``<task_id>-<run_id>`` directory: a brand-new path
    that is never deleted+recreated. That sidesteps a Docker Desktop
    file-sharing bug on macOS where re-creating a previously bind-mounted path
    leaves the daemon unable to see it ("bind source path does not exist") —
    the staleness also affects new children of a re-created parent, so the
    unique segment is kept flat under the (stable) workspaces directory.
    Hidden tests are placed next to the repo for post-run scoring.
    """
    task_ws = workspaces_dir / f"{task.task_id}-{run_id}"
    task_ws.mkdir(parents=True, exist_ok=True)

    ignore = shutil.ignore_patterns("__pycache__", ".pytest_cache")
    repo_dest = task_ws / "repo"
    shutil.copytree(task.repo_path, repo_dest, symlinks=True, ignore=ignore)
    if task.hidden_tests_path.parts and task.hidden_tests_path.exists():
        shutil.copytree(task.hidden_tests_path, task_ws / "hidden_tests", ignore=ignore)
    return repo_dest


def clean_workspace_repo(repo_dir: Path, workspaces_dir: Path) -> None:
    """Reset a per-run repo checkout to git HEAD and remove generated files.

    Baseline setup/tests can leave build artifacts in the workspace before the
    agent starts. SWE-agent rejects dirty local repos, and all agents should see
    the same clean task checkout.
    """
    repo = repo_dir.resolve()
    root = workspaces_dir.resolve()
    if not repo.is_relative_to(root):
        raise ValueError(f"refusing to clean repo outside workspaces_dir: {repo}")
    validate_local_git_config(repo)
    for command in (
        safe_git_command(repo, "reset", "--hard", "HEAD"),
        # -e: per-task test runners (SWE-bench Pro importer) live untracked
        # and git-excluded in the checkout; the agent and the evaluator both
        # invoke them, so cleanup must not sweep them away.
        safe_git_command(
            repo,
            "clean",
            "-fdx",
            "-e",
            "run_tests.sh",
            "-e",
            "run_tests_checked.sh",
        ),
    ):
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            env=safe_git_env(),
            timeout=120,
            check=False,
        )
        if result.returncode != 0:
            output = "\n".join(
                part for part in (result.stdout.strip(), result.stderr.strip()) if part
            )
            raise RuntimeError(
                f"workspace cleanup failed: {' '.join(command)}\n{output}"
            )


def write_agent_patch(
    repo_dir: Path,
    patch_path: Path,
    *,
    expected_head: str,
    excluded_paths: Iterable[str] = (),
    frozen_artifact_root: Path | None = None,
) -> list[str]:
    """Save the agent's diff vs the base commit for later quality scoring.

    Captured right after the agent finishes, before any evaluation overlay, so
    the patch is the agent's change alone. Failures are infrastructure errors:
    scoring a state that cannot be archived would make the result irreproducible.
    """
    verify_pristine_git_state(repo_dir, expected_head)
    add = subprocess.run(
        safe_git_command(repo_dir, "add", "-A"),
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
        env=safe_git_env(),
    )
    if add.returncode != 0:
        raise RuntimeError(f"failed to stage agent patch:\n{add.stderr[-1000:]}")
    staged_result = subprocess.run(
        safe_git_command(
            repo_dir,
            "diff",
            "--cached",
            "--name-only",
            "-z",
            "--no-renames",
            "HEAD",
            "--",
            ".",
        ),
        capture_output=True,
        timeout=60,
        check=False,
        env=safe_git_env(),
    )
    if staged_result.returncode != 0:
        error = staged_result.stderr.decode("utf-8", errors="replace")[-1000:]
        raise RuntimeError(f"failed to inventory staged agent patch:\n{error}")
    staged_paths = {
        item.decode("utf-8", errors="surrogateescape")
        for item in staged_result.stdout.split(b"\0")
        if item
    }
    setup_artifact_excludes: list[str] = []
    frozen_paths: set[PurePosixPath] = set()
    for relative in sorted(set(excluded_paths)):
        candidate = PurePosixPath(relative)
        if candidate.is_absolute() or ".." in candidate.parts or not candidate.parts:
            raise RuntimeError(f"unsafe agent-patch exclusion path: {relative!r}")
        frozen_paths.add(candidate)
        setup_artifact_excludes.append(f":(exclude,top,literal){candidate.as_posix()}")
    conflicts: set[str] = set()
    frozen_root = frozen_artifact_root.resolve() if frozen_artifact_root else None
    for raw in staged_paths:
        staged = PurePosixPath(raw)
        if staged.is_absolute() or ".." in staged.parts or not staged.parts:
            raise RuntimeError(f"unsafe staged agent-patch path: {raw!r}")
        for frozen in frozen_paths:
            if staged == frozen:
                if frozen_root is None or not _same_plain_file(
                    repo_dir,
                    frozen_root,
                    frozen,
                ):
                    conflicts.add(staged.as_posix())
                break
            if staged in frozen.parents or frozen in staged.parents:
                conflicts.add(staged.as_posix())
                break
    result = subprocess.run(
        safe_git_command(
            repo_dir,
            "diff",
            "--cached",
            "--binary",
            "--full-index",
            "--no-ext-diff",
            "HEAD",
            "--",
            ".",
            *_ARTIFACT_EXCLUDES,
            *setup_artifact_excludes,
        ),
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
        env=safe_git_env(),
    )
    if result.returncode != 0:
        raise RuntimeError(f"failed to render agent patch:\n{result.stderr[-1000:]}")
    patch_path.write_text(result.stdout, encoding="utf-8")
    return sorted(conflicts)


def _same_plain_file(
    left_root: Path, right_root: Path, relative: PurePosixPath
) -> bool:
    """Compare one frozen path without following model-controlled aliases."""
    candidates: list[Path] = []
    for root in (left_root.resolve(), right_root.resolve()):
        current = root
        for part in relative.parts[:-1]:
            current /= part
            try:
                metadata = current.lstat()
            except OSError:
                return False
            if not stat.S_ISDIR(metadata.st_mode):
                return False
        candidates.append(root / Path(relative))
    try:
        left_metadata = candidates[0].lstat()
        right_metadata = candidates[1].lstat()
    except OSError:
        return False
    if any(
        not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink > 1
        for metadata in (left_metadata, right_metadata)
    ):
        return False
    return (
        stat.S_IMODE(left_metadata.st_mode) == stat.S_IMODE(right_metadata.st_mode)
        and candidates[0].read_bytes() == candidates[1].read_bytes()
    )


def rebuild_scoring_checkout(
    repo_dir: Path,
    patch_path: Path,
    workspaces_dir: Path,
) -> None:
    """Reconstruct scoring state from the exact archived patch.

    Model-controlled shell commands can create executable/importable files in
    ignored dependency or build trees. Git omits those bytes from the patch,
    so evaluating the original dirty worktree would allow a success that no
    archived artifact can reproduce. Reset/clean removes that side channel;
    applying the binary patch restores only the submitted solution.
    """
    clean_workspace_repo(repo_dir, workspaces_dir)
    payload = patch_path.read_bytes()
    if not payload:
        return
    applied = subprocess.run(
        safe_git_command(
            repo_dir,
            "apply",
            "--index",
            "--binary",
            str(patch_path),
        ),
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
        env=safe_git_env(),
    )
    if applied.returncode != 0:
        raise RuntimeError(
            "archived agent patch cannot reconstruct the scoring checkout:\n"
            + applied.stderr[-1500:]
        )
