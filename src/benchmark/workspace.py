"""Per-run workspace: copy the pristine repo and capture the agent's patch."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from .collection import TaskSpec


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
    shutil.copytree(task.repo_path, repo_dest, ignore=ignore)
    if task.hidden_tests_path.parts and task.hidden_tests_path.exists():
        shutil.copytree(task.hidden_tests_path, task_ws / "hidden_tests", ignore=ignore)
    return repo_dest


def write_agent_patch(repo_dir: Path, patch_path: Path) -> None:
    """Save the agent's diff vs the base commit for later quality scoring.

    Captured right after the agent finishes, before any evaluation overlay, so
    the patch is the agent's change alone. Best-effort: a missing patch just
    means quality scoring has nothing to review.
    """
    try:
        subprocess.run(
            ["git", "-C", str(repo_dir), "add", "-A"],
            capture_output=True, text=True, timeout=60, check=False,
        )
        result = subprocess.run(
            ["git", "-C", str(repo_dir), "diff", "--cached", "HEAD"],
            capture_output=True, text=True, timeout=60, check=False,
        )
        patch_path.write_text(result.stdout, encoding="utf-8")
    except (OSError, subprocess.SubprocessError):
        pass
