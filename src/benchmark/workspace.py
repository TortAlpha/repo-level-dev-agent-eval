"""Per-run workspace: copy the pristine repo and capture the agent's patch."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

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
    for command in (
        ["git", "-C", str(repo), "reset", "--hard", "HEAD"],
        # -e: per-task test runners (SWE-bench Pro importer) live untracked
        # and git-excluded in the checkout; the agent and the evaluator both
        # invoke them, so cleanup must not sweep them away.
        ["git", "-C", str(repo), "clean", "-fdx",
         "-e", "run_tests.sh", "-e", "run_tests_checked.sh"],
    ):
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        if result.returncode != 0:
            output = "\n".join(
                part for part in (result.stdout.strip(), result.stderr.strip()) if part
            )
            raise RuntimeError(f"workspace cleanup failed: {' '.join(command)}\n{output}")


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
            ["git", "-C", str(repo_dir), "diff", "--cached", "HEAD", "--", ".",
             *_ARTIFACT_EXCLUDES],
            capture_output=True, text=True, timeout=60, check=False,
        )
        patch_path.write_text(result.stdout, encoding="utf-8")
    except (OSError, subprocess.SubprocessError):
        pass
