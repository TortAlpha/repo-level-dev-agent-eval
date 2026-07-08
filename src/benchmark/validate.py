"""CLI: validate collection tasks against the acceptance checklist.

For each task this checks, in a clean Docker sandbox, the three invariants a
row must satisfy before it can be marked verified (COLLECT_INSTRUCTION.md,
step 9):

- visible tests pass on ``base_commit``;
- required hidden tests behave correctly on ``base_commit`` (semantic /
  legacy / pr-parity suites fail because the behavior is missing; compat
  suites pass because they only pin existing behavior);
- every hidden suite passes on ``merge_commit`` (the reference state).

The reference state is produced by checking out ``merge_commit`` in a scratch
copy of the task repo; missing objects are fetched from ``repo_url``. With
``--update-status`` rows that pass all checks are marked
``pr_task_verified`` / ``manual_task_verified`` in ``collection.csv``.
"""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from ..agents.sandbox import DockerSandbox
from .collection import DEFAULT_COLLECTION, TaskSpec
from .evaluation import DEFAULT_SETUP_COMMANDS

DEFAULT_WORKSPACES = Path("experiments/workspaces")

# Suites that encode the PR's new behavior: they must FAIL on base_commit.
_MUST_FAIL_ON_BASE = {"hidden", "hidden_semantic", "hidden_pr_parity"}


@dataclass
class SuiteCheck:
    name: str
    base_passed: bool | None = None
    ref_passed: bool | None = None
    expect_fail_on_base: bool = True

    @property
    def base_ok(self) -> bool | None:
        if self.base_passed is None:
            return None
        expected = not self.expect_fail_on_base
        return self.base_passed is expected

    @property
    def ref_ok(self) -> bool | None:
        return None if self.ref_passed is None else self.ref_passed


@dataclass
class TaskVerdict:
    task_id: str
    visible_on_base: bool | None = None
    suites: list[SuiteCheck] = field(default_factory=list)
    error: str | None = None
    warnings: list[str] = field(default_factory=list)

    @property
    def verified(self) -> bool:
        if self.error or self.visible_on_base is not True or not self.suites:
            return False
        for suite in self.suites:
            if suite.name == "hidden_pr_parity" and suite.base_ok is False:
                # Parity tests do not gate task success; a parity suite that
                # already passes on base is only a discriminating-power warning.
                continue
            if suite.base_ok is not True or suite.ref_ok is not True:
                return False
        return True


def _run(command: list[str], timeout: int = 300) -> subprocess.CompletedProcess:
    return subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=False)


def _copy_repo(src: Path, dest: Path) -> None:
    ignore = shutil.ignore_patterns("__pycache__", ".pytest_cache", "*.egg-info")
    shutil.copytree(src, dest, symlinks=True, ignore=ignore)


def _checkout_reference(repo: Path, merge_commit: str, repo_url: str) -> None:
    """Move a scratch checkout to the reference state, fetching if needed."""
    have = _run(["git", "-C", str(repo), "cat-file", "-e", f"{merge_commit}^{{commit}}"])
    if have.returncode != 0:
        if not repo_url:
            raise RuntimeError(f"merge commit {merge_commit} unavailable and no repo_url to fetch from")
        fetched = _run(["git", "-C", str(repo), "fetch", "--quiet", repo_url, merge_commit], timeout=600)
        if fetched.returncode != 0:
            raise RuntimeError(f"could not fetch {merge_commit} from {repo_url}:\n{fetched.stderr.strip()}")
    for command in (
        ["git", "-C", str(repo), "checkout", "--force", merge_commit],
        ["git", "-C", str(repo), "clean", "-fdx"],
    ):
        result = _run(command)
        if result.returncode != 0:
            raise RuntimeError(f"{' '.join(command[3:])} failed:\n{result.stderr.strip()}")


def _repo_relative(command: str) -> str:
    return command.replace("../hidden_tests/", "")


def _overlay_hidden(hidden: Path, repo: Path) -> None:
    if hidden.exists():
        shutil.copytree(hidden, repo, dirs_exist_ok=True)


def _run_suites_in_sandbox(
    repo: Path,
    hidden: Path,
    task: TaskSpec,
    args: argparse.Namespace,
    visible: bool,
) -> tuple[bool | None, dict[str, bool]]:
    """Setup + optional visible run + hidden suites in one container."""
    sandbox = DockerSandbox(
        workdir=repo,
        image=args.docker_image,
        network_disabled=False,
        shell_timeout_seconds=args.shell_timeout,
        test_timeout_seconds=args.test_timeout,
    )
    setup_commands = args.setup_commands or task.setup_commands or list(DEFAULT_SETUP_COMMANDS)
    visible_passed: bool | None = None
    suite_results: dict[str, bool] = {}
    try:
        sandbox.start()
        for setup in setup_commands:
            result = sandbox.run_shell(setup, args.shell_timeout)
            if not result.success:
                raise RuntimeError(f"setup failed ({setup}):\n{result.output[-1500:]}")
        if visible:
            run = sandbox.run_tests(task.visible_test_command)
            visible_passed = run.success
            if not run.success:
                print(run.output[-1500:])
        _overlay_hidden(hidden, repo)
        for suite in task.hidden_suites():
            run = sandbox.run_tests(_repo_relative(suite.command))
            suite_results[suite.name] = run.success
    finally:
        sandbox.stop()
    return visible_passed, suite_results


def validate_task(row: dict[str, str], args: argparse.Namespace) -> TaskVerdict:
    task = TaskSpec.from_row(row)
    verdict = TaskVerdict(task_id=task.task_id)
    suites = task.hidden_suites()
    if not suites:
        verdict.error = "no hidden test suites configured"
        return verdict

    scratch = args.workspaces_dir / f"validate-{task.task_id}-{uuid.uuid4().hex[:8]}"
    base_repo = scratch / "base" / "repo"
    ref_repo = scratch / "ref" / "repo"
    hidden = Path(row["hidden_tests_path"]) if row.get("hidden_tests_path") else None
    if hidden is None or not hidden.exists():
        verdict.error = f"hidden tests path missing: {hidden}"
        return verdict

    try:
        # Base state: visible must pass, new-behavior suites must fail.
        _copy_repo(task.repo_path, base_repo)
        print(f"[{task.task_id}] base state ({row['base_commit'][:12]}) ...")
        visible_passed, base_results = _run_suites_in_sandbox(
            base_repo, hidden, task, args, visible=True
        )
        verdict.visible_on_base = visible_passed

        # Reference state: every hidden suite must pass.
        _copy_repo(task.repo_path, ref_repo)
        _checkout_reference(ref_repo, row["merge_commit"], row.get("repo_url", ""))
        print(f"[{task.task_id}] reference state ({row['merge_commit'][:12]}) ...")
        _, ref_results = _run_suites_in_sandbox(ref_repo, hidden, task, args, visible=False)

        for suite in suites:
            check = SuiteCheck(
                name=suite.name,
                base_passed=base_results.get(suite.name),
                ref_passed=ref_results.get(suite.name),
                expect_fail_on_base=suite.name in _MUST_FAIL_ON_BASE,
            )
            if check.name == "hidden_pr_parity" and check.base_passed:
                verdict.warnings.append(
                    "pr-parity suite already passes on base (no discriminating power)"
                )
            verdict.suites.append(check)
    except (RuntimeError, subprocess.TimeoutExpired, OSError) as exc:
        verdict.error = str(exc)
    finally:
        if args.keep_workdirs or (not verdict.verified and not args.clean_failures):
            print(f"[{task.task_id}] scratch kept at {scratch}")
        else:
            shutil.rmtree(scratch, ignore_errors=True)
    return verdict


def _print_verdict(verdict: TaskVerdict) -> None:
    status = "VERIFIED" if verdict.verified else "FAILED"
    print(f"\n=== {verdict.task_id}: {status} ===")
    if verdict.error:
        print(f"  error: {verdict.error}")
        return
    print(f"  visible on base: {'pass' if verdict.visible_on_base else 'FAIL (must pass)'}")
    for check in verdict.suites:
        expected = "fail" if check.expect_fail_on_base else "pass"
        base = "pass" if check.base_passed else "fail"
        base_note = "ok" if check.base_ok else f"MUST {expected}"
        ref_note = "ok" if check.ref_ok else "MUST pass"
        ref = "pass" if check.ref_passed else "fail"
        print(f"  {check.name}: base={base} ({base_note}), reference={ref} ({ref_note})")
    for warning in verdict.warnings:
        print(f"  warning: {warning}")


def _update_status(csv_path: Path, verified_ids: set[str]) -> None:
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    for row in rows:
        if row["task_id"] in verified_ids:
            manual = "_manual_" in row["task_id"]
            row["task_status"] = "manual_task_verified" if manual else "pr_task_verified"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate benchmark tasks (base/reference invariants).")
    parser.add_argument("--task-id", action="append", default=None, help="Task to validate (repeatable).")
    parser.add_argument(
        "--all-unverified", action="store_true",
        help="Validate every row whose task_status is not yet *_verified.",
    )
    parser.add_argument("--collection", type=Path, default=DEFAULT_COLLECTION)
    parser.add_argument("--workspaces-dir", type=Path, default=DEFAULT_WORKSPACES)
    parser.add_argument("--docker-image", default="python:3.11-slim")
    parser.add_argument(
        "--setup-command", action="append", dest="setup_commands", default=None,
        help="Override container setup (repeatable). Default: the task's "
        "setup_commands column, else pip install -e . && pip install pytest.",
    )
    parser.add_argument("--shell-timeout", type=int, default=300)
    parser.add_argument("--test-timeout", type=int, default=600)
    parser.add_argument("--update-status", action="store_true", help="Mark passing rows verified in the CSV.")
    parser.add_argument("--keep-workdirs", action="store_true", help="Keep scratch checkouts even on success.")
    parser.add_argument("--clean-failures", action="store_true", help="Remove scratch checkouts of failing tasks too.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    with args.collection.open(newline="", encoding="utf-8") as handle:
        rows = {row["task_id"]: row for row in csv.DictReader(handle) if row.get("task_id")}

    if args.task_id:
        missing = [task_id for task_id in args.task_id if task_id not in rows]
        if missing:
            raise SystemExit(f"unknown task ids: {', '.join(missing)}")
        selected = [rows[task_id] for task_id in args.task_id]
    elif args.all_unverified:
        selected = [
            row for row in rows.values()
            if row.get("task_status") not in ("pr_task_verified", "manual_task_verified")
            and row.get("hidden_test_command", "").strip()
        ]
    else:
        raise SystemExit("pass --task-id or --all-unverified")

    verdicts = [validate_task(row, args) for row in selected]
    for verdict in verdicts:
        _print_verdict(verdict)

    verified = {verdict.task_id for verdict in verdicts if verdict.verified}
    failed = [verdict.task_id for verdict in verdicts if not verdict.verified]
    print(f"\n{len(verified)}/{len(verdicts)} tasks verified.")
    if failed:
        print(f"failed: {', '.join(failed)}")
    if args.update_status and verified:
        _update_status(args.collection, verified)
        print(f"collection.csv updated: {', '.join(sorted(verified))} -> verified")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
