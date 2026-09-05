"""Freeze and validate task inputs with the kernel/v2 evaluator, before LLM calls.

This one-off campaign tool writes only below its plan directory. It never edits
source tasks, registry rows, or the harness. Docker images must already exist.
Every setup/test container is offline; unsupported images fail closed.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[3]
PLAN = Path(__file__).resolve().parent
OUT = PLAN / "strict_preflight"
WORKSPACES = PLAN / "strict_workspaces"
sys.path.insert(0, str(ROOT))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(argv: list[str], *, timeout: int = 120) -> subprocess.CompletedProcess:
    result = subprocess.run(argv, cwd=ROOT, capture_output=True, text=True,
                            timeout=timeout, check=False)
    if result.returncode:
        raise RuntimeError(f"Command failed ({argv[0]}): {result.stderr[-2000:]}")
    return result


def resolved_path(raw: str) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else ROOT / path


def tree_hash(root: Path, relative_paths: list[str] | None = None) -> str:
    digest = hashlib.sha256()
    names = relative_paths if relative_paths is not None else [
        p.relative_to(root).as_posix() for p in root.rglob("*")
        if p.is_file() or p.is_symlink()
    ]
    for name in sorted(set(names)):
        path = root / name
        digest.update(name.encode() + b"\0")
        if path.is_symlink():
            digest.update(b"symlink\0" + os.readlink(path).encode())
        elif path.is_file():
            digest.update(b"file\0" + hashlib.sha256(path.read_bytes()).digest())
        else:
            digest.update(b"missing\0")
    return digest.hexdigest()


def compute_input_hashes(row: dict[str, str]) -> dict[str, str]:
    """Reusable launcher contract: local read-only input identity, no Docker."""
    repo = resolved_path(row["repo_path"])
    tracked = run(["git", "-C", str(repo), "ls-files", "-z"]).stdout.split("\0")
    names = [name for name in tracked if name]
    names += [name for name in ("run_tests.sh", "run_tests_checked.sh")
              if (repo / name).is_file()]
    reference = repo.parent / "reference.patch"
    return {
        "registry_row_sha256": hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest(),
        "task_sha256": sha256(resolved_path(row["task_file_path"])),
        "reference_patch_sha256": sha256(reference) if reference.is_file() else "",
        "repository_tree_sha256": tree_hash(repo, names),
        "hidden_tree_sha256": tree_hash(resolved_path(row["hidden_tests_path"])),
    }


def wrapper_audit(repo: Path, commands: list[str]) -> dict:
    """Audit only invoked scripts; unused historical wrappers do not gate."""
    used = {}
    blocking = []
    for command in commands:
        tokens = shlex.split(command)
        for token in tokens:
            if not token.endswith(".sh") or token.startswith("-"):
                continue
            path = repo / token.removeprefix("./")
            if not path.is_file():
                continue
            content = path.read_text()
            used[token] = {"sha256": sha256(path), "source": str(path)}
            active_lines = [line for line in content.splitlines()
                            if line.strip() and not line.lstrip().startswith("#")]
            if any("PYTHONPATH" in line and "/app" in line for line in active_lines):
                blocking.append(f"Invoked wrapper {token} adds /app to PYTHONPATH; "
                                "kernel/v2 trusted pytest rejects paths outside /workspace.")
    return {"invoked_wrappers": used, "blocking_findings": blocking}


def image_audit(sandbox, base: Path, reference: Path, hidden: Path,
                changed_paths: list[str], expected_head: str) -> dict:
    """Check the baked /app checkout for exact hidden or reference bytes.

    This is a targeted audit, not proof that arbitrary image layers contain no
    additional information. No file contents are returned to model contexts.
    """
    names = set(changed_paths)
    names.update(p.relative_to(hidden).as_posix() for p in hidden.rglob("*")
                 if p.is_file() and not p.is_symlink())
    payload = json.dumps(sorted(names))
    script = '''import hashlib,json,pathlib,subprocess
app=pathlib.Path('/app')
names=json.loads(%r)
out={'app_exists':app.exists(),'app_git_exists':(app/'.git').exists(),'files':{}}
for name in names:
 p=app/name
 if p.is_file() and not p.is_symlink():
  out['files'][name]=hashlib.sha256(p.read_bytes()).hexdigest()
if (app/'.git').exists():
 try:
  r=subprocess.run(['git','-c','safe.directory=/app','-C','/app','rev-parse','HEAD'],capture_output=True,text=True,timeout=15)
  out['app_git_head']=r.stdout.strip() if r.returncode==0 else None
 except Exception as e:out['app_git_error']=str(e)
print(json.dumps(out,sort_keys=True))
''' % payload
    result = sandbox.run_shell("python -I -S -c " + shlex.quote(script), 60)
    if not result.success:
        raise RuntimeError("Cannot audit image /app source: " + result.output[-1500:])
    try:
        info = json.loads(result.output.strip().splitlines()[-1])
    except (ValueError, IndexError) as exc:
        raise RuntimeError("Image audit returned no valid JSON") from exc
    blocking = []
    for name, observed in info["files"].items():
        b, r, h = base / name, reference / name, hidden / name
        baseline = sha256(b) if b.is_file() and not b.is_symlink() else None
        if name in changed_paths and r.is_file() and sha256(r) != baseline and observed == sha256(r):
            blocking.append(f"Baked /app/{name} contains exact reference solution bytes.")
        if h.is_file() and not h.is_symlink() and sha256(h) != baseline and observed == sha256(h):
            blocking.append(f"Baked /app/{name} contains exact withheld fixture bytes.")
    baked_head = info.get("app_git_head")
    if info["app_git_exists"] and baked_head != expected_head:
        blocking.append("Baked /app Git checkout cannot be attested at the declared task base commit.")
    info["blocking_findings"] = blocking
    info["limitations"] = [
        "Targeted byte comparison covers changed production paths and withheld paths only.",
        "A readable baked checkout remains available to shell tools; this is not a guarantee of exclusive workspace source access.",
    ] if info["app_exists"] else []
    return info


def validate_one(task_id: str, args: argparse.Namespace) -> int:
    from src.benchmark import evaluation as ev
    from src.benchmark.collection import TaskSpec
    from src.benchmark.test_launch import pytest_selection_lower_bound
    from src.benchmark.workspace import prepare_workspace, clean_workspace_repo

    OUT.mkdir(parents=True, exist_ok=True)
    WORKSPACES.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:6]
    artifacts = OUT / f"{task_id}-{stamp}"
    artifacts.mkdir()
    result = {
        "task_id": task_id, "verified": False,
        "base_visible_passed": False, "base_hidden_semantic_failed": False,
        "reference_visible_passed": False, "reference_hidden_semantic_passed": False,
        "reference_hidden_compat_passed": False,
        "failure_category": None, "error": None, "evidence": [],
        "network_disabled_during_setup_and_tests": True,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "validator_sha256": sha256(Path(__file__)),
    }
    workdirs = []
    phase = "setup"
    current_command = {}
    original_parse = ev._parse_junit_evidence
    original_run_junit = ev._run_junit
    original_sandbox = ev._sandbox

    def offline_sandbox(repo, image, network_disabled, timeout, **kwargs):
        return original_sandbox(repo, image, True, timeout, **kwargs)

    def capture_parse(source):
        evidence = original_parse(source)
        serial = len(result["evidence"])
        item = {"phase": phase, **current_command, "junit": {
            "complete": evidence.complete, "collected": evidence.collected,
            "failures": evidence.failures, "errors": evidence.errors,
            "skipped": evidence.skipped, "error": evidence.error,
        }, "passing_ids": sorted(evidence.passing)}
        if isinstance(source, bytes):
            path = artifacts / f"{serial:02d}-{phase}.xml"
            path.write_bytes(source)
            item.update(path=str(path), sha256=sha256(path))
        result["evidence"].append(item)
        return evidence

    def capture_run_junit(sandbox, repo, command, timeout, **kwargs):
        current_command.clear()
        current_command.update(command=command, report_label=kwargs["report_label"])
        before = len(result["evidence"])
        passed, passing, output = original_run_junit(sandbox, repo, command, timeout, **kwargs)
        output_path = artifacts / f"{before:02d}-{phase}.txt"
        output_path.write_text(output)
        for item in result["evidence"][before:]:
            item.update(passed=passed, output_path=str(output_path),
                        output_sha256=sha256(output_path),
                        invocation_valid="pytest invocation evidence is missing" not in output,
                        selection_lower_bound=pytest_selection_lower_bound(command))
        return passed, passing, output

    def good_pass(item: dict, *, hidden: bool) -> bool:
        e = item["junit"]
        minimum = item.get("selection_lower_bound") or 1
        return bool(item.get("passed") and item.get("path") and e["complete"]
                    and e["collected"] >= minimum and item["passing_ids"]
                    and e["failures"] == 0 and e["errors"] == 0
                    and (not hidden or e["skipped"] == 0))

    try:
        rows = {r["task_id"]: r for r in csv.DictReader(args.collection.open())}
        row = rows[task_id]
        task = TaskSpec.from_row(row)
        task.repo_path = resolved_path(row["repo_path"])
        task.task_file_path = resolved_path(row["task_file_path"])
        task.hidden_tests_path = resolved_path(row["hidden_tests_path"])
        result["input_hashes"] = compute_input_hashes(row)
        image = task.docker_image or args.docker_image
        identity = json.loads(run(["docker", "image", "inspect", image]).stdout)[0]
        result["image_identity"] = {"requested": image, "id": identity["Id"],
                                    "repo_digests": identity.get("RepoDigests", []),
                                    "architecture": identity.get("Architecture")}
        # Use the immutable local image ID for every preflight container.
        image = identity["Id"]
        commands = [task.visible_test_command, *(s.command for s in task.hidden_suites())]
        result["wrapper_audit"] = wrapper_audit(task.repo_path, commands)
        if result["wrapper_audit"]["blocking_findings"]:
            result["failure_category"] = "unsupported_environment"
            raise RuntimeError("; ".join(result["wrapper_audit"]["blocking_findings"]))
        if not task.hidden_semantic_test_command:
            raise RuntimeError("This campaign requires an explicit hidden semantic command.")
        for label in ("base", "reference"):
            repo = prepare_workspace(task, WORKSPACES, stamp + "-" + label).resolve()
            workdirs.append(repo.parent)
            clean_workspace_repo(repo, WORKSPACES)
        base, reference = [directory / "repo" for directory in workdirs]
        actual_head = run(["git", "-C", str(base), "rev-parse", "HEAD"]).stdout.strip()
        if actual_head != task.base_commit:
            raise RuntimeError("Copied task HEAD differs from declared base_commit.")
        patch = task.repo_path.parent / "reference.patch"
        if not patch.is_file():
            raise RuntimeError("This campaign validator requires a local reference.patch.")
        run(["git", "-C", str(reference), "apply", "--whitespace=nowarn", str(patch)])
        changed_paths = re.findall(r"^diff --git a/\S+ b/(\S+)", patch.read_text(), re.M)
        oracle = workdirs[0] / "pristine_oracle"
        bootstrap = workdirs[0] / "pytest_bootstrap"
        dependencies = workdirs[0] / "scoring_dependencies"
        setup_artifacts = workdirs[0] / "setup_artifacts"
        ev.freeze_test_oracle(base, oracle, test_commands=commands)
        ev.freeze_pytest_bootstrap_packages(base, bootstrap)
        ev._sandbox = offline_sandbox
        sandbox = ev._sandbox(base, image, True, args.test_timeout)
        try:
            sandbox.start()
            result["image_audit"] = image_audit(sandbox, base, reference,
                                                 workdirs[0] / "hidden_tests",
                                                 changed_paths, task.base_commit)
        finally:
            sandbox.stop()
        if result["image_audit"]["blocking_findings"]:
            result["failure_category"] = "image_source_or_oracle_bypass"
            raise RuntimeError("; ".join(result["image_audit"]["blocking_findings"]))
        setup = task.setup_commands or list(ev.DEFAULT_SETUP_COMMANDS)
        result["setup_commands"] = setup
        ev.prepare_scoring_dependencies(
            base, docker_image=image, network_disabled=True, setup_commands=setup,
            shell_timeout=args.shell_timeout, test_timeout=args.test_timeout,
            dependency_environment=dependencies, setup_artifact_environment=setup_artifacts,
        )
        clean_workspace_repo(base, WORKSPACES)
        if ev.restore_test_oracle(base, pristine_repo=oracle, test_commands=commands):
            raise RuntimeError("Setup modified the frozen test oracle.")
        ev._parse_junit_evidence = capture_parse
        ev._run_junit = capture_run_junit
        common = dict(do_hidden=True, docker_image=image, network_disabled=True,
                      setup_commands=[], shell_timeout=args.shell_timeout,
                      test_timeout=args.test_timeout, pristine_repo=oracle,
                      dependency_environment=dependencies,
                      setup_artifact_environment=setup_artifacts,
                      pytest_bootstrap_environment=bootstrap)
        phase = "base"
        base_eval = ev.evaluate_solution(task, workdirs[0], task.visible_test_command,
                                         baseline_passing=None, **common)
        result["base_evaluation"] = asdict(base_eval)
        base_items = [e for e in result["evidence"] if e["phase"] == "base"]
        if not base_items:
            raise RuntimeError("No base JUnit evidence was produced.")
        result["base_visible_passed"] = good_pass(base_items[0], hidden=False)
        semantic_command = ev._repo_relative_hidden_command(task.hidden_semantic_test_command)
        semantic = [e for e in base_items if e["command"] == semantic_command]
        if len(semantic) == 1:
            item = semantic[0]
            e = item["junit"]
            result["base_hidden_semantic_failed"] = bool(
                not item["passed"] and item.get("path") and item["invocation_valid"]
                and e["complete"] and e["collected"] >= (item["selection_lower_bound"] or 1)
                and e["failures"] > 0 and e["errors"] == 0 and e["skipped"] == 0
            )
        compat_command = ev._repo_relative_hidden_command(task.hidden_compat_test_command)
        base_compat = [e for e in base_items if task.hidden_compat_test_command and e["command"] == compat_command]
        result["base_hidden_compat_passed"] = (all(good_pass(e, hidden=True) for e in base_compat)
                                                if task.hidden_compat_test_command else None)
        phase = "reference"
        reference_eval = ev.evaluate_solution(
            task, workdirs[1], task.visible_test_command,
            baseline_passing=set(base_items[0]["passing_ids"]), **common,
        )
        result["reference_evaluation"] = asdict(reference_eval)
        ref_items = [e for e in result["evidence"] if e["phase"] == "reference"]
        result["reference_visible_passed"] = bool(ref_items and good_pass(ref_items[0], hidden=False))
        ref_semantic = [e for e in ref_items if e["command"] == semantic_command]
        result["reference_hidden_semantic_passed"] = bool(len(ref_semantic) == 1 and good_pass(ref_semantic[0], hidden=True))
        ref_compat = [e for e in ref_items if task.hidden_compat_test_command and e["command"] == compat_command]
        result["reference_hidden_compat_passed"] = (
            bool(len(ref_compat) == 1 and good_pass(ref_compat[0], hidden=True))
            if task.hidden_compat_test_command else None
        )
        result["verified"] = bool(
            result["base_visible_passed"] and result["base_hidden_semantic_failed"]
            and result["base_hidden_compat_passed"] is not False
            and result["reference_visible_passed"] and result["reference_hidden_semantic_passed"]
            and result["reference_hidden_compat_passed"] is not False
            and reference_eval.task_success and not base_eval.test_oracle_tampered
            and not base_eval.setup_artifact_tampered
        )
        if not result["verified"]:
            result["failure_category"] = "oracle_or_reference_validation"
            result["error"] = "Required strict base/reference invariants did not all hold. See saved evidence."
        if compute_input_hashes(row) != result["input_hashes"]:
            result["verified"] = False
            result["failure_category"] = "input_changed_during_preflight"
            result["error"] = "Source inputs changed during preflight; discard this result."
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["failure_category"] = result["failure_category"] or "infrastructure_or_unsupported_environment"
    finally:
        ev._parse_junit_evidence = original_parse
        ev._run_junit = original_run_junit
        ev._sandbox = original_sandbox
        result["finished_at"] = datetime.now(timezone.utc).isoformat()
        result["artifact_directory"] = str(artifacts)
        encoded = json.dumps(result, indent=2) + "\n"
        (artifacts / "verdict.json").write_text(encoded)
        current = OUT / f"{task_id}.json"
        temporary = current.with_suffix(".json.tmp")
        temporary.write_text(encoded)
        temporary.replace(current)
        if not args.keep_workdirs:
            for directory in workdirs:
                shutil.rmtree(directory, ignore_errors=True)
        print(json.dumps({"task_id": task_id, "verified": result["verified"],
                          "failure_category": result["failure_category"],
                          "error": result["error"], "artifact_directory": str(artifacts)}), flush=True)
    return 0 if result["verified"] else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="+", required=False)
    parser.add_argument("--one")
    parser.add_argument("--collection", type=Path, default=ROOT / "repositories/collection.csv")
    parser.add_argument("--docker-image", default="python:3.11-slim")
    parser.add_argument("--shell-timeout", type=int, default=600)
    parser.add_argument("--test-timeout", type=int, default=600)
    parser.add_argument("--keep-workdirs", action="store_true")
    args = parser.parse_args()
    args.collection = args.collection.resolve()
    if args.one:
        return validate_one(args.one, args)
    if not args.tasks:
        parser.error("Supply --tasks TASK_ID ... or --one TASK_ID")
    if len(args.tasks) != len(set(args.tasks)):
        parser.error("Duplicate task IDs are not allowed")
    OUT.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:6]
    def execute(task_id):
        log = OUT / f"{task_id}-{stamp}.log"
        command = [sys.executable, "-u", str(Path(__file__).resolve()), "--one", task_id,
                   "--collection", str(args.collection), "--docker-image", args.docker_image,
                   "--shell-timeout", str(args.shell_timeout), "--test-timeout", str(args.test_timeout)]
        if args.keep_workdirs:
            command.append("--keep-workdirs")
        with log.open("w") as stream:
            child = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=False)
        item = {"task_id": task_id, "exit_code": child.returncode, "log": str(log)}
        print(json.dumps(item), flush=True)
        return item
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(execute, args.tasks))
    summary = {"tasks": args.tasks, "results": results,
               "all_verified": all(item["exit_code"] == 0 for item in results)}
    (OUT / f"batch-{stamp}.json").write_text(json.dumps(summary, indent=2) + "\n")
    return 0 if summary["all_verified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
