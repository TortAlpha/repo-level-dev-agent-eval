"""Freeze and launch two bounded, outcome-blind kernel/v2 repetition sweeps.

No models or tests run with --plan-only. --launch is the only public live action.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import random
import shlex
import subprocess
import sys
import traceback

PLAN = Path(__file__).resolve().parent
ROOT = PLAN.parents[2]
sys.path.insert(0, str(ROOT))
MODEL = "deepseek/deepseek-v4-flash-0731:nitro"
SEEDS = (20260904, 20260905)
AGENTS = ("single", "multi-graph")
ROLES = ("orchestrator", "planner", "developer", "tester", "reviewer")
CAMPAIGNS = tuple(f"kernel_v2_defense_new12_20260904_repeat{i}" for i in (1, 2))
ENVIRONMENT = {
    "MODEL_PROVIDER": "openrouter", "OPENROUTER_MODEL_NAME": MODEL,
    "TEMPERATURE": "0", "MAX_TOKENS": "16384",
    "OPENROUTER_CONTEXT_WINDOW_TOKENS": "1048576",
    "CONTEXT_BUDGET_TOKENS": "48000", "COMPACTION_MODE": "checkpoint",
    "REASONING_EFFORT": "low", "AGENT_ACTION_TRANSPORT": "text_json",
    "REQUEST_TIMEOUT_SECONDS": "300", "MAX_RETRIES": "0",
    "MAX_ITERATIONS": "5", "MAX_COST_USD": "0.30",
    "TEST_TIMEOUT_SECONDS": "600", "SHELL_TIMEOUT_SECONDS": "300",
    "DOCKER_ENABLED": "true", "DOCKER_NETWORK_DISABLED": "true",
    "SINGLE_RESEARCH_GUARD_ENABLED": "true",
    "SINGLE_RESEARCH_WARNING_STEPS": "12", "SINGLE_RESEARCH_HARD_LIMIT": "20",
    "SINGLE_POST_PLAN_RESEARCH_WARNING_STEPS": "5",
    "SINGLE_POST_PLAN_RESEARCH_HARD_LIMIT": "8",
    "SINGLE_COMPATIBILITY_GUARD_ENABLED": "true",
    "LANGSMITH_TRACING_ENABLED": "false", "LANGSMITH_TRACING": "false",
    "PYTHONUNBUFFERED": "1",
}
REQUIRED_VERDICTS = (
    "verified", "base_visible_passed", "base_hidden_semantic_failed",
    "reference_visible_passed", "reference_hidden_semantic_passed",
    "reference_hidden_compat_passed",
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temporary, path)


def module_at(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def primary_tasks() -> list[str]:
    # Read the parent's preselected list without executing its preflight program.
    for node in ast.parse((PLAN / "preflight.py").read_text()).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "PRIMARY"
            for target in node.targets
        ):
            tasks = ast.literal_eval(node.value)
            if len(tasks) != 12 or len(set(tasks)) != 12:
                raise ValueError("PRIMARY must contain exactly 12 distinct tasks")
            return tasks
    raise ValueError("No literal PRIMARY list in preflight.py")


def read_journals() -> list[dict]:
    result = []
    for name in ("runs.jsonl", "sweep_attempts.jsonl", "infrastructure_failures.jsonl"):
        path = ROOT / "experiments/results" / name
        if path.exists():
            result.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
    return result


def command(index: int, collection: Path, task_set: Path) -> list[str]:
    cmd = [
        sys.executable, "-u", "-m", "src.benchmark.sweep",
        "--task-set", str(task_set), "--collection", str(collection),
        "--agents", ",".join(AGENTS), "--models", MODEL,
        "--provider", "openrouter", "--session", CAMPAIGNS[index],
        "--campaign", CAMPAIGNS[index], "--concurrency", "1",
        "--order", "task-major", "--seed", str(SEEDS[index]),
        "--max-steps", "75", "--max-iterations", "5",
        "--max-cost-usd", "0.30", "--max-total-cost-usd", "8.00",
        "--reasoning-effort", "low", "--action-transport", "text_json",
        "--context-budget-tokens", "48000", "--no-network",
        "--archive-patches", "--infrastructure-retries", "0",
        "--single-research-guard", "--single-compatibility-guard",
        "--single-research-warning-steps", "12", "--single-research-hard-limit", "20",
        "--single-post-plan-research-warning-steps", "5",
        "--single-post-plan-research-hard-limit", "8",
    ]
    # CLI role overrides are forwarded only to multi-agent runs by sweep.py.
    for role in ROLES:
        cmd.extend(["--role-model", f"{role}={MODEL}", "--role-reasoning-effort", f"{role}=low"])
    return cmd


def validate() -> tuple[dict, list[dict], list[str]]:
    problems = []
    tasks = primary_tasks()
    source_collection = ROOT / "repositories/collection.csv"
    rows = {row["task_id"]: row for row in csv.DictReader(source_collection.open())}
    selected_rows = [rows[task] for task in tasks]
    classifications = {}
    report_hashes = {}
    for name in ("ansible", "openlibrary", "qutebrowser"):
        path = PLAN / f"{name}_candidates.json"
        report_hashes[str(path)] = digest(path)
        report = json.loads(path.read_text())
        for item in report.get("candidates", report.get("candidate_tasks", [])):
            classifications[item["task_id"]] = (name, item["classification"].replace("-", "_"))
    counts = Counter(classifications.get(task) for task in tasks)
    expected = Counter({(repo, kind): 2 for repo in ("ansible", "openlibrary", "qutebrowser")
                        for kind in ("localized", "cross_module")})
    if counts != expected:
        problems.append(f"Selection must have 2 localized + 2 cross_module tasks per repository: {dict(counts)}")
    exposed = sorted(set(tasks) & {row.get("task_id") for row in read_journals()})
    if exposed:
        problems.append(f"Selected tasks already have recorded attempts: {exposed}")
    preflight = None
    strict_path = PLAN / "strict_preflight.py"
    if strict_path.is_file():
        preflight = module_at("defense_strict_preflight", strict_path)
    else:
        problems.append("strict_preflight.py is not present")
    evidence_hashes = {}
    input_hashes = {}
    image_identities = {}
    for task, row in zip(tasks, selected_rows):
        path = PLAN / "strict_preflight" / f"{task}.json"
        if not path.is_file():
            problems.append(f"Missing strict preflight: {task}")
            continue
        verdict = json.loads(path.read_text())
        if verdict.get("task_id") != task or any(verdict.get(key) is not True for key in REQUIRED_VERDICTS):
            problems.append(f"Strict preflight did not pass all required checks: {task}")
        evidence_hashes[str(path)] = digest(path)
        if preflight is not None:
            actual = preflight.compute_input_hashes(row)
            if actual != verdict.get("input_hashes"):
                problems.append(f"Task inputs changed since strict preflight: {task}")
            input_hashes[task] = actual
        evidence = verdict.get("evidence", [])
        if not evidence:
            problems.append(f"No raw strict-preflight evidence: {task}")
        for item in evidence:
            evidence_path = Path(item.get("path", ""))
            if not evidence_path.is_file() or digest(evidence_path) != item.get("sha256"):
                problems.append(f"Missing or changed evidence file for {task}: {evidence_path}")
            else:
                evidence_hashes[str(evidence_path)] = digest(evidence_path)
        from src.run.reproducibility import docker_image_identity
        actual_image = docker_image_identity(row["docker_image"])
        saved_image = verdict.get("image_identity", {})
        if not actual_image.get("resolved") or actual_image.get("image_id") != saved_image.get("id"):
            problems.append(f"Docker image absent or changed since strict preflight: {task}")
        image_identities[task] = actual_image
    from src.config import load_config
    from src.benchmark.sweep import _harness_snapshot
    config = load_config("openrouter")
    forbidden = [key for key in (
        "reasoning_max_tokens", "developer_escalation_model", "developer_escalation_reasoning_effort",
        *[f"role_model_{role}" for role in ROLES],
        *[f"role_reasoning_effort_{role}" for role in ROLES],
    ) if getattr(config, key, None) is not None]
    if forbidden:
        problems.append(f"Ambient .env/environment overrides must be cleared before launch: {forbidden}")
    # Validate the exact route capability without sending a model request.
    spec = config.provider_spec(MODEL)
    if spec.reasoning_effort != "low" or spec.max_output_tokens != 16384:
        problems.append("Effective model settings do not match the protocol")
    harness = _harness_snapshot()
    if harness.get("dirty"):
        problems.append("Harness sources are dirty; freeze/commit them before this controlled study")
    if (PLAN / "launch_metadata.json").exists():
        problems.append("This bounded launch already has launch_metadata.json; duplicate dispatch is refused")
    orders = []
    for seed in SEEDS:
        order = []
        for task in tasks:
            modes = list(AGENTS)
            random.Random(f"{seed}:{task}").shuffle(modes)
            order.extend({"task_id": task, "agent_mode": mode} for mode in modes)
        orders.append(order)
    manifest = {
        "schema_version": 1, "created_at": now(), "task_ids": tasks,
        "classifications": {task: classifications.get(task) for task in tasks},
        "harness": harness, "selected_input_hashes": input_hashes,
        "image_identities": image_identities, "evidence_sha256": evidence_hashes,
        "candidate_reports_sha256": report_hashes,
        "source_collection_sha256": digest(source_collection),
        "launcher_sha256": digest(Path(__file__)), "protocol_sha256": digest(PLAN / "PROTOCOL.md"),
        "strict_preflight_sha256": digest(strict_path) if strict_path.exists() else None,
        "environment_overrides": ENVIRONMENT, "model": MODEL,
        "python_executable": sys.executable, "repetitions": 2,
        "runs_per_repetition": 24, "planned_runs": 48,
        "max_cost_usd_per_run": 0.30, "max_cost_usd_per_campaign": 8.0,
        "campaigns": list(CAMPAIGNS), "order_seeds": list(SEEDS),
        "ordered_combinations": orders,
        "host_note": "Two independent serial sweeps share one host and provider; timing is subject to contention.",
        "cost_note": "Soft guards check recorded/known cost and can overshoot; campaign accounting includes recorded infrastructure failures.",
    }
    return manifest, selected_rows, problems


def write_frozen_inputs(manifest: dict, rows: list[dict]) -> list[list[str]]:
    directory = PLAN / "frozen"
    directory.mkdir(exist_ok=True)
    collection = directory / "collection.csv"
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    contents = buffer.getvalue()
    if collection.exists() and collection.read_text() != contents:
        raise ValueError("Existing frozen collection differs; refuse silent replacement")
    collection.write_text(contents)
    task_set = directory / "task_set.json"
    payload = {"id": "kernel-v2-defense-new12-20260904", "version": 1,
               "primary_metric": "task_success", "groups": {"selected": manifest["task_ids"]},
               "previously_exercised": [], "calibration_excluded": []}
    if task_set.exists() and json.loads(task_set.read_text()) != payload:
        raise ValueError("Existing frozen task set differs; refuse silent replacement")
    atomic_json(task_set, payload)
    commands = [command(index, collection, task_set) for index in range(2)]
    manifest["frozen_collection_sha256"] = digest(collection)
    manifest["frozen_task_set_sha256"] = digest(task_set)
    manifest["commands"] = commands
    atomic_json(directory / "manifest.json", manifest)
    return commands


def worker(index: int) -> int:
    """Internal detached supervisor: one serial sweep and durable final status."""
    launch = json.loads((PLAN / "launch_metadata.json").read_text())
    frozen = PLAN / "frozen/manifest.json"
    if digest(frozen) != launch["manifest_sha256"]:
        raise ValueError("Frozen manifest changed after authorization")
    manifest = json.loads(frozen.read_text())
    if digest(Path(__file__)) != manifest["launcher_sha256"]:
        raise ValueError("Launcher changed after freeze")
    os.environ.update(manifest["environment_overrides"])
    status_path = PLAN / f"repeat{index + 1}_status.json"
    state = {"repeat": index + 1, "campaign": CAMPAIGNS[index], "session": CAMPAIGNS[index],
             "supervisor_pid": os.getpid(), "process_group_id": os.getpgrp(),
             "started_at": now(), "status": "starting", "command": manifest["commands"][index]}
    atomic_json(status_path, state)
    try:
        proc = subprocess.Popen(state["command"], cwd=ROOT, stdin=subprocess.DEVNULL)
        state.update(status="running", sweep_pid=proc.pid)
        atomic_json(status_path, state)
        code = proc.wait()
        state.update(status="finished" if code == 0 else "stopped_or_failed", returncode=code, finished_at=now())
        atomic_json(status_path, state)
        return code
    except BaseException as exc:
        state.update(status="supervisor_error", error=f"{type(exc).__name__}: {exc}", finished_at=now())
        atomic_json(status_path, state)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--plan-only", action="store_true")
    action.add_argument("--launch", action="store_true")
    action.add_argument("--worker-index", type=int, choices=(0, 1), help=argparse.SUPPRESS)
    args = parser.parse_args()
    os.chdir(ROOT)
    os.environ.update(ENVIRONMENT)
    if args.worker_index is not None:
        return worker(args.worker_index)
    with (PLAN / ".launcher.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest, rows, problems = validate()
        if problems:
            print(json.dumps({"status": "blocked", "model_calls_started": False, "problems": problems}, indent=2))
            return 2
        commands = write_frozen_inputs(manifest, rows)
        # Validate both complete frozen sweep plans before either may start.
        for index, cmd in enumerate(commands):
            log_path = PLAN / f"repeat{index + 1}_plan.log"
            with log_path.open("w") as log:
                result = subprocess.run(cmd + ["--plan-only"], cwd=ROOT, stdin=subprocess.DEVNULL,
                                        stdout=log, stderr=subprocess.STDOUT, check=False)
            if result.returncode:
                print(json.dumps({"status": "blocked", "model_calls_started": False,
                                  "reason": "Sweep plan validation failed", "log": str(log_path)}))
                return 2
        if args.plan_only:
            print(json.dumps({"status": "ready", "model_calls_started": False, "planned_runs": 48,
                              "manifest": str(PLAN / "frozen/manifest.json"),
                              "commands": [shlex.join(cmd) for cmd in commands]}, indent=2))
            return 0
        launch = {"created_at": now(), "status": "dispatching", "planned_runs": 48,
                  "manifest_sha256": digest(PLAN / "frozen/manifest.json"), "workers": []}
        atomic_json(PLAN / "launch_metadata.json", launch)
        try:
            for index in range(2):
                log_path = PLAN / f"repeat{index + 1}.log"
                with log_path.open("ab", buffering=0) as log:
                    proc = subprocess.Popen([sys.executable, "-u", str(Path(__file__)),
                                             "--worker-index", str(index)], cwd=ROOT,
                                            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                            start_new_session=True, close_fds=True)
                launch["workers"].append({"repeat": index + 1, "supervisor_pid": proc.pid,
                                          "process_group_id": proc.pid, "campaign": CAMPAIGNS[index],
                                          "session": CAMPAIGNS[index], "log": str(log_path),
                                          "status_file": str(PLAN / f"repeat{index + 1}_status.json")})
                atomic_json(PLAN / "launch_metadata.json", launch)
        except BaseException:
            launch.update(status="partial_dispatch_error", error=traceback.format_exc())
            atomic_json(PLAN / "launch_metadata.json", launch)
            raise
        launch.update(status="dispatched", dispatched_at=now())
        atomic_json(PLAN / "launch_metadata.json", launch)
        print(json.dumps(launch, indent=2))
        return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, KeyError) as exc:
        print(json.dumps({"status": "blocked", "error": f"{type(exc).__name__}: {exc}"}), file=sys.stderr)
        raise SystemExit(2)
