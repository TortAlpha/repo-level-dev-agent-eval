"""Reproducible task x model x architecture benchmark sweeps."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import subprocess
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from ..agents.swe_agent import MANAGED_BASE_IMAGE, ensure_managed_base_image
from ..config import REASONING_EFFORTS, apply_reasoning_overrides, load_config
from ..metrics.pricing import PRICING_CSV
from ..run.reproducibility import (
    HARNESS_SOURCE_PATHS,
    docker_image_identity,
    external_swe_runtime_snapshot,
    filesystem_tree_identity,
    harness_source_manifest,
    source_repository_identity,
    sweep_reproducibility_artifacts,
)
from ..run.results import _advisory_jsonl_lock, _append_jsonl_record
from .collection import DEFAULT_COLLECTION, load_collection
from .task_sets import TaskSet, load_task_set

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_DIR = Path("experiments/results")
HARNESS_PATHS = tuple(Path(path) for path in HARNESS_SOURCE_PATHS)
ROLE_NAMES = ("orchestrator", "planner", "developer", "tester", "reviewer")


@dataclass(frozen=True)
class Combination:
    task: str
    model_arg: str
    model_name: str
    agent: str


@dataclass
class AttemptResult:
    classification: str
    returncode: int
    record: dict | None
    output_tail: str


def _split(value: str | None) -> list[str]:
    return [item.strip() for item in (value or "").split(",") if item.strip()]


def _sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_safe(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a reproducible benchmark sweep into one session."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--tasks",
        default=None,
        help="Comma-separated task_ids, or explicit 'all' for every verified row.",
    )
    source.add_argument(
        "--task-set",
        default=None,
        help="Versioned task-set alias/path, e.g. final-v1.",
    )
    parser.add_argument(
        "--task-groups",
        default=None,
        help="Comma-separated manifest groups (requires --task-set).",
    )
    parser.add_argument(
        "--matrix",
        default=None,
        help="Named task/architecture matrix from the selected task set.",
    )
    parser.add_argument(
        "--models",
        default="",
        help="Comma-separated models (empty = provider default).",
    )
    parser.add_argument(
        "--agents",
        default=None,
        help=(
            "Comma-separated agents: single,single-decomposed,swe-agent,"
            "multi-graph,multi-orch,multi-orch-guided,multi-orch-guarded,multi-swe."
        ),
    )
    parser.add_argument("--session", default=None)
    parser.add_argument(
        "--campaign",
        default=None,
        help="Budget/reporting scope shared by multiple block sessions.",
    )
    parser.add_argument("--provider", default=None, choices=["local", "openrouter"])
    parser.add_argument(
        "--action-transport", choices=["text_json", "tools", "auto"], default=None
    )
    parser.add_argument("--reasoning-effort", choices=REASONING_EFFORTS, default=None)
    parser.add_argument(
        "--reasoning-max-tokens",
        type=int,
        default=None,
        help=(
            "Exact reasoning-token budget; requires a route whose frozen "
            "profile explicitly verifies end-to-end support."
        ),
    )
    parser.add_argument("--role-model", action="append", default=None)
    parser.add_argument("--role-reasoning-effort", action="append", default=None)
    parser.add_argument("--developer-escalation-model", default=None)
    parser.add_argument(
        "--developer-escalation-reasoning-effort",
        choices=REASONING_EFFORTS,
        default=None,
    )
    parser.add_argument("--developer-escalate-after-no-edit-episodes", type=int)
    parser.add_argument("--developer-escalate-after-failed-tests", type=int)
    parser.add_argument("--max-cost-usd", type=float, default=None)
    parser.add_argument(
        "--max-total-cost-usd",
        type=float,
        default=None,
        help="Stop before the next run once recorded session cost reaches this amount.",
    )
    parser.add_argument("--context-budget-tokens", type=int, default=None)
    parser.add_argument(
        "--single-research-guard", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument("--single-research-warning-steps", type=int, default=None)
    parser.add_argument("--single-research-hard-limit", type=int, default=None)
    parser.add_argument("--single-post-plan-research-warning-steps", type=int)
    parser.add_argument("--single-post-plan-research-hard-limit", type=int)
    parser.add_argument(
        "--single-compatibility-guard",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--max-subtasks", type=int, choices=range(4, 13), default=None)
    parser.add_argument("--decomposition-implementation-warning-steps", type=int)
    parser.add_argument("--decomposition-implementation-hard-limit", type=int)
    parser.add_argument("--decomposition-verification-step-reserve", type=int)
    parser.add_argument("--decomposition-max-repair-cycles", type=int)
    parser.add_argument("--collection", type=Path, default=DEFAULT_COLLECTION)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--max-iterations", type=int, default=None)
    parser.add_argument("--enable-review", action="store_true")
    parser.add_argument("--no-score", action="store_true")
    parser.add_argument("--no-regression", action="store_true")
    parser.add_argument("--no-network", action="store_true")
    parser.add_argument("--no-setup", action="store_true")
    parser.add_argument(
        "--order", choices=["task-major", "agent-major"], default="task-major"
    )
    parser.add_argument("--seed", type=int, default=20260712)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip combinations already recorded in this identical session snapshot.",
    )
    parser.add_argument("--infrastructure-retries", type=int, default=1)
    parser.add_argument(
        "--archive-patches", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--keep-workspaces", action="store_true")
    parser.add_argument("--allow-dirty-harness", action="store_true")
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Validate and print the ordered matrix without starting agents.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Legacy exploratory mode only; final task-set sweeps require 1.",
    )
    parser.add_argument("--stop-on-error", action="store_true")
    return parser.parse_args()


def _selected_tasks(
    args: argparse.Namespace,
) -> tuple[list[str], TaskSet | None]:
    collection = load_collection(args.collection)
    task_set = load_task_set(args.task_set) if args.task_set else None
    if task_set:
        if getattr(args, "matrix", None):
            tasks, _ = task_set.matrix(args.matrix)
        else:
            tasks = task_set.tasks(_split(args.task_groups) or None)
    elif (args.tasks or "").strip() == "all":
        tasks = [
            task_id
            for task_id, spec in collection.items()
            if spec.task_status.endswith("_verified")
        ]
        print(
            f"WARNING: --tasks all selects {len(tasks)} verified collection rows; "
            "it is not the frozen final-v1 set.",
            flush=True,
        )
    else:
        tasks = _split(args.tasks)
    unknown = [task_id for task_id in tasks if task_id not in collection]
    if unknown:
        raise SystemExit(f"unknown task_id(s): {', '.join(unknown)}")
    unverified = [
        task_id
        for task_id in tasks
        if not collection[task_id].task_status.endswith("_verified")
    ]
    if unverified:
        raise SystemExit(f"task set contains unverified tasks: {', '.join(unverified)}")
    return tasks, task_set


def _combinations(args: argparse.Namespace, tasks: list[str]) -> list[Combination]:
    config = load_config(args.provider)
    try:
        apply_reasoning_overrides(
            config,
            effort=getattr(args, "reasoning_effort", None),
            max_tokens=getattr(args, "reasoning_max_tokens", None),
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    model_args = _split(args.models) or [""]
    agents = _split(args.agents) or ["single"]

    def combo(task: str, model_arg: str, agent: str) -> Combination:
        return Combination(
            task=task,
            model_arg=model_arg,
            model_name=config.provider_spec(model_arg or None).model_name,
            agent=agent,
        )

    if args.order == "agent-major":
        return [
            combo(task, model, agent)
            for agent in agents
            for model in model_args
            for task in tasks
        ]

    result: list[Combination] = []
    for task in tasks:
        within_task = [
            combo(task, model, agent) for model in model_args for agent in agents
        ]
        random.Random(f"{args.seed}:{task}").shuffle(within_task)
        result.extend(within_task)
    return result


def _run_command(
    item: Combination,
    args: argparse.Namespace,
    *,
    run_id: str | None = None,
    expected_docker_image_id: str | None = None,
) -> list[str]:
    expected_task_input = getattr(args, "expected_task_inputs", {}).get(item.task, {})
    cmd = [
        sys.executable,
        "-m",
        "src.benchmark.runner",
        "--task-id",
        item.task,
        "--collection",
        str(args.collection),
        "--agent",
        item.agent,
    ]
    for flag, value in (
        ("--provider", args.provider),
        ("--model", item.model_arg),
        ("--session", args.session),
        ("--action-transport", args.action_transport),
        ("--reasoning-effort", args.reasoning_effort),
        ("--reasoning-max-tokens", args.reasoning_max_tokens),
        ("--developer-escalation-model", args.developer_escalation_model),
        (
            "--developer-escalation-reasoning-effort",
            args.developer_escalation_reasoning_effort,
        ),
        (
            "--developer-escalate-after-no-edit-episodes",
            args.developer_escalate_after_no_edit_episodes,
        ),
        (
            "--developer-escalate-after-failed-tests",
            args.developer_escalate_after_failed_tests,
        ),
        ("--max-cost-usd", args.max_cost_usd),
        ("--context-budget-tokens", args.context_budget_tokens),
        ("--max-subtasks", args.max_subtasks),
        (
            "--decomposition-implementation-warning-steps",
            args.decomposition_implementation_warning_steps,
        ),
        (
            "--decomposition-implementation-hard-limit",
            args.decomposition_implementation_hard_limit,
        ),
        (
            "--decomposition-verification-step-reserve",
            args.decomposition_verification_step_reserve,
        ),
        ("--decomposition-max-repair-cycles", args.decomposition_max_repair_cycles),
        ("--single-research-warning-steps", args.single_research_warning_steps),
        ("--single-research-hard-limit", args.single_research_hard_limit),
        (
            "--single-post-plan-research-warning-steps",
            args.single_post_plan_research_warning_steps,
        ),
        (
            "--single-post-plan-research-hard-limit",
            args.single_post_plan_research_hard_limit,
        ),
        ("--max-steps", args.max_steps),
        ("--max-iterations", args.max_iterations),
        ("--experiment-fingerprint", getattr(args, "experiment_fingerprint", None)),
        ("--task-set-id", getattr(args, "task_set_id", None)),
        ("--campaign-id", args.campaign),
        ("--run-id", run_id),
        ("--expected-docker-image-id", expected_docker_image_id),
        (
            "--expected-swe-env-image-id",
            getattr(args, "expected_swe_env_image_id", None),
        ),
        (
            "--expected-source-worktree-sha256",
            (expected_task_input.get("source_repository") or {}).get("worktree_sha256"),
        ),
        ("--expected-task-sha256", expected_task_input.get("task_sha256")),
        (
            "--expected-hidden-tree-sha256",
            expected_task_input.get("hidden_tree_sha256") or "absent",
        ),
        (
            "--expected-collection-sha256",
            getattr(args, "expected_collection_sha256", None),
        ),
        (
            "--expected-harness-tree-sha256",
            getattr(args, "expected_harness_tree_sha256", None),
        ),
        (
            "--expected-policy-kernel-sha256",
            getattr(args, "expected_policy_kernel_sha256", None),
        ),
        (
            "--expected-pricing-sha256",
            getattr(args, "expected_pricing_sha256", None),
        ),
        (
            "--expected-sweagent-distribution-sha256",
            getattr(args, "expected_sweagent_distribution_sha256", None),
        ),
        (
            "--expected-swerex-distribution-sha256",
            getattr(args, "expected_swerex_distribution_sha256", None),
        ),
        (
            "--expected-swe-runtime-sha256",
            getattr(args, "expected_swe_runtime_sha256", None),
        ),
    ):
        if value not in (None, ""):
            cmd.extend([flag, str(value)])
    if item.agent.startswith("multi"):
        for role_model in args.role_model or []:
            cmd.extend(["--role-model", role_model])
        for role_effort in args.role_reasoning_effort or []:
            cmd.extend(["--role-reasoning-effort", role_effort])
    if args.single_research_guard is not None:
        cmd.append(
            "--single-research-guard"
            if args.single_research_guard
            else "--no-single-research-guard"
        )
    if args.single_compatibility_guard is not None:
        cmd.append(
            "--single-compatibility-guard"
            if args.single_compatibility_guard
            else "--no-single-compatibility-guard"
        )
    for flag, enabled in (
        ("--enable-review", args.enable_review),
        ("--no-score", args.no_score),
        ("--no-regression", args.no_regression),
        ("--no-network", args.no_network),
        ("--no-setup", args.no_setup),
    ):
        if enabled:
            cmd.append(flag)
    return cmd


def _load_raw_runs(results_dir: Path = DEFAULT_RESULTS_DIR) -> list[dict]:
    path = results_dir / "runs.jsonl"
    if not path.is_file():
        return []
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _load_infrastructure_failures(
    results_dir: Path = DEFAULT_RESULTS_DIR,
) -> list[dict]:
    path = results_dir / "infrastructure_failures.jsonl"
    if not path.is_file():
        return []
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _combo_key(item: Combination) -> tuple[str, str, str]:
    return item.task, item.model_name, item.agent


def _record_key(row: dict) -> tuple[str, str, str]:
    return str(row.get("task_id")), str(row.get("model")), str(row.get("agent_mode"))


def _session_records(session: str | None) -> list[dict]:
    session_id = session or "default"
    return [
        row
        for row in _load_raw_runs()
        if row.get("session_id", "default") == session_id
    ]


def _effective_cost(row: dict) -> float | None:
    if (
        row.get("usage_accounting_complete") is False
        or row.get("cost_source") == "unknown_incomplete_usage"
    ):
        return None
    recorded = row.get("effective_cost_usd")
    if recorded is not None:
        return float(recorded)
    reported = row.get("provider_reported_cost_usd")
    reported_calls = row.get("provider_cost_calls")
    llm_calls = row.get("llm_calls")
    if (
        reported is not None
        and reported_calls
        and llm_calls
        and reported_calls >= llm_calls
    ):
        return float(reported)
    estimated = row.get("cost_usd")
    return float(estimated) if estimated is not None else None


def _budget_records(args: argparse.Namespace) -> list[dict]:
    if args.campaign:
        return [
            row
            for row in [*_load_raw_runs(), *_load_infrastructure_failures()]
            if row.get("campaign_id") == args.campaign
        ]
    return [
        row
        for row in [*_session_records(args.session), *_load_infrastructure_failures()]
        if row.get("session_id", "default") == (args.session or "default")
    ]


def _budget_cost(args: argparse.Namespace) -> float:
    rows = _budget_records(args)
    costs = [_effective_cost(row) for row in rows]
    unknown = [
        str(row.get("run_id") or row.get("task_id") or "unknown")
        for row, cost in zip(rows, costs, strict=False)
        if cost is None
    ]
    if unknown and args.max_total_cost_usd is not None:
        sample = ", ".join(unknown[:5])
        raise SystemExit(
            "Cannot enforce --max-total-cost-usd: effective cost is unknown "
            f"for {len(unknown)} recorded run(s): {sample}."
        )
    return sum(cost for cost in costs if cost is not None)


def _harness_snapshot() -> dict:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    ).stdout.strip()
    status = subprocess.run(
        [
            "git",
            "status",
            "--porcelain",
            "--untracked-files=all",
            "--",
            *map(str, HARNESS_PATHS),
        ],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.splitlines()
    source_manifest = harness_source_manifest(ROOT)
    return {
        "git_commit": commit or None,
        "dirty": bool(status),
        "dirty_entries": status,
        **source_manifest,
    }


def _task_input_snapshot(
    tasks: list[str],
    collection_path: Path,
    default_docker_image: str,
) -> dict[str, dict]:
    collection = load_collection(collection_path)
    result: dict[str, dict] = {}
    image_cache: dict[str, dict] = {}
    for task_id in tasks:
        task = collection[task_id]
        try:
            source = source_repository_identity(
                task.repo_path,
                declared_base_commit=task.base_commit or None,
                require_clean=True,
                # Visible test runners may be intentionally ignored by Git
                # and preserved by workspace cleanup. Their bytes are part of
                # the evaluator contract and must lock sweep resume too.
                include_ignored_files=True,
            )
        except (RuntimeError, ValueError) as exc:
            raise SystemExit(f"invalid source repository for {task_id}: {exc}") from exc
        image = task.docker_image or default_docker_image or "python:3.11-slim"
        if image not in image_cache:
            try:
                image_cache[image] = docker_image_identity(
                    image,
                    require_resolved=True,
                )
            except RuntimeError as exc:
                raise SystemExit(f"invalid Docker image for {task_id}: {exc}") from exc
        hidden_tree = filesystem_tree_identity(task.hidden_tests_path)
        result[task_id] = {
            "base_commit": source["head_commit"],
            "source_repository": source,
            "task_sha256": _sha256_file(task.task_file_path),
            "hidden_tree_sha256": (
                hidden_tree["sha256"] if hidden_tree is not None else None
            ),
            "hidden_fixture_tree": hidden_tree,
            "docker_image": image,
            "docker_image_identity": image_cache[image],
            "visible_test_command": task.visible_test_command,
            "required_hidden_suites": [
                {"name": suite.name, "command": suite.command}
                for suite in task.required_hidden_suites()
            ],
            "hidden_suites": [
                {
                    "name": suite.name,
                    "command": suite.command,
                    "required": suite.required,
                    "reuse_visible_result": suite.reuse_visible_result,
                }
                for suite in task.hidden_suites()
            ],
        }
    return result


def _resolved_role_assignments(
    config,
    raw_values: list[str] | None,
    *,
    field_prefix: str,
) -> dict[str, str | None]:
    values = {
        role: getattr(config, f"{field_prefix}_{role}", None) for role in ROLE_NAMES
    }
    for raw in raw_values or []:
        if "=" not in raw:
            raise SystemExit(f"Invalid role assignment {raw!r}; expected ROLE=VALUE.")
        role, value = (part.strip() for part in raw.split("=", 1))
        if role not in ROLE_NAMES or not value:
            raise SystemExit(
                f"Invalid role assignment {raw!r}; role must be one of "
                f"{', '.join(ROLE_NAMES)}."
            )
        values[role] = value
    return values


def _snapshot_payload(
    args: argparse.Namespace,
    tasks: list[str],
    task_set: TaskSet | None,
    combos: list[Combination],
) -> dict:
    config = load_config(args.provider)
    try:
        apply_reasoning_overrides(
            config,
            effort=args.reasoning_effort,
            max_tokens=args.reasoning_max_tokens,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    swe_execution_image_identity = None
    external_swe_distributions = None
    external_swe_runtime = None
    if any(item.agent in {"swe-agent", "multi-swe"} for item in combos):
        external_swe_runtime = external_swe_runtime_snapshot("sweagent")
        external_swe_distributions = {
            name: external_swe_runtime["dependency_closure"]["distributions"].get(
                name
            )
            for name in ("sweagent", "swe-rex")
        }
        if not external_swe_runtime["verifiable"]:
            closure = external_swe_runtime["dependency_closure"]
            missing = closure.get("missing_distributions") or []
            detail = f"; missing distributions: {', '.join(missing)}" if missing else ""
            raise SystemExit(
                "external SWE sweep requires a verifiable Python runtime and "
                f"resolved dependency closure{detail}"
            )
        try:
            ensure_managed_base_image()
            swe_execution_image_identity = docker_image_identity(
                MANAGED_BASE_IMAGE,
                require_resolved=True,
            )
        except RuntimeError as exc:
            raise SystemExit(
                f"invalid managed SWE-agent execution image: {exc}"
            ) from exc
    settings = {
        key: value
        for key, value in vars(args).items()
        if key not in {"resume", "plan_only", "allow_dirty_harness", "stop_on_error"}
    }
    payload = {
        "schema_version": 2,
        "session_id": args.session or "default",
        "created_at": _utc_now(),
        "task_set": task_set.task_set_id if task_set else None,
        "task_set_path": str(task_set.path) if task_set else None,
        "task_set_sha256": task_set.sha256 if task_set else None,
        "swe_execution_image_identity": swe_execution_image_identity,
        "external_swe_distributions": external_swe_distributions,
        "external_swe_runtime": external_swe_runtime,
        "previously_exercised": task_set.previously_exercised if task_set else [],
        "collection_sha256": _sha256_file(args.collection),
        "pricing_sha256": _sha256_file(PRICING_CSV),
        "tasks": tasks,
        "task_inputs": _task_input_snapshot(
            tasks,
            args.collection,
            config.docker_image or "python:3.11-slim",
        ),
        "ordered_combinations": [
            {
                "task": item.task,
                "model": item.model_name,
                "agent": item.agent,
                "model_route": {
                    "provider": spec.provider_name,
                    "base_url": spec.base_url,
                    "context_window_tokens": spec.context_window_tokens,
                    "max_output_tokens": spec.max_output_tokens,
                    "model_context_window_capability_tokens": (
                        spec.model_context_window_capability_tokens
                    ),
                    "model_max_output_tokens": spec.model_max_output_tokens,
                    "reasoning_effort": spec.reasoning_effort,
                    "reasoning_max_tokens": spec.reasoning_max_tokens,
                    "effective_reasoning_effort": (
                        spec.reasoning_effort
                        if spec.reasoning_effort is not None
                        else spec.provider_default_reasoning_effort
                    ),
                    "reasoning_configuration_source": (
                        "explicit_max_tokens"
                        if spec.reasoning_max_tokens is not None
                        else "explicit_effort"
                        if spec.reasoning_effort is not None
                        else "provider_default_snapshot"
                        if spec.provider_default_reasoning_effort is not None
                        else "capability_unknown"
                    ),
                    "supported_reasoning_efforts": list(
                        spec.supported_reasoning_efforts
                    ),
                    "provider_default_reasoning_effort": (
                        spec.provider_default_reasoning_effort
                    ),
                    "reasoning_capability_known": (spec.reasoning_capability_known),
                    "supports_reasoning_max_tokens": (
                        spec.supports_reasoning_max_tokens
                    ),
                    "model_profile_version": spec.model_profile_version,
                    "model_profile_source": spec.model_profile_source,
                    "model_profile_fetched_at": spec.model_profile_fetched_at,
                },
            }
            for item in combos
            for spec in [config.provider_spec(item.model_name)]
        ],
        "settings": _json_safe(settings),
        "resolved_config": {
            "model_provider": config.model_provider,
            "provider_default_model": config.provider_spec().model_name,
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
            "reasoning_effort": config.reasoning_effort,
            "reasoning_max_tokens": config.reasoning_max_tokens,
            "action_transport": (
                args.action_transport or config.agent_action_transport
            ),
            "request_timeout_seconds": config.request_timeout_seconds,
            "max_retries": config.max_retries,
            "max_iterations": args.max_iterations or config.max_iterations,
            "max_steps": args.max_steps or 50,
            "test_timeout_seconds": config.test_timeout_seconds,
            "shell_timeout_seconds": config.shell_timeout_seconds,
            "compaction_mode": config.compaction_mode,
            "context_budget_tokens": (
                args.context_budget_tokens or config.context_budget_tokens
            ),
            "docker_network_disabled": (
                args.no_network or config.docker_network_disabled
            ),
            "single_research_guard_enabled": (
                config.single_research_guard_enabled
                if args.single_research_guard is None
                else args.single_research_guard
            ),
            "single_compatibility_guard_enabled": (
                config.single_compatibility_guard_enabled
                if args.single_compatibility_guard is None
                else args.single_compatibility_guard
            ),
            "single_research_limits": {
                "warning": (
                    args.single_research_warning_steps
                    or config.single_research_warning_steps
                ),
                "hard": (
                    args.single_research_hard_limit or config.single_research_hard_limit
                ),
                "post_plan_warning": (
                    args.single_post_plan_research_warning_steps
                    or config.single_post_plan_research_warning_steps
                ),
                "post_plan_hard": (
                    args.single_post_plan_research_hard_limit
                    or config.single_post_plan_research_hard_limit
                ),
            },
            "role_models": _resolved_role_assignments(
                config,
                args.role_model,
                field_prefix="role_model",
            ),
            "role_reasoning_efforts": _resolved_role_assignments(
                config,
                args.role_reasoning_effort,
                field_prefix="role_reasoning_effort",
            ),
            "developer_escalation_model": (
                args.developer_escalation_model or config.developer_escalation_model
            ),
            "developer_escalation_reasoning_effort": (
                args.developer_escalation_reasoning_effort
                or config.developer_escalation_reasoning_effort
            ),
            "developer_escalate_after_no_edit_episodes": (
                args.developer_escalate_after_no_edit_episodes
                or config.developer_escalate_after_no_edit_episodes
            ),
            "developer_escalate_after_failed_tests": (
                args.developer_escalate_after_failed_tests
                or config.developer_escalate_after_failed_tests
            ),
            "max_cost_usd": (
                args.max_cost_usd
                if args.max_cost_usd is not None
                else config.max_cost_usd
            ),
            "docker_image": config.docker_image,
            "decomposition": {
                "max_subtasks": (
                    args.max_subtasks or config.single_decomposition_max_subtasks
                ),
                "implementation_warning_steps": (
                    args.decomposition_implementation_warning_steps
                    or config.single_decomposition_implementation_warning_steps
                ),
                "implementation_hard_limit": (
                    args.decomposition_implementation_hard_limit
                    or config.single_decomposition_implementation_hard_limit
                ),
                "verification_step_reserve": (
                    args.decomposition_verification_step_reserve
                    if args.decomposition_verification_step_reserve is not None
                    else config.single_decomposition_verification_step_reserve
                ),
                "max_repair_cycles": (
                    args.decomposition_max_repair_cycles
                    if args.decomposition_max_repair_cycles is not None
                    else config.single_decomposition_max_repair_cycles
                ),
            },
        },
        "reproducibility_artifacts": sweep_reproducibility_artifacts(),
        "harness": _harness_snapshot(),
    }
    comparable = {key: value for key, value in payload.items() if key != "created_at"}
    payload["fingerprint"] = hashlib.sha256(
        json.dumps(comparable, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return payload


def _session_snapshot_path(session: str | None) -> Path:
    safe = "".join(
        char if char.isalnum() or char in "-_" else "_"
        for char in (session or "default")
    )
    return DEFAULT_RESULTS_DIR / "sessions" / safe / "sweep.json"


def _prepare_snapshot(args: argparse.Namespace, payload: dict) -> None:
    strict = bool(payload.get("task_set"))
    if not strict:
        return
    path = _session_snapshot_path(args.session)
    previous = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None
    if previous and previous.get("fingerprint") != payload["fingerprint"]:
        raise SystemExit(
            f"session snapshot mismatch for {args.session!r}; use a new session name"
        )
    existing_runs = _session_records(args.session)
    if (previous or existing_runs) and not args.resume and not args.plan_only:
        raise SystemExit(
            f"session {args.session or 'default'!r} already exists; "
            "use --resume or a new session"
        )
    if payload["harness"]["dirty"] and not args.allow_dirty_harness:
        message = (
            "benchmark harness is dirty; commit/freeze it first or use "
            "--allow-dirty-harness for a non-final exploratory sweep"
        )
        if not args.plan_only:
            raise SystemExit(message)
        print(f"WARNING: {message}", flush=True)
    if not args.plan_only and not previous:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )


def _classify_missing_record(output: str) -> str:
    lower = output.lower()
    if any(
        marker in lower
        for marker in (
            "ratelimit",
            "openrouter",
            "api error",
            "provider",
            "server_error",
            "bad gateway",
        )
    ):
        return "provider_error"
    if any(
        marker in lower
        for marker in ("evaluation", "hidden tests", "test oracle", "junit")
    ):
        return "evaluation_error"
    if any(
        marker in lower
        for marker in (
            "docker",
            "setup failed",
            "environment",
            "failed to isolate",
            "no space left",
        )
    ):
        return "environment_error"
    return "infrastructure_error"


def _append_attempt(payload: dict) -> None:
    path = DEFAULT_RESULTS_DIR / "sweep_attempts.jsonl"
    _append_jsonl_record(path, payload)


def _archive_and_cleanup(record: dict, args: argparse.Namespace) -> None:
    workspace = Path(str(record.get("workspace") or ""))
    if not workspace.parts:
        return
    task_workspace = workspace.parent
    patch = task_workspace / "agent.patch"
    if args.archive_patches and patch.is_file():
        archive_dir = DEFAULT_RESULTS_DIR / "patches"
        archive_dir.mkdir(parents=True, exist_ok=True)
        target = archive_dir / f"{record['run_id']}.patch"
        shutil.copy2(patch, target)
        if (
            hashlib.sha256(patch.read_bytes()).digest()
            != hashlib.sha256(target.read_bytes()).digest()
        ):
            raise RuntimeError(
                f"patch archive verification failed for {record['run_id']}"
            )
        _append_jsonl_record(
            archive_dir / "index.jsonl",
            {
                "run_id": record["run_id"],
                "task_id": record.get("task_id"),
                "session_id": record.get("session_id"),
                "source": str(patch),
                "archive": str(target),
                "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                "bytes": target.stat().st_size,
            },
        )
    if not args.keep_workspaces and task_workspace.is_dir():
        root = Path("experiments/workspaces").resolve()
        resolved = task_workspace.resolve()
        if resolved.is_relative_to(root):
            shutil.rmtree(resolved)


def _remove_failed_workspace(
    task: str,
    run_id: str,
    args: argparse.Namespace,
    *,
    root: Path = Path("experiments/workspaces"),
) -> None:
    if args.keep_workspaces:
        return
    path = root / f"{task}-{run_id}"
    if path.is_dir() and path.resolve().is_relative_to(root.resolve()):
        shutil.rmtree(path, ignore_errors=True)


def _run_once(
    index: int,
    total: int,
    item: Combination,
    args: argparse.Namespace,
) -> AttemptResult:
    run_id = str(uuid.uuid4())
    expected_image_id = getattr(args, "expected_docker_image_ids", {}).get(item.task)
    header = f"[{index}/{total}] {item.agent} · {item.model_name} · {item.task}"
    print(f"\n===== {header} =====", flush=True)
    process = subprocess.Popen(
        _run_command(
            item,
            args,
            run_id=run_id,
            expected_docker_image_id=expected_image_id,
        ),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    output: list[str] = []
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
        output.append(line)
        if len(output) > 400:
            output = output[-400:]
    returncode = process.wait()
    matches = [row for row in _load_raw_runs() if str(row.get("run_id")) == run_id]
    record = matches[-1] if matches else None
    tail = "".join(output)[-6000:]
    if record is not None and record.get("run_record_complete") is True:
        classification = "task_success" if record.get("task_success") else "task_failed"
        _archive_and_cleanup(record, args)
    elif record is not None:
        # Kernel-v2 writes the scientific row only after every fallible
        # post-processing step and marks it explicitly. A stale/legacy partial
        # row for this freshly generated run id must never suppress an
        # infrastructure retry or be archived as a completed observation.
        classification = "postprocessing_error"
        record = None
        _remove_failed_workspace(item.task, run_id, args)
    else:
        classification = _classify_missing_record(tail)
        _remove_failed_workspace(item.task, run_id, args)
    return AttemptResult(classification, returncode, record, tail)


def _main_with_args(args: argparse.Namespace) -> int:
    if args.task_groups and not args.task_set:
        raise SystemExit("--task-groups requires --task-set")
    if args.matrix and not args.task_set:
        raise SystemExit("--matrix requires --task-set")
    if args.matrix and args.task_groups:
        raise SystemExit("--matrix and --task-groups are mutually exclusive")
    if args.matrix and args.agents:
        raise SystemExit("--matrix already fixes architecture coverage; omit --agents")
    if args.task_set and not args.session:
        raise SystemExit("versioned task-set sweeps require an explicit --session")
    if args.task_set and args.max_total_cost_usd is not None and not args.campaign:
        raise SystemExit(
            "final task-set budget guards require --campaign so the limit spans "
            "all block sessions"
        )
    if args.concurrency != 1 and (
        args.task_set or args.resume or args.max_total_cost_usd is not None
    ):
        raise SystemExit(
            "task-set/resumable/budgeted sweeps require --concurrency 1; parallel "
            "mode confounds provider timing and global budget accounting"
        )
    if args.concurrency != 1 and args.stop_on_error:
        raise SystemExit(
            "--stop-on-error requires --concurrency 1 because already-running "
            "parallel combinations cannot be stopped reliably"
        )

    tasks, task_set = _selected_tasks(args)
    if task_set and args.matrix:
        _, matrix_agents = task_set.matrix(args.matrix)
        args.agents = ",".join(matrix_agents)
    combos = _combinations(args, tasks)
    snapshot = _snapshot_payload(args, tasks, task_set, combos)
    args.experiment_fingerprint = snapshot["fingerprint"]
    args.task_set_id = task_set.task_set_id if task_set else None
    args.expected_docker_image_ids = {
        task_id: item["docker_image_identity"]["image_id"]
        for task_id, item in snapshot["task_inputs"].items()
    }
    swe_image = snapshot.get("swe_execution_image_identity") or {}
    args.expected_swe_env_image_id = swe_image.get("image_id")
    args.expected_task_inputs = snapshot["task_inputs"]
    args.expected_collection_sha256 = snapshot["collection_sha256"]
    args.expected_harness_tree_sha256 = snapshot["harness"]["tree_sha256"]
    args.expected_policy_kernel_sha256 = snapshot["reproducibility_artifacts"][
        "policy_kernel"
    ]["sources"]["aggregate_sha256"]
    args.expected_pricing_sha256 = snapshot["pricing_sha256"]
    swe_distributions = snapshot.get("external_swe_distributions") or {}
    args.expected_sweagent_distribution_sha256 = (
        swe_distributions.get("sweagent") or {}
    ).get("aggregate_sha256")
    args.expected_swerex_distribution_sha256 = (
        swe_distributions.get("swe-rex") or {}
    ).get("aggregate_sha256")
    args.expected_swe_runtime_sha256 = (
        snapshot.get("external_swe_runtime") or {}
    ).get("aggregate_sha256")
    _prepare_snapshot(args, snapshot)

    print(
        f"Sweep plan: {len(combos)} run(s), order={args.order}, seed={args.seed}, "
        f"session={args.session or 'default'}, "
        f"task_set={task_set.task_set_id if task_set else 'custom'}",
        flush=True,
    )
    if args.plan_only:
        for index, item in enumerate(combos, 1):
            print(f"{index:03d} {item.task} · {item.agent} · {item.model_name}")
        return 0

    completed = (
        {_record_key(row) for row in _session_records(args.session)}
        if args.resume
        else set()
    )
    skipped = 0
    successes = 0
    task_failures = 0
    infrastructure_failures = 0
    aborted = False

    def execute(index: int, item: Combination) -> AttemptResult:
        result: AttemptResult | None = None
        for attempt_number in range(args.infrastructure_retries + 1):
            result = _run_once(index, len(combos), item, args)
            _append_attempt(
                {
                    "finished_at": _utc_now(),
                    "session_id": args.session or "default",
                    "task_id": item.task,
                    "model": item.model_name,
                    "agent_mode": item.agent,
                    "attempt": attempt_number + 1,
                    "classification": result.classification,
                    "returncode": result.returncode,
                    "run_id": result.record.get("run_id") if result.record else None,
                    "output_tail": result.output_tail if result.record is None else "",
                }
            )
            if result.record is not None:
                break
            if attempt_number < args.infrastructure_retries:
                print(f"Retrying {result.classification} once...", flush=True)
        assert result is not None
        return result

    if args.concurrency > 1:
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = {
                pool.submit(execute, index, item): item
                for index, item in enumerate(combos, 1)
            }
            for future in as_completed(futures):
                result = future.result()
                if result.classification == "task_success":
                    successes += 1
                elif result.classification == "task_failed":
                    task_failures += 1
                else:
                    infrastructure_failures += 1
        spent = _budget_cost(args)
        print(
            "\nExploratory parallel sweep finished: "
            f"success={successes}, task_failed={task_failures}, "
            f"infrastructure_failed={infrastructure_failures}, "
            f"recorded_cost=${spent:.4f}.",
            flush=True,
        )
        return 2 if infrastructure_failures else 0

    for index, item in enumerate(combos, 1):
        if _combo_key(item) in completed:
            skipped += 1
            print(
                f"[{index}/{len(combos)}] resume skip · {item.agent} · {item.task}",
                flush=True,
            )
            continue
        if args.max_total_cost_usd is not None:
            spent = _budget_cost(args)
            if spent >= args.max_total_cost_usd:
                print(
                    f"Global cost guard reached: ${spent:.4f} >= "
                    f"${args.max_total_cost_usd:.4f}.",
                    flush=True,
                )
                aborted = True
                break

        result = execute(index, item)
        if result.classification == "task_success":
            successes += 1
        elif result.classification == "task_failed":
            task_failures += 1
        else:
            infrastructure_failures += 1
        if args.stop_on_error and result.classification != "task_success":
            aborted = True
            break

    spent = _budget_cost(args)
    print(
        "\nSweep finished: "
        f"success={successes}, task_failed={task_failures}, "
        f"infrastructure_failed={infrastructure_failures}, skipped={skipped}, "
        f"recorded_cost=${spent:.4f}.",
        flush=True,
    )
    return 2 if aborted or infrastructure_failures else 0


def _sweep_lease_paths(args: argparse.Namespace) -> list[Path]:
    scopes = [f"session-{args.session or 'default'}"]
    if args.campaign:
        scopes.append(f"campaign-{args.campaign}")
    paths: list[Path] = []
    for scope in scopes:
        safe = "".join(
            char if char.isalnum() or char in "-_" else "_" for char in scope
        )
        paths.append(DEFAULT_RESULTS_DIR / "leases" / safe)
    # Global ordering prevents a deadlock between campaign/session contenders.
    return sorted(paths, key=lambda path: path.as_posix())


def main() -> int:
    args = parse_args()
    # One process owns selection, resume claims, and global budget accounting
    # for the entire session/campaign. JSONL's per-append lock protects bytes;
    # this longer lease prevents duplicate combinations and budget races.
    with ExitStack() as stack:
        for path in _sweep_lease_paths(args):
            stack.enter_context(_advisory_jsonl_lock(path))
        return _main_with_args(args)


if __name__ == "__main__":
    raise SystemExit(main())
