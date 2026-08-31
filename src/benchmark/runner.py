"""CLI: run one benchmark task end to end.

Copies the pristine repo into a throwaway workspace, runs the agent against the
task's own visible test command, evaluates the solution (regressions + hidden
tests), optionally reviews its quality, and records everything. The originals
under ``repositories/`` are never modified.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import shutil
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

from ..agents.model import LangChainModel
from ..agents.single_agent import SingleAgent
from ..agents.state import State
from ..agents.swe_agent import (
    MANAGED_BASE_IMAGE,
    SweAgentAdapter,
    ensure_managed_base_image,
)
from ..agents.tracing import attach_run_feedback, configure_langsmith, run_trace_extra
from ..config import (
    REASONING_EFFORTS,
    Config,
    ProviderSpec,
    ReasoningEffort,
    apply_reasoning_overrides,
    load_config,
)
from ..metrics.compute import compute_metrics
from ..metrics.pricing import PRICING_CSV, estimate_cost
from ..metrics.records import RunRecord
from ..run import build_model, load_task_meta, record_run_result, run_metrics
from ..run.reproducibility import (
    build_run_reproducibility,
    distribution_content_snapshot,
    docker_image_identity,
    external_swe_runtime_snapshot,
    filesystem_tree_identity,
    harness_source_manifest,
    policy_kernel_source_manifest,
    source_repository_identity,
)
from ..run.results import _append_jsonl_record
from .collection import DEFAULT_COLLECTION, TaskSpec, load_collection
from .evaluation import (
    DEFAULT_SETUP_COMMANDS,
    EvalResult,
    collect_visible_passing,
    evaluate_solution,
    freeze_pytest_bootstrap_packages,
    freeze_test_oracle,
    frozen_setup_artifact_files,
    prepare_scoring_dependencies,
    restore_test_oracle,
)
from .evaluation import (
    task_success as compute_task_success,
)
from .review import DEFAULT_REVIEW_MODEL, review_solution
from .workspace import (
    clean_workspace_repo,
    prepare_workspace,
    rebuild_scoring_checkout,
    write_agent_patch,
)

TaskAgent = SingleAgent | SweAgentAdapter
ROLE_NAMES = ("orchestrator", "planner", "developer", "tester", "reviewer")


def agent_network_policy(agent: object) -> dict[str, object]:
    """Separate the editing boundary from the downstream evaluator.

    The built-in pseudo-SWE agents own their Docker sandbox and can enforce
    post-setup disconnection. External SWE-agent owns a SWE-ReX deployment
    whose network policy this adapter does not control; claiming otherwise
    would make an architecture comparison scientifically misleading.
    """
    nested = getattr(agent, "developer_adapter", None)
    external = isinstance(agent, SweAgentAdapter) or isinstance(nested, SweAgentAdapter)
    evaluation_disabled = bool(getattr(agent, "docker_network_disabled", False))
    return {
        "editing_network_disabled_after_setup": (
            evaluation_disabled if not external else False
        ),
        "evaluation_network_disabled_after_setup": evaluation_disabled,
        "editing_network_policy": (
            "external_swe_rex_unverified" if external else "harness_managed_docker"
        ),
    }


def configured_action_transport(agent: object) -> str:
    """Return the requested transport without triggering a live probe."""
    return str(getattr(agent, "action_transport", "external"))


def observed_action_transport(agent: object) -> str:
    """Return a cached runtime resolution, never initiating provider I/O."""
    cached = getattr(agent, "_transport_cache", None)
    if isinstance(cached, tuple) and cached:
        return str(cached[0])
    return configured_action_transport(agent)


def observed_transport_downgraded(agent: object) -> bool:
    cached = getattr(agent, "_transport_cache", None)
    return bool(isinstance(cached, tuple) and len(cached) > 1 and cached[1])


def validate_expected_input_contract(
    args: argparse.Namespace,
    task: TaskSpec,
    *,
    source_repository: dict | None = None,
    hidden_path: Path | None = None,
) -> dict:
    """Fail if a sweep input changed after its immutable parent snapshot."""
    source = source_repository or source_repository_identity(
        task.repo_path,
        declared_base_commit=task.base_commit or None,
        require_clean=True,
        include_ignored_files=True,
    )
    hidden_identity = filesystem_tree_identity(
        task.hidden_tests_path if hidden_path is None else hidden_path
    )
    actual = {
        "source_worktree": source.get("worktree_sha256"),
        "task": (
            hashlib.sha256(task.task_file_path.read_bytes()).hexdigest()
            if task.task_file_path.is_file()
            else None
        ),
        "hidden": (
            hidden_identity.get("sha256") if hidden_identity is not None else "absent"
        ),
        "collection": (
            hashlib.sha256(args.collection.read_bytes()).hexdigest()
            if args.collection.is_file()
            else None
        ),
        "harness": harness_source_manifest()["tree_sha256"],
        "policy_kernel": policy_kernel_source_manifest()["aggregate_sha256"],
        "pricing": (
            hashlib.sha256(PRICING_CSV.read_bytes()).hexdigest()
            if PRICING_CSV.is_file()
            else None
        ),
    }
    expected = {
        "source_worktree": args.expected_source_worktree_sha256,
        "task": args.expected_task_sha256,
        "hidden": args.expected_hidden_tree_sha256,
        "collection": args.expected_collection_sha256,
        "harness": args.expected_harness_tree_sha256,
        "policy_kernel": args.expected_policy_kernel_sha256,
        "pricing": args.expected_pricing_sha256,
    }
    mismatches = [
        f"{name}: {actual[name]!r} != {wanted!r}"
        for name, wanted in expected.items()
        if wanted is not None and actual[name] != wanted
    ]
    if mismatches:
        raise RuntimeError(
            "sweep input contract changed after snapshot; refusing mixed "
            "provenance:\n" + "\n".join(mismatches)
        )
    return source


def validate_frozen_evaluation_trees(
    expected: dict[str, dict | None],
    paths: dict[str, Path],
) -> None:
    """Attest scorer-only trees immediately before/after final evaluation."""
    mismatches = [
        name
        for name, path in paths.items()
        if filesystem_tree_identity(path) != expected.get(name)
    ]
    if mismatches:
        raise RuntimeError(
            "frozen evaluator input changed after preflight: "
            + ", ".join(sorted(mismatches))
        )


def agent_settings(agent: TaskAgent) -> dict:
    """One capability-safe settings view for traces, metrics, and dry-runs.

    Built-in agents expose guard/decomposition fields through ``SingleAgent``;
    external adapters intentionally do not. Central defaults keep adapters from
    breaking whenever a built-in-only setting is added.
    """
    return {
        **agent_network_policy(agent),
        "context_window_tokens": agent.context_window_tokens,
        "context_budget_tokens": getattr(agent, "context_budget_tokens", None),
        "max_response_tokens": agent.max_response_tokens,
        "action_transport": configured_action_transport(agent),
        "max_cost_usd": getattr(agent, "max_cost_usd", None),
        "research_guard_enabled": getattr(agent, "research_guard_enabled", False),
        "research_warning_steps": getattr(agent, "research_warning_steps", 0),
        "research_hard_limit": getattr(agent, "research_hard_limit", 0),
        "post_plan_research_warning_steps": getattr(
            agent, "post_plan_research_warning_steps", 0
        ),
        "post_plan_research_hard_limit": getattr(
            agent, "post_plan_research_hard_limit", 0
        ),
        "compatibility_guard_enabled": getattr(
            agent, "compatibility_guard_enabled", False
        ),
        "decomposition_enabled": getattr(agent, "decomposition_enabled", False),
        "max_subtasks": getattr(agent, "max_subtasks", 0),
        "decomposition_implementation_warning_steps": getattr(
            agent, "decomposition_implementation_warning_steps", 0
        ),
        "decomposition_implementation_hard_limit": getattr(
            agent, "decomposition_implementation_hard_limit", 0
        ),
        "decomposition_verification_step_reserve": getattr(
            agent, "decomposition_verification_step_reserve", 0
        ),
        "max_repair_cycles": getattr(agent, "max_repair_cycles", 0),
    }


def agent_model_policy(agent: TaskAgent) -> dict:
    if hasattr(agent, "model_policy_metadata"):
        return agent.model_policy_metadata()
    return {
        "type": "external" if isinstance(agent, SweAgentAdapter) else "single",
        "base_model": agent.model.model,
    }


def _swe_adapter_route(adapter: SweAgentAdapter, *, transport: str) -> dict:
    """Truthful route metadata for the external adapter boundary.

    SWE-agent 1.x receives the model id and base URL, but this adapter does not
    forward our LangChain context/output/reasoning request parameters. Keep the
    requested values for auditability while leaving the actual external values
    unknown (except for a reviewed provider default).
    """
    route = adapter.model.route_metadata(transport=transport)
    requested_effort = route.get("reasoning_effort")
    requested_reasoning_max = route.get("reasoning_max_tokens")
    requested_context = route.get("context_window_tokens")
    requested_output = route.get("max_output_tokens")
    provider_default = route.get("provider_default_reasoning_effort")
    return {
        **route,
        "context_window_tokens": None,
        "max_output_tokens": None,
        "reasoning_effort": None,
        "reasoning_max_tokens": None,
        "effective_reasoning_effort": provider_default,
        "reasoning_configuration_source": (
            "external_adapter_provider_default_snapshot"
            if provider_default is not None
            else "external_adapter_unverified"
        ),
        "requested_context_window_tokens": requested_context,
        "requested_max_output_tokens": requested_output,
        "requested_reasoning_effort": requested_effort,
        "requested_reasoning_max_tokens": requested_reasoning_max,
        "context_window_forwarded": False,
        "max_output_tokens_forwarded": False,
        "reasoning_parameters_forwarded": False,
    }


def _file_content_snapshot(path_value: str | None) -> dict:
    if not path_value:
        return {"path": None, "sha256": None, "available": False}
    path = Path(path_value).expanduser()
    return {
        "path": str(path),
        "sha256": (
            hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
        ),
        "available": path.is_file(),
    }


def swe_adapter_policy_snapshot(adapter: SweAgentAdapter) -> dict:
    """Hash the external invocation contract that this harness controls."""
    try:
        version = importlib.metadata.version("sweagent")
    except importlib.metadata.PackageNotFoundError:
        version = None
    resolved_binary = shutil.which(adapter.sweagent_bin)
    config_path = adapter._default_config()
    overlay = adapter._overlay_yaml() if adapter.env_image else None
    runtime = external_swe_runtime_snapshot(adapter.sweagent_bin)
    closure_distributions = runtime["dependency_closure"]["distributions"]
    return {
        "adapter": type(adapter).__name__,
        "sweagent_version": version,
        "sweagent_distribution": closure_distributions.get("sweagent")
        or distribution_content_snapshot("sweagent"),
        "swerex_distribution": closure_distributions.get("swe-rex")
        or distribution_content_snapshot("swe-rex"),
        "host_runtime": runtime,
        "binary": _file_content_snapshot(resolved_binary),
        "run_subcommand": adapter.run_subcommand,
        "config": _file_content_snapshot(config_path),
        "config_explicit": adapter.config_path is not None,
        "external_default_config_unverified": config_path is None,
        "env_image": adapter.env_image,
        "env_image_reference": adapter.env_image_reference,
        "env_image_identity": adapter.env_image_identity,
        "overlay_yaml_sha256": (
            hashlib.sha256(overlay.encode("utf-8")).hexdigest()
            if overlay is not None
            else None
        ),
        "post_startup_commands": list(adapter.post_startup_commands),
        "extra_args": list(adapter.extra_args),
        "parse_function": adapter.parse_function,
        "disable_cost_limit": adapter.disable_cost_limit,
        "per_instance_cost_limit": adapter.per_instance_cost_limit,
        "per_instance_call_limit": adapter.call_limit,
        "wall_timeout_seconds": adapter.wall_timeout_seconds,
        "container_label_policy": "unique_run_scoped_cleanup",
        "inherited_swe_environment": "python_litellm_loader_sanitized",
        "usage_accounting": "strict_trajectory_required",
        "editing_network_policy": "explicit_online_external_swe_rex_ablation",
    }


def external_adapter_policy_snapshot(agent: TaskAgent) -> dict | None:
    if isinstance(agent, SweAgentAdapter):
        return swe_adapter_policy_snapshot(agent)
    nested = getattr(agent, "developer_adapter", None)
    if isinstance(nested, SweAgentAdapter):
        return swe_adapter_policy_snapshot(nested)
    return None


def swe_execution_adapter(agent: object) -> SweAgentAdapter | None:
    """Return the external adapter that actually owns the SWE container."""
    if isinstance(agent, SweAgentAdapter):
        return agent
    nested = getattr(agent, "developer_adapter", None)
    return nested if isinstance(nested, SweAgentAdapter) else None


def pin_swe_execution_image(
    agent: object,
    *,
    require_resolved: bool,
    expected_image_id: str | None = None,
) -> dict | None:
    """Pin the fast SWE-agent execution image, separately from evaluation."""
    adapter = swe_execution_adapter(agent)
    if adapter is None or adapter.env_image is None:
        return None
    if require_resolved:
        for distribution_name in ("sweagent", "swe-rex"):
            distribution = distribution_content_snapshot(distribution_name)
            if not distribution["verifiable"]:
                raise RuntimeError(
                    f"{distribution_name} installation provenance is not "
                    "verifiable; install a pinned wheel or use a clean, "
                    "present Git editable checkout"
                )
    reference = adapter.env_image
    if require_resolved and reference == MANAGED_BASE_IMAGE:
        ensure_managed_base_image()
    identity = docker_image_identity(reference, require_resolved=require_resolved)
    actual_image_id = identity.get("image_id")
    if expected_image_id is not None and actual_image_id != expected_image_id:
        raise RuntimeError(
            "SWE-agent execution image changed after the sweep snapshot: "
            f"{reference!r} resolved to {actual_image_id!r}, "
            f"expected {expected_image_id!r}"
        )
    adapter.env_image_reference = reference
    adapter.env_image_identity = identity
    if actual_image_id is not None:
        adapter.env_image = str(actual_image_id)
    return identity


def validate_swe_installation_snapshot(
    agent: object,
    args: argparse.Namespace,
    *,
    require_verifiable: bool,
) -> None:
    adapter = swe_execution_adapter(agent)
    if adapter is None:
        return
    runtime = external_swe_runtime_snapshot(adapter.sweagent_bin)
    if require_verifiable and not runtime["verifiable"]:
        closure = runtime["dependency_closure"]
        missing = closure.get("missing_distributions") or []
        detail = f"; missing distributions: {', '.join(missing)}" if missing else ""
        raise RuntimeError(
            "external SWE Python runtime/dependency closure provenance is not "
            f"verifiable{detail}; run the benchmark from the same pinned Python "
            "environment that owns the sweagent entrypoint"
        )
    expected_runtime = getattr(args, "expected_swe_runtime_sha256", None)
    if (
        expected_runtime is not None
        and runtime["aggregate_sha256"] != expected_runtime
    ):
        raise RuntimeError(
            "external SWE Python runtime/dependency closure changed after the "
            f"sweep snapshot: {runtime['aggregate_sha256']!r} != "
            f"{expected_runtime!r}"
        )
    expected = {
        "sweagent": args.expected_sweagent_distribution_sha256,
        "swe-rex": args.expected_swerex_distribution_sha256,
    }
    for name, wanted in expected.items():
        snapshot = runtime["dependency_closure"]["distributions"].get(name)
        if snapshot is None:
            snapshot = distribution_content_snapshot(name)
        if require_verifiable and not snapshot["verifiable"]:
            raise RuntimeError(f"{name} installation provenance is not verifiable")
        if wanted is not None and snapshot["aggregate_sha256"] != wanted:
            raise RuntimeError(
                f"{name} changed after the sweep snapshot: "
                f"{snapshot['aggregate_sha256']!r} != {wanted!r}"
            )


def agent_model_routes(
    agent: TaskAgent,
    *,
    role_steps: dict[str, int] | None = None,
) -> dict[str, dict]:
    """Exact resolved provider/model/context/reasoning/transport per role."""

    def with_agent_context(route: dict) -> dict:
        return {
            **route,
            "agent_context_window_tokens": agent.context_window_tokens,
            "context_budget_tokens": getattr(agent, "context_budget_tokens", None),
            "max_response_tokens": agent.max_response_tokens,
        }

    if isinstance(agent, SweAgentAdapter):
        transport = (
            f"swe-agent:{agent.parse_function}"
            if agent.parse_function
            else "swe-agent:function_calling"
        )
        return {
            "agent": with_agent_context(_swe_adapter_route(agent, transport=transport))
        }

    if hasattr(agent, "role_models"):
        role_transports = (
            agent.role_transports() if hasattr(agent, "role_transports") else {}
        )
        role_usage = agent.role_usage() if hasattr(agent, "role_usage") else {}
        class_name = type(agent).__name__
        routes: dict[str, dict] = {}
        for role in ROLE_NAMES:
            developer_adapter = (
                getattr(agent, "developer_adapter", None)
                if role == "developer"
                else None
            )
            role_model = (
                developer_adapter.model
                if developer_adapter is not None
                else agent.role_models.get(role, agent.model)
            )
            resolved = (role_transports.get(role) or {}).get(role_model.model)
            if developer_adapter is not None:
                transport = (
                    f"swe-agent:{developer_adapter.parse_function}"
                    if developer_adapter.parse_function
                    else "swe-agent:function_calling"
                )
                route = _swe_adapter_route(
                    developer_adapter,
                    transport=transport,
                )
            else:
                route = role_model.route_metadata(
                    transport=resolved or f"unresolved:{agent.action_transport}"
                )
            routes[role] = {
                **with_agent_context(route),
                "configured_for_architecture": (
                    role != "orchestrator" or "Orchestrator" in class_name
                ),
                "invoked": role in role_usage or bool((role_steps or {}).get(role, 0)),
            }
        escalation = getattr(agent, "developer_escalation_model", None)
        if escalation is not None:
            resolved = (role_transports.get("developer") or {}).get(escalation.model)
            routes["developer_escalation"] = with_agent_context(
                escalation.route_metadata(
                    transport=resolved or f"unresolved:{agent.action_transport}"
                )
            )
        return routes
    return {
        "agent": with_agent_context(
            agent.model.route_metadata(transport=observed_action_transport(agent))
        )
    }


def configured_agent_model_routes(agent: TaskAgent) -> dict[str, dict]:
    """Stable pre-run routes used by the reproducibility fingerprint.

    Runtime routes intentionally contain observations such as which roles were
    invoked and which live tool probes succeeded. Those belong in run metrics,
    but hashing them would make identical configurations acquire different
    fingerprints based on the model's behavior. This snapshot contains only
    the configured transport/model/context/reasoning policy.
    """

    def with_agent_context(route: dict) -> dict:
        return {
            **route,
            "agent_context_window_tokens": agent.context_window_tokens,
            "context_budget_tokens": getattr(agent, "context_budget_tokens", None),
            "max_response_tokens": agent.max_response_tokens,
        }

    if isinstance(agent, SweAgentAdapter):
        transport = (
            f"swe-agent:{agent.parse_function}"
            if agent.parse_function
            else "swe-agent:function_calling"
        )
        return {
            "agent": with_agent_context(_swe_adapter_route(agent, transport=transport))
        }

    requested_transport = f"configured:{agent.action_transport}"
    if hasattr(agent, "role_models"):
        class_name = type(agent).__name__
        routes: dict[str, dict] = {}
        for role in ROLE_NAMES:
            developer_adapter = (
                getattr(agent, "developer_adapter", None)
                if role == "developer"
                else None
            )
            role_model = (
                developer_adapter.model
                if developer_adapter is not None
                else agent.role_models.get(role, agent.model)
            )
            route = (
                _swe_adapter_route(
                    developer_adapter,
                    transport=(
                        f"swe-agent:{developer_adapter.parse_function}"
                        if developer_adapter.parse_function
                        else "swe-agent:function_calling"
                    ),
                )
                if developer_adapter is not None
                else role_model.route_metadata(transport=requested_transport)
            )
            routes[role] = {
                **with_agent_context(route),
                "configured_for_architecture": (
                    role != "orchestrator" or "Orchestrator" in class_name
                ),
            }
        escalation = getattr(agent, "developer_escalation_model", None)
        if escalation is not None:
            routes["developer_escalation"] = with_agent_context(
                escalation.route_metadata(transport=requested_transport)
            )
        return routes
    return {
        "agent": with_agent_context(
            agent.model.route_metadata(transport=requested_transport)
        )
    }


def reasoning_policy_label(routes: dict[str, dict]) -> str:
    policies = {
        (
            route.get("effective_reasoning_effort"),
            route.get("reasoning_max_tokens"),
        )
        for name, route in routes.items()
        if route.get("configured_for_architecture", True)
    }
    return "shared" if len(policies) <= 1 else "role_specific"


def record_infrastructure_failure(
    config: Config,
    args: argparse.Namespace,
    task: TaskSpec,
    run_id: str,
    phase: str,
    exc: Exception,
    usage_models: list[LangChainModel],
    state: State,
    duration_s: float,
) -> None:
    text = str(exc).lower()
    if phase == "evaluation":
        classification = "evaluation_error"
    elif any(
        marker in text
        for marker in (
            "openrouter",
            "provider",
            "rate limit",
            "server_error",
            "bad gateway",
            "api",
        )
    ):
        classification = "provider_error"
    else:
        classification = "environment_error"
    payload = {
        "run_id": run_id,
        "task_id": task.task_id,
        "session_id": config.session_id,
        "campaign_id": args.campaign_id,
        "task_set_id": args.task_set_id,
        "experiment_fingerprint": args.experiment_fingerprint,
        "agent_mode": config.agent_mode,
        "model": usage_models[0].model,
        "phase": phase,
        "classification": classification,
        "error": str(exc)[-4000:],
        "finished_at": datetime.now(UTC).isoformat(),
        **run_metrics(usage_models, state, duration_s),
    }
    path = config.results_dir / "infrastructure_failures.jsonl"
    _append_jsonl_record(path, payload)
    print(
        f"INFRASTRUCTURE_FAILURE_RECORD {classification} {run_id}",
        flush=True,
    )


def resolve_setup_commands(args: argparse.Namespace, task: TaskSpec) -> list[str]:
    """CLI override > per-task ``setup_commands`` from the CSV > defaults."""
    return list(args.setup_command or task.setup_commands or DEFAULT_SETUP_COMMANDS)


def resolve_role_model_names(
    config: Config, args: argparse.Namespace
) -> dict[str, str]:
    """Config defaults overridden by repeatable ``--role-model ROLE=MODEL``."""
    names = {
        role: value
        for role in ROLE_NAMES
        if (value := getattr(config, f"role_model_{role}", None))
    }
    for raw in getattr(args, "role_model", None) or []:
        if "=" not in raw:
            raise SystemExit(f"Invalid --role-model {raw!r}; expected ROLE=MODEL.")
        role, model_name = (part.strip() for part in raw.split("=", 1))
        if role not in ROLE_NAMES or not model_name:
            raise SystemExit(
                f"Invalid --role-model {raw!r}; role must be one of "
                f"{', '.join(ROLE_NAMES)}."
            )
        names[role] = model_name
    return names


def resolve_role_reasoning_efforts(
    config: Config, args: argparse.Namespace
) -> dict[str, ReasoningEffort]:
    """Only explicit role overrides; absent roles inherit the shared policy."""
    efforts = {
        role: value
        for role in ROLE_NAMES
        if (value := getattr(config, f"role_reasoning_effort_{role}", None))
    }
    for raw in getattr(args, "role_reasoning_effort", None) or []:
        if "=" not in raw:
            raise SystemExit(
                f"Invalid --role-reasoning-effort {raw!r}; expected ROLE=EFFORT."
            )
        role, effort = (part.strip() for part in raw.split("=", 1))
        if role not in ROLE_NAMES or effort not in REASONING_EFFORTS:
            raise SystemExit(
                f"Invalid --role-reasoning-effort {raw!r}; role must be one of "
                f"{', '.join(ROLE_NAMES)} and effort one of "
                f"{', '.join(REASONING_EFFORTS)}."
            )
        efforts[role] = effort  # type: ignore[assignment]
    return efforts


def build_role_models(
    config: Config,
    names: dict[str, str],
    reasoning_efforts: dict[str, ReasoningEffort] | None = None,
) -> dict[str, LangChainModel]:
    efforts = reasoning_efforts or {}
    return {
        role: build_model(
            config,
            config.provider_spec(model_name, efforts.get(role)),
        )
        for role, model_name in names.items()
    }


def costed_role_usage(raw: dict, models: list[LangChainModel]) -> dict:
    result: dict = json.loads(json.dumps(raw))
    totals: dict[str, dict[str, int]] = {}
    for model in models:
        usage = totals.setdefault(
            model.model,
            {"calls": 0, "input_tokens": 0, "output_tokens": 0},
        )
        usage["calls"] += model.calls
        usage["input_tokens"] += model.input_tokens
        usage["output_tokens"] += model.output_tokens

    attributed: dict[str, dict[str, int]] = {}
    for role_models in result.values():
        for model_name, usage in role_models.items():
            item = attributed.setdefault(
                model_name,
                {"calls": 0, "input_tokens": 0, "output_tokens": 0},
            )
            for key in item:
                item[key] += int(usage.get(key, 0))

    for model_name, total in totals.items():
        residual = {
            key: max(total[key] - attributed.get(model_name, {}).get(key, 0), 0)
            for key in total
        }
        if any(residual.values()):
            residual["total_tokens"] = (
                residual["input_tokens"] + residual["output_tokens"]
            )
            result.setdefault("system", {})[model_name] = residual

    for role, role_models in raw.items():
        for model_name in role_models:
            item = result[role][model_name]
            cost = estimate_cost(
                model_name,
                int(item.get("input_tokens", 0)),
                int(item.get("output_tokens", 0)),
            )
            if cost is not None:
                item["cost_usd"] = round(cost, 8)
    for model_name, item in result.get("system", {}).items():
        cost = estimate_cost(
            model_name,
            int(item.get("input_tokens", 0)),
            int(item.get("output_tokens", 0)),
        )
        if cost is not None:
            item["cost_usd"] = round(cost, 8)
    return result


def _core_agent_kwargs(
    config: Config,
    args: argparse.Namespace,
    model: LangChainModel,
    spec: ProviderSpec,
    task: TaskSpec,
) -> dict:
    """Shared construction kwargs for the built-in loop agents (single/multi)."""
    setup_commands: list[str] = []
    if not args.no_setup:
        setup_commands = resolve_setup_commands(args, task)
    return dict(
        model=model,
        docker_image=args.docker_image
        or task.docker_image
        or config.docker_image
        or "python:3.11-slim",
        docker_network_disabled=(args.no_network or config.docker_network_disabled),
        shell_timeout_seconds=config.shell_timeout_seconds,
        test_timeout_seconds=config.test_timeout_seconds,
        stop_container=not args.keep_container,
        context_window_tokens=args.context_window_tokens or spec.context_window_tokens,
        context_budget_tokens=args.context_budget_tokens
        or config.context_budget_tokens,
        max_response_tokens=config.max_tokens,
        compaction_mode=config.compaction_mode,
        action_transport=args.action_transport or config.agent_action_transport,
        setup_commands=setup_commands,
        max_cost_usd=args.max_cost_usd or config.max_cost_usd,
    )


def _single_research_kwargs(config: Config, args: argparse.Namespace) -> dict:
    """Opt-in surface for the strong single baseline only."""
    return dict(
        research_guard_enabled=(
            config.single_research_guard_enabled
            if getattr(args, "single_research_guard", None) is None
            else args.single_research_guard
        ),
        research_warning_steps=(
            getattr(args, "single_research_warning_steps", None)
            or config.single_research_warning_steps
        ),
        research_hard_limit=(
            getattr(args, "single_research_hard_limit", None)
            or config.single_research_hard_limit
        ),
        post_plan_research_warning_steps=(
            getattr(args, "single_post_plan_research_warning_steps", None)
            or config.single_post_plan_research_warning_steps
        ),
        post_plan_research_hard_limit=(
            getattr(args, "single_post_plan_research_hard_limit", None)
            or config.single_post_plan_research_hard_limit
        ),
        compatibility_guard_enabled=(
            config.single_compatibility_guard_enabled
            if getattr(args, "single_compatibility_guard", None) is None
            else args.single_compatibility_guard
        ),
        max_subtasks=(
            getattr(args, "max_subtasks", None)
            or config.single_decomposition_max_subtasks
        ),
        decomposition_implementation_warning_steps=(
            getattr(args, "decomposition_implementation_warning_steps", None)
            or config.single_decomposition_implementation_warning_steps
        ),
        decomposition_implementation_hard_limit=(
            getattr(args, "decomposition_implementation_hard_limit", None)
            or config.single_decomposition_implementation_hard_limit
        ),
        decomposition_verification_step_reserve=(
            getattr(args, "decomposition_verification_step_reserve", None)
            if getattr(args, "decomposition_verification_step_reserve", None)
            is not None
            else config.single_decomposition_verification_step_reserve
        ),
        max_repair_cycles=(
            getattr(args, "decomposition_max_repair_cycles", None)
            if getattr(args, "decomposition_max_repair_cycles", None) is not None
            else config.single_decomposition_max_repair_cycles
        ),
    )


def build_task_agent(
    config: Config,
    args: argparse.Namespace,
    model: LangChainModel,
    spec: ProviderSpec,
    task: TaskSpec,
) -> TaskAgent:
    """Dispatch on ``config.agent_mode``. All agents expose the same interface
    the runner needs (docker settings + ``run(state)``), so the downstream
    evaluation and metrics are identical across agents."""
    mode = config.agent_mode
    if mode in ("single", "single-decomposed"):
        if (
            resolve_role_model_names(config, args)
            or resolve_role_reasoning_efforts(config, args)
            or (
                args.developer_escalation_model
                or config.developer_escalation_model
                or getattr(args, "developer_escalation_reasoning_effort", None)
                or config.developer_escalation_reasoning_effort
            )
        ):
            raise SystemExit(
                "Role-model overrides require a built-in multi-agent mode."
            )
        agent_class = SingleAgent
        if mode == "single-decomposed":
            from ..agents.decomposition import DecomposedSingleAgent

            agent_class = DecomposedSingleAgent
        return agent_class(
            **_core_agent_kwargs(config, args, model, spec, task),
            **_single_research_kwargs(config, args),
        )
    if mode in (
        "multi",
        "multi-graph",
        "multi-orch",
        "multi-orch-guided",
        "multi-orch-guarded",
        "multi-swe",
    ):
        from ..agents.multi_agent import (
            GuardedOrchestratorAgent,
            GuidedOrchestratorAgent,
            MultiGraphAgent,
            MultiOrchestratorAgent,
            MultiSweAgent,
        )

        kwargs = _core_agent_kwargs(config, args, model, spec, task)
        role_model_names = resolve_role_model_names(config, args)
        role_reasoning_efforts = resolve_role_reasoning_efforts(config, args)
        escalation_model_name = (
            args.developer_escalation_model or config.developer_escalation_model
        )
        escalation_effort = (
            getattr(args, "developer_escalation_reasoning_effort", None)
            or config.developer_escalation_reasoning_effort
        )
        if escalation_effort and not escalation_model_name:
            raise SystemExit(
                "--developer-escalation-reasoning-effort requires "
                "--developer-escalation-model."
            )
        for role in role_reasoning_efforts:
            role_model_names.setdefault(role, model.model)
        kwargs.update(
            role_models=build_role_models(
                config, role_model_names, role_reasoning_efforts
            ),
            developer_escalation_model=(
                build_model(
                    config,
                    config.provider_spec(
                        escalation_model_name,
                        escalation_effort,
                    ),
                )
                if escalation_model_name
                else None
            ),
            developer_escalate_after_no_edit_episodes=(
                args.developer_escalate_after_no_edit_episodes
                or config.developer_escalate_after_no_edit_episodes
            ),
            developer_escalate_after_failed_tests=(
                args.developer_escalate_after_failed_tests
                or config.developer_escalate_after_failed_tests
            ),
        )
        if mode in ("multi", "multi-graph"):
            return MultiGraphAgent(**kwargs)
        if mode == "multi-swe":
            developer_model = kwargs["role_models"].get("developer", model)
            developer_spec = config.provider_spec(
                developer_model.model,
                role_reasoning_efforts.get("developer"),
            )
            return MultiSweAgent(
                **kwargs,
                developer_adapter=_build_swe_adapter(
                    config,
                    args,
                    developer_model,
                    developer_spec,
                    task,
                ),
            )
        if mode == "multi-orch-guided":
            return GuidedOrchestratorAgent(**kwargs)
        if mode == "multi-orch-guarded":
            return GuardedOrchestratorAgent(**kwargs)
        return MultiOrchestratorAgent(**kwargs)
    if mode == "swe-agent":
        return _build_swe_adapter(config, args, model, spec, task)
    raise SystemExit(f"Unknown agent mode: {mode!r}")


def _build_swe_adapter(
    config: Config,
    args: argparse.Namespace,
    model: LangChainModel,
    spec: ProviderSpec,
    task: TaskSpec,
) -> SweAgentAdapter:
    # Fast path (default): prebuilt base image + skip standalone build, with
    # the repo installed via the same setup commands single-agent uses, so
    # startup drops from minutes to seconds without changing the comparison.
    if getattr(args, "swe_no_fast", False):
        raise SystemExit(
            "--swe-no-fast is not available for comparable benchmark runs: "
            "the external default deployment image cannot be resolved and "
            "pinned before execution"
        )
    if getattr(args, "no_network", False) or config.docker_network_disabled:
        raise SystemExit(
            "offline editing cannot be promised for external SWE-agent/SWE-ReX. "
            "Use a built-in pseudo-SWE agent, or run a separately declared "
            "all-online ablation with DOCKER_NETWORK_DISABLED=false"
        )
    if spec.reasoning_effort is not None or spec.reasoning_max_tokens is not None:
        raise SystemExit(
            "explicit reasoning controls are not forwarded through the external "
            "SWE-agent adapter; omit them or use a built-in pseudo-SWE agent"
        )
    fast = True
    post_startup: list[str] = []
    if fast and not args.no_setup:
        post_startup = resolve_setup_commands(args, task)
    return SweAgentAdapter(
        model=model,
        api_key=spec.api_key,
        base_url=spec.base_url,
        docker_image=args.docker_image
        or task.docker_image
        or config.docker_image
        or "python:3.11-slim",
        docker_network_disabled=False,
        context_window_tokens=args.context_window_tokens or spec.context_window_tokens,
        max_response_tokens=config.max_tokens,
        env_image=MANAGED_BASE_IMAGE if fast else None,
        post_startup_commands=post_startup,
        extra_args=list(args.sweagent_arg or []),
    )


def main() -> int:
    args = parse_args()
    if args.sweagent_arg:
        raise SystemExit(
            "--sweagent-arg is not accepted for comparable runs; add a typed, "
            "fingerprinted adapter field instead"
        )
    config = load_config(args.provider)
    if args.agent:
        config.agent_mode = args.agent
    if args.session:
        config.session_id = args.session
    if args.action_transport:
        config.agent_action_transport = args.action_transport
    try:
        apply_reasoning_overrides(
            config,
            effort=args.reasoning_effort,
            max_tokens=args.reasoning_max_tokens,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    configure_langsmith(config)

    collection = load_collection(args.collection)
    if args.task_id not in collection:
        raise SystemExit(
            f"Unknown task_id '{args.task_id}'. "
            f"Available: {', '.join(sorted(collection))}"
        )
    task = collection[args.task_id]

    try:
        source_input_identity = source_repository_identity(
            task.repo_path,
            declared_base_commit=task.base_commit or None,
            require_clean=True,
            include_ignored_files=True,
        )
        validate_expected_input_contract(
            args,
            task,
            source_repository=source_input_identity,
        )
    except (RuntimeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    if args.run_id:
        try:
            run_id = str(uuid.UUID(args.run_id))
        except ValueError as exc:
            raise SystemExit("--run-id must be a UUID") from exc
    else:
        run_id = str(uuid.uuid4())
    repo_dir = (
        task.repo_path.resolve()
        if args.dry_run
        else prepare_workspace(task, config.workspaces_dir, run_id).resolve()
    )
    if not args.dry_run:
        # A task mirror may retain ignored output from an earlier local setup.
        # Every executable run begins from HEAD plus the two explicitly
        # preserved runner shims before any oracle/setup input is frozen.
        clean_workspace_repo(repo_dir, config.workspaces_dir)
        try:
            validate_expected_input_contract(
                args,
                task,
                source_repository=source_input_identity,
                hidden_path=repo_dir.parent / "hidden_tests",
            )
        except RuntimeError as exc:
            raise SystemExit(str(exc)) from exc
    test_command = args.test_command or task.visible_test_command
    scoring_commands = [
        test_command,
        *(suite.command for suite in task.hidden_suites()),
    ]
    frozen_oracle = repo_dir.parent / "pristine_oracle"
    pytest_bootstrap_environment = repo_dir.parent / "pytest_bootstrap"
    if not args.dry_run:
        try:
            freeze_test_oracle(
                repo_dir,
                frozen_oracle,
                test_commands=scoring_commands,
            )
            freeze_pytest_bootstrap_packages(
                repo_dir,
                pytest_bootstrap_environment,
            )
        except RuntimeError as exc:
            raise SystemExit(str(exc)) from exc

    spec = config.provider_spec(args.model)
    model = build_model(config, spec)
    agent = build_task_agent(config, args, model, spec, task)
    execution_adapter = swe_execution_adapter(agent)
    if execution_adapter is not None:
        execution_adapter.container_label_value = run_id
    try:
        validate_swe_installation_snapshot(
            agent,
            args,
            require_verifiable=not args.dry_run,
        )
        swe_execution_image = pin_swe_execution_image(
            agent,
            require_resolved=not args.dry_run,
            expected_image_id=args.expected_swe_env_image_id,
        )
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    requested_docker_image = agent.docker_image
    try:
        resolved_docker_image = docker_image_identity(
            requested_docker_image,
            require_resolved=not args.dry_run,
        )
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    expected_docker_image_id = args.expected_docker_image_id
    actual_docker_image_id = resolved_docker_image["image_id"]
    if (
        expected_docker_image_id is not None
        and actual_docker_image_id != expected_docker_image_id
    ):
        raise SystemExit(
            "Docker image changed after the sweep snapshot: "
            f"{requested_docker_image!r} resolved to {actual_docker_image_id!r}, "
            f"expected {expected_docker_image_id!r}"
        )
    if actual_docker_image_id is not None:
        # Execute against the immutable object resolved before any setup or
        # model call, even if another process retags the mutable reference.
        agent.docker_image = actual_docker_image_id
    usage_models = agent.usage_models() if hasattr(agent, "usage_models") else [model]
    if execution_adapter is not None:
        unpriced_external = [
            item.model
            for item in usage_models
            if item.input_cost_per_1m is None or item.output_cost_per_1m is None
        ]
        if unpriced_external:
            raise SystemExit(
                "external SWE-agent trajectories provide token counts rather "
                "than authoritative billing; comparable runs require static "
                "pricing for every configured model. Missing: "
                + ", ".join(sorted(set(unpriced_external)))
            )
    max_cost_usd = args.max_cost_usd or config.max_cost_usd
    if max_cost_usd is not None:
        if execution_adapter is not None:
            raise SystemExit(
                "--max-cost-usd cannot be enforced inside an external SWE-agent "
                "episode; use a built-in pseudo-SWE agent for a hard per-run cap"
            )
        unpriced = [
            item.model
            for item in usage_models
            if item.input_cost_per_1m is None or item.output_cost_per_1m is None
        ]
        if unpriced:
            raise SystemExit(
                "--max-cost-usd requires known pricing for every configured "
                f"model; missing: {', '.join(sorted(set(unpriced)))}"
            )

    state = State(
        task=task.task_file_path.read_text(encoding="utf-8"),
        workdir=repo_dir,
        test_command=test_command,
        max_iterations=args.max_iterations or config.max_iterations,
        max_steps=args.max_steps,
    )

    if args.dry_run:
        print(plan(task, repo_dir, test_command, agent, args))
        return 0

    scoring_setup = resolve_setup_commands(args, task)
    dependency_environment = repo_dir.parent / "scoring_dependencies"
    setup_artifact_environment = repo_dir.parent / "setup_artifacts"
    baseline = None
    clean_workspace_repo(repo_dir, config.workspaces_dir)
    try:
        prepare_scoring_dependencies(
            repo_dir,
            docker_image=agent.docker_image,
            network_disabled=agent.docker_network_disabled,
            setup_commands=scoring_setup,
            shell_timeout=config.shell_timeout_seconds,
            test_timeout=config.test_timeout_seconds,
            dependency_environment=dependency_environment,
            setup_artifact_environment=setup_artifact_environment,
        )
        clean_workspace_repo(repo_dir, config.workspaces_dir)
        post_setup_tampered = restore_test_oracle(
            repo_dir,
            pristine_repo=frozen_oracle,
            test_commands=scoring_commands,
        )
        if post_setup_tampered:
            raise RuntimeError(
                "pristine setup changed the frozen test oracle: "
                + ", ".join(post_setup_tampered[:20])
            )
        if not args.no_regression:
            baseline = collect_visible_passing(
                repo_dir,
                test_command,
                docker_image=agent.docker_image,
                network_disabled=agent.docker_network_disabled,
                test_timeout=config.test_timeout_seconds,
                dependency_environment=dependency_environment,
                setup_artifact_environment=setup_artifact_environment,
                pytest_bootstrap_environment=pytest_bootstrap_environment,
                pristine_repo=frozen_oracle,
                test_commands=scoring_commands,
            )
    except Exception as exc:
        record_infrastructure_failure(
            config, args, task, run_id, "preflight", exc, usage_models, state, 0.0
        )
        raise
    clean_workspace_repo(repo_dir, config.workspaces_dir)
    try:
        source_repository = source_repository_identity(
            repo_dir,
            declared_base_commit=task.base_commit or None,
            require_clean=True,
            include_ignored_files=True,
        )
    except (RuntimeError, ValueError) as exc:
        raise SystemExit(
            f"prepared workspace identity validation failed: {exc}"
        ) from exc
    frozen_evaluation_paths = {
        "hidden_fixture_tree": repo_dir.parent / "hidden_tests",
        "frozen_scoring_oracle_tree": frozen_oracle,
        "frozen_dependency_environment_tree": dependency_environment,
        "frozen_setup_artifact_tree": setup_artifact_environment,
        "frozen_pytest_bootstrap_tree": pytest_bootstrap_environment,
    }
    frozen_evaluation_trees = {
        name: filesystem_tree_identity(path)
        for name, path in frozen_evaluation_paths.items()
    }
    evaluation_inputs = {
        "hidden_fixture_tree": frozen_evaluation_trees["hidden_fixture_tree"],
        "hidden_suites": [
            {
                "name": suite.name,
                "command": suite.command,
                "required": suite.required,
                "reuse_visible_result": suite.reuse_visible_result,
            }
            for suite in task.hidden_suites()
        ],
        "hidden_scoring_enabled": not args.no_score,
        "regression_check_enabled": not args.no_regression,
        "evaluation_setup_commands": scoring_setup,
        "frozen_scoring_oracle_tree": frozen_evaluation_trees[
            "frozen_scoring_oracle_tree"
        ],
        "frozen_dependency_environment_tree": frozen_evaluation_trees[
            "frozen_dependency_environment_tree"
        ],
        "frozen_setup_artifact_tree": frozen_evaluation_trees[
            "frozen_setup_artifact_tree"
        ],
        "frozen_pytest_bootstrap_tree": frozen_evaluation_trees[
            "frozen_pytest_bootstrap_tree"
        ],
    }

    meta = load_task_meta(task.task_file_path)
    runtime_settings = agent_settings(agent)
    action_transport = runtime_settings["action_transport"]
    configured_routes = configured_agent_model_routes(agent)
    configured_model_policy = agent_model_policy(agent)
    configured_model_policy["reasoning_policy"] = reasoning_policy_label(
        configured_routes
    )
    configured_model_policy["routes"] = configured_routes
    configured_external_adapter = external_adapter_policy_snapshot(agent)
    policy_configuration = {
        "agent_mode": config.agent_mode,
        "agent_class": type(agent).__name__,
        "runtime_settings": {
            **runtime_settings,
            "action_transport": getattr(agent, "action_transport", "external"),
        },
        "compaction_mode": config.compaction_mode,
        "provider_call_settings": {
            "temperature": config.temperature,
            "request_timeout_seconds": config.request_timeout_seconds,
            "max_retries": config.max_retries,
        },
        "loop_limits": {
            "max_repeated_actions": getattr(agent, "max_repeated_actions", None),
            "max_parse_failures": getattr(agent, "max_parse_failures", None),
            "shell_timeout_seconds": getattr(agent, "shell_timeout_seconds", None),
            "test_timeout_seconds": getattr(agent, "test_timeout_seconds", None),
        },
        "external_adapter": configured_external_adapter,
        "model_policy": configured_model_policy,
        "max_steps": state.max_steps,
        "max_iterations": state.max_iterations,
        "test_command": test_command,
        "setup_commands": list(getattr(agent, "setup_commands", [])),
        "docker_image": requested_docker_image,
        "docker_image_identity": resolved_docker_image,
        "swe_execution_image_identity": swe_execution_image,
        **agent_network_policy(agent),
        "evaluation_inputs": evaluation_inputs,
    }
    # Freeze the configured protocol before the first model-controlled action.
    # Runtime observations (actual routes invoked, outcomes, usage) remain
    # separate metrics and cannot retroactively alter this fingerprint.
    reproducibility = build_run_reproducibility(
        task_content=state.task,
        agent=agent,
        policy_configuration=policy_configuration,
        model_routes=configured_routes,
        source_repository=source_repository,
    )
    frozen_kernel_sha256 = reproducibility["policy_kernel"]["sources"][
        "aggregate_sha256"
    ]
    started = time.perf_counter()
    try:
        final_state = agent.run(
            state,
            langsmith_extra=run_trace_extra(
                task_id=task.task_id,
                agent_mode=config.agent_mode,
                repo_dir=str(repo_dir),
                model_name=model.model,
                run_id=run_id,
                complexity=meta.get("complexity") or task.size,
                settings={
                    "session_id": config.session_id,
                    "compaction_mode": config.compaction_mode,
                    **runtime_settings,
                    "reasoning_effort": config.reasoning_effort,
                    "reasoning_max_tokens": config.reasoning_max_tokens,
                    "max_steps": state.max_steps,
                    "max_iterations": state.max_iterations,
                    "docker_image": agent.docker_image,
                    "visible_test_command": test_command,
                    "size": task.size,
                    "task_type": task.task_type,
                    "model_policy": configured_model_policy,
                },
            ),
        )
    except Exception as exc:
        record_infrastructure_failure(
            config,
            args,
            task,
            run_id,
            "agent",
            exc,
            usage_models,
            state,
            time.perf_counter() - started,
        )
        raise
    duration_s = time.perf_counter() - started
    current_kernel_sha256 = policy_kernel_source_manifest()["aggregate_sha256"]
    if current_kernel_sha256 != frozen_kernel_sha256:
        error = RuntimeError(
            "benchmark policy kernel changed while the agent was running; "
            "discarding the scientifically ambiguous result"
        )
        record_infrastructure_failure(
            config,
            args,
            task,
            run_id,
            "harness_drift",
            error,
            usage_models,
            final_state,
            duration_s,
        )
        raise error
    if (
        configured_external_adapter is not None
        and external_adapter_policy_snapshot(agent) != configured_external_adapter
    ):
        error = RuntimeError(
            "external SWE Python runtime, dependency closure, or invocation "
            "configuration changed while the agent was running; discarding "
            "the scientifically ambiguous result"
        )
        record_infrastructure_failure(
            config,
            args,
            task,
            run_id,
            "external_adapter_drift",
            error,
            usage_models,
            final_state,
            duration_s,
        )
        raise error
    print(final_state.to_string())

    # Capture and replay the agent's diff now, before evaluation overlays
    # hidden tests. Scoring only HEAD + this archived patch prevents ignored
    # build/dependency artifacts from becoming unrecorded solution state.
    patch_path = repo_dir.parent / "agent.patch"
    try:
        setup_artifact_conflicts = write_agent_patch(
            repo_dir,
            patch_path,
            expected_head=str(source_repository["head_commit"]),
            excluded_paths=frozen_setup_artifact_files(setup_artifact_environment),
            frozen_artifact_root=setup_artifact_environment,
        )
        rebuild_scoring_checkout(repo_dir, patch_path, config.workspaces_dir)
    except Exception as exc:
        record_infrastructure_failure(
            config,
            args,
            task,
            run_id,
            "patch_replay",
            exc,
            usage_models,
            final_state,
            duration_s,
        )
        raise

    metrics = run_metrics(usage_models, final_state, duration_s)
    metrics["provider"] = config.model_provider
    metrics["requested_action_transport"] = action_transport
    metrics["action_transport"] = observed_action_transport(agent)
    if config.reasoning_effort is not None:
        metrics["reasoning_effort"] = config.reasoning_effort
    if config.reasoning_max_tokens is not None:
        metrics["reasoning_max_tokens"] = config.reasoning_max_tokens
    fallback_calls = sum(item.fallback_calls for item in usage_models)
    if fallback_calls:
        metrics["tools_fallback_calls"] = fallback_calls
    if observed_transport_downgraded(agent):
        metrics["transport_downgraded"] = True
    empty_retries = sum(item.empty_retries for item in usage_models)
    if empty_retries:
        metrics["no_action_retries"] = empty_retries
    model_routes = agent_model_routes(agent, role_steps=final_state.role_steps)
    model_policy = agent_model_policy(agent)
    model_policy["reasoning_policy"] = reasoning_policy_label(model_routes)
    model_policy["routes"] = model_routes
    metrics["model_policy"] = model_policy
    metrics["model_routes"] = model_routes
    metrics["research_guard"] = {
        "enabled": runtime_settings["research_guard_enabled"],
        "warning_steps": runtime_settings["research_warning_steps"],
        "hard_limit": runtime_settings["research_hard_limit"],
        "post_plan_warning_steps": runtime_settings["post_plan_research_warning_steps"],
        "post_plan_hard_limit": runtime_settings["post_plan_research_hard_limit"],
    }
    metrics["compatibility_guard"] = {
        "enabled": runtime_settings["compatibility_guard_enabled"],
        "passed": final_state.compatibility_check_passed,
        "command": final_state.last_compatibility_command,
    }
    metrics["decomposition"] = {
        "enabled": runtime_settings["decomposition_enabled"],
        "subtask_count": len(final_state.subtasks),
        "completed": sum(item.status == "completed" for item in final_state.subtasks),
        "active_subtask": final_state.active_subtask_id,
        "repair_cycles": final_state.repair_cycles,
        "max_repair_cycles": final_state.max_repair_cycles,
        "workspace_revision": final_state.workspace_revision,
    }
    if max_cost_usd is not None:
        metrics["max_cost_usd"] = max_cost_usd
    if hasattr(agent, "role_usage"):
        metrics["role_usage"] = costed_role_usage(agent.role_usage(), usage_models)
    if hasattr(agent, "role_transports"):
        metrics["role_transports"] = agent.role_transports()
    developer_escalations = getattr(agent, "developer_escalations", 0)
    if developer_escalations:
        metrics["developer_escalations"] = developer_escalations
    if final_state.role_steps:
        metrics["role_steps"] = final_state.role_steps
    metrics["workspace"] = str(repo_dir)
    metrics["task_type"] = task.task_type
    metrics["docker_image"] = requested_docker_image
    metrics["docker_image_id"] = actual_docker_image_id
    metrics["docker_image_identity"] = resolved_docker_image
    if swe_execution_image is not None:
        metrics["swe_execution_image_identity"] = swe_execution_image
    metrics.update(agent_network_policy(agent))
    metrics["reproducibility"] = reproducibility
    metrics["run_fingerprint"] = reproducibility["run_fingerprint"]
    if args.experiment_fingerprint:
        metrics["experiment_fingerprint"] = args.experiment_fingerprint
    if args.task_set_id:
        metrics["task_set_id"] = args.task_set_id
    if args.campaign_id:
        metrics["campaign_id"] = args.campaign_id
    try:
        validate_frozen_evaluation_trees(
            frozen_evaluation_trees,
            frozen_evaluation_paths,
        )
        eval_result = evaluate_and_report(
            task,
            repo_dir.parent,
            test_command,
            baseline,
            config,
            args,
            agent,
            run_id,
            setup_artifact_conflicts=setup_artifact_conflicts,
        )
        validate_frozen_evaluation_trees(
            frozen_evaluation_trees,
            frozen_evaluation_paths,
        )
    except Exception as exc:
        record_infrastructure_failure(
            config,
            args,
            task,
            run_id,
            "evaluation",
            exc,
            usage_models,
            final_state,
            duration_s,
        )
        raise
    if final_state.test_oracle_tamper_attempts:
        # The executor restores protected tests immediately, before final
        # evaluation can observe the diff. Preserve the attempted violation in
        # the benchmark outcome rather than treating the clean rollback as if
        # no attempt occurred.
        eval_result.test_oracle_tampered = True
        if (
            eval_result.visible_passed is not None
            and eval_result.hidden_passed is not None
        ):
            eval_result.task_success = compute_task_success(
                eval_result.visible_passed,
                eval_result.hidden_passed,
                eval_result.regressions,
                test_oracle_tampered=True,
                setup_artifact_tampered=eval_result.setup_artifact_tampered,
            )
        elif eval_result.task_success is not None:
            eval_result.task_success = False
        else:
            # --no-score deliberately leaves scientific success unknown, but
            # an observed policy violation must still fail the process rather
            # than falling back to the agent's self-reported solved status.
            metrics["policy_failure"] = "test_oracle_tampering"
    if eval_result.setup_artifact_tampered and eval_result.task_success is None:
        metrics["policy_failure"] = "setup_artifact_tampering"
    if config.langsmith_tracing_enabled and eval_result.task_success is not None:
        try:
            attach_run_feedback(
                run_id,
                "task_success",
                1.0 if eval_result.task_success else 0.0,
            )
        except Exception as exc:  # noqa: BLE001 - feedback is best-effort
            print(f"(could not attach LangSmith task-success feedback: {exc})")
    metrics.update(eval_metrics(eval_result))
    if eval_result.regressions is not None:
        metrics["regressions"] = eval_result.regressions

    try:
        quality_score = None
        if args.enable_review:
            quality_score = review_solution(task, repo_dir, config, args, run_id)

        metrics["run_record_complete"] = True
        record = record_run_result(
            config,
            task_id=task.task_id,
            run_id=run_id,
            model_name=model.model,
            final_state=final_state,
            test_passed=eval_result.visible_passed,
            hidden_tests_passed=eval_result.hidden_passed,
            extra=metrics,
            append=False,
        )
        metrics_path = write_run_metrics(record, config.results_dir, quality_score)
        if policy_kernel_source_manifest()["aggregate_sha256"] != frozen_kernel_sha256:
            raise RuntimeError(
                "benchmark policy kernel changed during evaluation/post-processing; "
                "discarding the scientifically ambiguous result"
            )
        if external_adapter_policy_snapshot(agent) != configured_external_adapter:
            raise RuntimeError(
                "external SWE-agent installation/configuration changed during "
                "the run; discarding mixed provenance"
            )
        validate_expected_input_contract(args, task)
        exit_code = benchmark_exit_code(final_state, eval_result)
        print(f"Metrics written to {metrics_path}")
        if config.langsmith_tracing_enabled:
            print(f"\nLangSmith run_id: {run_id}")
        # Commit the scientific row last. After this durable append, no
        # fallible post-processing remains that could turn a complete-looking
        # row into an infrastructure failure.
        _append_jsonl_record(config.results_dir / "runs.jsonl", record)
        return exit_code
    except Exception as exc:
        record_infrastructure_failure(
            config,
            args,
            task,
            run_id,
            "postprocessing",
            exc,
            usage_models,
            final_state,
            duration_s,
        )
        raise


def evaluate_and_report(
    task: TaskSpec,
    task_ws: Path,
    visible_command: str,
    baseline_passing: set[str] | None,
    config: Config,
    args: argparse.Namespace,
    agent: TaskAgent,
    run_id: str,
    *,
    setup_artifact_conflicts: list[str] | None = None,
) -> EvalResult:
    """Run the post-run evaluation (regressions + hidden tests) and push a
    LangSmith feedback score for the hidden verdict."""
    result = evaluate_solution(
        task,
        task_ws,
        visible_command,
        baseline_passing=baseline_passing,
        do_hidden=not args.no_score,
        docker_image=agent.docker_image,
        network_disabled=agent.docker_network_disabled,
        setup_commands=resolve_setup_commands(args, task),
        shell_timeout=config.shell_timeout_seconds,
        test_timeout=config.test_timeout_seconds,
        pristine_repo=task_ws / "pristine_oracle",
        dependency_environment=task_ws / "scoring_dependencies",
        setup_artifact_environment=task_ws / "setup_artifacts",
        pytest_bootstrap_environment=task_ws / "pytest_bootstrap",
        setup_artifact_conflicts=setup_artifact_conflicts,
    )
    if config.langsmith_tracing_enabled and result.hidden_passed is not None:
        try:
            attach_run_feedback(
                run_id, "hidden_tests_passed", 1.0 if result.hidden_passed else 0.0
            )
            for name, passed in result.hidden_suite_results.items():
                attach_run_feedback(run_id, f"{name}_passed", 1.0 if passed else 0.0)
        except Exception as exc:  # noqa: BLE001 - feedback is best-effort
            print(f"(could not attach LangSmith feedback: {exc})")
    return result


def eval_metrics(result: EvalResult) -> dict:
    """Run-record fields derived from post-run evaluation."""
    out: dict = {
        "excluded_agent_test_files": list(result.excluded_agent_test_files),
        "excluded_agent_test_file_count": len(result.excluded_agent_test_files),
        "setup_artifact_tampered": result.setup_artifact_tampered,
        "setup_artifact_conflicts": list(result.setup_artifact_conflicts),
    }
    if result.task_success is not None:
        out["task_success"] = result.task_success
    out["test_oracle_tampered"] = result.test_oracle_tampered
    if result.hidden_passed is not None:
        out["hidden_required_tests_passed"] = result.hidden_passed
    if result.hidden_semantic_passed is not None:
        out["hidden_semantic_tests_passed"] = result.hidden_semantic_passed
    if result.hidden_compat_passed is not None:
        out["hidden_compat_tests_passed"] = result.hidden_compat_passed
    if result.hidden_pr_parity_passed is not None:
        out["hidden_pr_parity_tests_passed"] = result.hidden_pr_parity_passed
    if result.hidden_suite_results:
        out["hidden_suite_results"] = dict(result.hidden_suite_results)
    return out


def benchmark_exit_code(final_state: State, result: EvalResult) -> int:
    """Keep unscored oracle tampering from becoming a successful process."""
    if result.task_success is not None:
        return 0 if result.task_success else 1
    if final_state.test_oracle_tamper_attempts:
        return 1
    if result.setup_artifact_tampered:
        return 1
    return 0 if final_state.status == "solved" else 1


def write_run_metrics(
    record: dict, results_dir: Path, quality_score: int | None = None
) -> Path:
    """Compute this run's metrics and write them to metrics/<run_id>.json."""
    run_record = RunRecord.from_dict(record)
    run_record.quality_score = quality_score
    payload = {
        "run_id": run_record.run_id,
        "task_id": run_record.task_id,
        "model": run_record.model,
        "status": run_record.status,
        "task_success": run_record.task_success,
        "hidden_tests_passed": run_record.hidden_tests_passed,
        "hidden_required_tests_passed": run_record.hidden_required_tests_passed,
        "hidden_semantic_tests_passed": run_record.hidden_semantic_tests_passed,
        "hidden_compat_tests_passed": run_record.hidden_compat_tests_passed,
        "hidden_pr_parity_tests_passed": run_record.hidden_pr_parity_tests_passed,
        "quality_score": run_record.quality_score,
        "metrics": compute_metrics([run_record]).as_dict(),
    }
    metrics_dir = results_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    path = metrics_dir / f"{run_record.run_id}.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def plan(
    task: TaskSpec,
    repo_dir: Path,
    test_command: str,
    agent: TaskAgent,
    args: argparse.Namespace,
) -> str:
    setup = "\n".join(f"    {c}" for c in agent.setup_commands) or "    (none)"
    settings = agent_settings(agent)
    network_policy = agent_network_policy(agent)
    editing_network = (
        "disabled"
        if network_policy["editing_network_disabled_after_setup"]
        else "not guaranteed"
    )
    evaluation_network = (
        "disabled"
        if network_policy["evaluation_network_disabled_after_setup"]
        else "enabled"
    )
    model_policy = agent_model_policy(agent)
    hidden = (
        "\n".join(
            f"    {suite.name}"
            f"{'' if suite.required else ' (optional)'}: {suite.command}"
            for suite in task.hidden_suites()
        )
        or "    (none)"
    )
    decomposition_details = (
        "  decomp impl:   "
        f"{settings['decomposition_implementation_warning_steps']}/"
        f"{settings['decomposition_implementation_hard_limit']}, reserve "
        f"{settings['decomposition_verification_step_reserve']}, repairs "
        f"{settings['max_repair_cycles']}\n"
        if settings["decomposition_enabled"]
        else ""
    )
    return (
        "Dry run — prepared, not executed:\n"
        f"  task_id:       {task.task_id} ({task.size})\n"
        f"  workspace:     {repo_dir}\n"
        f"  test command:  {test_command}\n"
        f"  edit network:  {editing_network}\n"
        f"  eval network:  {evaluation_network}\n"
        f"  docker image:  {agent.docker_image}\n"
        f"  action ACI:    {configured_action_transport(agent)} (requested)\n"
        f"  model policy:  {json.dumps(model_policy, ensure_ascii=False)}\n"
        f"  max cost USD:  {getattr(agent, 'max_cost_usd', None) or '(none)'}\n"
        f"  research guard:{'on' if settings['research_guard_enabled'] else 'off'} "
        f"({settings['research_warning_steps']}/{settings['research_hard_limit']}, "
        f"post-plan {settings['post_plan_research_warning_steps']}/"
        f"{settings['post_plan_research_hard_limit']})\n"
        "  compatibility: "
        f"{'on' if settings['compatibility_guard_enabled'] else 'off'}\n"
        "  decomposition: "
        f"{'on' if settings['decomposition_enabled'] else 'off'} "
        f"(max {settings['max_subtasks']})\n"
        f"{decomposition_details}"
        f"  setup:\n{setup}\n"
        f"  hidden score:  {'on' if not args.no_score else 'off'}\n"
        f"  regression:    {'on' if not args.no_regression else 'off'}\n"
        f"  quality review:{'on' if args.enable_review else 'off'}\n"
        f"  hidden tests:\n{hidden}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one benchmark task.")
    parser.add_argument("--task-id", default="h11_pr_181")
    parser.add_argument("--collection", type=Path, default=DEFAULT_COLLECTION)
    parser.add_argument("--provider", choices=["local", "openrouter"], default=None)
    parser.add_argument(
        "--agent",
        choices=[
            "single",
            "single-decomposed",
            "swe-agent",
            "multi",
            "multi-graph",
            "multi-orch",
            "multi-orch-guided",
            "multi-orch-guarded",
            "multi-swe",
        ],
        default=None,
        help="Which agent solves the task (default: config.agent_mode). "
        "swe-agent runs the external SWE-agent scaffold; `multi` is the "
        "legacy alias for the deterministic multi-graph architecture.",
    )
    parser.add_argument(
        "--sweagent-arg",
        action="append",
        default=None,
        help="Extra flag passed through to `sweagent run` (repeatable); "
        "use to match your installed SWE-agent version.",
    )
    parser.add_argument(
        "--swe-no-fast",
        action="store_true",
        help="Disable the swe-agent fast path (prebuilt image + skipped "
        "standalone build); fall back to SWE-agent's slow default startup.",
    )
    parser.add_argument(
        "--session",
        default=None,
        help="Experiment/session name recorded with the run; group or filter by "
        "it in the report (`--session NAME` / `--list-sessions`). "
        "New name = clean slate without deleting history.",
    )
    parser.add_argument("--model", default=None)
    parser.add_argument(
        "--role-model",
        action="append",
        default=None,
        metavar="ROLE=MODEL",
        help=(
            "Override a built-in multi-agent role model; repeat for multiple "
            "roles. Without this flag every role uses --model (legacy mode)."
        ),
    )
    parser.add_argument(
        "--role-reasoning-effort",
        action="append",
        default=None,
        metavar="ROLE=EFFORT",
        help=(
            "Explicit role-only reasoning override; repeat for multiple roles. "
            "Without it every role shares --reasoning-effort."
        ),
    )
    parser.add_argument(
        "--developer-escalation-model",
        default=None,
        help=(
            "Optional stronger developer model used after a no-edit episode "
            "or failed test threshold."
        ),
    )
    parser.add_argument(
        "--developer-escalation-reasoning-effort",
        choices=REASONING_EFFORTS,
        default=None,
    )
    parser.add_argument(
        "--developer-escalate-after-no-edit-episodes",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--developer-escalate-after-failed-tests",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--max-cost-usd",
        type=float,
        default=None,
        help=(
            "Stop the agent when known model-token cost reaches this limit. "
            "Multi-agent checks occur between role episodes."
        ),
    )
    parser.add_argument(
        "--context-budget-tokens",
        type=int,
        default=None,
        help="Working history budget for the compactor, in tokens (defaults "
        "to CONTEXT_BUDGET_TOKENS, else the model window). Too big: reasoning "
        "models burn the completion budget re-thinking a huge history; too "
        "small: the agent loses findings to summarization and re-explores.",
    )
    parser.add_argument(
        "--single-research-guard",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Enable the strong single-agent read-only progress guard. Use "
            "--no-single-research-guard to reproduce the legacy unbounded "
            "single loop. Multi-agent role quotas are unaffected."
        ),
    )
    parser.add_argument("--single-research-warning-steps", type=int, default=None)
    parser.add_argument("--single-research-hard-limit", type=int, default=None)
    parser.add_argument(
        "--single-post-plan-research-warning-steps", type=int, default=None
    )
    parser.add_argument(
        "--single-post-plan-research-hard-limit", type=int, default=None
    )
    parser.add_argument(
        "--single-compatibility-guard",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Require a focused passing legacy-behavior probe before single "
            "can finish. Use --no-single-compatibility-guard to reproduce "
            "the earlier baseline. Multi-agent modes are unaffected."
        ),
    )
    parser.add_argument(
        "--max-subtasks",
        type=int,
        choices=range(4, 13),
        default=None,
        help="Maximum total work units for single-decomposed (4-12).",
    )
    parser.add_argument(
        "--decomposition-implementation-warning-steps", type=int, default=None
    )
    parser.add_argument(
        "--decomposition-implementation-hard-limit", type=int, default=None
    )
    parser.add_argument(
        "--decomposition-verification-step-reserve", type=int, default=None
    )
    parser.add_argument("--decomposition-max-repair-cycles", type=int, default=None)
    parser.add_argument(
        "--reasoning-effort",
        choices=REASONING_EFFORTS,
        default=None,
        help="Reasoning budget for reasoning models via OpenRouter. Hidden "
        "reasoning shares max_tokens with the answer; on "
        "long contexts an uncapped model can burn the whole budget and return "
        "empty/truncated actions. Applies to both transports.",
    )
    parser.add_argument(
        "--reasoning-max-tokens",
        type=int,
        default=None,
        help=(
            "Exact hidden-reasoning token budget for an OpenRouter route with "
            "an explicitly verified frozen profile; mutually exclusive with "
            "--reasoning-effort. Unknown, local, and translated-only routes "
            "fail closed."
        ),
    )
    parser.add_argument(
        "--action-transport",
        choices=["text_json", "tools", "auto"],
        default=None,
        help="Action interface for built-in agents. text_json is the JSON-text "
        "protocol; tools uses native tool calls with the history replayed in "
        "tool protocol; auto probes the endpoint once and picks tools only if "
        "the probe succeeds. An explicit tools request that fails the probe "
        "downgrades to text_json (recorded as transport_downgraded).",
    )
    parser.add_argument(
        "--test-command",
        default=None,
        help="Override the task's visible test command.",
    )
    parser.add_argument(
        "--setup-command",
        action="append",
        default=None,
        help="Setup command to run before the agent (repeatable).",
    )
    parser.add_argument(
        "--no-setup",
        action="store_true",
        help="Skip dependency installation.",
    )
    parser.add_argument(
        "--no-network",
        action="store_true",
        help=(
            "Require the agent/test container to be offline after dependency "
            "setup. DOCKER_NETWORK_DISABLED applies the same policy by default."
        ),
    )
    parser.add_argument("--docker-image", default=None)
    parser.add_argument("--context-window-tokens", type=int, default=None)
    parser.add_argument("--max-iterations", type=int, default=None)
    # One full multi-agent pass has soft role quotas totalling 46 steps. Use a
    # common 50-step budget for every architecture so the default can complete
    # one pass while comparisons still share the same hard cap.
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--keep-container", action="store_true")
    parser.add_argument(
        "--experiment-fingerprint", default=None, help=argparse.SUPPRESS
    )
    parser.add_argument("--task-set-id", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--campaign-id", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--run-id", default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        "--expected-docker-image-id",
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--expected-swe-env-image-id",
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--expected-source-worktree-sha256", default=None, help=argparse.SUPPRESS
    )
    parser.add_argument("--expected-task-sha256", default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        "--expected-hidden-tree-sha256", default=None, help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--expected-collection-sha256", default=None, help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--expected-harness-tree-sha256", default=None, help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--expected-policy-kernel-sha256", default=None, help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--expected-pricing-sha256", default=None, help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--expected-sweagent-distribution-sha256",
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--expected-swerex-distribution-sha256",
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--expected-swe-runtime-sha256",
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--no-score",
        action="store_true",
        help="Skip running the hidden tests after the agent finishes.",
    )
    parser.add_argument(
        "--no-regression",
        action="store_true",
        help="Skip the before/after visible-test regression check.",
    )
    parser.add_argument(
        "--enable-review",
        action="store_true",
        help="Score the solution's quality (0-5) with a reviewer model.",
    )
    parser.add_argument(
        "--review-model",
        default=DEFAULT_REVIEW_MODEL,
        help=f"Reviewer model for --enable-review (default: {DEFAULT_REVIEW_MODEL}).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Prepare the workspace and print the plan without running.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
