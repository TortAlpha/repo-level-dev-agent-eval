"""CLI: run one benchmark task end to end.

Copies the pristine repo into a throwaway workspace, runs the agent against the
task's own visible test command, evaluates the solution (regressions + hidden
tests), optionally reviews its quality, and records everything. The originals
under ``repositories/`` are never modified.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
import uuid
from pathlib import Path

from ..agents.model import LangChainModel
from ..agents.single_agent import SingleAgent
from ..agents.state import State
from ..agents.swe_agent import MANAGED_BASE_IMAGE, SweAgentAdapter
from ..agents.tracing import attach_run_feedback, configure_langsmith, run_trace_extra
from ..config import Config, ProviderSpec, load_config
from ..metrics.compute import compute_metrics
from ..metrics.pricing import estimate_cost
from ..metrics.records import RunRecord
from ..run import build_model, load_task_meta, record_run_result, run_metrics
from .collection import DEFAULT_COLLECTION, TaskSpec, load_collection
from .evaluation import (
    DEFAULT_SETUP_COMMANDS,
    EvalResult,
    collect_visible_passing,
    evaluate_solution,
)
from .review import DEFAULT_REVIEW_MODEL, review_solution
from .workspace import clean_workspace_repo, prepare_workspace, write_agent_patch

TaskAgent = SingleAgent | SweAgentAdapter
ROLE_NAMES = ("orchestrator", "planner", "developer", "tester", "reviewer")


def docker_image_id(image: str) -> str | None:
    result = subprocess.run(
        ["docker", "image", "inspect", image, "--format", "{{.Id}}"],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    return result.stdout.strip() or None


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
        for marker in ("openrouter", "rate limit", "server_error", "bad gateway", "api")
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
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
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


def build_role_models(
    config: Config, names: dict[str, str]
) -> dict[str, LangChainModel]:
    return {
        role: build_model(config, config.provider_spec(model_name))
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
        if resolve_role_model_names(config, args) or (
            args.developer_escalation_model or config.developer_escalation_model
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
        kwargs.update(
            role_models=build_role_models(config, role_model_names),
            developer_escalation_model=(
                build_model(
                    config,
                    config.provider_spec(
                        args.developer_escalation_model
                        or config.developer_escalation_model
                    ),
                )
                if (
                    args.developer_escalation_model or config.developer_escalation_model
                )
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
            return MultiSweAgent(
                **kwargs,
                developer_adapter=_build_swe_adapter(config, args, model, spec, task),
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
    fast = not args.swe_no_fast
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
        docker_network_disabled=(args.no_network or config.docker_network_disabled),
        context_window_tokens=args.context_window_tokens or spec.context_window_tokens,
        max_response_tokens=config.max_tokens,
        env_image=MANAGED_BASE_IMAGE if fast else None,
        post_startup_commands=post_startup,
        extra_args=list(args.sweagent_arg or []),
    )


def main() -> int:
    args = parse_args()
    config = load_config(args.provider)
    if args.agent:
        config.agent_mode = args.agent
    if args.session:
        config.session_id = args.session
    if args.action_transport:
        config.agent_action_transport = args.action_transport
    if args.reasoning_effort:
        config.reasoning_effort = args.reasoning_effort
    configure_langsmith(config)

    collection = load_collection(args.collection)
    if args.task_id not in collection:
        raise SystemExit(
            f"Unknown task_id '{args.task_id}'. "
            f"Available: {', '.join(sorted(collection))}"
        )
    task = collection[args.task_id]

    run_id = str(uuid.uuid4())
    repo_dir = (
        task.repo_path.resolve()
        if args.dry_run
        else prepare_workspace(task, config.workspaces_dir, run_id).resolve()
    )
    test_command = args.test_command or task.visible_test_command

    spec = config.provider_spec(args.model)
    model = build_model(config, spec)
    agent = build_task_agent(config, args, model, spec, task)
    usage_models = agent.usage_models() if hasattr(agent, "usage_models") else [model]
    max_cost_usd = args.max_cost_usd or config.max_cost_usd
    if max_cost_usd is not None:
        if isinstance(agent, SweAgentAdapter):
            raise SystemExit(
                "--max-cost-usd is enforced only by the built-in agents; "
                "use SWE-agent's own per-instance cost limit for --agent swe-agent."
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
    baseline = None
    if not args.no_regression:
        try:
            baseline = collect_visible_passing(
                repo_dir,
                test_command,
                docker_image=agent.docker_image,
                network_disabled=agent.docker_network_disabled,
                setup_commands=scoring_setup,
                shell_timeout=config.shell_timeout_seconds,
                test_timeout=config.test_timeout_seconds,
            )
        except Exception as exc:
            record_infrastructure_failure(
                config, args, task, run_id, "preflight", exc, usage_models, state, 0.0
            )
            raise
    clean_workspace_repo(repo_dir, config.workspaces_dir)

    meta = load_task_meta(task.task_file_path)
    action_transport = getattr(agent, "resolved_action_transport", "external")
    started = time.perf_counter()
    model_policy = (
        agent.model_policy_metadata()
        if hasattr(agent, "model_policy_metadata")
        else {"type": "single", "base_model": model.model}
    )
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
                    "context_window_tokens": agent.context_window_tokens,
                    "context_budget_tokens": getattr(
                        agent, "context_budget_tokens", None
                    ),
                    "max_response_tokens": agent.max_response_tokens,
                    "action_transport": action_transport,
                    "reasoning_effort": config.reasoning_effort,
                    "max_steps": state.max_steps,
                    "max_iterations": state.max_iterations,
                    "docker_image": agent.docker_image,
                    "visible_test_command": test_command,
                    "size": task.size,
                    "task_type": task.task_type,
                    "model_policy": model_policy,
                    "max_cost_usd": max_cost_usd,
                    "research_guard_enabled": agent.research_guard_enabled,
                    "research_warning_steps": agent.research_warning_steps,
                    "research_hard_limit": agent.research_hard_limit,
                    "post_plan_research_warning_steps": (
                        agent.post_plan_research_warning_steps
                    ),
                    "post_plan_research_hard_limit": (
                        agent.post_plan_research_hard_limit
                    ),
                    "compatibility_guard_enabled": agent.compatibility_guard_enabled,
                    "decomposition_enabled": agent.decomposition_enabled,
                    "max_subtasks": agent.max_subtasks,
                    "decomposition_implementation_warning_steps": (
                        agent.decomposition_implementation_warning_steps
                    ),
                    "decomposition_implementation_hard_limit": (
                        agent.decomposition_implementation_hard_limit
                    ),
                    "decomposition_verification_step_reserve": (
                        agent.decomposition_verification_step_reserve
                    ),
                    "max_repair_cycles": agent.max_repair_cycles,
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
    print(final_state.to_string())

    # Capture the agent's diff now, before the evaluation overlays hidden tests.
    write_agent_patch(repo_dir, repo_dir.parent / "agent.patch")

    metrics = run_metrics(usage_models, final_state, duration_s)
    metrics["provider"] = config.model_provider
    metrics["action_transport"] = action_transport
    if config.reasoning_effort:
        metrics["reasoning_effort"] = config.reasoning_effort
    fallback_calls = sum(item.fallback_calls for item in usage_models)
    if fallback_calls:
        metrics["tools_fallback_calls"] = fallback_calls
    if getattr(agent, "transport_downgraded", False):
        metrics["transport_downgraded"] = True
    empty_retries = sum(item.empty_retries for item in usage_models)
    if empty_retries:
        metrics["no_action_retries"] = empty_retries
    metrics["model_policy"] = model_policy
    metrics["research_guard"] = {
        "enabled": agent.research_guard_enabled,
        "warning_steps": agent.research_warning_steps,
        "hard_limit": agent.research_hard_limit,
        "post_plan_warning_steps": agent.post_plan_research_warning_steps,
        "post_plan_hard_limit": agent.post_plan_research_hard_limit,
    }
    metrics["compatibility_guard"] = {
        "enabled": agent.compatibility_guard_enabled,
        "passed": final_state.compatibility_check_passed,
        "command": final_state.last_compatibility_command,
    }
    metrics["decomposition"] = {
        "enabled": agent.decomposition_enabled,
        "subtask_count": len(final_state.subtasks),
        "completed": sum(
            item.status == "completed" for item in final_state.subtasks
        ),
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
    if getattr(agent, "developer_escalations", 0):
        metrics["developer_escalations"] = agent.developer_escalations
    if final_state.role_steps:
        metrics["role_steps"] = final_state.role_steps
    metrics["workspace"] = str(repo_dir)
    metrics["task_type"] = task.task_type
    metrics["docker_image"] = agent.docker_image
    metrics["docker_image_id"] = docker_image_id(agent.docker_image)
    metrics["agent_network_disabled_after_setup"] = agent.docker_network_disabled
    if args.experiment_fingerprint:
        metrics["experiment_fingerprint"] = args.experiment_fingerprint
    if args.task_set_id:
        metrics["task_set_id"] = args.task_set_id
    if args.campaign_id:
        metrics["campaign_id"] = args.campaign_id
    try:
        eval_result = evaluate_and_report(
            task, repo_dir.parent, test_command, baseline, config, args, agent, run_id
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
    metrics.update(eval_metrics(eval_result))
    if eval_result.regressions is not None:
        metrics["regressions"] = eval_result.regressions

    record = record_run_result(
        config,
        task_id=task.task_id,
        run_id=run_id,
        model_name=model.model,
        final_state=final_state,
        test_passed=eval_result.visible_passed,
        hidden_tests_passed=eval_result.hidden_passed,
        extra=metrics,
    )

    quality_score = None
    if args.enable_review:
        quality_score = review_solution(task, repo_dir, config, args, run_id)

    metrics_path = write_run_metrics(record, config.results_dir, quality_score)
    print(f"Metrics written to {metrics_path}")
    if config.langsmith_tracing_enabled:
        print(f"\nLangSmith run_id: {run_id}")
    if eval_result.task_success is not None:
        return 0 if eval_result.task_success else 1
    return 0 if final_state.status == "solved" else 1


def evaluate_and_report(
    task: TaskSpec,
    task_ws: Path,
    visible_command: str,
    baseline_passing: set[str] | None,
    config: Config,
    args: argparse.Namespace,
    agent: TaskAgent,
    run_id: str,
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
    )
    if config.langsmith_tracing_enabled and result.hidden_passed is not None:
        try:
            attach_run_feedback(
                run_id, "hidden_tests_passed", 1.0 if result.hidden_passed else 0.0
            )
            if result.task_success is not None:
                attach_run_feedback(
                    run_id, "task_success", 1.0 if result.task_success else 0.0
                )
            for name, passed in result.hidden_suite_results.items():
                attach_run_feedback(run_id, f"{name}_passed", 1.0 if passed else 0.0)
        except Exception as exc:  # noqa: BLE001 - feedback is best-effort
            print(f"(could not attach LangSmith feedback: {exc})")
    return result


def eval_metrics(result: EvalResult) -> dict:
    """Run-record fields derived from post-run evaluation."""
    out: dict = {}
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
    agent: SingleAgent,
    args: argparse.Namespace,
) -> str:
    setup = "\n".join(f"    {c}" for c in agent.setup_commands) or "    (none)"
    network = "disabled" if agent.docker_network_disabled else "enabled"
    model_policy = (
        agent.model_policy_metadata()
        if hasattr(agent, "model_policy_metadata")
        else {"type": "single", "base_model": agent.model.model}
    )
    hidden = (
        "\n".join(
            f"    {suite.name}{'' if suite.required else ' (optional)'}: {suite.command}"
            for suite in task.hidden_suites()
        )
        or "    (none)"
    )
    decomposition_details = (
        f"  decomp impl:   {agent.decomposition_implementation_warning_steps}/"
        f"{agent.decomposition_implementation_hard_limit}, reserve "
        f"{agent.decomposition_verification_step_reserve}, repairs "
        f"{agent.max_repair_cycles}\n"
        if agent.decomposition_enabled
        else ""
    )
    return (
        "Dry run — prepared, not executed:\n"
        f"  task_id:       {task.task_id} ({task.size})\n"
        f"  workspace:     {repo_dir}\n"
        f"  test command:  {test_command}\n"
        f"  network:       {network}\n"
        f"  docker image:  {agent.docker_image}\n"
        f"  action ACI:    {getattr(agent, 'resolved_action_transport', 'external')}\n"
        f"  model policy:  {json.dumps(model_policy, ensure_ascii=False)}\n"
        f"  max cost USD:  {getattr(agent, 'max_cost_usd', None) or '(none)'}\n"
        f"  research guard:{'on' if agent.research_guard_enabled else 'off'} "
        f"({agent.research_warning_steps}/{agent.research_hard_limit}, "
        f"post-plan {agent.post_plan_research_warning_steps}/"
        f"{agent.post_plan_research_hard_limit})\n"
        f"  compatibility: {'on' if agent.compatibility_guard_enabled else 'off'}\n"
        f"  decomposition: {'on' if agent.decomposition_enabled else 'off'} "
        f"(max {agent.max_subtasks})\n"
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
        "--developer-escalation-model",
        default=None,
        help=(
            "Optional stronger developer model used after a no-edit episode "
            "or failed test threshold."
        ),
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
    parser.add_argument(
        "--decomposition-max-repair-cycles", type=int, default=None
    )
    parser.add_argument(
        "--reasoning-effort",
        default=None,
        help="Reasoning budget for reasoning models via OpenRouter (e.g. low/"
        "medium/high). Hidden reasoning shares max_tokens with the answer; on "
        "long contexts an uncapped model can burn the whole budget and return "
        "empty/truncated actions. Applies to both transports.",
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
    parser.add_argument("--experiment-fingerprint", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--task-set-id", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--campaign-id", default=None, help=argparse.SUPPRESS)
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
