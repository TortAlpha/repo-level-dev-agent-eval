"""CLI: run the single-agent prototype on one task file + repo directory."""

from __future__ import annotations

import argparse
import time
import uuid
from pathlib import Path

from ..agents.prompts import RUN_SUMMARY_PROMPT
from ..agents.state import State
from ..agents.tracing import configure_langsmith, run_trace_extra
from ..config import (
    PROVIDER_CONFIGS,
    REASONING_EFFORTS,
    apply_reasoning_overrides,
    load_config,
)
from .models import build_agent, build_model
from .reproducibility import (
    build_run_reproducibility,
    docker_image_identity,
    policy_kernel_source_manifest,
    source_repository_identity,
    text_sha256,
)
from .results import (
    _append_jsonl_record,
    load_task_meta,
    record_run_result,
    run_metrics,
)

DEFAULT_TEST_COMMAND = "python -m unittest discover -s tests"


def main() -> int:
    args = parse_args()
    config = load_config(args.provider)
    try:
        apply_reasoning_overrides(
            config,
            effort=args.reasoning_effort,
            max_tokens=args.reasoning_max_tokens,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    configure_langsmith(config)

    task_file = args.task_file.resolve()
    repo_dir = args.repo_dir.resolve()
    try:
        source_repository = source_repository_identity(
            repo_dir,
            include_ignored_files=True,
        )
    except (RuntimeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    spec = config.provider_spec(args.model)
    model = build_model(config, spec)
    agent = build_agent(config, args, model, spec)
    requested_docker_image = agent.docker_image
    try:
        resolved_docker_image = docker_image_identity(
            requested_docker_image,
            require_resolved=True,
        )
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    agent.docker_image = str(resolved_docker_image["image_id"])
    if agent.max_cost_usd is not None and (
        model.input_cost_per_1m is None or model.output_cost_per_1m is None
    ):
        raise SystemExit(
            "--max-cost-usd requires a model with known pricing in static/pricing.csv"
        )

    state = State(
        task=task_file.read_text(encoding="utf-8"),
        workdir=repo_dir,
        test_command=args.test_command,
        max_iterations=args.max_iterations or config.max_iterations,
        max_steps=args.max_steps,
    )

    task_id = task_file.stem if task_file.stem != "task" else task_file.parent.name
    run_id = str(uuid.uuid4())
    meta = load_task_meta(task_file)

    configured_model_routes = {
        "agent": {
            **model.route_metadata(transport=f"configured:{agent.action_transport}"),
            "agent_context_window_tokens": agent.context_window_tokens,
            "context_budget_tokens": agent.context_budget_tokens,
            "max_response_tokens": agent.max_response_tokens,
        }
    }
    policy_configuration = {
        "agent_mode": config.agent_mode,
        "agent_class": type(agent).__name__,
        "compaction_mode": config.compaction_mode,
        "action_transport": agent.action_transport,
        "provider_call_settings": {
            "temperature": config.temperature,
            "request_timeout_seconds": config.request_timeout_seconds,
            "max_retries": config.max_retries,
        },
        "loop_limits": {
            "max_repeated_actions": agent.max_repeated_actions,
            "max_parse_failures": agent.max_parse_failures,
            "shell_timeout_seconds": agent.shell_timeout_seconds,
            "test_timeout_seconds": agent.test_timeout_seconds,
        },
        "max_steps": state.max_steps,
        "max_iterations": state.max_iterations,
        "test_command": state.test_command,
        "docker_image": requested_docker_image,
        "docker_image_identity": resolved_docker_image,
        "network_disabled_after_setup": agent.docker_network_disabled,
        "post_run_summary": {
            "enabled": args.summarize == "enabled",
            "accounting_scope": "included_evaluator_overhead",
            "system_prompt_sha256": (
                text_sha256(RUN_SUMMARY_PROMPT) if args.summarize == "enabled" else None
            ),
            "system_prompt_chars": (
                len(RUN_SUMMARY_PROMPT) if args.summarize == "enabled" else 0
            ),
        },
    }
    reproducibility = build_run_reproducibility(
        task_content=state.task,
        agent=agent,
        policy_configuration=policy_configuration,
        model_routes=configured_model_routes,
        source_repository=source_repository,
    )
    frozen_kernel_sha256 = reproducibility["policy_kernel"]["sources"][
        "aggregate_sha256"
    ]

    # Freeze provenance and begin timing before transport probing or any other
    # provider call. Dry/configuration metadata above is strictly local.
    started = time.perf_counter()
    final_state = agent.run(
        state,
        langsmith_extra=run_trace_extra(
            task_id=task_id,
            agent_mode=config.agent_mode,
            repo_dir=str(repo_dir),
            model_name=model.model,
            run_id=run_id,
            complexity=meta.get("complexity"),
            settings={
                "compaction_mode": config.compaction_mode,
                "context_window_tokens": agent.context_window_tokens,
                "max_response_tokens": agent.max_response_tokens,
                "action_transport": agent.action_transport,
                "reasoning_effort": spec.reasoning_effort,
                "reasoning_max_tokens": spec.reasoning_max_tokens,
                "max_steps": state.max_steps,
                "max_iterations": state.max_iterations,
                "docker_image": agent.docker_image,
            },
        ),
    )
    print(final_state.to_string())

    summary: str | None = None
    if args.summarize == "enabled":
        # This optional reporting call is evaluator overhead rather than an
        # agent action. It is nevertheless included in run usage/cost below and
        # fingerprinted explicitly so it can never disappear from billing.
        summary = model.summarize(RUN_SUMMARY_PROMPT, final_state.to_string())
        print("\n=== Run summary ===")
        print(summary)

    duration_s = time.perf_counter() - started
    cached_transport = getattr(agent, "_transport_cache", None)
    observed_transport = (
        str(cached_transport[0])
        if isinstance(cached_transport, tuple) and cached_transport
        else str(agent.action_transport)
    )
    transport_downgraded = bool(
        isinstance(cached_transport, tuple)
        and len(cached_transport) > 1
        and cached_transport[1]
    )

    metrics = run_metrics(model, final_state, duration_s)
    metrics["provider"] = config.model_provider
    metrics["requested_action_transport"] = agent.action_transport
    metrics["action_transport"] = observed_transport
    if transport_downgraded:
        metrics["transport_downgraded"] = True
    metrics["docker_image"] = requested_docker_image
    metrics["docker_image_id"] = resolved_docker_image["image_id"]
    metrics["docker_image_identity"] = resolved_docker_image
    model_routes = {
        "agent": {
            **model.route_metadata(transport=observed_transport),
            "agent_context_window_tokens": agent.context_window_tokens,
            "context_budget_tokens": agent.context_budget_tokens,
            "max_response_tokens": agent.max_response_tokens,
        }
    }
    metrics["model_routes"] = model_routes
    metrics["reproducibility"] = reproducibility
    metrics["run_fingerprint"] = reproducibility["run_fingerprint"]
    if config.reasoning_effort is not None:
        metrics["reasoning_effort"] = config.reasoning_effort
    if config.reasoning_max_tokens is not None:
        metrics["reasoning_max_tokens"] = config.reasoning_max_tokens
    if args.summarize == "enabled":
        metrics["post_run_summary"] = {
            "enabled": True,
            "accounting_scope": "included_evaluator_overhead",
            "system_prompt_sha256": text_sha256(RUN_SUMMARY_PROMPT),
        }

    record = record_run_result(
        config,
        task_id=task_id,
        run_id=run_id,
        model_name=model.model,
        final_state=final_state,
        summary=summary,
        extra=metrics,
        append=False,
    )
    if policy_kernel_source_manifest()["aggregate_sha256"] != frozen_kernel_sha256:
        raise RuntimeError(
            "agent policy kernel changed during the run; result was not committed"
        )
    _append_jsonl_record(config.results_dir / "runs.jsonl", record)
    if config.langsmith_tracing_enabled:
        print(f"\nLangSmith run_id: {run_id}")
    return 0 if final_state.status == "solved" else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the single-agent prototype.")
    parser.add_argument("--task-file", type=Path, default=None, required=True)
    parser.add_argument("--repo-dir", type=Path, default=None, required=True)
    parser.add_argument("--test-command", default=DEFAULT_TEST_COMMAND)
    parser.add_argument(
        "--provider",
        choices=list(PROVIDER_CONFIGS),
        default=None,
        help="Model provider to use. Overrides MODEL_PROVIDER from .env.",
    )
    parser.add_argument("--model", default=None)
    parser.add_argument("--reasoning-effort", choices=REASONING_EFFORTS, default=None)
    parser.add_argument(
        "--reasoning-max-tokens",
        type=int,
        default=None,
        help=(
            "Exact reasoning-token budget; requires an OpenRouter route whose "
            "frozen profile explicitly verifies end-to-end support."
        ),
    )
    parser.add_argument("--max-cost-usd", type=float, default=None)
    parser.add_argument(
        "--action-transport",
        choices=["text_json", "tools", "auto"],
        default=None,
        help="How the single-agent obtains actions from the model. "
        "text_json keeps the legacy JSON-text protocol; tools uses model "
        "tool calls; auto enables tools only for known tool-capable families.",
    )
    parser.add_argument(
        "--summarize",
        choices=["enabled", "disabled"],
        default="disabled",
        help="When enabled, call the model at the end to summarize what was done.",
    )
    parser.add_argument("--max-iterations", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--docker-image", default=None)
    parser.add_argument("--context-window-tokens", type=int, default=None)
    parser.add_argument(
        "--docker-network",
        action="store_true",
        help="Enable network access in the Docker sandbox.",
    )
    parser.add_argument(
        "--keep-container",
        action="store_true",
        help="Leave the Docker container running after the agent stops.",
    )
    return parser.parse_args()
