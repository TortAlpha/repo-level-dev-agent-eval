"""CLI: run one benchmark task end to end.

Copies the pristine repo into a throwaway workspace, runs the agent against the
task's own visible test command, evaluates the solution (regressions + hidden
tests), optionally reviews its quality, and records everything. The originals
under ``repositories/`` are never modified.
"""

from __future__ import annotations

import argparse
import json
import time
import uuid
from pathlib import Path

from ..agents.model import LangChainModel
from ..agents.single_agent import SingleAgent
from ..agents.state import State
from ..agents.tracing import attach_run_feedback, configure_langsmith, run_trace_extra
from ..config import Config, ProviderSpec, load_config
from ..metrics.compute import compute_metrics
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
from .workspace import prepare_workspace, write_agent_patch


def build_task_agent(
    config: Config,
    args: argparse.Namespace,
    model: LangChainModel,
    spec: ProviderSpec,
) -> SingleAgent:
    setup_commands: list[str] = []
    if not args.no_setup:
        setup_commands = args.setup_command or list(DEFAULT_SETUP_COMMANDS)
    return SingleAgent(
        model=model,
        docker_image=args.docker_image or config.docker_image or "python:3.11-slim",
        docker_network_disabled=args.no_network,
        shell_timeout_seconds=config.shell_timeout_seconds,
        test_timeout_seconds=config.test_timeout_seconds,
        stop_container=not args.keep_container,
        context_window_tokens=args.context_window_tokens or spec.context_window_tokens,
        max_response_tokens=config.max_tokens,
        compaction_mode=config.compaction_mode,
        setup_commands=setup_commands,
    )


def main() -> int:
    args = parse_args()
    config = load_config(args.provider)
    configure_langsmith(config)

    collection = load_collection(args.collection)
    if args.task_id not in collection:
        raise SystemExit(
            f"Unknown task_id '{args.task_id}'. "
            f"Available: {', '.join(sorted(collection))}"
        )
    task = collection[args.task_id]

    run_id = str(uuid.uuid4())
    repo_dir = prepare_workspace(task, config.workspaces_dir, run_id).resolve()
    test_command = args.test_command or task.visible_test_command

    spec = config.provider_spec(args.model)
    model = build_model(config, spec)
    agent = build_task_agent(config, args, model, spec)

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

    scoring_setup = args.setup_command or list(DEFAULT_SETUP_COMMANDS)
    baseline = None
    if not args.no_regression:
        baseline = collect_visible_passing(
            repo_dir,
            test_command,
            docker_image=agent.docker_image,
            network_disabled=agent.docker_network_disabled,
            setup_commands=scoring_setup,
            shell_timeout=config.shell_timeout_seconds,
            test_timeout=config.test_timeout_seconds,
        )

    meta = load_task_meta(task.task_file_path)
    started = time.perf_counter()
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
                "compaction_mode": config.compaction_mode,
                "context_window_tokens": agent.context_window_tokens,
                "max_response_tokens": agent.max_response_tokens,
                "max_steps": state.max_steps,
                "max_iterations": state.max_iterations,
                "docker_image": agent.docker_image,
                "visible_test_command": test_command,
                "size": task.size,
            },
        ),
    )
    duration_s = time.perf_counter() - started
    print(final_state.to_string())

    # Capture the agent's diff now, before the evaluation overlays hidden tests.
    write_agent_patch(repo_dir, repo_dir.parent / "agent.patch")

    metrics = run_metrics(model, final_state, duration_s)
    metrics["workspace"] = str(repo_dir)
    eval_result = evaluate_and_report(
        task, repo_dir.parent, test_command, baseline, config, args, agent, run_id
    )
    if eval_result.regressions is not None:
        metrics["regressions"] = eval_result.regressions

    record = record_run_result(
        config,
        task_id=task.task_id,
        run_id=run_id,
        model_name=model.model,
        final_state=final_state,
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
    return 0 if final_state.status == "solved" else 1


def evaluate_and_report(
    task: TaskSpec,
    task_ws: Path,
    visible_command: str,
    baseline_passing: set[str] | None,
    config: Config,
    args: argparse.Namespace,
    agent: SingleAgent,
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
        setup_commands=args.setup_command or list(DEFAULT_SETUP_COMMANDS),
        shell_timeout=config.shell_timeout_seconds,
        test_timeout=config.test_timeout_seconds,
    )
    if config.langsmith_tracing_enabled and result.hidden_passed is not None:
        try:
            attach_run_feedback(
                run_id, "hidden_tests_passed", 1.0 if result.hidden_passed else 0.0
            )
        except Exception as exc:  # noqa: BLE001 - feedback is best-effort
            print(f"(could not attach LangSmith feedback: {exc})")
    return result


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
        "hidden_tests_passed": run_record.hidden_tests_passed,
        "quality_score": run_record.quality_score,
        "metrics": compute_metrics([run_record]).as_dict(),
    }
    metrics_dir = results_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    path = metrics_dir / f"{run_record.run_id}.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def plan(task: TaskSpec, repo_dir: Path, test_command: str, agent: SingleAgent,
         args: argparse.Namespace) -> str:
    setup = "\n".join(f"    {c}" for c in agent.setup_commands) or "    (none)"
    network = "disabled" if agent.docker_network_disabled else "enabled"
    return (
        "Dry run — prepared, not executed:\n"
        f"  task_id:       {task.task_id} ({task.size})\n"
        f"  workspace:     {repo_dir}\n"
        f"  test command:  {test_command}\n"
        f"  network:       {network}\n"
        f"  docker image:  {agent.docker_image}\n"
        f"  setup:\n{setup}\n"
        f"  hidden score:  {'on' if not args.no_score else 'off'}\n"
        f"  regression:    {'on' if not args.no_regression else 'off'}\n"
        f"  quality review:{'on' if args.enable_review else 'off'}\n"
        f"  hidden tests:  {task.hidden_test_command or '(none)'}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one benchmark task.")
    parser.add_argument("--task-id", default="h11_pr_181")
    parser.add_argument("--collection", type=Path, default=DEFAULT_COLLECTION)
    parser.add_argument("--provider", choices=["local", "openrouter"], default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument(
        "--test-command", default=None,
        help="Override the task's visible test command.",
    )
    parser.add_argument(
        "--setup-command", action="append", default=None,
        help="Setup command to run before the agent (repeatable).",
    )
    parser.add_argument(
        "--no-setup", action="store_true", help="Skip dependency installation.",
    )
    parser.add_argument(
        "--no-network", action="store_true",
        help="Disable network in the sandbox (setup will fail without it).",
    )
    parser.add_argument("--docker-image", default=None)
    parser.add_argument("--context-window-tokens", type=int, default=None)
    parser.add_argument("--max-iterations", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=25)
    parser.add_argument("--keep-container", action="store_true")
    parser.add_argument(
        "--no-score", action="store_true",
        help="Skip running the hidden tests after the agent finishes.",
    )
    parser.add_argument(
        "--no-regression", action="store_true",
        help="Skip the before/after visible-test regression check.",
    )
    parser.add_argument(
        "--enable-review", action="store_true",
        help="Score the solution's quality (0-5) with a reviewer model.",
    )
    parser.add_argument(
        "--review-model", default=DEFAULT_REVIEW_MODEL,
        help=f"Reviewer model for --enable-review (default: {DEFAULT_REVIEW_MODEL}).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Prepare the workspace and print the plan without running.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
