"""CLI: run the single-agent prototype on one task file + repo directory."""

from __future__ import annotations

import argparse
import time
import uuid
from pathlib import Path

from ..agents.prompts import RUN_SUMMARY_PROMPT
from ..agents.state import State
from ..agents.tracing import configure_langsmith, run_trace_extra
from ..config import PROVIDER_CONFIGS, load_config
from .models import build_agent, build_model
from .results import load_task_meta, record_run_result, run_metrics

DEFAULT_TEST_COMMAND = "python -m unittest discover -s tests"


def main() -> int:
    args = parse_args()
    config = load_config(args.provider)
    configure_langsmith(config)

    task_file = args.task_file.resolve()
    repo_dir = args.repo_dir.resolve()

    spec = config.provider_spec(args.model)
    model = build_model(config, spec)
    agent = build_agent(config, args, model, spec)

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
                "max_steps": state.max_steps,
                "max_iterations": state.max_iterations,
                "docker_image": agent.docker_image,
            },
        ),
    )
    duration_s = time.perf_counter() - started
    print(final_state.to_string())

    metrics = run_metrics(model, final_state, duration_s)

    summary: str | None = None
    if args.summarize == "enabled":
        summary = model.summarize(RUN_SUMMARY_PROMPT, final_state.to_string())
        print("\n=== Run summary ===")
        print(summary)

    record_run_result(
        config,
        task_id=task_id,
        run_id=run_id,
        model_name=model.model,
        final_state=final_state,
        summary=summary,
        extra=metrics,
    )
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
    parser.add_argument(
        "--summarize",
        choices=["enabled", "disabled"],
        default="disabled",
        help="When enabled, call the model at the end to summarize what was done.",
    )
    parser.add_argument("--max-iterations", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=25)
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
