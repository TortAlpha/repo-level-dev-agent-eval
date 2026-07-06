"""Sweep runner: run tasks x models x agents sequentially into one session.

Each combination is a fresh ``python -m src.benchmark.runner`` subprocess (full
isolation, same as a single console launch), run one at a time so Docker /
SWE-ReX aren't contended. Used by the web console's "full benchmark" / multi-model
launch. Results land in ``runs.jsonl`` tagged with ``--session``.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .collection import DEFAULT_COLLECTION, load_collection


def _split(value: str) -> list[str]:
    return [item.strip() for item in (value or "").split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep benchmark runs across tasks x models x agents into a session."
    )
    parser.add_argument("--tasks", default="all", help="Comma-separated task_ids, or 'all'.")
    parser.add_argument("--models", default="", help="Comma-separated models ('' = provider default).")
    parser.add_argument("--agents", default="single", help="Comma-separated agents: single,swe-agent,multi.")
    parser.add_argument("--session", default=None, help="Session name recorded with every run.")
    parser.add_argument("--provider", default=None, choices=["local", "openrouter"])
    parser.add_argument(
        "--action-transport",
        choices=["text_json", "tools", "auto"],
        default=None,
        help="Forwarded to each built-in agent run.",
    )
    parser.add_argument("--collection", type=Path, default=DEFAULT_COLLECTION)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--max-iterations", type=int, default=None)
    parser.add_argument("--enable-review", action="store_true")
    parser.add_argument("--no-score", action="store_true")
    parser.add_argument("--no-regression", action="store_true")
    parser.add_argument("--no-network", action="store_true")
    parser.add_argument("--no-setup", action="store_true")
    parser.add_argument(
        "--concurrency", type=int, default=1,
        help="Run this many combinations at once (default 1 = sequential). Each "
        "run is model-latency/Docker bound, so 2-4 speeds a sweep up a lot; "
        "swe-agent is heavier (a container per run), so keep it modest.",
    )
    parser.add_argument(
        "--stop-on-error", action="store_true",
        help="Abort the sweep on the first failing run (sequential mode only).",
    )
    return parser.parse_args()


def _combos(args: argparse.Namespace) -> list[tuple[str, str, str]]:
    collection = load_collection(args.collection)
    all_tasks = list(collection)
    tasks = all_tasks if args.tasks.strip() == "all" else _split(args.tasks)
    unknown = [t for t in tasks if t not in collection]
    if unknown:
        raise SystemExit(f"unknown task_id(s): {', '.join(unknown)}")
    models = _split(args.models) or [""]  # "" -> provider default
    agents = _split(args.agents) or ["single"]
    # agent-major, then model, then task: keeps same-agent runs adjacent.
    return [(t, m, a) for a in agents for m in models for t in tasks]


def _run_command(task: str, model: str, agent: str, args: argparse.Namespace) -> list[str]:
    cmd = [sys.executable, "-m", "src.benchmark.runner", "--task-id", task, "--agent", agent]
    if args.provider:
        cmd += ["--provider", args.provider]
    if model:
        cmd += ["--model", model]
    if args.session:
        cmd += ["--session", args.session]
    if args.action_transport:
        cmd += ["--action-transport", args.action_transport]
    if args.max_steps:
        cmd += ["--max-steps", str(args.max_steps)]
    if args.max_iterations:
        cmd += ["--max-iterations", str(args.max_iterations)]
    for flag, on in (
        ("--enable-review", args.enable_review),
        ("--no-score", args.no_score),
        ("--no-regression", args.no_regression),
        ("--no-network", args.no_network),
        ("--no-setup", args.no_setup),
    ):
        if on:
            cmd.append(flag)
    return cmd


def _run_one(
    index: int, total: int, task: str, model: str, agent: str,
    args: argparse.Namespace, capture: bool,
) -> int:
    header = f"[{index}/{total}] {agent} · {model or 'default'} · {task}"
    cmd = _run_command(task, model, agent, args)
    if capture:
        # Parallel mode: collect the whole run and print it as one block so
        # concurrent runs' output doesn't interleave into noise.
        proc = subprocess.run(cmd, capture_output=True, text=True)
        body = ((proc.stdout or "") + (proc.stderr or "")).rstrip()
        print(f"\n===== {header} (rc={proc.returncode}) =====\n{body}", flush=True)
        return proc.returncode
    print(f"\n===== {header} =====", flush=True)
    return subprocess.run(cmd).returncode


def main() -> int:
    args = parse_args()
    combos = _combos(args)
    total = len(combos)
    concurrency = max(1, args.concurrency)
    print(
        f"Sweep: {total} run(s), concurrency={concurrency} — "
        f"session={args.session or 'default'} [{args.agents}] x [{args.models or 'default'}]",
        flush=True,
    )
    unsolved = 0
    aborted = False
    if concurrency == 1:
        for index, (task, model, agent) in enumerate(combos, start=1):
            if _run_one(index, total, task, model, agent, args, capture=False) != 0:
                unsolved += 1
                if args.stop_on_error:
                    print("Stopping sweep (--stop-on-error).", flush=True)
                    aborted = True
                    break
    else:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = [
                pool.submit(_run_one, index, total, task, model, agent, args, True)
                for index, (task, model, agent) in enumerate(combos, start=1)
            ]
            for future in as_completed(futures):
                if future.result() != 0:
                    unsolved += 1
    # A completed sweep succeeds even if some runs didn't solve — the per-run
    # solved/unsolved outcome is data in runs.jsonl, not a sweep failure. Only a
    # --stop-on-error abort (incomplete sweep) is a real failure.
    solved = total - unsolved
    print(
        f"\nSweep {'aborted' if aborted else 'finished'}: {solved}/{total} runs "
        f"solved ({unsolved} unsolved/errored). See runs.jsonl / the report.",
        flush=True,
    )
    return 1 if aborted else 0


if __name__ == "__main__":
    raise SystemExit(main())
