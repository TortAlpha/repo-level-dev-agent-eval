"""Single-agent runner: model/agent construction, run recording, and CLI.

- ``models``  — build the chat model and the agent.
- ``results`` — record the outcome to runs.jsonl and derive per-run metrics.
- ``cli``     — the ``python -m src.run`` entry point.
"""

from .cli import main
from .models import build_agent, build_chat_model, build_model
from .results import load_task_meta, record_run_result, run_metrics

__all__ = [
    "main",
    "build_model",
    "build_agent",
    "build_chat_model",
    "run_metrics",
    "load_task_meta",
    "record_run_result",
]
