"""Prompt texts for the agents.

The prompt bodies live as plain-text files under ``static/prompts/`` so they
can be read and edited without touching Python. This module loads them once at
import time and exposes them under the same names the rest of the code imports.
"""

from __future__ import annotations

from pathlib import Path

PROMPTS_DIR = Path(__file__).resolve().parents[2] / "static" / "prompts"


def load_prompt(name: str) -> str:
    """Read a prompt file from ``static/prompts`` (``.txt`` extension implied)."""
    return (PROMPTS_DIR / f"{name}.txt").read_text(encoding="utf-8").strip()


# Instructs the single agent which action to take next (see actions.py).
SINGLE_AGENT_PROMPT = load_prompt("single_agent")
DECOMPOSED_SINGLE_AGENT_PROMPT = (
    SINGLE_AGENT_PROMPT + "\n\n" + load_prompt("decomposed_single_agent")
)

# Compresses dropped history into a factual digest; takes a {max_chars} slot.
CONTEXT_SUMMARY_PROMPT = load_prompt("context_summary")

# Produces the end-of-run report emitted when --summarize is enabled.
RUN_SUMMARY_PROMPT = load_prompt("run_summary")
