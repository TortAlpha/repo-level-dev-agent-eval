"""The ``make_traced`` mechanism: ``traceable`` plus dynamic span naming.

LangSmith's static ``name`` tells you which method ran; it does not tell you
what happened. ``make_traced`` keeps the static name as a fallback but lets a
span rename and tag itself from its own arguments (``on_enter``) and result
(``on_exit``), so a trace reads as "execute edit_file: src/pricing.py" or
"run_tests: pytest [FAILED]" instead of "ActionExecutor.execute". The traced
methods stay untouched — all the wiring lives in this package.
"""

from __future__ import annotations

import functools
import inspect
import logging
from typing import Any, Callable

from langsmith import traceable

try:
    from langsmith import get_current_run_tree
except ImportError:  # pragma: no cover - older langsmith layout
    from langsmith.run_helpers import get_current_run_tree

logger = logging.getLogger(__name__)

LANGSMITH_PROJECT_NAME = "repo-level-dev-agent-eval"

# A span describer: mutates the live run from its arguments or its result.
RunHook = Callable[[Any, Any], None]


def annotate(
    run: Any,
    *,
    name: str | None = None,
    tags: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Set the span's name and append tags/metadata on the live run tree."""
    if name is not None:
        run.name = name
    if tags:
        run.tags = [*(run.tags or []), *tags]
    if metadata:
        run.metadata.update(metadata)


def make_traced(
    *,
    name: str,
    run_type: str,
    process_inputs: Callable[[dict], dict] | None = None,
    process_outputs: Callable[[Any], dict] | None = None,
    on_enter: RunHook | None = None,
    on_exit: RunHook | None = None,
):
    """Build a tracing decorator, like ``traceable`` but with dynamic naming.

    ``on_enter(run, arguments)`` runs before the body with the bound call
    arguments, so even a failing call is named after what it attempted;
    ``on_exit(run, result)`` runs after a successful body to refine the span
    from its result. With neither hook this is exactly ``traceable``.
    """
    base = traceable(
        name=name,
        run_type=run_type,
        project_name=LANGSMITH_PROJECT_NAME,
        process_inputs=process_inputs,
        process_outputs=process_outputs,
    )
    if on_enter is None and on_exit is None:
        return base

    def decorator(func):
        signature = inspect.signature(func)

        @functools.wraps(func)
        def inner(*args, **kwargs):
            run = get_current_run_tree()
            if run is not None and on_enter is not None:
                _run_hook(on_enter, run, _bind(signature, args, kwargs))
            result = func(*args, **kwargs)
            if run is not None and on_exit is not None:
                _run_hook(on_exit, run, result)
            return result

        return base(inner)

    return decorator


def _run_hook(hook: RunHook, run: Any, payload: Any) -> None:
    try:
        hook(run, payload)
    except Exception:  # noqa: BLE001 - annotation must never break the run
        logger.debug("trace annotation hook failed", exc_info=True)


def _bind(signature: inspect.Signature, args: tuple, kwargs: dict) -> dict:
    bound = signature.bind(*args, **kwargs)
    bound.apply_defaults()
    return bound.arguments
