from __future__ import annotations

from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from .model import LangChainModel

ActionTransport = Literal["text_json", "tools", "auto"]


def resolve_action_transport(
    requested: ActionTransport,
    model: LangChainModel,
) -> tuple[Literal["text_json", "tools"], bool]:
    """Resolve the transport with a live capability probe.

    Model-name heuristics lie twice over: metadata claims support the backend
    doesn't honor, and OpenRouter routes each request to a different backend.
    So ``auto`` and ``tools`` fire one cheap probe request instead:

    - ``text_json``: no probe, use text.
    - ``auto``: probe; use tools when it works, else text.
    - ``tools``: probe; on failure *downgrade* to text rather than burning the
      whole run on a broken endpoint.

    Returns ``(transport, downgraded)`` where ``downgraded`` is True only for
    an explicit ``tools`` request that the endpoint failed.
    """
    if requested == "text_json":
        return "text_json", False
    supported = model.probe_tools()
    if supported:
        return "tools", False
    return "text_json", requested == "tools"
