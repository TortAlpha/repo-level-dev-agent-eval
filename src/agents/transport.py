from __future__ import annotations

from typing import Literal

ActionTransport = Literal["text_json", "tools", "auto"]


def resolve_action_transport(
    requested: ActionTransport,
    model_name: str,
) -> Literal["text_json", "tools"]:
    """Resolve an action transport without surprising non-tool models.

    ``auto`` is intentionally conservative. Text JSON remains the fallback for
    unknown and known non-tool-friendly models; tool calling is enabled only for
    model families where the provider path usually supports it.
    """
    if requested != "auto":
        return requested

    normalized = model_name.lower()
    if normalized.startswith("z-ai/") or "glm" in normalized:
        return "text_json"
    if normalized.startswith("openai/") or normalized.startswith("anthropic/"):
        return "tools"
    if "gpt-" in normalized or "claude" in normalized:
        return "tools"
    return "text_json"
