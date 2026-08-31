"""One entry in the agent's action history."""

import json
from copy import deepcopy
from typing import Any

from pydantic import BaseModel, field_validator


class ContextEntry(BaseModel):
    step: int
    kind: str
    path: str | None = None
    text: str
    action_json: str | None = None
    # Multi-agent: which role produced this entry (None for single-agent).
    role: str | None = None
    # Provider-issued tool call metadata, set only when the action arrived via
    # the tools transport. Lets the history be replayed natively as an
    # assistant tool_call + tool result pair instead of flattened text.
    tool_call_id: str | None = None
    tool_name: str | None = None
    # Opaque, JSON-safe provider message used only for native protocol replay.
    # Provider reasoning must not be copied into ``text`` or returned by
    # ``render`` because it is replay metadata rather than a user-visible fact.
    provider_assistant_message: dict[str, Any] | None = None
    provider_reasoning_details: dict[str, Any] | list[Any] | None = None

    @field_validator("provider_assistant_message")
    @classmethod
    def validate_provider_assistant_message(
        cls, value: dict[str, Any] | None
    ) -> dict[str, Any] | None:
        if value is None:
            return None
        detached = _json_safe_copy(value, "provider assistant message")
        if not isinstance(detached, dict):
            raise ValueError("provider assistant message must be an object")
        return detached

    @field_validator("provider_reasoning_details")
    @classmethod
    def validate_provider_reasoning_details(
        cls, value: dict[str, Any] | list[Any] | None
    ) -> dict[str, Any] | list[Any] | None:
        if value is None:
            return None
        detached = _json_safe_copy(value, "provider reasoning details")
        if not isinstance(detached, (dict, list)):
            raise ValueError("provider reasoning details must be an object or array")
        return detached

    @property
    def hidden_context_chars(self) -> int:
        """Upper-bound replay payload omitted from ``render``.

        ``action_json`` is replayed as assistant text in text mode and as tool
        arguments in native-tool mode. A preserved provider message replaces
        that assistant message rather than appearing beside it, so take the
        largest possible replay shape instead of double-counting alternatives.
        """
        action_chars = len(self.action_json or "")
        arguments = self.replay_tool_arguments()
        tool_chars = _serialized_chars(
            _canonical_tool_assistant_message(self, arguments)
            if arguments is not None
            else None
        )
        provider_chars = _serialized_chars(self.provider_message_for_replay(arguments))
        return max(action_chars, tool_chars, provider_chars)

    def replay_tool_arguments(self) -> dict[str, Any] | None:
        """Validated action arguments used by native tool-history replay."""
        if not (self.tool_call_id and self.tool_name and self.action_json):
            return None
        try:
            arguments = json.loads(self.action_json)
            if not isinstance(arguments, dict):
                return None
            arguments.pop("action", None)
        except (json.JSONDecodeError, AttributeError):
            return None
        return arguments

    def provider_message_for_replay(
        self,
        arguments: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        """Provider message exactly as reasoning-history replay will send it."""
        if isinstance(self.provider_assistant_message, dict):
            replay = deepcopy(self.provider_assistant_message)
        elif self.provider_reasoning_details is not None:
            replay = {"role": "assistant", "content": ""}
        else:
            return None
        replay["role"] = "assistant"
        replay.pop("function_call", None)
        if self.provider_reasoning_details is not None:
            replay["reasoning_details"] = deepcopy(self.provider_reasoning_details)
        if arguments is None:
            replay.pop("tool_calls", None)
        else:
            replay["tool_calls"] = _canonical_tool_calls(self, arguments)
        return replay

    @property
    def context_chars(self) -> int:
        return len(self.text) + self.hidden_context_chars

    def render(self, index: int) -> str:
        label = f"{self.kind} {self.path}" if self.path else self.kind
        if self.role:
            label = f"{self.role}: {label}"
        return f"({index}) [step {self.step}] [{label}]\n{self.text}"


def _json_safe_copy(value: Any, label: str) -> Any:
    try:
        return json.loads(json.dumps(value, ensure_ascii=False, separators=(",", ":")))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be JSON-safe") from exc


def _canonical_tool_calls(
    entry: ContextEntry,
    arguments: dict[str, Any],
) -> list[dict[str, Any]]:
    return [
        {
            "id": entry.tool_call_id,
            "type": "function",
            "function": {
                "name": entry.tool_name,
                "arguments": json.dumps(
                    arguments,
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
            },
        }
    ]


def _canonical_tool_assistant_message(
    entry: ContextEntry,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": _canonical_tool_calls(entry, arguments),
    }


def _serialized_chars(value: dict[str, Any] | None) -> int:
    if value is None:
        return 0
    try:
        return len(json.dumps(value, ensure_ascii=False, separators=(",", ":")))
    except (TypeError, ValueError):
        # ContextEntry validation normally prevents this path. Keep the budget
        # conservative if a custom caller bypassed validation.
        return len(repr(value))
