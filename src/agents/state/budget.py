"""Context-window sizing math used by compaction."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from .core import State


class ContextBudget(BaseModel):
    """Sizing math for keeping the rendered prompt inside the model window.

    Estimates the whole prompt in characters (system prompt via
    ``overhead_chars``, plus the rendered state including the action
    history) and reserves room for the model response.

    ``chars_per_token`` is deliberately conservative: file paths and code
    tokenize at roughly 2 characters per token, far denser than prose.
    """

    window_tokens: int = Field(gt=0)
    reserved_output_tokens: int = Field(default=4096, ge=0)
    overhead_chars: int = Field(default=0, ge=0)
    chars_per_token: float = 2.0
    high_watermark: float = 0.8
    low_watermark: float = 0.6

    @property
    def input_budget_chars(self) -> float:
        input_tokens = max(self.window_tokens - self.reserved_output_tokens, 1)
        return input_tokens * self.chars_per_token

    def prompt_chars(self, state: State) -> int:
        # ``to_string`` intentionally omits native provider metadata so hidden
        # reasoning cannot leak into logs/prompts. It is nevertheless sent in
        # tool history and must count toward compaction thresholds.
        return self.overhead_chars + len(state.to_string()) + state.hidden_context_chars

    def needs_compaction(self, state: State) -> bool:
        threshold = self.input_budget_chars * self.high_watermark
        return self.prompt_chars(state) > threshold

    def history_target_chars(self, state: State, reserve_chars: int = 0) -> int:
        """How many history characters fit once compaction brings the prompt
        down to the low watermark, keeping ``reserve_chars`` for a digest."""
        fixed_chars = self.prompt_chars(state) - state.context_chars
        target = int(self.input_budget_chars * self.low_watermark)
        return max(target - fixed_chars - reserve_chars, 0)
