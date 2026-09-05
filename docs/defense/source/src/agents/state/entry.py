"""One entry in the agent's action history."""

from pydantic import BaseModel


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

    def render(self, index: int) -> str:
        label = f"{self.kind} {self.path}" if self.path else self.kind
        if self.role:
            label = f"{self.role}: {label}"
        return f"({index}) [step {self.step}] [{label}]\n{self.text}"
