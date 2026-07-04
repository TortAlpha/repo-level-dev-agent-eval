"""One entry in the agent's action history."""

from pydantic import BaseModel


class ContextEntry(BaseModel):
    step: int
    kind: str
    path: str | None = None
    text: str
    action_json: str | None = None

    def render(self, index: int) -> str:
        label = f"{self.kind} {self.path}" if self.path else self.kind
        return f"({index}) [step {self.step}] [{label}]\n{self.text}"
