from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel

from .state import State
from .tracing import trace_model_call, trace_summarize_call


class LangChainModel(BaseModel):
    model: str
    chat: BaseChatModel

    # Cumulative usage over a run, for efficiency metrics.
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @trace_model_call
    def generate(self, prompt: str, state: State) -> str:
        messages: list[BaseMessage] = [SystemMessage(content=prompt)]
        for role, text in state.to_messages():
            if role == "assistant":
                messages.append(AIMessage(content=text))
            else:
                messages.append(HumanMessage(content=text))
        response = self.chat.invoke(messages)
        self._record_usage(response)
        return self._text(response)

    @trace_summarize_call
    def summarize(self, instructions: str, content: str) -> str:
        response = self.chat.invoke([
            SystemMessage(content=instructions),
            HumanMessage(content=content),
        ])
        self._record_usage(response)
        return self._text(response)

    @staticmethod
    def _text(response: BaseMessage) -> str:
        """Extract the assistant text. Reasoning models return ``content`` as a
        list of blocks (reasoning + text); take the text parts, drop the rest."""
        content = response.content
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            # Blocks are plain strings or dicts like
            # {"type": "text" | "reasoning", "text": ...}. Keep the text ones,
            # drop reasoning (and anything else) and concatenate.
            parts: list[str] = []
            for block in content:
                if isinstance(block, str):
                    parts.append(block)
                elif isinstance(block, dict) and block.get("type") == "text":
                    parts.append(str(block.get("text", "")))
            if parts:
                return "".join(parts)
        return str(content)

    def _record_usage(self, response: BaseMessage) -> None:
        self.calls += 1
        usage = getattr(response, "usage_metadata", None) or {}
        self.input_tokens += int(usage.get("input_tokens") or 0)
        self.output_tokens += int(usage.get("output_tokens") or 0)
