from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel

from .actions import (
    ActionParseError,
    AgentAction,
    action_tool_schemas,
    parse_action,
    parse_tool_action,
)
from .state import State
from .tracing import trace_model_call, trace_summarize_call
from .transport import ActionTransport


TOOL_MODE_PROMPT = """

Action transport override:
- Use exactly one of the provided action tools for your next action.
- Do not answer with JSON text unless tool calling is unavailable.
- Do not provide reasoning text outside the tool call.
""".rstrip()


class GeneratedAction(BaseModel):
    action: AgentAction
    transport: str
    text: str = ""
    action_json: str


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
        response = self.chat.invoke(self._messages(prompt, state))
        self._record_usage(response)
        return self._text(response)

    @trace_model_call
    def generate_action(
        self,
        prompt: str,
        state: State,
        transport: ActionTransport,
    ) -> GeneratedAction:
        if transport == "text_json" or transport == "auto":
            response = self.chat.invoke(self._messages(prompt, state))
            self._record_usage(response)
            text = self._text(response)
            action = parse_action(text)
            return GeneratedAction(
                action=action,
                transport="text_json",
                text=text,
                action_json=action.model_dump_json(),
            )

        if not hasattr(self.chat, "bind_tools"):
            raise ActionParseError(
                "invalid_action",
                "Configured action transport is tools, but this chat model "
                "does not expose bind_tools().",
            )

        tool_chat = self.chat.bind_tools(
            action_tool_schemas(),
            tool_choice="required",
            parallel_tool_calls=False,
            strict=False,
        )
        response = tool_chat.invoke(
            self._messages(f"{prompt}\n{TOOL_MODE_PROMPT}", state)
        )
        self._record_usage(response)

        text = self._text(response)
        try:
            action = parse_tool_action(getattr(response, "tool_calls", []) or [])
            return GeneratedAction(
                action=action,
                transport="tools",
                text=text,
                action_json=action.model_dump_json(),
            )
        except ActionParseError as tool_error:
            if text.strip():
                action = parse_action(text)
                return GeneratedAction(
                    action=action,
                    transport="tools_text_fallback",
                    text=text,
                    action_json=action.model_dump_json(),
                )
            tool_error.raw_response = text
            raise tool_error

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
        """Extract the assistant's *answer* text, dropping reasoning.

        Reasoning models (codex, o-series, deepseek-r) return ``content`` as
        blocks: plain strings, answer text (``type`` in {"text", "output_text"}),
        or reasoning-only items (``type: "reasoning"`` / a ``{"id", "summary"}``
        Responses-API item). Keep only the answer text. Crucially, when a turn is
        reasoning-only, return "" rather than ``str(content)``: dumping the raw
        reasoning blob would (a) be parsed as an action and rejected, and (b)
        flood the conversation history (a single reasoning item can be tens of
        thousands of characters), blowing up token usage on every later turn.
        """
        content = response.content
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for block in content:
                if isinstance(block, str):
                    parts.append(block)
                elif isinstance(block, dict) and block.get("type") in (
                    "text", "output_text",
                ):
                    parts.append(str(block.get("text", "")))
                # else: reasoning item / unknown block -> drop it.
            return "".join(parts)
        return str(content)

    @staticmethod
    def _messages(prompt: str, state: State) -> list[BaseMessage]:
        messages: list[BaseMessage] = [SystemMessage(content=prompt)]
        for role, text in state.to_messages():
            if role == "assistant":
                messages.append(AIMessage(content=text))
            else:
                messages.append(HumanMessage(content=text))
        return messages

    def _record_usage(self, response: BaseMessage) -> None:
        self.calls += 1
        usage = getattr(response, "usage_metadata", None) or {}
        self.input_tokens += int(usage.get("input_tokens") or 0)
        self.output_tokens += int(usage.get("output_tokens") or 0)
