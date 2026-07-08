import json
import time

from langchain_core.language_models import BaseChatModel
from langsmith import tracing_context
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from pydantic import BaseModel

from .actions import (
    ActionParseError,
    AgentAction,
    action_tool_schemas,
    parse_action,
    parse_tool_action,
)
from .state import State
from .state.rendering import render_header, render_progress
from .tracing import trace_model_call, trace_summarize_call
from .transport import ActionTransport


TOOL_MODE_PROMPT = """

Action transport override:
- Use exactly one of the provided action tools for your next action.
- Do not answer with JSON text unless tool calling is unavailable.
- Put brief reasoning (1-3 sentences) in the tool's optional `thought`
  argument rather than in free-form text.
""".rstrip()

# In-call retry nudges for reasoning-only turns (no action produced). Appended
# as an extra user message so a temperature-0 retry actually sees new input;
# never persisted into the agent's durable history.
_EMPTY_TURN_NUDGE_TEXT = (
    "Your previous turn produced no action. Respond now with exactly one JSON "
    "action object and nothing else — no further deliberation."
)
_EMPTY_TURN_NUDGE_TOOLS = (
    "Your previous turn produced no action. Call exactly one of the provided "
    "action tools now — no further deliberation."
)

# Upstream/provider hiccups that arrive as a 200 with an embedded error, so
# the OpenAI client's own HTTP retries never see them (e.g. langchain raises
# them as ValueError from the Responses API result). Matched case-insensitively
# against the exception text.
_TRANSIENT_ERROR_MARKERS = (
    "idle timeout",
    "server_error",
    "overloaded",
    "temporarily unavailable",
    "bad gateway",
    "service unavailable",
)


def _is_transient_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return any(marker in message for marker in _TRANSIENT_ERROR_MARKERS)


class GeneratedAction(BaseModel):
    action: AgentAction
    transport: str
    text: str = ""
    action_json: str
    # Provider-issued id/name when the action arrived as a native tool call;
    # persisted on the history entry so the replay can stay in tool protocol.
    tool_call_id: str = ""
    tool_name: str = ""


class LangChainModel(BaseModel):
    model: str
    chat: BaseChatModel

    # Cumulative usage over a run, for efficiency metrics.
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    # Steps where the tools transport had to fall back to text-JSON parsing;
    # recorded per run so tools-mode results stay interpretable.
    fallback_calls: int = 0
    # Reasoning-only turns recovered by the in-call nudged retry (either
    # transport), before the agent-level parse-failure streak gets involved.
    empty_retries: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @trace_model_call
    def generate(self, prompt: str, state: State) -> str:
        response = self._invoke_with_transient_retry(
            self.chat, self._messages(prompt, state)
        )
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
            return self._generate_text_action(prompt, state)
        return self._generate_tool_action(prompt, state)

    def _generate_text_action(self, prompt: str, state: State) -> GeneratedAction:
        messages = self._messages(prompt, state)
        boost: dict = {}
        for attempt in (0, 1):
            response = self._invoke_with_transient_retry(self.chat, messages, **boost)
            self._record_usage(response)
            text = self._text(response)
            try:
                action = parse_action(text)
            except ActionParseError as error:
                # In-call recovery, once, before charging the agent a step:
                # a reasoning-only turn gets an explicit nudge (at temperature
                # 0 the retry must alter the input to matter), and a response
                # cut off by the completion budget gets a doubled budget —
                # hidden reasoning spends the same tokens as the answer, so
                # re-asking with the same cap would just burn out again.
                truncated = self._hit_output_limit(response)
                if attempt == 0 and (error.kind == "no_action" or truncated):
                    if error.kind == "no_action":
                        messages = [*messages, HumanMessage(content=_EMPTY_TURN_NUDGE_TEXT)]
                    if truncated:
                        boost = {"max_tokens": self._escalated_max_tokens()}
                    self.empty_retries += 1
                    continue
                raise self._with_output_limit_hint(error, response, text)
            return GeneratedAction(
                action=action,
                transport="text_json",
                text=text,
                action_json=action.model_dump_json(exclude_none=True),
            )
        raise AssertionError("unreachable")

    def _generate_tool_action(self, prompt: str, state: State) -> GeneratedAction:
        if not hasattr(self.chat, "bind_tools"):
            raise ActionParseError(
                "invalid_action",
                "Configured action transport is tools, but this chat model "
                "does not expose bind_tools().",
            )

        # No parallel_tool_calls=False: several OpenRouter backends reject the
        # parameter outright. parse_tool_action tolerates multi-call responses
        # by executing the first call instead.
        tool_chat = self.chat.bind_tools(
            action_tool_schemas(),
            tool_choice="required",
        )
        messages = self._tool_history_messages(f"{prompt}\n{TOOL_MODE_PROMPT}", state)
        boost: dict = {}
        for attempt in (0, 1):
            response = self._invoke_with_transient_retry(tool_chat, messages, **boost)
            self._record_usage(response)

            text = self._text(response)
            calls = getattr(response, "tool_calls", []) or []
            try:
                action = parse_tool_action(calls)
            except ActionParseError as tool_error:
                if text.strip():
                    try:
                        action = parse_action(text)
                    except ActionParseError as text_error:
                        raise self._with_output_limit_hint(
                            text_error, response, text
                        ) from text_error
                    self.fallback_calls += 1
                    return GeneratedAction(
                        action=action,
                        transport="tools_text_fallback",
                        text=text,
                        action_json=action.model_dump_json(exclude_none=True),
                    )
                # In-call recovery (see _generate_text_action): nudge an empty
                # reasoning-only turn, double the budget on a truncated one;
                # after that the agent's parse-failure streak takes over.
                truncated = self._hit_output_limit(response)
                if attempt == 0 and (tool_error.kind == "no_action" or truncated):
                    if tool_error.kind == "no_action":
                        messages = [*messages, HumanMessage(content=_EMPTY_TURN_NUDGE_TOOLS)]
                    if truncated:
                        boost = {"max_tokens": self._escalated_max_tokens()}
                    self.empty_retries += 1
                    continue
                tool_error.raw_response = text
                raise self._with_output_limit_hint(tool_error, response, text)
            first = calls[0]
            return GeneratedAction(
                action=action,
                transport="tools",
                text=text,
                action_json=action.model_dump_json(exclude_none=True),
                tool_call_id=str(first.get("id") or ""),
                tool_name=str(first.get("name") or ""),
            )
        raise AssertionError("unreachable")

    def probe_tools(self) -> bool:
        """One cheap live request to check the endpoint really honors native
        tool calling (metadata lies; OpenRouter routes per request). Used to
        resolve the ``auto``/``tools`` transports before a run."""
        if not hasattr(self.chat, "bind_tools"):
            return False
        try:
            bound = self.chat.bind_tools(action_tool_schemas(), tool_choice="required")
            # Harness infrastructure, not agent behavior: keep the probe out
            # of the LangSmith trace (same convention as evaluation.py). Its
            # tokens still count toward the run's usage metrics.
            with tracing_context(enabled=False):
                response = bound.invoke(
                    [
                        SystemMessage(content="You are a coding agent."),
                        HumanMessage(
                            content="First step: list the repository root. "
                            "Use exactly one action tool."
                        ),
                    ]
                )
            self._record_usage(response)
            parse_tool_action(getattr(response, "tool_calls", []) or [])
            return True
        except Exception:  # noqa: BLE001 - any failure means "don't use tools"
            return False

    @trace_summarize_call
    def summarize(self, instructions: str, content: str) -> str:
        response = self._invoke_with_transient_retry(self.chat, [
            SystemMessage(content=instructions),
            HumanMessage(content=content),
        ])
        self._record_usage(response)
        return self._text(response)

    @staticmethod
    def _invoke_with_transient_retry(chat, messages, **kwargs):
        """Invoke, retrying provider hiccups the HTTP client can't see.

        Some upstream failures come back as a 200 whose body carries the error
        (Responses API), bypassing the client's status-code retries. Those are
        worth two quick retries before the failure burns an agent step.
        """
        last: Exception | None = None
        for attempt in range(3):
            if attempt:
                time.sleep(2 * attempt)
            try:
                return chat.invoke(messages, **kwargs)
            except Exception as exc:  # noqa: BLE001 - filtered just below
                if not _is_transient_error(exc):
                    raise
                last = exc
        assert last is not None
        raise last

    def _escalated_max_tokens(self) -> int:
        """Quadrupled completion budget for the one-shot truncation retry.

        Doubling proved insufficient in practice: a stuck reasoning turn burned
        a 2x budget whole. 4x is the last cheap resort before the failure
        reaches the agent loop and costs a step.
        """
        configured = getattr(self.chat, "max_tokens", None) or 4096
        return min(configured * 4, 65536)

    @staticmethod
    def _hit_output_limit(response: BaseMessage) -> bool:
        """Whether the response was cut off by the completion token budget.

        Covers both chat-completions (``finish_reason: length``) and
        Responses-API metadata (``status: incomplete`` /
        ``incomplete_details.reason: max_output_tokens``). Reasoning models
        spend the same budget on hidden reasoning, so a truncated action JSON
        usually means reasoning ate most of ``max_tokens``.
        """
        meta = getattr(response, "response_metadata", None) or {}
        if str(meta.get("finish_reason") or "") == "length":
            return True
        if str(meta.get("status") or "") == "incomplete":
            return True
        details = meta.get("incomplete_details")
        if isinstance(details, dict) and details.get("reason") == "max_output_tokens":
            return True
        return False

    def _with_output_limit_hint(
        self, error: ActionParseError, response: BaseMessage, text: str
    ) -> ActionParseError:
        """Tell the model *why* its action broke when the cause is truncation.

        A bare "malformed JSON" error invites retrying the same oversized
        action; naming the token limit steers the retry toward smaller edits.
        An entirely empty truncated turn means hidden reasoning consumed the
        whole completion budget — that needs a different instruction (and is
        the operator's cue to raise MAX_TOKENS or set REASONING_EFFORT).
        """
        if not self._hit_output_limit(response):
            return error
        if text.strip():
            hint = (
                "The response was cut off by the output token limit "
                "mid-action. Re-issue a smaller action: shorter text, or "
                "change the file in several smaller edit_file steps."
            )
        else:
            hint = (
                "The output token limit was exhausted before any action was "
                "produced (reasoning used the whole budget). Answer "
                "immediately with one short action and minimal deliberation."
            )
        return ActionParseError(
            error.kind,
            f"{error} {hint}",
            raw_response=text,
        )

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

    @staticmethod
    def _tool_history_messages(prompt: str, state: State) -> list[BaseMessage]:
        """Replay the history in the provider's native tool protocol.

        Steps that arrived as tool calls become an assistant ``tool_calls``
        message followed by a ``tool`` result message — the shape tool-trained
        models saw in post-training. Forcing ``tool_choice="required"`` on top
        of a flattened text transcript is off-distribution and is what caused
        reasoning models to emit action-less turns. Entries without tool
        metadata (setup, parse failures, compaction summaries) fall back to
        the text rendering.
        """
        messages: list[BaseMessage] = [
            SystemMessage(content=prompt),
            HumanMessage(content=render_header(state)),
        ]

        def append_human(text: str) -> None:
            # Providers differ on consecutive user messages; merge them.
            if messages and isinstance(messages[-1], HumanMessage):
                messages[-1] = HumanMessage(content=f"{messages[-1].content}\n\n{text}")
            else:
                messages.append(HumanMessage(content=text))

        for entry in state.context:
            args: dict | None = None
            if entry.tool_call_id and entry.tool_name and entry.action_json:
                try:
                    args = json.loads(entry.action_json)
                    args.pop("action", None)
                except (json.JSONDecodeError, AttributeError):
                    args = None
            if args is not None:
                messages.append(
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "id": entry.tool_call_id,
                                "name": entry.tool_name,
                                "args": args,
                                "type": "tool_call",
                            }
                        ],
                    )
                )
                messages.append(
                    ToolMessage(content=entry.text, tool_call_id=entry.tool_call_id)
                )
            else:
                if entry.action_json:
                    messages.append(AIMessage(content=entry.action_json))
                append_human(entry.text)

        append_human(render_progress(state))
        return messages

    def _record_usage(self, response: BaseMessage) -> None:
        self.calls += 1
        usage = getattr(response, "usage_metadata", None) or {}
        self.input_tokens += int(usage.get("input_tokens") or 0)
        self.output_tokens += int(usage.get("output_tokens") or 0)
