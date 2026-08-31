import json
import math
import re
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langsmith import tracing_context
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
    "error code: 429",
    "status code: 429",
    "too many requests",
    "rate limit",
    "idle timeout",
    "server_error",
    "provider returned error",
    "the upstream provider returned an error",
    "overloaded",
    "temporarily unavailable",
    "bad gateway",
    "service unavailable",
)
_RETRY_AFTER_RE = re.compile(
    r"retry[_-]after(?:[_-]seconds)?['\"]?\s*[:=]\s*['\"]?(\d+(?:\.\d+)?)",
    flags=re.IGNORECASE,
)
_MAX_RETRY_DELAY_SECONDS = 30.0


def _provider_status_code(exc: Exception) -> int | None:
    response = getattr(exc, "response", None)
    for candidate in (
        getattr(exc, "status_code", None),
        getattr(response, "status_code", None),
    ):
        if isinstance(candidate, int):
            return candidate
    return None


def _is_transient_error(exc: Exception) -> bool:
    status = _provider_status_code(exc)
    if status == 429 or (status is not None and 500 <= status < 600):
        return True
    # Avoid a hard dependency on one HTTP stack while still treating network
    # transport exceptions from httpx/requests/aiohttp as retryable.
    error_type = type(exc)
    module = error_type.__module__.lower()
    name = error_type.__name__.lower()
    if isinstance(exc, (ConnectionError, TimeoutError)) or (
        module.startswith(("httpx", "httpcore", "requests"))
        and any(marker in name for marker in ("connect", "timeout", "transport"))
    ):
        return True
    message = str(exc).lower()
    return any(marker in message for marker in _TRANSIENT_ERROR_MARKERS)


def _retry_after_seconds(exc: Exception) -> float | None:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if headers is not None:
        try:
            value = headers.get("Retry-After")
        except Exception:  # pragma: no cover - defensive SDK boundary
            value = None
        if value:
            try:
                return float(str(value).strip())
            except ValueError:
                pass
    texts = [str(exc)]
    try:
        texts.append(str(getattr(response, "text", "") or ""))
    except Exception:  # pragma: no cover - unreadable response bodies
        pass
    for text in texts:
        match = _RETRY_AFTER_RE.search(text)
        if match is not None:
            try:
                return float(match.group(1))
            except ValueError:  # pragma: no cover - guarded by the regex
                pass
    return None


class GeneratedAction(BaseModel):
    action: AgentAction
    transport: str
    text: str = ""
    action_json: str
    # Provider-issued id/name when the action arrived as a native tool call;
    # persisted on the history entry so the replay can stay in tool protocol.
    tool_call_id: str = ""
    tool_name: str = ""
    # Opaque provider protocol state (e.g. reasoning_details). It is never
    # rendered as prompt text; ContextEntry reattaches it only when the
    # corresponding assistant turn is replayed.
    provider_assistant_message: dict[str, Any] | None = None
    provider_reasoning_details: dict[str, Any] | list[Any] | None = None


@dataclass(frozen=True)
class ActionGenerationFailure:
    """A recoverable invalid model response, with its transport metadata.

    This is deliberately a return value rather than an exception escaping the
    traced model call.  The single-agent loop can then feed back the error in
    the correct protocol without every normal model-formatting slip becoming
    a trace failure.
    """

    error: ActionParseError
    transport: str
    action_json: str
    tool_call_id: str = ""
    tool_name: str = ""
    provider_assistant_message: dict[str, Any] | None = None
    provider_reasoning_details: dict[str, Any] | list[Any] | None = None


class LangChainModel(BaseModel):
    model: str
    chat: BaseChatModel
    provider: str = "unknown"
    base_url: str | None = None
    context_window_tokens: int | None = None
    max_output_tokens: int | None = None
    model_context_window_capability_tokens: int | None = None
    model_max_output_tokens: int | None = None
    reasoning_effort: str | None = None
    reasoning_max_tokens: int | None = None
    supported_reasoning_efforts: tuple[str, ...] = ()
    provider_default_reasoning_effort: str | None = None
    reasoning_capability_known: bool = False
    supports_reasoning_max_tokens: bool | None = None
    model_profile_version: str | None = None
    model_profile_source: str | None = None
    model_profile_fetched_at: str | None = None
    preserve_reasoning_history: bool = False
    input_cost_per_1m: float | None = None
    output_cost_per_1m: float | None = None
    cached_input_cost_per_1m: float | None = None
    cache_write_input_cost_per_1m: float | None = None

    # Cumulative usage over a run, for efficiency metrics.
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0
    cache_write_input_tokens: int = 0
    reasoning_tokens: int = 0
    # OpenRouter can report billed generation cost in detailed usage. Keep it
    # separate from the static token-price estimate: absence is not zero.
    provider_reported_cost_usd: float = 0.0
    provider_cost_calls: int = 0
    # False means an invoked external/runtime boundary could not provide a
    # complete usage record.  Absence is not zero: scientific cost guards and
    # campaign accounting must fail closed instead of pricing that call at $0.
    usage_accounting_complete: bool = True
    # Steps where the tools transport had to fall back to text-JSON parsing;
    # recorded per run so tools-mode results stay interpretable.
    fallback_calls: int = 0
    # Reasoning-only turns recovered by the in-call nudged retry (either
    # transport), before the agent-level parse-failure streak gets involved.
    empty_retries: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def estimated_cost_usd(self) -> float | None:
        if not self.usage_accounting_complete:
            return None
        # Configured-but-unused role/escalation models must not make aggregate
        # run cost unknowable merely because their static price is unavailable.
        if self.calls == 0 and self.input_tokens == 0 and self.output_tokens == 0:
            return 0.0
        if self.input_cost_per_1m is None or self.output_cost_per_1m is None:
            return None
        cache_reads = min(self.cached_input_tokens, self.input_tokens)
        cache_writes = min(
            self.cache_write_input_tokens,
            max(self.input_tokens - cache_reads, 0),
        )
        uncached = max(self.input_tokens - cache_reads - cache_writes, 0)
        read_rate = (
            self.cached_input_cost_per_1m
            if self.cached_input_cost_per_1m is not None
            else self.input_cost_per_1m
        )
        write_rate = (
            self.cache_write_input_cost_per_1m
            if self.cache_write_input_cost_per_1m is not None
            else self.input_cost_per_1m
        )
        return (
            uncached / 1_000_000 * self.input_cost_per_1m
            + cache_reads / 1_000_000 * read_rate
            + cache_writes / 1_000_000 * write_rate
            + self.output_tokens / 1_000_000 * self.output_cost_per_1m
        )

    @property
    def effective_cost_usd(self) -> float | None:
        """Actual cache-discounted billing only when every call reported it.

        Partial provider usage must not make a hard guard optimistic. Until
        coverage is complete, use the conservative static estimate for all
        tokens (and never less than the cost already reported).
        """
        if not self.usage_accounting_complete:
            return None
        if self.calls > 0 and self.provider_cost_calls >= self.calls:
            return self.provider_reported_cost_usd
        estimate = self.estimated_cost_usd
        if estimate is None:
            return None
        if 0 < self.provider_cost_calls < self.calls:
            # Without per-call token attribution we cannot subtract the static
            # estimate for calls already covered by actual billing. Adding the
            # complete static estimate is an explicit upper bound; taking max
            # would incorrectly price every unreported call at zero whenever
            # the reported subset happened to be more expensive.
            return estimate + self.provider_reported_cost_usd
        return max(estimate, self.provider_reported_cost_usd)

    @property
    def cost_source(self) -> str:
        if not self.usage_accounting_complete:
            return "unknown_incomplete_usage"
        if self.calls > 0 and self.provider_cost_calls >= self.calls:
            return "provider_actual"
        if self.provider_cost_calls:
            return "actual_plus_static_upper_bound"
        return "static_estimate"

    def route_metadata(self, *, transport: str | None = None) -> dict[str, Any]:
        effective_effort = (
            self.reasoning_effort
            if self.reasoning_effort is not None
            else self.provider_default_reasoning_effort
        )
        if self.reasoning_max_tokens is not None:
            reasoning_source = "explicit_max_tokens"
        elif self.reasoning_effort is not None:
            reasoning_source = "explicit_effort"
        elif self.provider_default_reasoning_effort is not None:
            reasoning_source = "provider_default_snapshot"
        else:
            reasoning_source = "capability_unknown"
        return {
            "provider": self.provider,
            "base_url": self.base_url,
            "model": self.model,
            "context_window_tokens": self.context_window_tokens,
            "max_output_tokens": self.max_output_tokens,
            "model_context_window_capability_tokens": (
                self.model_context_window_capability_tokens
            ),
            "model_max_output_tokens": self.model_max_output_tokens,
            "reasoning_effort": self.reasoning_effort,
            "reasoning_max_tokens": self.reasoning_max_tokens,
            "effective_reasoning_effort": effective_effort,
            "reasoning_configuration_source": reasoning_source,
            "supported_reasoning_efforts": list(self.supported_reasoning_efforts),
            "provider_default_reasoning_effort": (
                self.provider_default_reasoning_effort
            ),
            "reasoning_capability_known": self.reasoning_capability_known,
            "supports_reasoning_max_tokens": self.supports_reasoning_max_tokens,
            "model_profile_version": self.model_profile_version,
            "model_profile_source": self.model_profile_source,
            "model_profile_fetched_at": self.model_profile_fetched_at,
            "pricing": {
                "input_cost_per_1m": self.input_cost_per_1m,
                "output_cost_per_1m": self.output_cost_per_1m,
                "cached_input_cost_per_1m": self.cached_input_cost_per_1m,
                "cache_write_input_cost_per_1m": (
                    self.cache_write_input_cost_per_1m
                ),
            },
            "action_transport": transport,
        }

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
        space: str | None = None,
    ) -> GeneratedAction | ActionGenerationFailure:
        try:
            if transport == "text_json" or transport == "auto":
                generated = self._generate_text_action(prompt, state, space)
            else:
                generated = self._generate_tool_action(prompt, state, space)
        except ActionParseError as error:
            generated = self._generation_failure(error, transport)
        return generated

    @staticmethod
    def _generation_failure(
        error: ActionParseError,
        transport: ActionTransport,
        provider_assistant_message: dict[str, Any] | None = None,
        provider_reasoning_details: dict[str, Any] | list[Any] | None = None,
    ) -> ActionGenerationFailure:
        """Normalize a failed response for replay in its original protocol."""
        action_json = error.raw_response
        if error.tool_call_id and error.tool_name:
            args = error.tool_arguments
            # Native tool calls require an object of valid JSON arguments on
            # replay.  A provider may have supplied malformed JSON text, in
            # which case the tool error itself retains the raw diagnostic and
            # the replay uses an empty argument object.
            if isinstance(args, Mapping):
                action_json = json.dumps({"action": error.tool_name, **args})
            else:
                action_json = json.dumps({"action": error.tool_name})
        return ActionGenerationFailure(
            error=error,
            transport=transport,
            action_json=action_json,
            tool_call_id=error.tool_call_id,
            tool_name=error.tool_name,
            provider_assistant_message=provider_assistant_message,
            provider_reasoning_details=provider_reasoning_details,
        )

    def _generate_text_action(
        self, prompt: str, state: State, space: str | None = None
    ) -> GeneratedAction | ActionGenerationFailure:
        messages = self._messages(prompt, state)
        boost: dict = {}
        for attempt in (0, 1):
            response = self._invoke_with_transient_retry(self.chat, messages, **boost)
            self._record_usage(response)
            text = self._text(response)
            provider_message = self._opaque_provider_assistant_message(response)
            reasoning_details = self._provider_reasoning_details(response)
            try:
                action = parse_action(text, space)
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
                        messages = [
                            *messages,
                            *([response] if provider_message else []),
                            HumanMessage(content=_EMPTY_TURN_NUDGE_TEXT),
                        ]
                    if truncated:
                        boost = {"max_tokens": self._escalated_max_tokens()}
                    self.empty_retries += 1
                    continue
                return self._generation_failure(
                    self._with_output_limit_hint(error, response, text),
                    "text_json",
                    provider_message,
                    reasoning_details,
                )
            return GeneratedAction(
                action=action,
                transport="text_json",
                text=text,
                action_json=action.model_dump_json(exclude_none=True),
                provider_assistant_message=provider_message,
                provider_reasoning_details=reasoning_details,
            )
        raise AssertionError("unreachable")

    def _generate_tool_action(
        self, prompt: str, state: State, space: str | None = None
    ) -> GeneratedAction | ActionGenerationFailure:
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
            action_tool_schemas(space),
            tool_choice="required",
        )
        messages = self._tool_history_messages(f"{prompt}\n{TOOL_MODE_PROMPT}", state)
        boost: dict = {}
        for attempt in (0, 1):
            response = self._invoke_with_transient_retry(tool_chat, messages, **boost)
            self._record_usage(response)

            text = self._text(response)
            provider_message = self._opaque_provider_assistant_message(response)
            reasoning_details = self._provider_reasoning_details(response)
            calls = getattr(response, "tool_calls", []) or []
            try:
                action = parse_tool_action(calls, space)
            except ActionParseError as tool_error:
                if text.strip():
                    try:
                        action = parse_action(text, space)
                    except ActionParseError as text_error:
                        combined = ActionParseError(
                            text_error.kind,
                            str(text_error),
                            raw_response=text,
                            tool_call_id=tool_error.tool_call_id,
                            tool_name=tool_error.tool_name,
                            tool_arguments=tool_error.tool_arguments,
                        )
                        return self._generation_failure(
                            self._with_output_limit_hint(combined, response, text),
                            "tools",
                            provider_message,
                            reasoning_details,
                        )
                    self.fallback_calls += 1
                    return GeneratedAction(
                        action=action,
                        transport="tools_text_fallback",
                        text=text,
                        action_json=action.model_dump_json(exclude_none=True),
                        provider_assistant_message=provider_message,
                        provider_reasoning_details=reasoning_details,
                    )
                # In-call recovery (see _generate_text_action): nudge an empty
                # reasoning-only turn, double the budget on a truncated one;
                # after that the agent's parse-failure streak takes over.
                truncated = self._hit_output_limit(response)
                if attempt == 0 and (tool_error.kind == "no_action" or truncated):
                    if tool_error.kind == "no_action":
                        messages = [
                            *messages,
                            *([response] if provider_message else []),
                            HumanMessage(content=_EMPTY_TURN_NUDGE_TOOLS),
                        ]
                    if truncated:
                        boost = {"max_tokens": self._escalated_max_tokens()}
                    self.empty_retries += 1
                    continue
                tool_error.raw_response = text
                return self._generation_failure(
                    self._with_output_limit_hint(tool_error, response, text),
                    "tools",
                    provider_message,
                    reasoning_details,
                )
            first = calls[0]
            return GeneratedAction(
                action=action,
                transport="tools",
                text=text,
                action_json=action.model_dump_json(exclude_none=True),
                tool_call_id=str(first.get("id") or ""),
                tool_name=str(first.get("name") or ""),
                provider_assistant_message=provider_message,
                provider_reasoning_details=reasoning_details,
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
        response = self._invoke_with_transient_retry(
            self.chat,
            [
                SystemMessage(content=instructions),
                HumanMessage(content=content),
            ],
        )
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
                time.sleep(LangChainModel._retry_delay_seconds(last, attempt))
            try:
                return chat.invoke(messages, **kwargs)
            except Exception as exc:  # noqa: BLE001 - filtered just below
                if not _is_transient_error(exc):
                    raise
                last = exc
        assert last is not None
        raise last

    @staticmethod
    def _retry_delay_seconds(last: Exception | None, attempt: int) -> float:
        if last is None:
            return 2.0 * attempt
        hinted = _retry_after_seconds(last) or 0.0
        status = _provider_status_code(last)
        message = str(last).lower()
        rate_limited = status == 429 or any(
            marker in message
            for marker in ("rate limit", "too many requests", "error code: 429")
        )
        base = 5.0 * attempt if rate_limited else 2.0 * attempt
        return min(max(hinted, base), _MAX_RETRY_DELAY_SECONDS)

    def _escalated_max_tokens(self) -> int:
        """Up to 4x completion budget for the one-shot truncation retry.

        Doubling proved insufficient in practice: a stuck reasoning turn burned
        a 2x budget whole. 4x is the last cheap resort before the failure
        reaches the agent loop and costs a step.
        """
        configured = int(getattr(self.chat, "max_tokens", None) or 4096)
        capability = self.model_max_output_tokens
        if capability is not None:
            return min(configured * 4, capability)
        # Unknown routes keep the historical 65k safety ceiling, but a retry
        # must never request fewer tokens than the already accepted baseline.
        return max(configured, min(configured * 4, 65536))

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
            tool_call_id=error.tool_call_id,
            tool_name=error.tool_name,
            tool_arguments=error.tool_arguments,
        )

    def _opaque_provider_assistant_message(
        self, response: BaseMessage
    ) -> dict[str, Any] | None:
        if not self.preserve_reasoning_history:
            return None
        raw = (getattr(response, "additional_kwargs", None) or {}).get(
            "_provider_assistant_message"
        )
        if not isinstance(raw, dict):
            return None
        try:
            detached = json.loads(
                json.dumps(raw, ensure_ascii=False, separators=(",", ":"))
            )
        except (TypeError, ValueError):
            return None
        return detached if isinstance(detached, dict) else None

    def _provider_reasoning_details(
        self, response: BaseMessage
    ) -> dict[str, Any] | list[Any] | None:
        if not self.preserve_reasoning_history:
            return None
        additional = getattr(response, "additional_kwargs", None) or {}
        raw = additional.get("reasoning_details")
        if raw is None:
            provider_message = additional.get("_provider_assistant_message")
            if isinstance(provider_message, dict):
                raw = provider_message.get("reasoning_details")
        if not isinstance(raw, (dict, list)):
            return None
        try:
            detached = json.loads(
                json.dumps(raw, ensure_ascii=False, separators=(",", ":"))
            )
        except (TypeError, ValueError):
            return None
        return detached if isinstance(detached, (dict, list)) else None

    def _provider_replay_kwargs(self, entry, args: dict | None) -> dict[str, Any]:
        if not self.preserve_reasoning_history:
            return {}
        replay = (
            entry.provider_message_for_replay(args)
            if hasattr(entry, "provider_message_for_replay")
            else None
        )
        if replay is None:
            return {}
        return {"_provider_assistant_message": replay}

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
                    "text",
                    "output_text",
                ):
                    parts.append(str(block.get("text", "")))
                # else: reasoning item / unknown block -> drop it.
            return "".join(parts)
        return str(content)

    def _messages(self, prompt: str, state: State) -> list[BaseMessage]:
        messages: list[BaseMessage] = [
            SystemMessage(content=prompt),
            HumanMessage(content=render_header(state)),
        ]

        def append_human(text: str) -> None:
            if messages and isinstance(messages[-1], HumanMessage):
                messages[-1] = HumanMessage(content=f"{messages[-1].content}\n\n{text}")
            else:
                messages.append(HumanMessage(content=text))

        for entry in state.context:
            if entry.action_json:
                kwargs = self._provider_replay_kwargs(entry, None)
                messages.append(
                    AIMessage(content=entry.action_json, additional_kwargs=kwargs)
                )
            append_human(entry.text)
        append_human(render_progress(state))
        return messages

    def _tool_history_messages(self, prompt: str, state: State) -> list[BaseMessage]:
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
                provider_kwargs = self._provider_replay_kwargs(entry, args)
                messages.append(
                    AIMessage(
                        content="",
                        additional_kwargs=provider_kwargs,
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
                provider_kwargs = self._provider_replay_kwargs(entry, None)
                if provider_kwargs:
                    messages.append(
                        AIMessage(content="", additional_kwargs=provider_kwargs)
                    )
                elif entry.action_json:
                    messages.append(AIMessage(content=entry.action_json))
                append_human(entry.text)

        append_human(render_progress(state))
        return messages

    def _record_usage(self, response: BaseMessage) -> None:
        self.calls += 1
        raw_usage = getattr(response, "usage_metadata", None) or {}
        usage = raw_usage if isinstance(raw_usage, Mapping) else {}
        raw_response_meta = getattr(response, "response_metadata", None) or {}
        response_meta = (
            raw_response_meta if isinstance(raw_response_meta, Mapping) else {}
        )
        token_usage = (
            response_meta.get("token_usage") or response_meta.get("usage") or {}
        )
        token_usage = token_usage if isinstance(token_usage, Mapping) else {}
        input_tokens = self._usage_count(
            (usage, ("input_tokens",)),
            (token_usage, ("prompt_tokens", "input_tokens")),
        )
        output_tokens = self._usage_count(
            (usage, ("output_tokens",)),
            (token_usage, ("completion_tokens", "output_tokens")),
        )
        if (
            input_tokens is None
            or output_tokens is None
            or input_tokens + output_tokens < 1
        ):
            self.usage_accounting_complete = False
            input_tokens = input_tokens or 0
            output_tokens = output_tokens or 0
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        input_details = usage.get("input_token_details") or {}
        prompt_details = token_usage.get("prompt_tokens_details") or {}
        self.cached_input_tokens += int(
            input_details.get("cache_read")
            or input_details.get("cached_tokens")
            or prompt_details.get("cached_tokens")
            or token_usage.get("cached_tokens")
            or 0
        )
        self.cache_write_input_tokens += int(
            input_details.get("cache_write")
            or input_details.get("cache_write_tokens")
            or prompt_details.get("cache_write_tokens")
            or token_usage.get("cache_write_tokens")
            or 0
        )
        output_details = usage.get("output_token_details") or {}
        completion_details = token_usage.get("completion_tokens_details") or {}
        self.reasoning_tokens += int(
            output_details.get("reasoning")
            or output_details.get("reasoning_tokens")
            or completion_details.get("reasoning_tokens")
            or token_usage.get("reasoning_tokens")
            or 0
        )

        cost = self._reported_cost(usage, response_meta)
        if cost is not None:
            self.provider_reported_cost_usd += cost
            self.provider_cost_calls += 1

    @staticmethod
    def _usage_count(
        *sources: tuple[Mapping, tuple[str, ...]],
    ) -> int | None:
        """Return an explicit non-negative token count, never a missing zero."""
        for container, keys in sources:
            for key in keys:
                if key not in container:
                    continue
                try:
                    value = int(container[key])
                except (TypeError, ValueError, OverflowError):
                    continue
                if value >= 0:
                    return value
        return None

    @staticmethod
    def _reported_cost(usage: Mapping, response_meta: Mapping) -> float | None:
        """Extract a provider-reported billed total when one is explicit.

        ``upstream_inference_cost`` is deliberately excluded: OpenRouter uses
        that name for its own upstream/provider expense, which is not the same
        quantity as the amount billed to this benchmark account.
        """
        containers = [
            usage,
            response_meta.get("usage") or {},
            response_meta.get("token_usage") or {},
        ]
        for container in containers:
            for key in ("cost", "total_cost"):
                value = container.get(key) if isinstance(container, Mapping) else None
                if value is not None:
                    try:
                        parsed = float(value)
                    except (TypeError, ValueError):
                        continue
                    if math.isfinite(parsed) and parsed >= 0:
                        return parsed
        return None
