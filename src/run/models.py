"""Build the chat model and agent from config, provider spec, and CLI args."""

from __future__ import annotations

import argparse
from copy import deepcopy
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage

from ..agents.model import LangChainModel
from ..agents.single_agent import SingleAgent
from ..config import Config, ProviderSpec
from ..metrics.pricing import PRICES


def build_model(config: Config, spec: ProviderSpec) -> LangChainModel:
    """Wrap the provider's chat endpoint, failing early if no key is set."""
    if not spec.api_key:
        raise SystemExit(
            f"No API key configured for provider '{config.model_provider}'. "
            "Set the matching *_API_KEY in your .env file."
        )
    price = PRICES.get(spec.model_name)
    return LangChainModel(
        model=spec.model_name,
        chat=build_chat_model(config, spec),
        provider=spec.provider_name,
        base_url=spec.base_url,
        context_window_tokens=spec.context_window_tokens,
        max_output_tokens=spec.max_output_tokens,
        model_context_window_capability_tokens=(
            spec.model_context_window_capability_tokens
        ),
        model_max_output_tokens=spec.model_max_output_tokens,
        reasoning_effort=spec.reasoning_effort,
        reasoning_max_tokens=spec.reasoning_max_tokens,
        supported_reasoning_efforts=spec.supported_reasoning_efforts,
        provider_default_reasoning_effort=spec.provider_default_reasoning_effort,
        reasoning_capability_known=spec.reasoning_capability_known,
        supports_reasoning_max_tokens=spec.supports_reasoning_max_tokens,
        model_profile_version=spec.model_profile_version,
        model_profile_source=spec.model_profile_source,
        model_profile_fetched_at=spec.model_profile_fetched_at,
        # OpenRouter may expose interleaved reasoning_details for any routed
        # backend. Capturing/replaying the opaque message is harmless when the
        # field is absent and avoids a model-id routing table in the kernel.
        preserve_reasoning_history=spec.provider_name == "openrouter",
        input_cost_per_1m=price.input_per_1m if price else None,
        output_cost_per_1m=price.output_per_1m if price else None,
        cached_input_cost_per_1m=(
            price.cache_read_input_per_1m if price else None
        ),
        cache_write_input_cost_per_1m=(
            price.cache_write_input_per_1m if price else None
        ),
    )


def build_agent(
    config: Config,
    args: argparse.Namespace,
    model: LangChainModel,
    spec: ProviderSpec,
) -> SingleAgent:
    """Assemble the agent from config defaults, CLI overrides, and the model."""
    return SingleAgent(
        model=model,
        docker_image=args.docker_image or config.docker_image or "python:3.11-slim",
        docker_network_disabled=not args.docker_network,
        shell_timeout_seconds=config.shell_timeout_seconds,
        test_timeout_seconds=config.test_timeout_seconds,
        stop_container=not args.keep_container,
        context_window_tokens=args.context_window_tokens or spec.context_window_tokens,
        context_budget_tokens=getattr(args, "context_budget_tokens", None)
        or config.context_budget_tokens,
        max_response_tokens=config.max_tokens,
        compaction_mode=config.compaction_mode,
        action_transport=getattr(args, "action_transport", None)
        or config.agent_action_transport,
        max_cost_usd=getattr(args, "max_cost_usd", None) or config.max_cost_usd,
    )


def build_chat_model(config: Config, spec: ProviderSpec) -> BaseChatModel:
    try:
        from langchain_openai import ChatOpenAI
    except ImportError as exc:
        raise RuntimeError(
            "This runner requires langchain-openai. "
            "Install dependencies with: python -m pip install -e ."
        ) from exc

    class ReasoningHistoryChatOpenAI(ChatOpenAI):
        """Round-trip opaque provider assistant fields across agent turns."""

        def _create_chat_result(
            self,
            response: Any,
            generation_info: dict[str, Any] | None = None,
        ) -> Any:
            response_dict = (
                response
                if isinstance(response, dict)
                else response.model_dump(
                    exclude={"choices": {"__all__": {"message": {"parsed"}}}}
                )
            )
            result = super()._create_chat_result(response, generation_info)
            choices = response_dict.get("choices") or []
            provider_usage = response_dict.get("usage")
            for generation, choice in zip(result.generations, choices, strict=False):
                raw_message = choice.get("message") or {}
                if isinstance(generation.message, AIMessage):
                    generation.message.additional_kwargs[
                        "_provider_assistant_message"
                    ] = deepcopy(raw_message)
                    # LangChain normalizes known token fields into
                    # ``usage_metadata`` but drops OpenRouter extensions such
                    # as exact ``usage.cost`` and cache-write tokens. Preserve
                    # the provider payload so run accounting can use the exact
                    # billed amount when every call reports it.
                    if isinstance(provider_usage, dict):
                        generation.message.response_metadata["usage"] = deepcopy(
                            provider_usage
                        )
            return result

        def _get_request_payload(
            self,
            input_: Any,
            *,
            stop: list[str] | None = None,
            **kwargs: Any,
        ) -> dict[str, Any]:
            payload = super()._get_request_payload(input_, stop=stop, **kwargs)
            sources = self._convert_input(input_).to_messages()
            encoded = payload.get("messages") or []
            for source, item in zip(sources, encoded, strict=False):
                if not isinstance(source, AIMessage) or not isinstance(item, dict):
                    continue
                raw = source.additional_kwargs.get("_provider_assistant_message")
                if isinstance(raw, dict):
                    item.clear()
                    item.update(deepcopy(raw))
            return payload

    # NOTE: OpenRouter's provider.require_parameters was tried here to pin
    # tools-capable backends and rejected: endpoints under-declare common
    # params (max_tokens, temperature), so it 404s legitimate models. Backend
    # variance is handled at the transport layer instead: a live probe at run
    # start, per-step text fallback (tools_fallback_calls), and the
    # parse-failure streak.
    extra_body: dict[str, Any] | None = None
    chat_class: type[ChatOpenAI] = ChatOpenAI
    if spec.provider_name == "openrouter":
        # ``usage.cost`` is the only exact billed amount and already includes
        # provider/cache discounts. Ask OpenRouter to return it on every call.
        extra_body = {"usage": {"include": True}}
        if spec.reasoning_effort is not None:
            extra_body["reasoning"] = {"effort": spec.reasoning_effort}
        elif spec.reasoning_max_tokens is not None:
            extra_body["reasoning"] = {"max_tokens": spec.reasoning_max_tokens}
        chat_class = ReasoningHistoryChatOpenAI

    # Both "local" and "openrouter" speak the OpenAI-compatible API, so the
    # same client works for either — only the spec (url/key/headers) differs.
    kwargs: dict[str, Any] = dict(
        model=spec.model_name,
        api_key=spec.api_key,
        base_url=spec.base_url,
        temperature=config.temperature,
        max_tokens=spec.max_output_tokens,
        timeout=config.request_timeout_seconds,
        max_retries=config.max_retries,
        default_headers=spec.default_headers or None,
        extra_body=extra_body,
    )
    if spec.provider_name != "openrouter":
        if spec.reasoning_effort is not None:
            kwargs["reasoning_effort"] = spec.reasoning_effort
        elif spec.reasoning_max_tokens is not None:
            kwargs["extra_body"] = {
                "reasoning": {"max_tokens": spec.reasoning_max_tokens}
            }
    return chat_class(**kwargs)
