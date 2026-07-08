"""Build the chat model and agent from config, provider spec, and CLI args."""

from __future__ import annotations

import argparse

from langchain_core.language_models.chat_models import BaseChatModel

from ..agents.model import LangChainModel
from ..agents.single_agent import SingleAgent
from ..config import Config, ProviderSpec


def build_model(config: Config, spec: ProviderSpec) -> LangChainModel:
    """Wrap the provider's chat endpoint, failing early if no key is set."""
    if not spec.api_key:
        raise SystemExit(
            f"No API key configured for provider '{config.model_provider}'. "
            "Set the matching *_API_KEY in your .env file."
        )
    return LangChainModel(model=spec.model_name, chat=build_chat_model(config, spec))


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
        action_transport=getattr(
            args, "action_transport", None
        ) or config.agent_action_transport,
    )


def build_chat_model(config: Config, spec: ProviderSpec) -> BaseChatModel:
    try:
        from langchain_openai import ChatOpenAI
    except ImportError as exc:
        raise RuntimeError(
            "This runner requires langchain-openai. "
            "Install dependencies with: python -m pip install -e ."
        ) from exc

    # NOTE: OpenRouter's provider.require_parameters was tried here to pin
    # tools-capable backends and rejected: endpoints under-declare common
    # params (max_tokens, temperature), so it 404s legitimate models. Backend
    # variance is handled at the transport layer instead: a live probe at run
    # start, per-step text fallback (tools_fallback_calls), and the
    # parse-failure streak.
    extra_body = None
    if config.model_provider == "openrouter" and config.reasoning_effort:
        extra_body = {"reasoning": {"effort": config.reasoning_effort}}

    # Both "local" and "openrouter" speak the OpenAI-compatible API, so the
    # same client works for either — only the spec (url/key/headers) differs.
    return ChatOpenAI(
        model=spec.model_name,
        api_key=spec.api_key,
        base_url=spec.base_url,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        timeout=config.request_timeout_seconds,
        max_retries=config.max_retries,
        default_headers=spec.default_headers or None,
        extra_body=extra_body,
    )
