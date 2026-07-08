from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from .agents.transport import ActionTransport

AgentMode = Literal["single", "multi", "swe-agent"]
ModelProvider = Literal["local", "openrouter"]


@dataclass
class ProviderSpec:
    """Resolved connection details for an OpenAI-compatible chat endpoint."""

    model_name: str
    api_key: str
    base_url: str
    context_window_tokens: int
    default_headers: dict[str, str] = field(default_factory=dict)


class Config(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_ignore_empty=True,
        extra="ignore",
    )

    model_provider: ModelProvider = "openrouter"
    agent_mode: AgentMode = "single"
    # Groups runs into a named experiment. Every run records its session_id so
    # the report can filter/aggregate one batch without deleting history — a new
    # session gives a clean slate; old sessions stay selectable.
    session_id: str = "default"

    temperature: float = 0.2
    max_tokens: int = 4096
    # Reasoning-model budget control, forwarded to OpenRouter as
    # ``reasoning: {effort: ...}`` (e.g. "low"/"medium"/"high"). Hidden
    # reasoning spends the same completion budget as the answer; on long
    # contexts codex/o-series models can burn all of ``max_tokens`` reasoning
    # and return an empty or truncated action. Applies identically to both
    # action transports so transport comparisons stay fair. None = provider
    # default. Ignored for the local provider.
    reasoning_effort: str | None = None
    agent_action_transport: ActionTransport = "text_json"
    request_timeout_seconds: int = 120
    max_retries: int = 10

    max_iterations: int = 5
    test_timeout_seconds: int = 600
    shell_timeout_seconds: int = 300

    compaction_mode: Literal["drop", "summarize"] = "summarize"
    # Working context budget for the agent's history compactor, in tokens.
    # Distinct from the model's physical context window: the window is what
    # the model *can* take, the budget is what is *useful* to carry — too big
    # and reasoning models burn their completion budget re-thinking a huge
    # history (empty/truncated actions), too small and the agent loses its
    # findings to summarization and re-explores in circles. None = fall back
    # to the provider's context window (compaction effectively off for
    # large-window models). 48000 was a good middle for codex-mini.
    context_budget_tokens: int | None = None

    docker_enabled: bool = True
    docker_image: str | None = None
    docker_network_disabled: bool = True

    benchmark_dir: Path = Path("eval/tasks")
    workspaces_dir: Path = Path("experiments/workspaces")
    results_dir: Path = Path("experiments/results")

    logger_enabled: bool = True
    logger_level: str = "INFO"

    langsmith_api_key: str | None = Field(default=None, repr=False)
    langsmith_project_name: str = "repo-level-dev-agent-eval"
    langsmith_tracing_enabled: bool = False
    langsmith_endpoint: str | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "LANGSMITH_ENDPOINT",
            "LANGSMITH_TRACING_ENDPOINT",
        ),
    )

    def provider_spec(self, model_override: str | None = None) -> ProviderSpec:
        raise NotImplementedError(
            f"provider_spec is not implemented for {type(self).__name__}"
        )


class LocalConfig(Config):
    local_api_url: str = "http://localhost:1234/v1"
    local_api_key: str = Field(default="dummy", repr=False)
    local_model_name: str = "qwen3-coder:30b"
    local_context_window_tokens: int = 8192

    def provider_spec(self, model_override: str | None = None) -> ProviderSpec:
        return ProviderSpec(
            model_name=model_override or self.local_model_name,
            api_key=self.local_api_key,
            base_url=self.local_api_url,
            context_window_tokens=self.local_context_window_tokens,
        )


class OpenRouterConfig(Config):
    openrouter_api_url: str = "https://openrouter.ai/api/v1"
    openrouter_api_key: str = Field(default="", repr=False)
    openrouter_model_name: str = "openai/gpt-4o-mini"
    openrouter_context_window_tokens: int = 128000
    openrouter_http_referer: str | None = None
    openrouter_app_title: str = "repo-level-dev-agent-eval"

    def provider_spec(self, model_override: str | None = None) -> ProviderSpec:
        # Optional attribution headers OpenRouter uses for its dashboards.
        headers: dict[str, str] = {}
        if self.openrouter_http_referer:
            headers["HTTP-Referer"] = self.openrouter_http_referer
        if self.openrouter_app_title:
            headers["X-Title"] = self.openrouter_app_title
        return ProviderSpec(
            model_name=model_override or self.openrouter_model_name,
            api_key=self.openrouter_api_key,
            base_url=self.openrouter_api_url,
            context_window_tokens=self.openrouter_context_window_tokens,
            default_headers=headers,
        )


PROVIDER_CONFIGS: dict[ModelProvider, type[Config]] = {
    "local": LocalConfig,
    "openrouter": OpenRouterConfig,
}


def load_config(provider_override: ModelProvider | None = None) -> Config:
    """Select the provider-specific config.

    A ``provider_override`` (e.g. from a CLI flag) wins over the
    ``MODEL_PROVIDER`` value read from the environment / ``.env``.
    """
    provider = provider_override or Config().model_provider
    return PROVIDER_CONFIGS[provider](model_provider=provider)
