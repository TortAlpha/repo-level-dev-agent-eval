from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from .agents.transport import ActionTransport

AgentMode = Literal[
    "single",
    "single-decomposed",
    "swe-agent",
    # Multi-agent variants (see src/agents/multi_agent.py). "multi" is the
    # legacy alias for the spec's deterministic graph variant.
    "multi",
    "multi-graph",
    "multi-orch",
    "multi-orch-guided",
    "multi-orch-guarded",
    "multi-swe",
]
ModelProvider = Literal["local", "openrouter"]
ReasoningEffort = Literal[
    "none",
    "minimal",
    "low",
    "medium",
    "high",
    "xhigh",
    "max",
]
REASONING_EFFORTS: tuple[ReasoningEffort, ...] = (
    "none",
    "minimal",
    "low",
    "medium",
    "high",
    "xhigh",
    "max",
)
MODEL_PROFILE_REGISTRY_VERSION = "model-profile-registry-v2"
MODEL_PROFILE_SOURCE = "https://openrouter.ai/api/v1/models"
MODEL_PROFILE_FETCHED_AT = "2026-08-31"


@dataclass(frozen=True)
class ModelProfile:
    """Reviewed calling-contract facts for one model route.

    Pricing deliberately remains in ``static/pricing.csv``.  These facts
    describe request compatibility and are included in run fingerprints, so a
    model alias cannot silently change context/reasoning behavior between
    otherwise comparable runs.
    """

    model_id: str
    context_window_tokens: int | None = None
    max_output_tokens: int | None = None
    supported_reasoning_efforts: tuple[ReasoningEffort, ...] = ()
    default_reasoning_effort: ReasoningEffort | None = None
    # This describes an end-to-end exact token-budget guarantee, not merely a
    # provider accepting and possibly translating the field. ``None`` means
    # the reviewed snapshot does not establish exactness, so comparable runs
    # must reject ``reasoning_max_tokens`` until this is explicitly ``True``.
    supports_reasoning_max_tokens: bool | None = None


MODEL_PROFILES: dict[str, ModelProfile] = {
    "z-ai/glm-5.2": ModelProfile(
        model_id="z-ai/glm-5.2",
        context_window_tokens=1_048_576,
        max_output_tokens=262_144,
        supported_reasoning_efforts=("xhigh", "high"),
        default_reasoning_effort="high",
    ),
    "deepseek/deepseek-v4-flash": ModelProfile(
        model_id="deepseek/deepseek-v4-flash",
        context_window_tokens=1_048_576,
        max_output_tokens=384_000,
        supported_reasoning_efforts=("xhigh", "high"),
        default_reasoning_effort="high",
    ),
    # Pin the current target of OpenRouter's floating
    # ``~deepseek/deepseek-v4-flash-latest`` alias for reproducible runs.
    # The conservative completion cap comes from the reviewed model page;
    # individual providers may advertise a larger transport-level limit.
    "deepseek/deepseek-v4-flash-0731": ModelProfile(
        model_id="deepseek/deepseek-v4-flash-0731",
        context_window_tokens=1_310_720,
        max_output_tokens=131_072,
        supported_reasoning_efforts=("max", "high", "low"),
        default_reasoning_effort="high",
    ),
    "openai/gpt-5.6-terra-pro": ModelProfile(
        model_id="openai/gpt-5.6-terra-pro",
        context_window_tokens=1_050_000,
        max_output_tokens=128_000,
        supported_reasoning_efforts=(
            "none",
            "low",
            "medium",
            "high",
            "xhigh",
            "max",
        ),
        default_reasoning_effort="medium",
    ),
}


def resolve_model_profile(model_id: str) -> ModelProfile:
    """Resolve a reviewed snapshot; unknown effort routes remain permissive.

    Exact token budgets are validated separately and fail closed on the empty
    profile returned for an unknown route.
    """
    exact = MODEL_PROFILES.get(model_id)
    if exact is not None:
        return exact
    return ModelProfile(model_id=model_id)


@dataclass
class ProviderSpec:
    """Resolved connection details for an OpenAI-compatible chat endpoint."""

    model_name: str
    api_key: str
    base_url: str
    context_window_tokens: int
    default_headers: dict[str, str] = field(default_factory=dict)
    provider_name: ModelProvider = "openrouter"
    max_output_tokens: int = 4096
    model_context_window_capability_tokens: int | None = None
    model_max_output_tokens: int | None = None
    reasoning_effort: ReasoningEffort | None = None
    reasoning_max_tokens: int | None = None
    supported_reasoning_efforts: tuple[ReasoningEffort, ...] = ()
    provider_default_reasoning_effort: ReasoningEffort | None = None
    reasoning_capability_known: bool = False
    supports_reasoning_max_tokens: bool | None = None
    model_profile_version: str = MODEL_PROFILE_REGISTRY_VERSION
    model_profile_source: str | None = MODEL_PROFILE_SOURCE
    model_profile_fetched_at: str | None = MODEL_PROFILE_FETCHED_AT


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
    # default. OpenAI-compatible local endpoints receive the same explicit
    # setting when configured.
    reasoning_effort: ReasoningEffort | None = None
    # Alternative exact reasoning budget. It is mutually exclusive with effort,
    # shares the completion budget with the visible answer, and fails closed
    # unless the reviewed route profile explicitly guarantees exact support.
    reasoning_max_tokens: int | None = Field(default=None, gt=0)
    agent_action_transport: ActionTransport = "text_json"
    request_timeout_seconds: int = 120
    max_retries: int = 10

    max_iterations: int = 5
    max_cost_usd: float | None = Field(default=None, gt=0)
    test_timeout_seconds: int = 600
    shell_timeout_seconds: int = 300

    # ``checkpoint`` is deterministic and therefore the benchmark default.
    # ``drop`` and model-generated ``summarize`` remain explicit ablations.
    compaction_mode: Literal["drop", "checkpoint", "summarize"] = "checkpoint"
    # Working context budget for the agent's history compactor, in tokens.
    # Distinct from the model's physical context window: the window is what
    # the model *can* take, the budget is what is *useful* to carry — too big
    # and reasoning models burn their completion budget re-thinking a huge
    # history (empty/truncated actions), too small and the agent loses its
    # findings to summarization and re-explores in circles. None = fall back
    # to the provider's context window (compaction effectively off for
    # large-window models). 48000 was a good middle for codex-mini.
    context_budget_tokens: int | None = None
    # Strong single-agent baseline: bound read-only exploration without
    # changing the shorter, independently budgeted multi-agent role loops.
    single_research_guard_enabled: bool = True
    single_research_warning_steps: int = Field(default=12, gt=0)
    single_research_hard_limit: int = Field(default=20, gt=0)
    single_post_plan_research_warning_steps: int = Field(default=5, gt=0)
    single_post_plan_research_hard_limit: int = Field(default=8, gt=0)
    single_compatibility_guard_enabled: bool = True
    single_decomposition_max_subtasks: int = Field(default=8, ge=4, le=12)
    single_decomposition_implementation_warning_steps: int = Field(default=8, gt=0)
    single_decomposition_implementation_hard_limit: int = Field(default=14, gt=0)
    single_decomposition_verification_step_reserve: int = Field(default=20, ge=0)
    single_decomposition_max_repair_cycles: int = Field(default=2, ge=0)

    docker_enabled: bool = True
    docker_image: str | None = None
    docker_network_disabled: bool = True

    # Optional role-specific model routing. Empty values preserve the legacy
    # behavior where every role shares the primary model.
    role_model_orchestrator: str | None = None
    role_model_planner: str | None = None
    role_model_developer: str | None = None
    role_model_tester: str | None = None
    role_model_reviewer: str | None = None
    # The homogeneous comparison uses ``reasoning_effort`` for every role.
    # These fields are opt-in exceptions and are always surfaced in route
    # metadata/fingerprints rather than silently inferred from a role name.
    role_reasoning_effort_orchestrator: ReasoningEffort | None = None
    role_reasoning_effort_planner: ReasoningEffort | None = None
    role_reasoning_effort_developer: ReasoningEffort | None = None
    role_reasoning_effort_tester: ReasoningEffort | None = None
    role_reasoning_effort_reviewer: ReasoningEffort | None = None
    developer_escalation_model: str | None = None
    developer_escalation_reasoning_effort: ReasoningEffort | None = None
    developer_escalate_after_no_edit_episodes: int = Field(default=1, gt=0)
    developer_escalate_after_failed_tests: int = Field(default=1, gt=0)

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

    def provider_spec(
        self,
        model_override: str | None = None,
        reasoning_effort_override: ReasoningEffort | None = None,
    ) -> ProviderSpec:
        raise NotImplementedError(
            f"provider_spec is not implemented for {type(self).__name__}"
        )


class LocalConfig(Config):
    local_api_url: str = "http://localhost:1234/v1"
    local_api_key: str = Field(default="dummy", repr=False)
    local_model_name: str = "qwen3-coder:30b"
    local_context_window_tokens: int = 8192

    def provider_spec(
        self,
        model_override: str | None = None,
        reasoning_effort_override: ReasoningEffort | None = None,
    ) -> ProviderSpec:
        model_name = model_override or self.local_model_name
        # Capability/default reasoning data in MODEL_PROFILES describes the
        # reviewed OpenRouter route, not an arbitrary OpenAI-compatible local
        # endpoint that happens to reuse the same model id.
        profile = ModelProfile(model_id=model_name)
        effort = _validated_reasoning_effort(
            model_name,
            profile,
            self.reasoning_effort
            if reasoning_effort_override is None
            else reasoning_effort_override,
        )
        reasoning_max_tokens = _validated_reasoning_max_tokens(
            model_name,
            profile,
            effort,
            self.reasoning_max_tokens,
            self.max_tokens,
        )
        _validate_max_output_tokens(model_name, self.max_tokens, profile)
        return ProviderSpec(
            model_name=model_name,
            api_key=self.local_api_key,
            base_url=self.local_api_url,
            context_window_tokens=self.local_context_window_tokens,
            provider_name="local",
            max_output_tokens=self.max_tokens,
            model_context_window_capability_tokens=profile.context_window_tokens,
            model_max_output_tokens=profile.max_output_tokens,
            reasoning_effort=effort,
            reasoning_max_tokens=reasoning_max_tokens,
            supported_reasoning_efforts=profile.supported_reasoning_efforts,
            provider_default_reasoning_effort=profile.default_reasoning_effort,
            reasoning_capability_known=(
                bool(profile.supported_reasoning_efforts)
                or profile.supports_reasoning_max_tokens is not None
            ),
            supports_reasoning_max_tokens=profile.supports_reasoning_max_tokens,
            model_profile_version="local-endpoint-capability-unknown-v1",
            model_profile_source=None,
            model_profile_fetched_at=None,
        )


class OpenRouterConfig(Config):
    openrouter_api_url: str = "https://openrouter.ai/api/v1"
    openrouter_api_key: str = Field(default="", repr=False)
    openrouter_model_name: str = "openai/gpt-4o-mini"
    openrouter_context_window_tokens: int = 128000
    openrouter_http_referer: str | None = None
    openrouter_app_title: str = "repo-level-dev-agent-eval"

    def provider_spec(
        self,
        model_override: str | None = None,
        reasoning_effort_override: ReasoningEffort | None = None,
    ) -> ProviderSpec:
        # Optional attribution headers OpenRouter uses for its dashboards.
        headers: dict[str, str] = {}
        if self.openrouter_http_referer:
            headers["HTTP-Referer"] = self.openrouter_http_referer
        if self.openrouter_app_title:
            headers["X-Title"] = self.openrouter_app_title
        model_name = model_override or self.openrouter_model_name
        profile = resolve_model_profile(model_name)
        effort = _validated_reasoning_effort(
            model_name,
            profile,
            self.reasoning_effort
            if reasoning_effort_override is None
            else reasoning_effort_override,
        )
        reasoning_max_tokens = _validated_reasoning_max_tokens(
            model_name,
            profile,
            effort,
            self.reasoning_max_tokens,
            self.max_tokens,
        )
        _validate_max_output_tokens(model_name, self.max_tokens, profile)
        return ProviderSpec(
            model_name=model_name,
            api_key=self.openrouter_api_key,
            base_url=self.openrouter_api_url,
            context_window_tokens=self.openrouter_context_window_tokens,
            default_headers=headers,
            provider_name="openrouter",
            max_output_tokens=self.max_tokens,
            model_context_window_capability_tokens=profile.context_window_tokens,
            model_max_output_tokens=profile.max_output_tokens,
            reasoning_effort=effort,
            reasoning_max_tokens=reasoning_max_tokens,
            supported_reasoning_efforts=profile.supported_reasoning_efforts,
            provider_default_reasoning_effort=profile.default_reasoning_effort,
            reasoning_capability_known=(
                bool(profile.supported_reasoning_efforts)
                or profile.supports_reasoning_max_tokens is not None
            ),
            supports_reasoning_max_tokens=profile.supports_reasoning_max_tokens,
        )


def _validated_reasoning_effort(
    model_name: str,
    profile: ModelProfile,
    effort: ReasoningEffort | None,
) -> ReasoningEffort | None:
    if effort is None:
        return None
    if effort not in REASONING_EFFORTS:
        raise ValueError(
            f"reasoning effort {effort!r} is not one of {', '.join(REASONING_EFFORTS)}"
        )
    if (
        profile.supported_reasoning_efforts
        and effort not in profile.supported_reasoning_efforts
    ):
        supported = ", ".join(profile.supported_reasoning_efforts) or "none declared"
        raise ValueError(
            f"reasoning effort {effort!r} is not supported by {model_name!r}; "
            f"supported: {supported}"
        )
    return effort


def _validate_max_output_tokens(
    model_name: str, configured: int, profile: ModelProfile
) -> None:
    if profile.max_output_tokens is not None and configured > profile.max_output_tokens:
        raise ValueError(
            f"max_tokens={configured} exceeds {model_name!r} capability "
            f"({profile.max_output_tokens})"
        )


def _validated_reasoning_max_tokens(
    model_name: str,
    profile: ModelProfile,
    effort: ReasoningEffort | None,
    reasoning_max_tokens: int | None,
    max_output_tokens: int,
) -> int | None:
    if reasoning_max_tokens is None:
        return None
    if reasoning_max_tokens <= 0:
        raise ValueError("reasoning_max_tokens must be greater than zero")
    if effort is not None:
        raise ValueError(
            "reasoning_effort and reasoning_max_tokens are mutually exclusive"
        )
    if profile.supports_reasoning_max_tokens is not True:
        capability = (
            "is explicitly unsupported"
            if profile.supports_reasoning_max_tokens is False
            else "has no reviewed exact-token guarantee"
        )
        raise ValueError(
            "reasoning_max_tokens requires an explicitly verified exact-token "
            f"route; {model_name!r} {capability}. Use reasoning_effort for a "
            "comparable run."
        )
    if reasoning_max_tokens > max_output_tokens:
        raise ValueError(
            f"reasoning_max_tokens={reasoning_max_tokens} exceeds the shared "
            f"max_tokens={max_output_tokens} completion budget"
        )
    return reasoning_max_tokens


def apply_reasoning_overrides(
    config: Config,
    *,
    effort: ReasoningEffort | None,
    max_tokens: int | None,
) -> None:
    """Apply explicit CLI settings over environment defaults atomically."""
    if effort is not None and max_tokens is not None:
        raise ValueError(
            "reasoning_effort and reasoning_max_tokens are mutually exclusive"
        )
    if effort is not None:
        config.reasoning_effort = effort
        config.reasoning_max_tokens = None
    elif max_tokens is not None:
        config.reasoning_effort = None
        config.reasoning_max_tokens = max_tokens


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
