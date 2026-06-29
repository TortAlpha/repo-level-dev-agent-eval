from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()

ModelProvider = Literal["local", "openrouter"]
AgentMode = Literal["single", "multi"]


class Config(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_ignore_empty=True,
        extra="ignore",
    )

    model_provider: ModelProvider = "openrouter"
    agent_mode: AgentMode = "single"

    temperature: float = 0.2
    max_tokens: int = 4096
    request_timeout_seconds: int = 120
    max_retries: int = 10

    max_iterations: int = 5
    test_timeout_seconds: int = 600
    shell_timeout_seconds: int = 300

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


class LocalConfig(Config):
    local_api_url: str = "http://localhost:1234/v1"
    local_api_key: str = Field(default="dummy", repr=False)
    local_model_name: str | None = None


class OpenRouterConfig(Config):
    openrouter_api_key: str | None = Field(default=None, repr=False)
    openrouter_model_name: str | None = None
    openrouter_api_url: str = "https://openrouter.ai/api/v1"
    openrouter_http_referer: str | None = None
    openrouter_app_title: str = "repo-level-dev-agent-eval"
