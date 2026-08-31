from __future__ import annotations

import os
import subprocess
import sys
from importlib.metadata import PackageNotFoundError
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from pydantic import ValidationError

from src.agents.model import LangChainModel
from src.agents.state import State
from src.agents.swe_agent import SweAgentAdapter, SweAgentError, SweAgentUsage
from src.benchmark.runner import validate_swe_installation_snapshot
from src.run.reproducibility import distribution_dependency_closure_snapshot


def _model() -> LangChainModel:
    return LangChainModel(
        model="priced-model",
        chat=FakeListChatModel(responses=["unused"]),
        input_cost_per_1m=1.0,
        output_cost_per_1m=1.0,
    )


def _state(repo: Path) -> State:
    return State(task="fix behavior", workdir=repo, max_steps=20)


def test_arbitrary_external_adapter_arguments_are_rejected() -> None:
    with pytest.raises(ValidationError, match="typed, fingerprinted field"):
        SweAgentAdapter(
            model=_model(),
            api_key="secret",
            extra_args=["--agent.model.name=attacker/model"],
        )

    launched = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.benchmark.runner",
            "--sweagent-arg=--agent.model.name=attacker/model",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert launched.returncode != 0
    assert "not accepted for comparable runs" in launched.stderr


def test_missing_external_trajectory_marks_accounting_unknown(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    adapter = SweAgentAdapter(model=_model(), api_key="secret")
    completed = subprocess.CompletedProcess(["sweagent"], 0)

    with (
        patch.object(SweAgentAdapter, "_ensure_available"),
        patch.object(SweAgentAdapter, "_ensure_base_image"),
        patch.object(SweAgentAdapter, "_invoke", return_value=completed),
        patch.object(SweAgentAdapter, "_find_patch", return_value=None),
        patch.object(
            SweAgentAdapter, "_read_trajectory", return_value=SweAgentUsage()
        ),
        pytest.raises(SweAgentError, match="valid token-usage trajectory"),
    ):
        adapter.run(_state(repo))

    assert adapter.model.calls == 1
    assert not adapter.model.usage_accounting_complete
    assert adapter.model.effective_cost_usd is None


def test_external_episode_directory_is_never_reused(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    adapter = SweAgentAdapter(model=_model(), api_key="secret")
    completed = subprocess.CompletedProcess(["sweagent"], 0)
    usage = SweAgentUsage(
        steps=1,
        calls=1,
        input_tokens=10,
        output_tokens=2,
        accounting_complete=True,
    )

    with (
        patch.object(SweAgentAdapter, "_ensure_available"),
        patch.object(SweAgentAdapter, "_ensure_base_image"),
        patch.object(SweAgentAdapter, "_invoke", return_value=completed),
        patch.object(SweAgentAdapter, "_find_patch", return_value=None),
        patch.object(SweAgentAdapter, "_read_trajectory", return_value=usage),
    ):
        first = adapter.run(_state(repo))
        with pytest.raises(SweAgentError, match="refusing to reuse"):
            adapter.run(_state(repo))

    assert first.status == "handoff"
    assert (tmp_path / "swe_agent/episode-0000/problem.md").is_file()


def test_external_overlay_labels_only_the_current_run() -> None:
    adapter = SweAgentAdapter(
        model=_model(),
        api_key="secret",
        env_image="sha256:immutable",
        container_label_value="run-123",
    )

    overlay = adapter._overlay_yaml()

    assert "repo-level-dev-agent-eval.swe-run=run-123" in overlay
    assert adapter.wall_timeout_seconds == 7200


def test_external_environment_drops_python_and_litellm_injection() -> None:
    adapter = SweAgentAdapter(
        model=_model(),
        api_key="controlled-key",
        base_url="https://controlled.invalid/api",
    )
    inherited = {
        "PYTHONPATH": "/attacker",
        "PYTHONHOME": "/attacker/runtime",
        "PYTHONSTARTUP": "/attacker/startup.py",
        "LITELLM_CONFIG_PATH": "/attacker/litellm.yaml",
        "LITELLM_LOCAL_MODEL_COST_MAP": "https://attacker.invalid/map.json",
        "SWE_AGENT_CONFIG": "/attacker/swe.yaml",
        "LD_PRELOAD": "/attacker/inject.so",
        "DYLD_INSERT_LIBRARIES": "/attacker/inject.dylib",
        "BASH_ENV": "/attacker/bashrc",
        "KEEP_ME": "benign",
    }

    with patch.dict(os.environ, inherited, clear=True):
        environment = adapter._subprocess_environment()

    assert environment["KEEP_ME"] == "benign"
    assert environment["OPENROUTER_API_KEY"] == "controlled-key"
    assert environment["OPENROUTER_API_BASE"] == "https://controlled.invalid/api"
    assert environment["PYTHONNOUSERSITE"] == "1"
    assert environment["PYTHONSAFEPATH"] == "1"
    assert environment["PYTHONDONTWRITEBYTECODE"] == "1"
    assert environment["PYTHONHASHSEED"] == "0"
    assert environment["PYTHON_DOTENV_DISABLED"] == "1"
    for hostile_key in inherited.keys() - {"KEEP_ME"}:
        assert hostile_key not in environment


def test_external_invoke_uses_resolved_entrypoint_and_episode_cwd(
    tmp_path: Path,
) -> None:
    adapter = SweAgentAdapter(model=_model(), api_key="secret")
    repo = tmp_path / "repo"
    episode = tmp_path / "episode"
    repo.mkdir()
    episode.mkdir()
    problem = episode / "problem.md"
    problem.write_text("fix behavior", encoding="utf-8")
    process = MagicMock()
    process.wait.return_value = 0

    with (
        patch.object(
            SweAgentAdapter,
            "_resolved_binary",
            return_value="/trusted/bin/sweagent",
        ),
        patch("src.agents.swe_agent.subprocess.Popen", return_value=process) as popen,
    ):
        adapter._invoke(repo, problem, episode)

    command = popen.call_args.args[0]
    assert command[0] == "/trusted/bin/sweagent"
    assert popen.call_args.kwargs["cwd"] == episode
    assert popen.call_args.kwargs["start_new_session"] is True


class _FakeDistribution:
    def __init__(self, name: str, version: str, requires: list[str]) -> None:
        self.metadata = {"Name": name}
        self.version = version
        self.requires = requires


def test_external_dependency_snapshot_follows_litellm_transitive_closure() -> None:
    distributions = {
        "sweagent": _FakeDistribution(
            "sweagent",
            "1.1.0",
            ["litellm>=1", "dev-only; extra == 'dev'"],
        ),
        "litellm": _FakeDistribution("litellm", "1.2.0", ["httpx>=0.28"]),
        "httpx": _FakeDistribution("httpx", "0.28.1", []),
    }

    def find_distribution(name: str) -> _FakeDistribution:
        try:
            return distributions[name]
        except KeyError as exc:
            raise PackageNotFoundError(name) from exc

    def content_snapshot(name: str) -> dict[str, object]:
        distribution = distributions[name]
        return {
            "name": name,
            "version": distribution.version,
            "aggregate_sha256": f"content-{name}-{distribution.version}",
            "file_count": 1,
            "missing_files": [],
            "editable_source": None,
            "verifiable": True,
        }

    with (
        patch(
            "src.run.reproducibility.importlib.metadata.distribution",
            side_effect=find_distribution,
        ),
        patch(
            "src.run.reproducibility.distribution_content_snapshot",
            side_effect=content_snapshot,
        ),
    ):
        snapshot = distribution_dependency_closure_snapshot(["sweagent"])
        distributions["litellm"].version = "1.3.0"
        changed = distribution_dependency_closure_snapshot(["sweagent"])

    assert set(snapshot["distributions"]) == {"sweagent", "litellm", "httpx"}
    assert "dev-only" not in snapshot["distributions"]
    assert snapshot["verifiable"]
    assert snapshot["aggregate_sha256"] != changed["aggregate_sha256"]


def test_sweep_runtime_contract_rejects_dependency_closure_drift() -> None:
    adapter = SweAgentAdapter(model=_model(), api_key="secret")
    runtime = {
        "verifiable": True,
        "aggregate_sha256": "changed-runtime",
        "dependency_closure": {
            "missing_distributions": [],
            "distributions": {
                "sweagent": {"aggregate_sha256": "swe-hash", "verifiable": True},
                "swe-rex": {"aggregate_sha256": "rex-hash", "verifiable": True},
            },
        },
    }
    args = SimpleNamespace(
        expected_swe_runtime_sha256="frozen-runtime",
        expected_sweagent_distribution_sha256="swe-hash",
        expected_swerex_distribution_sha256="rex-hash",
    )

    with (
        patch(
            "src.benchmark.runner.external_swe_runtime_snapshot",
            return_value=runtime,
        ),
        pytest.raises(RuntimeError, match="dependency closure changed"),
    ):
        validate_swe_installation_snapshot(
            adapter,
            args,
            require_verifiable=True,
        )
