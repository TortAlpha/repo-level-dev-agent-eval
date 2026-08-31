from __future__ import annotations

from http import HTTPStatus

import pytest

from src.config import MODEL_PROFILES, REASONING_EFFORTS, ModelProfile
from src.console import server
from src.metrics.records import RunRecord


def _catalog(models: dict[str, dict]) -> dict:
    return {"source": "test", "complete": True, "models": models}


def _capability(
    *,
    efforts: list[str] | None = None,
    supports_reasoning: bool = True,
    supports_effort: bool = True,
    supports_max_tokens: bool | None = None,
) -> dict:
    return {
        "supports_reasoning": supports_reasoning,
        "supports_effort": supports_effort,
        "supported_efforts": efforts,
        "default_effort": None,
        "mandatory": None,
        "default_enabled": None,
        "supports_max_tokens": supports_max_tokens,
    }


def test_console_uses_the_canonical_full_effort_scale() -> None:
    assert REASONING_EFFORTS == (
        "none",
        "minimal",
        "low",
        "medium",
        "high",
        "xhigh",
        "max",
    )
    assert server.REASONING_EFFORTS is REASONING_EFFORTS


def test_catalog_parser_preserves_reasoning_capabilities() -> None:
    parsed = server.parse_reasoning_catalog(
        {
            "data": [
                {
                    "id": "vendor/reasoner",
                    "supported_parameters": ["reasoning", "reasoning_effort"],
                    "reasoning": {
                        "supported_efforts": ["high", "none", "low"],
                        "default_effort": "high",
                        "mandatory": False,
                        "default_enabled": True,
                        "supports_max_tokens": True,
                    },
                },
                {"id": "vendor/plain", "supported_parameters": ["tools"]},
            ]
        }
    )

    assert parsed["vendor/reasoner"] == {
        "supports_reasoning": True,
        "supports_effort": True,
        "supported_efforts": ["none", "low", "high"],
        "default_effort": "high",
        "mandatory": False,
        "default_enabled": True,
        "supports_max_tokens": True,
    }
    assert parsed["vendor/plain"]["supports_reasoning"] is False


def test_effort_is_rejected_if_any_selected_route_does_not_support_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        server,
        "reasoning_catalog",
        lambda: _catalog(
            {
                "vendor/base": _capability(efforts=["low", "high"]),
                "vendor/reviewer": _capability(efforts=["low"]),
            }
        ),
    )
    command: list[str] = []

    with pytest.raises(server.ApiError) as caught:
        server.append_reasoning_flags(
            command,
            {
                "model": "vendor/base",
                "role_models": ["reviewer=vendor/reviewer"],
                "reasoning_effort": "high",
            },
            "openrouter",
        )

    assert caught.value.status == HTTPStatus.BAD_REQUEST
    assert "vendor/reviewer" in caught.value.message
    assert command == []


def test_supported_shared_effort_is_forwarded_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        server,
        "reasoning_catalog",
        lambda: _catalog(
            {
                "vendor/base": _capability(efforts=["low", "high"]),
                "vendor/escalation": _capability(efforts=["low"]),
            }
        ),
    )
    command: list[str] = []

    server.append_reasoning_flags(
        command,
        {
            "models": "vendor/base",
            "developer_escalation_model": "vendor/escalation",
            "reasoning_effort": "low",
        },
        "openrouter",
    )

    assert command == ["--reasoning-effort", "low"]


def test_max_tokens_is_mutually_exclusive_and_capability_checked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(
        MODEL_PROFILES,
        "vendor/reasoner",
        ModelProfile(
            model_id="vendor/reasoner",
            supports_reasoning_max_tokens=True,
        ),
    )
    monkeypatch.setattr(
        server,
        "reasoning_catalog",
        lambda: _catalog(
            {"vendor/reasoner": _capability(efforts=["low"], supports_max_tokens=False)}
        ),
    )

    with pytest.raises(server.ApiError, match="mutually exclusive"):
        server.append_reasoning_flags(
            [],
            {
                "model": "vendor/reasoner",
                "reasoning_effort": "low",
                "reasoning_max_tokens": 512,
            },
            "openrouter",
        )
    with pytest.raises(server.ApiError, match="reasoning_max_tokens"):
        server.append_reasoning_flags(
            [],
            {"model": "vendor/reasoner", "reasoning_max_tokens": 512},
            "openrouter",
        )


def test_max_tokens_requires_a_frozen_exactness_guarantee(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        server,
        "reasoning_catalog",
        lambda: _catalog(
            {
                "vendor/unreviewed": _capability(supports_max_tokens=True),
                "vendor/reviewed": _capability(supports_max_tokens=True),
            }
        ),
    )

    with pytest.raises(server.ApiError, match="exactness is not verified"):
        server.append_reasoning_flags(
            [],
            {"model": "vendor/unreviewed", "reasoning_max_tokens": 512},
            "openrouter",
        )

    monkeypatch.setitem(
        MODEL_PROFILES,
        "vendor/reviewed",
        ModelProfile(
            model_id="vendor/reviewed",
            supports_reasoning_max_tokens=True,
        ),
    )
    command: list[str] = []
    server.append_reasoning_flags(
        command,
        {"model": "vendor/reviewed", "reasoning_max_tokens": 512},
        "openrouter",
    )
    assert command == ["--reasoning-max-tokens", "512"]

    with pytest.raises(server.ApiError, match="local route capabilities"):
        server.append_reasoning_flags(
            [],
            {"model": "local-reasoner", "reasoning_max_tokens": 512},
            "local",
        )


def test_live_exact_token_claim_is_unverified_without_frozen_profile() -> None:
    capabilities = server._apply_reviewed_exact_token_capabilities(
        {"vendor/live-only": _capability(supports_max_tokens=True)}
    )

    assert capabilities["vendor/live-only"]["supports_max_tokens"] is None


def test_offline_catalog_uses_incomplete_reviewed_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def offline(*_args, **_kwargs):
        raise OSError("offline")

    monkeypatch.setattr(server, "urlopen", offline)
    monkeypatch.setattr(server, "_reasoning_catalog_cache", (0.0, None))

    catalog = server.reasoning_catalog()

    assert catalog["source"] == "model_profiles"
    assert catalog["complete"] is False
    assert "z-ai/glm-5.2" in catalog["models"]
    assert catalog["models"]["z-ai/glm-5.2"]["default_effort"] == "high"


def test_run_serializer_uses_recorded_effective_cost_and_new_metadata(tmp_path) -> None:
    record = RunRecord.from_dict(
        {
            "task_id": "task",
            "run_id": "run",
            "session_id": "session",
            "agent_mode": "single",
            "model": "vendor/reasoner",
            "status": "solved",
            "steps": 1,
            "iterations": 1,
            "test_passed": True,
            "hidden_tests_passed": True,
            "llm_calls": 2,
            "cost_usd": 0.25,
            "provider_reported_cost_usd": 0.10,
            "provider_cost_calls": 1,
            "usage_accounting_complete": True,
            "effective_cost_usd": 0.25,
            "cost_source": "static_estimate",
            "reasoning_effort": "high",
            "reasoning_tokens": 900,
            "cache_write_input_tokens": 50,
            "model_routes": {"agent": {"model": "vendor/reasoner"}},
            "run_fingerprint": "abc123",
            "test_oracle_tamper_attempts": 2,
        }
    )

    payload = server.serialize_run(record, tmp_path, {})

    assert payload["cost_usd"] == 0.25
    assert payload["effective_cost_usd"] == 0.25
    assert payload["provider_reported_cost_usd"] == 0.10
    assert payload["provider_cost_complete"] is False
    assert payload["usage_accounting_complete"] is True
    assert payload["cost_source"] == "static_estimate"
    assert payload["reasoning_tokens"] == 900
    assert payload["cache_write_input_tokens"] == 50
    assert payload["run_fingerprint"] == "abc123"
    assert payload["test_oracle_tamper_attempts"] == 2
