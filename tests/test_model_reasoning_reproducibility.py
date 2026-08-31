from __future__ import annotations

import json
import multiprocessing
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from langchain_core.language_models.fake_chat_models import (
    FakeListChatModel,
    FakeMessagesListChatModel,
)
from langchain_core.messages import AIMessage

from src.agents.model import GeneratedAction, LangChainModel
from src.agents.single_agent import SingleAgent
from src.agents.state import State
from src.benchmark.runner import (
    configured_agent_model_routes,
    reasoning_policy_label,
    validate_frozen_evaluation_trees,
)
from src.benchmark.sweep import _budget_cost, _effective_cost
from src.config import (
    MODEL_PROFILES,
    REASONING_EFFORTS,
    LocalConfig,
    ModelProfile,
    OpenRouterConfig,
    apply_reasoning_overrides,
)
from src.metrics.pricing import ModelPrice, estimate_cost
from src.metrics.records import load_runs
from src.run.models import build_chat_model
from src.run.reproducibility import (
    HARNESS_SOURCE_PATHS,
    POLICY_KERNEL_FILES,
    agent_prompt_manifest,
    build_run_reproducibility,
    docker_image_identity,
    filesystem_tree_identity,
    source_repository_identity,
)
from src.run.results import _append_jsonl_record, run_metrics


def _append_jsonl_worker(path: str, worker: int, count: int) -> None:
    for index in range(count):
        _append_jsonl_record(
            Path(path),
            {"worker": worker, "index": index, "payload": "x" * 2048},
        )


class _ToolMessagesModel(FakeMessagesListChatModel):
    def bind_tools(self, tools, *, tool_choice=None, **kwargs):  # noqa: ANN001
        return self


class ModelReasoningReproducibilityTests(unittest.TestCase):
    def state(self, root: Path) -> State:
        return State(task="fix the exact behavior", workdir=root, max_steps=10)

    def test_effort_scale_and_reviewed_capability_validation(self) -> None:
        self.assertEqual(
            REASONING_EFFORTS,
            ("none", "minimal", "low", "medium", "high", "xhigh", "max"),
        )
        known = OpenRouterConfig(
            _env_file=None,
            openrouter_api_key="key",
            openrouter_model_name="z-ai/glm-5.2",
            reasoning_effort="high",
        ).provider_spec()
        self.assertTrue(known.reasoning_capability_known)
        self.assertEqual(known.reasoning_effort, "high")
        self.assertEqual(known.context_window_tokens, 128_000)
        self.assertEqual(known.model_context_window_capability_tokens, 1_048_576)
        self.assertEqual(known.model_max_output_tokens, 262_144)
        self.assertEqual(known.provider_default_reasoning_effort, "high")
        with self.assertRaisesRegex(ValueError, "not supported"):
            OpenRouterConfig(
                _env_file=None,
                openrouter_api_key="key",
                openrouter_model_name="z-ai/glm-5.2",
                reasoning_effort="medium",
            ).provider_spec()

        deepseek = OpenRouterConfig(
            _env_file=None,
            openrouter_api_key="key",
            openrouter_model_name="deepseek/deepseek-v4-flash-0731",
            reasoning_effort="high",
        ).provider_spec()
        self.assertTrue(deepseek.reasoning_capability_known)
        self.assertEqual(deepseek.model_context_window_capability_tokens, 1_310_720)
        self.assertEqual(deepseek.model_max_output_tokens, 131_072)
        self.assertEqual(deepseek.supported_reasoning_efforts, ("max", "high", "low"))
        self.assertEqual(deepseek.provider_default_reasoning_effort, "high")
        with self.assertRaisesRegex(ValueError, "not supported"):
            OpenRouterConfig(
                _env_file=None,
                openrouter_api_key="key",
                openrouter_model_name="deepseek/deepseek-v4-flash-0731",
                reasoning_effort="medium",
            ).provider_spec()

        unknown = OpenRouterConfig(
            _env_file=None,
            openrouter_api_key="key",
            openrouter_model_name="vendor/new-model",
            reasoning_effort="minimal",
        ).provider_spec()
        self.assertFalse(unknown.reasoning_capability_known)
        self.assertEqual(unknown.reasoning_effort, "minimal")

    def test_reasoning_effort_and_exact_budget_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            OpenRouterConfig(
                _env_file=None,
                openrouter_api_key="key",
                openrouter_model_name="deepseek/deepseek-v4-flash",
                reasoning_effort="high",
                reasoning_max_tokens=1024,
            ).provider_spec()

        for model_name in (
            "deepseek/deepseek-v4-flash",
            "vendor/capability-unknown",
        ):
            with self.subTest(model_name=model_name):
                with self.assertRaisesRegex(
                    ValueError, "explicitly verified exact-token route"
                ):
                    OpenRouterConfig(
                        _env_file=None,
                        openrouter_api_key="key",
                        openrouter_model_name=model_name,
                        reasoning_max_tokens=1024,
                    ).provider_spec()

        exact_model = "vendor/reviewed-exact-reasoner"
        with patch.dict(
            MODEL_PROFILES,
            {
                exact_model: ModelProfile(
                    model_id=exact_model,
                    max_output_tokens=4096,
                    supports_reasoning_max_tokens=True,
                )
            },
        ):
            spec = OpenRouterConfig(
                _env_file=None,
                openrouter_api_key="key",
                openrouter_model_name=exact_model,
                reasoning_max_tokens=1024,
            ).provider_spec()
        self.assertEqual(spec.reasoning_max_tokens, 1024)
        self.assertTrue(spec.supports_reasoning_max_tokens)
        self.assertTrue(spec.reasoning_capability_known)

        config = OpenRouterConfig(
            _env_file=None,
            openrouter_api_key="key",
            reasoning_max_tokens=1,
        )
        config.reasoning_max_tokens = -7
        with self.assertRaisesRegex(ValueError, "greater than zero"):
            config.provider_spec()

    def test_cli_override_is_atomic_and_local_capability_remains_unknown(self) -> None:
        config = OpenRouterConfig(
            _env_file=None,
            openrouter_api_key="key",
            reasoning_effort="high",
        )
        apply_reasoning_overrides(config, effort=None, max_tokens=512)
        self.assertIsNone(config.reasoning_effort)
        self.assertEqual(config.reasoning_max_tokens, 512)

        local = LocalConfig(
            _env_file=None,
            local_model_name="z-ai/glm-5.2",
            reasoning_effort="medium",
        ).provider_spec()
        self.assertFalse(local.reasoning_capability_known)
        self.assertIsNone(local.provider_default_reasoning_effort)
        self.assertIsNone(local.model_profile_source)

        effort_config = LocalConfig(
            _env_file=None,
            local_api_key="key",
            reasoning_effort="high",
        )
        effort_chat = build_chat_model(effort_config, effort_config.provider_spec())
        self.assertEqual(effort_chat.reasoning_effort, "high")

        exact_config = LocalConfig(
            _env_file=None,
            local_api_key="key",
            reasoning_max_tokens=321,
        )
        with self.assertRaisesRegex(ValueError, "no reviewed exact-token guarantee"):
            exact_config.provider_spec()

    def test_openrouter_request_includes_usage_and_exact_reasoning_budget(self) -> None:
        model_name = "vendor/reviewed-exact-reasoner"
        with patch.dict(
            MODEL_PROFILES,
            {
                model_name: ModelProfile(
                    model_id=model_name,
                    max_output_tokens=4096,
                    supports_reasoning_max_tokens=True,
                )
            },
        ):
            config = OpenRouterConfig(
                _env_file=None,
                openrouter_api_key="key",
                openrouter_model_name=model_name,
                reasoning_max_tokens=777,
            )
            chat = build_chat_model(config, config.provider_spec())

        self.assertEqual(chat.extra_body["usage"], {"include": True})
        self.assertEqual(chat.extra_body["reasoning"], {"max_tokens": 777})

        raw_response = {
            "id": "completion-1",
            "object": "chat.completion",
            "created": 0,
            "model": model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": '{"action":"list_dir","path":"."}',
                        "reasoning_details": [
                            {"type": "reasoning.text", "id": "reasoning-1"}
                        ],
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 2,
                "total_tokens": 12,
                "cached_tokens": 4,
                "cache_write_tokens": 3,
                "cost": 0.000012,
            },
        }
        message = chat._create_chat_result(raw_response).generations[0].message
        model = LangChainModel(
            model=model_name,
            chat=chat,
            preserve_reasoning_history=True,
        )
        model._record_usage(message)

        self.assertEqual(model.calls, 1)
        self.assertEqual(model.cached_input_tokens, 4)
        self.assertEqual(model.cache_write_input_tokens, 3)
        self.assertEqual(model.provider_cost_calls, 1)
        self.assertEqual(model.effective_cost_usd, 0.000012)
        self.assertEqual(
            message.additional_kwargs["_provider_assistant_message"],
            raw_response["choices"][0]["message"],
        )
        self.assertEqual(
            chat._get_request_payload([message])["messages"][0],
            raw_response["choices"][0]["message"],
        )

    def test_transient_retry_honors_bounded_provider_retry_after(self) -> None:
        error = RuntimeError("Error code: 429 - retry_after_seconds: 12")

        class FlakyChat:
            def __init__(self) -> None:
                self.calls = 0

            def invoke(self, messages, **kwargs):  # noqa: ANN001
                self.calls += 1
                if self.calls == 1:
                    raise error
                return AIMessage(content="ok")

        chat = FlakyChat()
        with patch("src.agents.model.time.sleep") as sleep:
            result = LangChainModel._invoke_with_transient_retry(chat, [])

        self.assertEqual(result.content, "ok")
        self.assertEqual(chat.calls, 2)
        sleep.assert_called_once_with(12.0)

    def test_truncation_retry_respects_model_output_capability(self) -> None:
        config = OpenRouterConfig(
            _env_file=None,
            openrouter_api_key="key",
            openrouter_model_name="openai/gpt-5.6-terra-pro",
            max_tokens=100_000,
        )
        spec = config.provider_spec()
        model = LangChainModel(
            model=spec.model_name,
            chat=build_chat_model(config, spec),
            model_max_output_tokens=spec.model_max_output_tokens,
        )

        self.assertEqual(model._escalated_max_tokens(), 128_000)

    def test_provider_reasoning_payload_survives_durable_state_replay(self) -> None:
        raw = {
            "role": "assistant",
            "content": '{"action":"list_dir","path":"."}',
            "reasoning_details": [{"type": "reasoning.text", "id": "r1"}],
        }
        response = AIMessage(
            content=raw["content"],
            additional_kwargs={
                "_provider_assistant_message": raw,
                "reasoning_details": raw["reasoning_details"],
            },
        )
        with tempfile.TemporaryDirectory() as directory:
            state = self.state(Path(directory))
            model = LangChainModel(
                model="reasoner",
                chat=FakeMessagesListChatModel(responses=[response]),
                preserve_reasoning_history=True,
            )
            generated = model.generate_action("system", state, "text_json")
            self.assertIsInstance(generated, GeneratedAction)
            assert isinstance(generated, GeneratedAction)
            stored = state.with_context(
                kind="list_dir", text="file.py"
            ).with_last_action_json(
                generated.action_json,
                generated.tool_call_id or None,
                generated.tool_name or None,
                generated.provider_assistant_message,
                generated.provider_reasoning_details,
            )

            # A fresh wrapper proves replay comes from durable ContextEntry,
            # not an in-process sidecar owned by the generating model.
            fresh = LangChainModel(
                model="reasoner",
                chat=FakeListChatModel(responses=["unused"]),
                preserve_reasoning_history=True,
            )
            assistant = next(
                message
                for message in fresh._messages("system", stored)
                if isinstance(message, AIMessage)
            )
            replay = assistant.additional_kwargs["_provider_assistant_message"]

        self.assertEqual(replay["reasoning_details"], raw["reasoning_details"])
        self.assertEqual(replay["content"], raw["content"])

    def test_native_tool_response_records_usage_once(self) -> None:
        response = AIMessage(
            content="",
            tool_calls=[{"id": "call-1", "name": "list_dir", "args": {"path": "."}}],
            usage_metadata={
                "input_tokens": 10,
                "output_tokens": 2,
                "total_tokens": 12,
            },
        )
        with tempfile.TemporaryDirectory() as directory:
            model = LangChainModel(
                model="tool-model",
                chat=_ToolMessagesModel(responses=[response]),
            )
            generated = model.generate_action(
                "system", self.state(Path(directory)), "tools"
            )

        self.assertIsInstance(generated, GeneratedAction)
        self.assertEqual(model.calls, 1)
        self.assertEqual(model.input_tokens, 10)
        self.assertEqual(model.output_tokens, 2)

    def test_usage_tracks_reasoning_cache_write_and_actual_cost(self) -> None:
        model = LangChainModel(
            model="priced",
            chat=FakeListChatModel(responses=["unused"]),
            input_cost_per_1m=1.0,
            output_cost_per_1m=2.0,
            cached_input_cost_per_1m=0.0,
            cache_write_input_cost_per_1m=0.0,
        )
        model._record_usage(
            AIMessage(
                content="ok",
                usage_metadata={
                    "input_tokens": 100,
                    "output_tokens": 20,
                    "total_tokens": 120,
                    "input_token_details": {
                        "cache_read": 40,
                        "cache_write_tokens": 10,
                    },
                    "output_token_details": {"reasoning_tokens": 12},
                },
                response_metadata={"usage": {"cost": 0.000031}},
            )
        )

        self.assertEqual(model.cached_input_tokens, 40)
        self.assertEqual(model.cache_write_input_tokens, 10)
        self.assertEqual(model.reasoning_tokens, 12)
        self.assertAlmostEqual(model.estimated_cost_usd or 0, 0.00009)
        self.assertAlmostEqual(model.effective_cost_usd or 0, 0.000031)
        self.assertAlmostEqual(
            estimate_cost(
                "x",
                100,
                20,
                override=ModelPrice(1.0, 2.0, 0.0, 0.0),
                cached_input_tokens=40,
                cache_write_input_tokens=10,
            )
            or 0,
            0.00009,
        )

    def test_partial_provider_cost_coverage_uses_conservative_estimate(self) -> None:
        model = LangChainModel(
            model="priced",
            chat=FakeListChatModel(responses=["unused"]),
            calls=2,
            input_tokens=1_000_000,
            provider_cost_calls=1,
            provider_reported_cost_usd=0.01,
            input_cost_per_1m=1.0,
            output_cost_per_1m=1.0,
        )
        agent = SingleAgent(model=model, max_cost_usd=0.5)
        with tempfile.TemporaryDirectory() as directory:
            stopped = agent._enforce_cost_limit(self.state(Path(directory)))
            metrics = run_metrics(model, stopped, 1.0)

        self.assertEqual(stopped.status, "handoff")
        self.assertEqual(model.cost_source, "actual_plus_static_upper_bound")
        self.assertEqual(model.effective_cost_usd, 1.01)
        self.assertFalse(metrics["provider_cost_complete"])
        self.assertEqual(metrics["effective_cost_usd"], 1.01)
        self.assertEqual(metrics["cost_source"], "actual_plus_static_upper_bound")
        self.assertEqual(_effective_cost(metrics), 1.01)

    def test_upstream_provider_expense_is_not_treated_as_billed_cost(self) -> None:
        model = LangChainModel(
            model="priced",
            chat=FakeListChatModel(responses=["unused"]),
            input_cost_per_1m=1.0,
            output_cost_per_1m=1.0,
        )
        model._record_usage(
            AIMessage(
                content="ok",
                usage_metadata={
                    "input_tokens": 10,
                    "output_tokens": 5,
                    "total_tokens": 15,
                },
                response_metadata={
                    "usage": {"cost_details": {"upstream_inference_cost": 0.000001}}
                },
            )
        )

        self.assertEqual(model.provider_cost_calls, 0)
        self.assertEqual(model.cost_source, "static_estimate")
        self.assertAlmostEqual(model.effective_cost_usd or 0, 0.000015)

        self.assertIsNone(LangChainModel._reported_cost({"cost": -1}, {}))
        self.assertIsNone(LangChainModel._reported_cost({"cost": "nan"}, {}))

    def test_unused_unpriced_role_does_not_hide_complete_actual_cost(self) -> None:
        called = LangChainModel(
            model="called",
            chat=FakeListChatModel(responses=["unused"]),
            calls=1,
            provider_cost_calls=1,
            provider_reported_cost_usd=0.2,
        )
        unused = LangChainModel(
            model="unused-unpriced",
            chat=FakeListChatModel(responses=["unused"]),
        )
        with tempfile.TemporaryDirectory() as directory:
            metrics = run_metrics(
                [called, unused],
                self.state(Path(directory)),
                1.0,
            )

        self.assertEqual(unused.effective_cost_usd, 0.0)
        self.assertTrue(metrics["provider_cost_complete"])
        self.assertEqual(metrics["effective_cost_usd"], 0.2)
        self.assertEqual(metrics["cost_source"], "provider_actual")

        missing_usage = LangChainModel(
            model="called-without-usage",
            chat=FakeListChatModel(responses=["unused"]),
            calls=1,
        )
        self.assertIsNone(missing_usage.effective_cost_usd)

    def test_budgeted_sweep_fails_closed_when_recorded_cost_is_unknown(self) -> None:
        args = SimpleNamespace(max_total_cost_usd=1.0)
        with (
            patch(
                "src.benchmark.sweep._budget_records",
                return_value=[{"run_id": "unknown-cost"}],
            ),
            self.assertRaisesRegex(SystemExit, "effective cost is unknown"),
        ):
            _budget_cost(args)

    def test_incomplete_usage_overrides_recorded_and_estimated_costs(self) -> None:
        row = {
            "run_id": "incomplete-usage",
            "usage_accounting_complete": False,
            "effective_cost_usd": 0.25,
            "cost_usd": 0.20,
            "provider_reported_cost_usd": 0.10,
            "provider_cost_calls": 1,
            "llm_calls": 1,
        }

        self.assertIsNone(_effective_cost(row))
        args = SimpleNamespace(max_total_cost_usd=1.0)
        with (
            patch("src.benchmark.sweep._budget_records", return_value=[row]),
            self.assertRaisesRegex(SystemExit, "effective cost is unknown"),
        ):
            _budget_cost(args)

        model = LangChainModel(
            model="priced",
            chat=FakeListChatModel(responses=["unused"]),
            calls=1,
            input_tokens=10,
            output_tokens=5,
            input_cost_per_1m=1.0,
            output_cost_per_1m=1.0,
            usage_accounting_complete=False,
        )
        with tempfile.TemporaryDirectory() as directory:
            metrics = run_metrics(model, self.state(Path(directory)), 1.0)
        self.assertFalse(metrics["usage_accounting_complete"])
        self.assertEqual(metrics["cost_source"], "unknown_incomplete_usage")
        self.assertNotIn("effective_cost_usd", metrics)

    def test_unknown_cost_source_overrides_recorded_effective_cost(self) -> None:
        row = {
            "run_id": "unknown-source",
            "usage_accounting_complete": True,
            "cost_source": "unknown_incomplete_usage",
            "effective_cost_usd": 0.25,
        }

        self.assertIsNone(_effective_cost(row))

    def test_run_record_loads_effective_cost_without_property_collision(
        self,
    ) -> None:
        row = {
            "task_id": "t",
            "run_id": "r",
            "agent_mode": "single",
            "model": "m",
            "status": "failed",
            "steps": 0,
            "iterations": 0,
            "test_passed": False,
            "effective_cost_usd": 0.25,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "runs.jsonl"
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            record = load_runs(path)[0]
        self.assertEqual(record.effective_cost_usd, 0.25)

    def test_run_record_parses_incomplete_usage_and_fails_cost_closed(self) -> None:
        row = {
            "task_id": "t",
            "run_id": "r",
            "agent_mode": "single",
            "model": "m",
            "status": "failed",
            "steps": 1,
            "iterations": 1,
            "test_passed": False,
            "usage_accounting_complete": False,
            "effective_cost_usd": 0.25,
            "cost_usd": 0.20,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "runs.jsonl"
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            record = load_runs(path)[0]

        self.assertFalse(record.usage_accounting_complete)
        self.assertIsNone(record.effective_cost_usd)

    def test_run_fingerprint_is_stable_and_changes_with_semantics(self) -> None:
        model = LangChainModel(
            model="m",
            chat=FakeListChatModel(responses=["unused"]),
            provider="openrouter",
            base_url="https://openrouter.ai/api/v1",
            context_window_tokens=128_000,
            max_output_tokens=4096,
            reasoning_effort="none",
        )
        agent = SingleAgent(model=model, compaction_mode="drop")
        routes = {"agent": model.route_metadata(transport="text_json")}
        first = build_run_reproducibility(
            task_content="task body",
            agent=agent,
            policy_configuration={"max_steps": 50, "guard": True},
            model_routes=routes,
        )
        second = build_run_reproducibility(
            task_content="task body",
            agent=agent,
            policy_configuration={"max_steps": 50, "guard": True},
            model_routes=routes,
        )
        changed = build_run_reproducibility(
            task_content="changed task body",
            agent=agent,
            policy_configuration={"max_steps": 50, "guard": True},
            model_routes=routes,
        )

        self.assertEqual(first["run_fingerprint"], second["run_fingerprint"])
        self.assertNotEqual(first["run_fingerprint"], changed["run_fingerprint"])
        self.assertTrue(first["system_prompts"]["aggregate_sha256"])
        self.assertTrue(first["tool_schemas"]["aggregate_sha256"])
        self.assertTrue(first["policy_kernel"]["sources"]["aggregate_sha256"])
        for source in (
            "src/agents/state/rendering.py",
            "src/benchmark/runner.py",
            "src/benchmark/evaluation.py",
            "src/benchmark/evaluator_bin/python",
            "src/benchmark/evaluator_bin/pytest",
            "src/run/results.py",
            "src/metrics/pricing.py",
            "static/pricing.csv",
        ):
            self.assertIn(source, POLICY_KERNEL_FILES)
        self.assertIn("src/git_safety.py", HARNESS_SOURCE_PATHS)

    def test_source_repository_identity_locks_clean_and_dirty_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
            subprocess.run(
                ["git", "config", "user.email", "eval@example.test"],
                cwd=repo,
                check=True,
            )
            subprocess.run(
                ["git", "config", "user.name", "Eval Test"],
                cwd=repo,
                check=True,
            )
            (repo / "tracked.txt").write_text("base\n", encoding="utf-8")
            subprocess.run(["git", "add", "tracked.txt"], cwd=repo, check=True)
            subprocess.run(["git", "commit", "-q", "-m", "base"], cwd=repo, check=True)
            head = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=repo,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()

            clean = source_repository_identity(
                repo,
                declared_base_commit=head,
                require_clean=True,
            )
            (repo / "tracked.txt").write_text("changed\n", encoding="utf-8")
            (repo / "untracked.bin").write_bytes(b"\x00new")
            dirty = source_repository_identity(repo)

            self.assertTrue(clean["base_commit_validated"])
            self.assertFalse(clean["dirty"])
            self.assertEqual(clean["head_commit"], head)
            self.assertTrue(dirty["dirty"])
            self.assertNotEqual(clean["worktree_sha256"], dirty["worktree_sha256"])
            with self.assertRaisesRegex(ValueError, "is dirty"):
                source_repository_identity(repo, require_clean=True)

    def test_run_fingerprint_uses_pre_run_source_snapshot(self) -> None:
        model = LangChainModel(
            model="m",
            chat=FakeListChatModel(responses=["unused"]),
        )
        agent = SingleAgent(model=model)
        routes = {"agent": model.route_metadata(transport="text_json")}
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
            subprocess.run(
                ["git", "config", "user.email", "eval@example.test"],
                cwd=repo,
                check=True,
            )
            subprocess.run(
                ["git", "config", "user.name", "Eval Test"],
                cwd=repo,
                check=True,
            )
            (repo / "file.py").write_text("value = 1\n", encoding="utf-8")
            subprocess.run(["git", "add", "file.py"], cwd=repo, check=True)
            subprocess.run(["git", "commit", "-q", "-m", "base"], cwd=repo, check=True)
            before = source_repository_identity(repo, require_clean=True)
            first = build_run_reproducibility(
                task_content="task",
                agent=agent,
                policy_configuration={"max_steps": 5},
                model_routes=routes,
                source_repository=before,
            )
            (repo / "file.py").write_text("value = 2\n", encoding="utf-8")
            replay = build_run_reproducibility(
                task_content="task",
                agent=agent,
                policy_configuration={"max_steps": 5},
                model_routes=routes,
                source_repository=before,
            )
            changed = build_run_reproducibility(
                task_content="task",
                agent=agent,
                policy_configuration={"max_steps": 5},
                model_routes=routes,
                source_repository=source_repository_identity(repo),
            )

        self.assertEqual(first["run_fingerprint"], replay["run_fingerprint"])
        self.assertNotEqual(first["run_fingerprint"], changed["run_fingerprint"])
        self.assertEqual(first["source_repository"], before)

    def test_parallel_jsonl_append_never_interleaves_records(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "runs.jsonl"
            methods = multiprocessing.get_all_start_methods()
            context = multiprocessing.get_context(
                "fork" if "fork" in methods else methods[0]
            )
            processes = [
                context.Process(
                    target=_append_jsonl_worker,
                    args=(str(path), worker, 16),
                )
                for worker in range(4)
            ]
            for process in processes:
                process.start()
            try:
                for process in processes:
                    process.join(30)
                    self.assertEqual(process.exitcode, 0)
            finally:
                for process in processes:
                    if process.is_alive():
                        process.terminate()
                        process.join(5)

            rows = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
            ]

        self.assertEqual(len(rows), 64)
        self.assertEqual(
            {(row["worker"], row["index"]) for row in rows},
            {(worker, index) for worker in range(4) for index in range(16)},
        )

    def test_jsonl_append_fails_closed_without_an_os_lock(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "runs.jsonl"
            with (
                patch("src.run.results._fcntl", None),
                patch("src.run.results._msvcrt", None),
                self.assertRaisesRegex(RuntimeError, "No supported OS file lock"),
            ):
                _append_jsonl_record(path, {"run_id": "unsafe"})

            self.assertFalse(path.exists())

    def test_docker_image_identity_resolves_immutable_id(self) -> None:
        first = subprocess.CompletedProcess(
            [],
            0,
            stdout=json.dumps(
                [{"Id": "sha256:first", "RepoDigests": ["repo@sha256:first"]}]
            ),
            stderr="",
        )
        second = subprocess.CompletedProcess(
            [],
            0,
            stdout=json.dumps(
                [{"Id": "sha256:second", "RepoDigests": ["repo@sha256:second"]}]
            ),
            stderr="",
        )
        with patch(
            "src.run.reproducibility.subprocess.run",
            side_effect=[first, second],
        ):
            before = docker_image_identity("repo:latest", require_resolved=True)
            after = docker_image_identity("repo:latest", require_resolved=True)

        self.assertEqual(before["reference"], after["reference"])
        self.assertNotEqual(before["image_id"], after["image_id"])

    def test_evaluator_fixture_tree_identity_changes_with_content(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "hidden_test.py").write_text("assert True\n", encoding="utf-8")
            first = filesystem_tree_identity(root)
            (root / "hidden_test.py").write_text("assert False\n", encoding="utf-8")
            second = filesystem_tree_identity(root)

        self.assertIsNotNone(first)
        self.assertIsNotNone(second)
        self.assertNotEqual(first["sha256"], second["sha256"])

    def test_frozen_evaluator_tree_drift_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = root / "fixture.py"
            fixture.write_text("VALUE = 1\n", encoding="utf-8")
            expected = {"fixture": filesystem_tree_identity(root)}
            paths = {"fixture": root}

            validate_frozen_evaluation_trees(expected, paths)
            fixture.write_text("VALUE = 2\n", encoding="utf-8")

            with self.assertRaisesRegex(RuntimeError, "frozen evaluator input changed"):
                validate_frozen_evaluation_trees(expected, paths)

    def test_configured_routes_are_outcome_free_and_escalation_is_heterogeneous(
        self,
    ) -> None:
        base = LangChainModel(
            model="base",
            chat=FakeListChatModel(responses=["unused"]),
            reasoning_effort="high",
        )
        escalation = LangChainModel(
            model="escalation",
            chat=FakeListChatModel(responses=["unused"]),
            reasoning_effort="low",
        )
        agent = SimpleNamespace(
            model=base,
            role_models={},
            developer_escalation_model=escalation,
            developer_adapter=None,
            context_window_tokens=128_000,
            context_budget_tokens=48_000,
            max_response_tokens=4096,
            action_transport="tools",
        )

        routes = configured_agent_model_routes(agent)

        self.assertTrue(all("invoked" not in route for route in routes.values()))
        self.assertEqual(reasoning_policy_label(routes), "role_specific")
        self.assertEqual(
            routes["developer_escalation"]["action_transport"],
            "configured:tools",
        )

    def test_tool_prompt_only_affects_tools_capable_configurations(self) -> None:
        model = LangChainModel(
            model="m",
            chat=FakeListChatModel(responses=["unused"]),
        )
        text_agent = SingleAgent(model=model, action_transport="text_json")
        tools_agent = SingleAgent(model=model, action_transport="tools")

        text_prompts = agent_prompt_manifest(text_agent)
        tool_prompts = agent_prompt_manifest(tools_agent)

        self.assertNotIn("agent:native_tools", text_prompts["prompts"])
        self.assertIn("agent:native_tools", tool_prompts["prompts"])
        self.assertNotEqual(
            text_prompts["aggregate_sha256"],
            tool_prompts["aggregate_sha256"],
        )


if __name__ == "__main__":
    unittest.main()
