from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage

from src.agents.actions import action_tool_schemas
from src.agents.decomposition import DecomposedSingleAgent
from src.agents.model import LangChainModel
from src.agents.sandbox import DockerSandbox
from src.agents.state import State
from src.agents.swe_agent import SweAgentAdapter
from src.benchmark.collection import DEFAULT_COLLECTION, TaskSpec
from src.benchmark.evaluation import EvalResult
from src.benchmark.runner import (
    agent_model_policy,
    agent_settings,
    benchmark_exit_code,
    eval_metrics,
    plan,
    record_infrastructure_failure,
)
from src.benchmark.sweep import (
    Combination,
    _combinations,
    _remove_failed_workspace,
    _run_command,
    _selected_tasks,
    _task_input_snapshot,
)
from src.benchmark.sweep import (
    parse_args as parse_sweep_args,
)
from src.benchmark.task_sets import load_task_set
from src.config import OpenRouterConfig
from src.console.server import AGENTS, ApiError, ConsoleHandler, _origin_allowed
from src.run.results import run_metrics


class BenchmarkFairnessTests(unittest.TestCase):
    def test_final_v1_is_exact_and_excludes_calibration(self) -> None:
        task_set = load_task_set("final-v1")

        self.assertEqual(task_set.task_set_id, "final-v1")
        self.assertEqual(len(task_set.tasks()), 41)
        self.assertEqual(len(task_set.groups["core_small"]), 12)
        self.assertEqual(len(task_set.groups["core_medium"]), 12)
        self.assertFalse(set(task_set.tasks()) & set(task_set.calibration_excluded))
        core_tasks, core_agents = task_set.matrix("core-all")
        self.assertEqual(len(core_tasks), 24)
        self.assertEqual(core_agents, ["single", "multi-graph", "multi-orch-guarded"])
        swe_single_tasks, swe_single_agents = task_set.matrix("swepro-all-single50")
        swe_graph_tasks, swe_graph_agents = task_set.matrix("swepro-all-graph75")
        self.assertEqual(len(swe_single_tasks), 15)
        self.assertEqual(swe_single_tasks, swe_graph_tasks)
        self.assertEqual(swe_single_agents, ["single"])
        self.assertEqual(swe_graph_agents, ["multi-graph"])
        self.assertEqual(len(task_set.previously_exercised), 21)
        unexercised = set(task_set.tasks()) - set(task_set.previously_exercised)
        self.assertEqual(len(unexercised), 20)

    def test_manifest_selection_never_expands_to_all_verified_rows(self) -> None:
        args = argparse.Namespace(
            collection=DEFAULT_COLLECTION,
            task_set="final-v1",
            task_groups="core_small,core_medium",
            tasks=None,
        )

        tasks, task_set = _selected_tasks(args)

        self.assertIsNotNone(task_set)
        self.assertEqual(len(tasks), 24)

    def test_task_major_order_is_deterministic_and_interleaved(self) -> None:
        args = argparse.Namespace(
            provider="openrouter",
            models="z-ai/glm-5.2",
            agents="single,single-decomposed,multi-graph,multi-orch-guarded",
            order="task-major",
            seed=20260712,
        )
        tasks = ["h11_pr_181", "humanize_pr_329"]

        first = _combinations(args, tasks)
        second = _combinations(args, tasks)

        self.assertEqual(first, second)
        self.assertEqual({item.task for item in first[:4]}, {tasks[0]})
        self.assertEqual(
            {item.agent for item in first[:4]}, set(args.agents.split(","))
        )

    def test_setup_container_can_be_disconnected_before_agent_actions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            sandbox = DockerSandbox(
                workdir=Path(directory),
                container_id="container-1",
                network_disabled=False,
            )
            completed = subprocess.CompletedProcess([], 0, stdout="", stderr="")
            with patch(
                "src.agents.sandbox.subprocess.run", return_value=completed
            ) as run:
                sandbox.disable_network()

        self.assertTrue(sandbox.network_disabled)
        self.assertEqual(
            run.call_args.args[0],
            ["docker", "network", "disconnect", "bridge", "container-1"],
        )

    def test_provider_cost_and_cached_tokens_are_separate_from_estimate(self) -> None:
        model = LangChainModel(
            model="priced",
            chat=FakeListChatModel(responses=["unused"]),
            input_cost_per_1m=1.0,
            output_cost_per_1m=2.0,
        )
        response = AIMessage(
            content="ok",
            usage_metadata={
                "input_tokens": 100,
                "output_tokens": 10,
                "total_tokens": 110,
                "input_token_details": {"cache_read": 40},
            },
            response_metadata={"usage": {"cost": 0.00009}},
        )
        model._record_usage(response)
        with tempfile.TemporaryDirectory() as directory:
            state = State(task="x", workdir=Path(directory))
            metrics = run_metrics(model, state, 1.0)

        self.assertEqual(metrics["cached_input_tokens"], 40)
        self.assertEqual(metrics["provider_reported_cost_usd"], 0.00009)
        self.assertNotEqual(metrics["provider_reported_cost_usd"], metrics["cost_usd"])

    def test_infrastructure_failure_preserves_original_error_and_cost(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = OpenRouterConfig(
                openrouter_api_key="dummy",
                results_dir=Path(directory),
            )
            args = argparse.Namespace(
                campaign_id="campaign",
                task_set_id="final-v1",
                experiment_fingerprint="fingerprint",
            )
            task = TaskSpec(
                task_id="task-1",
                repo_path=Path(directory),
                task_file_path=Path(directory) / "task.md",
                size="small",
                task_type="bugfix",
                visible_test_command="pytest",
                hidden_test_command="pytest hidden",
                hidden_semantic_test_command="",
                hidden_compat_test_command="",
                hidden_pr_parity_test_command="",
                hidden_tests_path=Path(directory),
                base_commit="base",
                task_status="pr_task_verified",
                setup_commands=[],
                docker_image="",
            )
            model = LangChainModel(
                model="priced",
                chat=FakeListChatModel(responses=["unused"]),
                input_tokens=100,
                input_cost_per_1m=1.0,
                output_cost_per_1m=1.0,
            )
            state = State(task="x", workdir=Path(directory))

            record_infrastructure_failure(
                config,
                args,
                task,
                "run-1",
                "agent",
                RuntimeError("original provider failure"),
                [model],
                state,
                1.0,
            )
            row = json.loads(
                (Path(directory) / "infrastructure_failures.jsonl")
                .read_text(encoding="utf-8")
                .strip()
            )

        self.assertEqual(row["error"], "original provider failure")
        self.assertEqual(row["classification"], "provider_error")
        self.assertEqual(row["finished_at"][-6:], "+00:00")
        self.assertGreater(row["cost_usd"], 0)

    def test_swe_adapter_uses_safe_settings_and_dry_run_plan(self) -> None:
        model = LangChainModel(
            model="fake", chat=FakeListChatModel(responses=["unused"])
        )
        adapter = SweAgentAdapter(model=model, api_key="dummy")
        settings = agent_settings(adapter)
        task = TaskSpec(
            task_id="task-1",
            repo_path=Path("repo"),
            task_file_path=Path("task.md"),
            size="small",
            task_type="bugfix",
            visible_test_command="pytest",
            hidden_test_command="pytest hidden",
            hidden_semantic_test_command="",
            hidden_compat_test_command="",
            hidden_pr_parity_test_command="",
            hidden_tests_path=Path("hidden"),
            base_commit="base",
            task_status="pr_task_verified",
            setup_commands=[],
            docker_image="",
        )
        output = plan(
            task,
            Path("repo"),
            "pytest",
            adapter,
            argparse.Namespace(
                no_score=False, no_regression=False, enable_review=False
            ),
        )

        self.assertFalse(settings["research_guard_enabled"])
        self.assertFalse(settings["decomposition_enabled"])
        self.assertEqual(agent_model_policy(adapter)["type"], "external")
        self.assertIn("research guard:off", output)

    def test_decomposed_tool_schema_has_no_dangling_refs(self) -> None:
        space = DecomposedSingleAgent.model_fields["action_space"].default
        schemas = action_tool_schemas(space)
        set_subtasks = next(
            item for item in schemas if item["function"]["name"] == "set_subtasks"
        )

        serialized = json.dumps(set_subtasks["function"]["parameters"])
        self.assertNotIn('"$ref"', serialized)
        self.assertIn("objective", serialized)

    def test_console_rejects_foreign_origins_and_exposes_decomposed_agent(self) -> None:
        self.assertTrue(_origin_allowed(None, "127.0.0.1:8765"))
        self.assertTrue(
            _origin_allowed("http://127.0.0.1:8765", "127.0.0.1:8765")
        )
        self.assertFalse(_origin_allowed("https://evil.example", "127.0.0.1:8765"))
        self.assertIn("single-decomposed", AGENTS)

    def test_console_handler_rejects_foreign_origin_before_launch(self) -> None:
        handler = object.__new__(ConsoleHandler)
        handler.headers = {
            "Origin": "https://evil.example",
            "Host": "127.0.0.1:8765",
        }

        with self.assertRaises(ApiError) as raised:
            handler._require_allowed_origin()

        self.assertEqual(raised.exception.status.value, 403)

    def test_parallel_stop_on_error_is_rejected_explicitly(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "src.benchmark.sweep",
                "--tasks",
                "h11_pr_181",
                "--concurrency",
                "2",
                "--stop-on-error",
                "--plan-only",
            ],
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("--stop-on-error requires --concurrency 1", result.stderr)

    def test_parallel_cleanup_removes_only_the_preassigned_run(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "task-11111111-1111-1111-1111-111111111111"
            concurrent = root / "task-22222222-2222-2222-2222-222222222222"
            target.mkdir()
            concurrent.mkdir()

            _remove_failed_workspace(
                "task",
                "11111111-1111-1111-1111-111111111111",
                argparse.Namespace(keep_workspaces=False),
                root=root,
            )

            self.assertFalse(target.exists())
            self.assertTrue(concurrent.exists())

    def test_parallel_command_propagates_run_and_image_identity(self) -> None:
        with patch.object(sys, "argv", ["sweep", "--tasks", "task"]):
            args = parse_sweep_args()
        args.expected_task_inputs = {
            "task": {
                "source_repository": {"worktree_sha256": "source-hash"},
                "task_sha256": "task-hash",
                "hidden_tree_sha256": "hidden-hash",
            }
        }
        args.expected_collection_sha256 = "collection-hash"
        args.expected_harness_tree_sha256 = "harness-hash"
        args.expected_policy_kernel_sha256 = "kernel-hash"
        args.expected_pricing_sha256 = "pricing-hash"
        args.expected_sweagent_distribution_sha256 = "sweagent-hash"
        args.expected_swerex_distribution_sha256 = "swerex-hash"
        args.expected_swe_runtime_sha256 = "swe-runtime-hash"
        command = _run_command(
            Combination("task", "model", "model", "single"),
            args,
            run_id="11111111-1111-1111-1111-111111111111",
            expected_docker_image_id="sha256:locked",
        )

        self.assertEqual(
            command[command.index("--run-id") + 1],
            "11111111-1111-1111-1111-111111111111",
        )
        self.assertEqual(
            command[command.index("--expected-docker-image-id") + 1],
            "sha256:locked",
        )
        expected_contract = {
            "--expected-source-worktree-sha256": "source-hash",
            "--expected-task-sha256": "task-hash",
            "--expected-hidden-tree-sha256": "hidden-hash",
            "--expected-collection-sha256": "collection-hash",
            "--expected-harness-tree-sha256": "harness-hash",
            "--expected-policy-kernel-sha256": "kernel-hash",
            "--expected-pricing-sha256": "pricing-hash",
            "--expected-sweagent-distribution-sha256": "sweagent-hash",
            "--expected-swerex-distribution-sha256": "swerex-hash",
            "--expected-swe-runtime-sha256": "swe-runtime-hash",
        }
        for flag, expected in expected_contract.items():
            with self.subTest(flag=flag):
                self.assertEqual(command[command.index(flag) + 1], expected)

    def test_task_snapshot_records_resolved_image_object(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            task_file = root / "task.md"
            task_file.write_text("task", encoding="utf-8")
            task = TaskSpec(
                task_id="task",
                repo_path=root / "repo",
                task_file_path=task_file,
                size="small",
                task_type="bugfix",
                visible_test_command="pytest",
                hidden_test_command="",
                hidden_semantic_test_command="",
                hidden_compat_test_command="",
                hidden_pr_parity_test_command="",
                hidden_tests_path=Path(),
                base_commit="abc",
                task_status="pr_task_verified",
                setup_commands=[],
                docker_image="repo:latest",
            )
            image = {
                "reference": "repo:latest",
                "resolved": True,
                "image_id": "sha256:locked",
                "repo_digests": ["repo@sha256:locked"],
            }
            with (
                patch(
                    "src.benchmark.sweep.load_collection",
                    return_value={"task": task},
                ),
                patch(
                    "src.benchmark.sweep.source_repository_identity",
                    return_value={"head_commit": "abc"},
                ),
                patch(
                    "src.benchmark.sweep.docker_image_identity",
                    return_value=image,
                ),
            ):
                snapshot = _task_input_snapshot(
                    ["task"], Path("collection.csv"), "python:3.11-slim"
                )

        self.assertEqual(snapshot["task"]["docker_image_identity"], image)

    def test_unscored_test_oracle_tampering_is_a_process_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            state = State(
                task="x",
                workdir=Path(directory),
                status="solved",
                test_oracle_tamper_attempts=1,
            )
            result = EvalResult(task_success=None, test_oracle_tampered=True)

            exit_code = benchmark_exit_code(state, result)

        self.assertEqual(exit_code, 1)
        self.assertIsNone(result.task_success)

    def test_eval_metrics_records_excluded_agent_tests(self) -> None:
        metrics = eval_metrics(
            EvalResult(excluded_agent_test_files=["tests/test_agent_added.py"])
        )

        self.assertEqual(metrics["excluded_agent_test_file_count"], 1)
        self.assertEqual(
            metrics["excluded_agent_test_files"], ["tests/test_agent_added.py"]
        )


if __name__ == "__main__":
    unittest.main()
