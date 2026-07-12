from __future__ import annotations

import argparse
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage

from src.agents.model import LangChainModel
from src.agents.sandbox import DockerSandbox
from src.agents.state import State
from src.benchmark.collection import DEFAULT_COLLECTION
from src.benchmark.sweep import _combinations, _selected_tasks
from src.benchmark.task_sets import load_task_set
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
        self.assertEqual(len(core_agents), 4)
        self.assertIn("single-decomposed", core_agents)
        self.assertEqual(len(task_set.previously_exercised), 21)
        self.assertEqual(len(set(task_set.tasks()) - set(task_set.previously_exercised)), 20)

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
        self.assertEqual({item.agent for item in first[:4]}, set(args.agents.split(",")))

    def test_setup_container_can_be_disconnected_before_agent_actions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            sandbox = DockerSandbox(
                workdir=Path(directory),
                container_id="container-1",
                network_disabled=False,
            )
            completed = subprocess.CompletedProcess([], 0, stdout="", stderr="")
            with patch("src.agents.sandbox.subprocess.run", return_value=completed) as run:
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


if __name__ == "__main__":
    unittest.main()
