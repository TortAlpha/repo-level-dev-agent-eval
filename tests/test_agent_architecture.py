from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from langchain_core.language_models.fake_chat_models import FakeListChatModel

import src.agents.roles  # noqa: F401 - registers role action spaces
from src.agents.actions import (
    CompleteSubtaskAction,
    EditFileAction,
    FinishAction,
    ReopenSubtaskAction,
    RunShellAction,
    RunTestsAction,
    SearchAction,
    SetSubtasksAction,
    SubtaskDraft,
    WriteFileAction,
    resolve_space,
)
from src.agents.context import ContextCompactor
from src.agents.decomposition import (
    DecomposedSingleAgent,
    complete_subtask,
    initialize_decomposition,
    reopen_subtask,
)
from src.agents.executor import ActionExecutor, _is_full_test_command
from src.agents.model import LangChainModel
from src.agents.multi_agent import (
    GuardedOrchestratorAgent,
    MultiGraphAgent,
    developer_instruction,
    guarded_phase_budgets,
    guarded_role_step_limit,
    orchestration_next_action,
    planner_handoff_plan,
    swe_developer_call_limit,
)
from src.agents.roles import ROLES, build_role_agent, run_role
from src.agents.sandbox import CommandResult, DockerSandbox
from src.agents.single_agent import SingleAgent
from src.agents.state import ContextBudget, State
from src.agents.tracing import run_trace_extra
from src.agents.workspace import Workspace
from src.benchmark.evaluation import restore_test_oracle, task_success
from src.run.results import run_metrics


class AgentArchitectureInvariantTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.executor = ActionExecutor(
            workspace=Workspace(root=self.root),
            sandbox=DockerSandbox(workdir=self.root),
        )

    def tearDown(self) -> None:
        self.temp.cleanup()

    def state(self, **updates: object) -> State:
        state = State(
            task="fix the behavior",
            workdir=self.root,
            test_command="python -m pytest tests",
            max_steps=50,
        )
        return state.model_copy(update=updates)

    def commit_workspace(self) -> None:
        subprocess.run(["git", "init", "-q", str(self.root)], check=True)
        subprocess.run(["git", "-C", str(self.root), "add", "-A"], check=True)
        subprocess.run(
            [
                "git",
                "-C",
                str(self.root),
                "-c",
                "user.name=Harness Test",
                "-c",
                "user.email=harness@example.invalid",
                "commit",
                "-qm",
                "base",
            ],
            check=True,
        )

    def test_focused_test_does_not_unlock_finish(self) -> None:
        state = self.state().with_changed_files(["src/module.py"])
        state = state.record_test_result(
            True,
            "1 passed",
            command="python -m pytest tests/test_module.py -x",
            full_suite=False,
        )

        result = self.executor.execute(
            state, FinishAction(action="finish", summary="done")
        )

        self.assertTrue(state.test_passed)
        self.assertFalse(state.full_test_passed)
        self.assertNotEqual(result.status, "solved")
        self.assertIn("full visible test command", result.last_error or "")

    def test_full_suite_accepts_safe_reporting_flags(self) -> None:
        configured = "python -m pytest tests"

        self.assertTrue(_is_full_test_command(configured, configured))
        self.assertTrue(_is_full_test_command("python -m pytest tests -v", configured))
        self.assertTrue(
            _is_full_test_command(
                "python -m pytest tests --tb=short --maxfail=1", configured
            )
        )
        self.assertFalse(
            _is_full_test_command("python -m pytest tests/test_parse.py", configured)
        )
        self.assertFalse(
            _is_full_test_command("python -m pytest tests -k window", configured)
        )

    def test_edit_invalidates_full_suite_verification(self) -> None:
        state = self.state().with_changed_files(["src/module.py"])
        state = state.record_test_result(
            True,
            "all passed",
            command=state.test_command or "",
            full_suite=True,
        )

        changed = state.with_changed_files(["src/other.py"])

        self.assertTrue(state.full_test_passed)
        self.assertFalse(changed.test_passed)
        self.assertFalse(changed.full_test_passed)

    def test_full_suite_on_current_revision_unlocks_finish(self) -> None:
        state = self.state().with_changed_files(["src/module.py"])
        state = state.record_test_result(
            True,
            "all passed",
            command=state.test_command or "",
            full_suite=True,
        )

        result = self.executor.execute(
            state, FinishAction(action="finish", summary="verified")
        )

        self.assertEqual(result.status, "solved")

    def test_single_compatibility_gate_requires_distinct_probe(self) -> None:
        state = (
            self.state()
            .with_compatibility_requirement(True)
            .with_changed_files(["src/module.py"])
            .record_test_result(
                True,
                "all passed",
                command="python -m pytest tests",
                full_suite=True,
            )
        )

        blocked = self.executor.execute(
            state, FinishAction(action="finish", summary="verified")
        )
        self.assertNotEqual(blocked.status, "solved")
        self.assertIn("compatibility check", blocked.last_error or "")

        verified = state.record_test_result(
            True,
            "legacy behavior passed",
            command="python -c 'assert legacy()'",
            compatibility_check=True,
            consume_iteration=False,
        )
        finished = self.executor.execute(
            verified, FinishAction(action="finish", summary="verified")
        )
        self.assertTrue(verified.compatibility_check_passed)
        self.assertEqual(verified.iteration, state.iteration)
        self.assertEqual(finished.status, "solved")

    def test_decomposed_single_enforces_and_checkpoints_subtasks(self) -> None:
        compatibility_command = "python -c 'assert legacy()'"
        full_command = "python -m pytest tests"
        action = SetSubtasksAction(
            action="set_subtasks",
            subtasks=[
                SubtaskDraft(
                    id="understand",
                    objective="Locate the existing implementation behavior",
                    kind="investigate",
                ),
                SubtaskDraft(
                    id="implement",
                    objective="Implement the required production behavior",
                    kind="implement",
                    dependencies=["understand"],
                ),
            ],
        )
        state = (
            self.state()
            .with_compatibility_requirement(True)
            .with_decomposition_requirement(True)
        )

        state = initialize_decomposition(state, action)
        self.assertEqual(state.active_subtask_id, "understand")
        self.assertEqual(
            [item.id for item in state.subtasks],
            ["understand", "implement", "system_compat", "system_verify"],
        )
        guard_agent = DecomposedSingleAgent(
            model=LangChainModel(
                model="fake",
                chat=FakeListChatModel(responses=["unused"]),
            )
        )
        self.assertEqual(
            guard_agent._research_limits(state),
            ("subtask investigation", 12, 20),
        )
        warned = guard_agent._with_research_warning(
            state.model_copy(update={"research_streak": 12})
        )
        self.assertIn("complete_subtask", warned.context[-1].text)
        self.assertEqual(
            guard_agent._action_space_for_state(state),
            "decomposed_investigate",
        )
        self.assertEqual(
            guard_agent._action_space_for_state(
                state.model_copy(update={"research_streak": 20})
            ),
            "decomposed_force_investigate",
        )
        state = state.with_context(
            kind="inspect_file",
            path="src/module.py",
            text="exact implementation target",
        )
        state = complete_subtask(
            state,
            CompleteSubtaskAction(
                action="complete_subtask",
                subtask_id="understand",
                evidence="Parser and generator paths were identified.",
            ),
        )
        self.assertEqual(state.active_subtask_id, "implement")
        self.assertEqual(
            guard_agent._research_limits(state),
            ("implementation subtask", 8, 14),
        )
        self.assertEqual(state.context[-1].kind, "subtask_checkpoint")
        self.assertTrue(
            any(entry.text == "exact implementation target" for entry in state.context)
        )
        self.assertEqual(
            guard_agent._action_space_for_state(state),
            "decomposed_implement",
        )
        self.assertEqual(
            guard_agent._action_space_for_state(
                state.model_copy(update={"research_streak": 14})
            ),
            "decomposed_force_implement",
        )

        with self.assertRaisesRegex(ValueError, "production edit"):
            complete_subtask(
                state,
                CompleteSubtaskAction(
                    action="complete_subtask",
                    subtask_id="implement",
                    evidence="Implementation is claimed without a code edit.",
                ),
            )

        state = state.with_changed_files(["src/module.py"])
        with self.assertRaisesRegex(ValueError, "passing focused test"):
            complete_subtask(
                state,
                CompleteSubtaskAction(
                    action="complete_subtask",
                    subtask_id="implement",
                    evidence="Production parser implementation was changed.",
                ),
            )
        state = state.record_test_result(
            True,
            "focused passed",
            command="python -m pytest tests/test_new_behavior.py",
        )
        state = complete_subtask(
            state,
            CompleteSubtaskAction(
                action="complete_subtask",
                subtask_id="implement",
                evidence="Production parser implementation was changed.",
            ),
        )
        state = state.record_test_result(
            False,
            "legacy behavior failed",
            command=compatibility_command,
            compatibility_check=True,
        )
        state = reopen_subtask(
            state,
            ReopenSubtaskAction(
                action="reopen_subtask",
                subtask_id="implement",
                reason="The legacy compatibility probe exposed a regression.",
            ),
        )
        self.assertEqual(state.active_subtask_id, "implement")
        self.assertEqual(state.repair_cycles, 1)
        self.assertEqual(
            next(item for item in state.subtasks if item.id == "system_compat").status,
            "pending",
        )
        state = state.with_changed_files(["src/module.py"])
        state = state.record_test_result(
            True,
            "focused repair passed",
            command="python -m pytest tests/test_new_behavior.py",
        )
        state = complete_subtask(
            state,
            CompleteSubtaskAction(
                action="complete_subtask",
                subtask_id="implement",
                evidence="The compatibility regression was repaired and tested.",
            ),
        )
        state = state.record_test_result(
            True,
            "legacy passed",
            command=compatibility_command,
            compatibility_check=True,
            consume_iteration=False,
        )
        state = complete_subtask(
            state,
            CompleteSubtaskAction(
                action="complete_subtask",
                subtask_id="system_compat",
                evidence="The explicit legacy behavior probe passed.",
            ),
        )
        state = state.record_test_result(
            True,
            "all passed",
            command=full_command,
            full_suite=True,
        )
        state = complete_subtask(
            state,
            CompleteSubtaskAction(
                action="complete_subtask",
                subtask_id="system_verify",
                evidence="The configured full visible suite passed.",
            ),
        )
        result = self.executor.execute(
            state, FinishAction(action="finish", summary="all subtasks verified")
        )

        self.assertTrue(state.decomposition_complete)
        self.assertIsNone(state.active_subtask_id)
        self.assertEqual(result.status, "solved")

    def test_decomposition_rejects_cycles(self) -> None:
        state = self.state().with_decomposition_requirement(True)
        action = SetSubtasksAction(
            action="set_subtasks",
            subtasks=[
                SubtaskDraft(
                    id="investigate",
                    objective="Investigate the production behavior change",
                    kind="investigate",
                    dependencies=["implement"],
                ),
                SubtaskDraft(
                    id="implement",
                    objective="Implement the production behavior change",
                    kind="implement",
                    dependencies=["investigate"],
                ),
            ],
        )

        with self.assertRaisesRegex(ValueError, "cycle"):
            initialize_decomposition(state, action)

    def test_decomposition_actions_are_isolated_from_default_single(self) -> None:
        self.assertNotIn("set_subtasks", resolve_space(None).models)
        self.assertNotIn("complete_subtask", resolve_space(None).models)
        self.assertIn("set_subtasks", resolve_space("decomposed_init").models)
        self.assertIn(
            "complete_subtask", resolve_space("decomposed_investigate").models
        )
        self.assertIn(
            "reopen_subtask", resolve_space("decomposed_verify").models
        )

    def test_edit_invalidates_compatibility_check(self) -> None:
        state = (
            self.state()
            .with_compatibility_requirement(True)
            .with_changed_files(["src/module.py"])
            .record_test_result(
                True,
                "legacy behavior passed",
                command="python -c 'assert legacy()'",
                compatibility_check=True,
            )
        )

        changed = state.with_changed_files(["src/other.py"])

        self.assertTrue(state.compatibility_check_passed)
        self.assertFalse(changed.compatibility_check_passed)

    def test_new_regression_file_cannot_certify_compatibility(self) -> None:
        state = (
            self.state()
            .with_compatibility_requirement(True)
            .with_changed_files(["src/module.py", "tests/test_new_behavior.py"])
        )

        result = self.executor.execute(
            state,
            RunTestsAction(
                action="run_tests",
                command="python -m pytest tests/test_new_behavior.py",
                purpose="compatibility",
            ),
        )

        self.assertEqual(result.context[-1].kind, "policy_rejection")
        self.assertIn("newly created", result.last_error or "")

    def test_any_failed_test_clears_full_suite_verification(self) -> None:
        state = self.state().with_changed_files(["src/module.py"])
        state = state.record_test_result(
            True,
            "all passed",
            command=state.test_command or "",
            full_suite=True,
        )

        state = state.record_test_result(
            False,
            "focused regression failed",
            command="python -m pytest tests/test_module.py -x",
            full_suite=False,
        )

        self.assertFalse(state.full_test_passed)

    def test_existing_tests_are_read_only(self) -> None:
        path = self.root / "tests" / "test_existing.py"
        path.parent.mkdir(parents=True)
        path.write_text("def test_existing():\n    assert True\n", encoding="utf-8")
        self.commit_workspace()
        state = self.state()

        edited = self.executor.execute(
            state,
            EditFileAction(
                action="edit_file",
                path="tests/test_existing.py",
                old_string="assert True",
                new_string="assert False",
            ),
        )
        self.assertIn("read-only", edited.last_error or "")
        self.assertEqual(edited.context[-1].kind, "policy_rejection")

        overwritten = self.executor.execute(
            state,
            WriteFileAction(
                action="write_file",
                path="tests/test_existing.py",
                content="",
            ),
        )
        self.assertIn("read-only", overwritten.last_error or "")
        self.assertEqual(overwritten.context[-1].kind, "policy_rejection")

    def test_tester_can_create_tests_but_not_production_files(self) -> None:
        state = self.state(active_role="tester")
        result = self.executor.execute(
            state,
            WriteFileAction(
                action="write_file",
                path="tests/test_new_behavior.py",
                content="def test_new_behavior():\n    assert True\n",
            ),
        )
        self.assertIn(Path("tests/test_new_behavior.py"), result.changed_files)
        revised = self.executor.execute(
            result,
            WriteFileAction(
                action="write_file",
                path="tests/test_new_behavior.py",
                content="def test_new_behavior():\n    assert 1 == 1\n",
            ),
        )
        self.assertIn(Path("tests/test_new_behavior.py"), revised.changed_files)

        rejected = self.executor.execute(
            state,
            WriteFileAction(
                action="write_file",
                path="src/production.py",
                content="value = 1\n",
            ),
        )
        self.assertIn("production code", rejected.last_error or "")
        self.assertEqual(rejected.context[-1].kind, "policy_rejection")

        developer_rejected = self.executor.execute(
            self.state(active_role="developer"),
            WriteFileAction(
                action="write_file",
                path="tests/test_developer_owned.py",
                content="def test_new_behavior():\n    assert True\n",
            ),
        )
        self.assertIn("Developer role cannot", developer_rejected.last_error or "")
        self.assertEqual(developer_rejected.context[-1].kind, "policy_rejection")

    def test_reviewer_shell_is_read_only(self) -> None:
        state = self.state(active_role="reviewer")

        rejected = self.executor.execute(
            state,
            RunShellAction(action="run_shell", command="touch src/changed.py"),
        )
        self.assertIn("read-only", rejected.last_error or "")
        self.assertEqual(rejected.context[-1].kind, "policy_rejection")

    def test_regressions_make_evaluation_unsuccessful(self) -> None:
        self.assertTrue(task_success(True, True, 0))
        self.assertTrue(task_success(True, True, None))
        self.assertFalse(task_success(True, True, 1))
        self.assertFalse(task_success(True, True, 0, test_oracle_tampered=True))

    def test_developer_handoff_contains_planner_and_review_feedback(self) -> None:
        instruction = developer_instruction(
            "change parser.py and preserve escapes",
            "REVISE: handle an empty input",
        )

        self.assertIn("change parser.py", instruction)
        self.assertIn("REVISE: handle an empty input", instruction)

    def test_guided_orchestrator_phase_uses_verified_state(self) -> None:
        state = self.state(plan="inspect then implement")
        self.assertEqual(
            orchestration_next_action(state, "planner", "done"),
            ("implementation", "developer"),
        )
        self.assertEqual(
            orchestration_next_action(state, "developer", "changed code"),
            ("verification", "tester"),
        )

        failed = state.model_copy(update={"changed_files": [Path("src/a.py")]})
        self.assertEqual(
            orchestration_next_action(failed, "tester", "FAIL: old output"),
            ("repair", "developer"),
        )
        passed = failed.model_copy(update={"full_test_passed": True})
        self.assertEqual(
            orchestration_next_action(passed, "tester", "PASS"),
            ("review", "reviewer"),
        )
        self.assertEqual(
            orchestration_next_action(passed, "reviewer", "APPROVE: clean"),
            ("ready", "finish"),
        )
        unchanged = passed.model_copy(update={"changed_files": []})
        self.assertEqual(
            orchestration_next_action(unchanged, "reviewer", "APPROVE: clean"),
            ("revision", "developer"),
        )

    def test_guarded_orchestrator_reserves_verification_budget(self) -> None:
        self.assertEqual(
            guarded_phase_budgets(100),
            {
                "planner": 20,
                "implementation": 40,
                "verification": 15,
                "orchestration": 5,
            },
        )
        self.assertEqual(guarded_role_step_limit("developer", 40), 20)
        self.assertEqual(guarded_role_step_limit("developer", 20), 9)
        self.assertEqual(guarded_role_step_limit("tester", 10), 3)
        self.assertEqual(guarded_role_step_limit("reviewer", 2), 1)
        self.assertEqual(
            guarded_role_step_limit("planner", 80, max_steps=100, role_steps=10),
            10,
        )
        self.assertEqual(
            guarded_role_step_limit("planner", 70, max_steps=100, role_steps=20),
            0,
        )

    def test_guarded_orchestrator_promotes_or_forces_planner_handoff(self) -> None:
        explored = self.state(relevant_files=[Path("sqlglot/parsers/clickhouse.py")])
        promoted = planner_handoff_plan(
            explored,
            "planner stopped at its step budget without an explicit report.",
        )
        self.assertIn("promoted automatically", promoted)

        fallback = planner_handoff_plan(self.state(), "", force=True)
        self.assertIn("Planner budget exhausted", fallback)

        exhausted = self.state(
            max_steps=100,
            role_steps={"planner": 20},
        )
        self.assertEqual(
            orchestration_next_action(
                exhausted,
                "planner",
                "",
                planner_step_cap=20,
            ),
            ("implementation", "developer"),
        )

    def test_role_model_policy_is_opt_in_and_developer_can_escalate(self) -> None:
        cheap = LangChainModel(
            model="cheap",
            chat=FakeListChatModel(responses=["unused"]),
            input_cost_per_1m=1.0,
            output_cost_per_1m=2.0,
        )
        strong = LangChainModel(
            model="strong",
            chat=FakeListChatModel(responses=["unused"]),
            input_cost_per_1m=3.0,
            output_cost_per_1m=4.0,
        )

        legacy = GuardedOrchestratorAgent(model=cheap)
        self.assertEqual(legacy.model_policy_metadata()["type"], "shared")
        self.assertEqual(
            set(legacy.model_policy_metadata()["roles"].values()), {"cheap"}
        )

        adaptive = GuardedOrchestratorAgent(
            model=cheap,
            role_models={"reviewer": strong},
            developer_escalation_model=strong,
        )
        self.assertEqual(adaptive.model_policy_metadata()["type"], "adaptive")
        self.assertIs(adaptive._select_role_model("developer", self.state()), cheap)

        adaptive._developer_episodes = 1
        passed_state = self.state().record_test_result(True, "ok")
        self.assertEqual(passed_state.failed_test_runs, 0)
        self.assertIs(adaptive._select_role_model("developer", passed_state), cheap)
        failed_state = passed_state.record_test_result(False, "failed")
        self.assertEqual(failed_state.failed_test_runs, 1)
        self.assertIs(adaptive._select_role_model("developer", failed_state), strong)

        adaptive = GuardedOrchestratorAgent(
            model=cheap,
            role_models={"reviewer": strong},
            developer_escalation_model=strong,
        )
        adaptive._developer_no_edit_episodes = 1
        self.assertIs(adaptive._select_role_model("developer", self.state()), strong)
        self.assertTrue(adaptive.developer_escalation_pending())
        self.assertIs(adaptive._select_role_model("reviewer", self.state()), strong)
        self.assertEqual(len(adaptive.usage_models()), 2)

    def test_failed_tester_episode_preserves_failure_for_adaptive_escalation(
        self,
    ) -> None:
        cheap = LangChainModel(
            model="cheap",
            chat=FakeListChatModel(
                responses=[
                    '{"action":"run_tests"}',
                    '{"action":"report","summary":"tests failed"}',
                ]
            ),
        )
        strong = LangChainModel(
            model="strong",
            chat=FakeListChatModel(responses=["unused"]),
        )
        role_agent = build_role_agent(ROLES["tester"], cheap, "text_json")
        compactor = ContextCompactor(
            budget=ContextBudget(window_tokens=8192, reserved_output_tokens=1024),
            mode="drop",
            model=cheap,
        )
        with patch.object(
            DockerSandbox,
            "run_tests",
            return_value=CommandResult(returncode=1, output="1 failed"),
        ):
            merged, _ = run_role(
                ROLES["tester"],
                agent=role_agent,
                executor=self.executor,
                compactor=compactor,
                state=self.state(),
                instruction="verify",
            )

        self.assertEqual(merged.failed_test_runs, 1)
        adaptive = GuardedOrchestratorAgent(
            model=cheap,
            developer_escalation_model=strong,
        )
        adaptive._developer_episodes = 1
        self.assertIs(adaptive._select_role_model("developer", merged), strong)

    def test_policy_rejection_does_not_count_as_parse_failure(self) -> None:
        state = self.state(active_role="developer")
        rejected = self.executor.execute(
            state,
            WriteFileAction(
                action="write_file",
                path="tests/test_forbidden.py",
                content="def test_x():\n    assert True\n",
            ),
        )
        model = LangChainModel(
            model="fake", chat=FakeListChatModel(responses=["unused"])
        )
        agent = SingleAgent(model=model)

        self.assertEqual(rejected.context[-1].kind, "policy_rejection")
        self.assertEqual(agent._parse_failure_streak(rejected), 0)

    def test_trace_name_and_tags_expose_model_policy(self) -> None:
        extra = run_trace_extra(
            task_id="task-1",
            agent_mode="multi-orch-guarded",
            repo_dir="/workspace",
            model_name="base",
            settings={
                "model_policy": {
                    "type": "adaptive",
                    "base_model": "z-ai/glm-5.2",
                    "roles": {"developer": "z-ai/glm-5.2"},
                    "developer_escalation_model": "anthropic/claude-sonnet-4.5",
                }
            },
        )

        self.assertEqual(
            extra["name"],
            "multi-orch-guarded/adaptive_dev_glm-5.2_to_claude-sonnet-4.5.run[task-1]",
        )
        self.assertIn("model_policy:adaptive", extra["tags"])
        self.assertIn(
            "model_policy_label:adaptive_dev_glm-5.2_to_claude-sonnet-4.5",
            extra["tags"],
        )
        self.assertEqual(extra["metadata"]["model_policy"]["type"], "adaptive")

        routed = run_trace_extra(
            task_id="task-1",
            agent_mode="multi-orch-guarded",
            repo_dir="/workspace",
            model_name="z-ai/glm-5.2",
            settings={
                "model_policy": {
                    "type": "role_routed",
                    "base_model": "z-ai/glm-5.2",
                    "roles": {
                        "developer": "anthropic/claude-sonnet-4.5",
                    },
                }
            },
        )
        self.assertEqual(
            routed["name"],
            "multi-orch-guarded/dev_claude-sonnet-4.5.run[task-1]",
        )

    def test_multi_model_usage_and_cost_are_aggregated(self) -> None:
        first = LangChainModel(
            model="first",
            chat=FakeListChatModel(responses=["unused"]),
            calls=2,
            input_tokens=100,
            output_tokens=20,
            input_cost_per_1m=1.0,
            output_cost_per_1m=2.0,
        )
        second = LangChainModel(
            model="second",
            chat=FakeListChatModel(responses=["unused"]),
            calls=3,
            input_tokens=200,
            output_tokens=30,
            input_cost_per_1m=3.0,
            output_cost_per_1m=4.0,
        )

        metrics = run_metrics([first, second], self.state(), 1.25)

        self.assertEqual(metrics["llm_calls"], 5)
        self.assertEqual(metrics["input_tokens"], 300)
        self.assertEqual(metrics["output_tokens"], 50)
        self.assertAlmostEqual(metrics["cost_usd"], 0.00086)
        self.assertEqual(metrics["compaction_count"], 0)
        self.assertEqual(metrics["research_guard_violations"], 0)

    def test_search_retries_basic_grep_alternation(self) -> None:
        path = self.root / "module.py"
        path.write_text('AGG_FUNCTIONS = {"count"}\n', encoding="utf-8")

        output = self.executor.workspace.search(
            r'"count"\|groupConcat', "module.py"
        )

        self.assertIn('"count"', output)

    def test_repeated_search_result_survives_outside_context(self) -> None:
        path = self.root / "module.py"
        path.write_text("needle = 1\n", encoding="utf-8")
        action = SearchAction(
            action="search", query="needle", path="module.py"
        )

        first = self.executor.execute(self.state(), action)
        # Simulate compaction removing the rendered search entry. The durable
        # search cache must still recover the exact result.
        compacted = first.model_copy(update={"context": []})
        second = self.executor.execute(compacted, action)

        self.assertEqual(second.context[-1].kind, "search_reused")
        self.assertIn("needle = 1", second.context[-1].text)

    def test_compaction_preserves_search_ledger_and_counts_itself(self) -> None:
        state = self.state()
        state = state.with_context(
            kind="search",
            path="src",
            text="query='GroupConcat'\nsrc/aggregate.py:133:class GroupConcat",
        )
        state = state.with_context(
            kind="inspect_file",
            path="src/parser.py",
            text="parser details\n" + ("x" * 3000),
        )
        state = state.with_context(
            kind="inspect_file",
            path="src/other.py",
            text="latest details\n" + ("y" * 3000),
        )
        model = LangChainModel(
            model="summarizer",
            chat=FakeListChatModel(responses=["Parser and aggregate locations found."]),
        )
        compactor = ContextCompactor(
            budget=ContextBudget(window_tokens=1000, reserved_output_tokens=100),
            model=model,
            summary_max_chars=500,
        )

        compacted = compactor.apply(state)

        self.assertEqual(compacted.compaction_count, 1)
        self.assertEqual(compacted.context[0].kind, "context_summary")
        self.assertIn("Recent searches before compaction", compacted.context[0].text)
        self.assertIn("GroupConcat", compacted.context[0].text)

    def test_single_agent_stops_ignored_research_guard(self) -> None:
        model = LangChainModel(
            model="fake",
            chat=FakeListChatModel(
                responses=[
                    '{"action":"search","query":"again","path":"."}',
                    '{"action":"search","query":"again","path":"."}',
                ]
            ),
        )
        agent = SingleAgent(
            model=model,
            research_warning_steps=1,
            research_hard_limit=2,
            max_research_guard_violations=2,
        )
        state = self.state(research_streak=2)

        state, previous = agent._step(state, self.executor, None)
        self.assertEqual(state.research_guard_violations, 1)
        self.assertNotEqual(state.status, "handoff")
        self.assertEqual(state.context[-1].kind, "research_guard")

        state, _ = agent._step(state, self.executor, previous)
        self.assertEqual(state.research_guard_violations, 2)
        self.assertEqual(state.status, "handoff")

    def test_single_agent_cost_limit_uses_known_model_price(self) -> None:
        model = LangChainModel(
            model="priced",
            chat=FakeListChatModel(responses=["unused"]),
            input_tokens=1_000_000,
            input_cost_per_1m=1.0,
            output_cost_per_1m=1.0,
        )
        agent = GuardedOrchestratorAgent(model=model, max_cost_usd=0.5)

        stopped = agent._enforce_cost_limit(self.state())

        self.assertEqual(stopped.status, "handoff")
        self.assertEqual(stopped.context[-1].kind, "cost_limit")

    def test_swe_developer_respects_shared_role_budget(self) -> None:
        self.assertEqual(swe_developer_call_limit(75, 50), 20)
        self.assertEqual(swe_developer_call_limit(75, 7), 7)

    def test_swe_patch_is_submitted_not_preemptively_solved(self) -> None:
        from src.agents.swe_agent import SweAgentAdapter

        model = LangChainModel(
            model="fake",
            chat=FakeListChatModel(responses=["unused"]),
        )
        adapter = SweAgentAdapter(model=model, api_key="dummy")

        result = adapter._to_state(
            self.state(),
            ["src/module.py"],
            3,
            subprocess.CompletedProcess([], 0),
        )

        self.assertEqual(result.status, "submitted")
        self.assertFalse(result.test_passed)

    def test_multi_graph_compiles_with_handoff_state(self) -> None:
        model = LangChainModel(
            model="fake",
            chat=FakeListChatModel(responses=['{"action":"report","summary":"ok"}']),
        )
        agent = MultiGraphAgent(model=model)
        compactor = ContextCompactor(
            budget=ContextBudget(
                window_tokens=8192,
                reserved_output_tokens=1024,
            ),
            mode="drop",
            model=model,
        )

        graph = agent._build_graph(self.executor, compactor)

        self.assertIsNotNone(graph)

    def test_evaluator_restores_tracked_test_oracle(self) -> None:
        test_path = self.root / "tests" / "test_original.py"
        test_path.parent.mkdir(parents=True)
        test_path.write_text(
            "def test_original():\n    assert True\n", encoding="utf-8"
        )
        self.commit_workspace()
        test_path.write_text("", encoding="utf-8")

        tampered = restore_test_oracle(self.root)

        self.assertEqual(tampered, ["tests/test_original.py"])
        self.assertIn("assert True", test_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
