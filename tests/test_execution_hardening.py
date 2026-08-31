from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from subprocess import CompletedProcess
from unittest.mock import patch

from pydantic import ValidationError

from src.agents.actions import (
    EditFileAction,
    InspectFileAction,
    RunShellAction,
    RunTestsAction,
    SearchAction,
    WriteFileAction,
)
from src.agents.executor import ActionExecutor
from src.agents.sandbox import CommandResult, DockerSandbox
from src.agents.state import State
from src.agents.workspace import Workspace


class ExecutionHardeningTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / "src").mkdir()
        (self.root / "tests").mkdir()
        (self.root / "src" / "app.py").write_text("value = 1\n", encoding="utf-8")
        (self.root / "tests" / "test_app.py").write_text(
            "def test_value():\n    assert True\n", encoding="utf-8"
        )
        (self.root / "vendor").mkdir()
        (self.root / "vendor" / "tracked.txt").write_text(
            "original\n", encoding="utf-8"
        )
        (self.root / "external-link").symlink_to("/tmp/outside-workspace")
        subprocess.run(["git", "init", "-q", str(self.root)], check=True)
        subprocess.run(["git", "-C", str(self.root), "add", "-A"], check=True)
        subprocess.run(
            [
                "git",
                "-C",
                str(self.root),
                "-c",
                "user.name=Hardening Test",
                "-c",
                "user.email=hardening@example.invalid",
                "commit",
                "-qm",
                "base",
            ],
            check=True,
        )
        self.workspace = Workspace(root=self.root)
        self.sandbox = DockerSandbox(workdir=self.root)
        self.executor = ActionExecutor(
            workspace=self.workspace,
            sandbox=self.sandbox,
        )

    def tearDown(self) -> None:
        self.temp.cleanup()

    def state(self) -> State:
        return State(
            task="fix behavior",
            workdir=self.root,
            test_command="python -m pytest tests",
            max_steps=20,
        )

    def test_shell_mutation_is_recorded_in_shared_state(self) -> None:
        def mutate(*_args: object, **_kwargs: object) -> CommandResult:
            (self.root / "src" / "app.py").write_text(
                "value = 2\n", encoding="utf-8"
            )
            return CommandResult(returncode=0, output="changed")

        with patch.object(DockerSandbox, "run_shell", side_effect=mutate):
            result = self.executor.execute(
                self.state(),
                RunShellAction(action="run_shell", command="sed -i change src/app.py"),
            )

        self.assertIn(Path("src/app.py"), result.changed_files)
        self.assertEqual(result.workspace_revision, 1)
        self.assertFalse(result.full_test_passed)

    def test_shell_mutation_invalidates_prior_file_inspection(self) -> None:
        inspected = self.executor.execute(
            self.state(),
            InspectFileAction(action="inspect_file", path="src/../src/app.py"),
        )
        self.assertEqual(inspected.relevant_files, [Path("src/app.py")])
        self.assertEqual(inspected.context[-1].path, "src/app.py")

        def mutate(*_args: object, **_kwargs: object) -> CommandResult:
            (self.root / "src" / "app.py").write_text(
                "value = 2\n", encoding="utf-8"
            )
            return CommandResult(returncode=0, output="changed")

        with patch.object(DockerSandbox, "run_shell", side_effect=mutate):
            result = self.executor.execute(
                inspected,
                RunShellAction(action="run_shell", command="sed -i change src/app.py"),
            )

        old_snapshot = next(
            entry
            for entry in result.context
            if entry.kind == "inspect_file" and entry.path == "src/app.py"
        )
        self.assertEqual(old_snapshot.text, "[stale: file changed by a later edit]")

    def test_write_invalidates_dot_prefixed_file_inspection(self) -> None:
        inspected = self.executor.execute(
            self.state(),
            InspectFileAction(action="inspect_file", path="./src/app.py"),
        )

        result = self.executor.execute(
            inspected,
            WriteFileAction(
                action="write_file",
                path="src/app.py",
                content="value = 2\n",
            ),
        )

        old_snapshot = next(
            entry for entry in result.context if entry.kind == "inspect_file"
        )
        self.assertEqual(old_snapshot.path, "src/app.py")
        self.assertEqual(old_snapshot.text, "[stale: file changed by a later edit]")

    def test_shell_test_oracle_tampering_is_restored_and_rejected(self) -> None:
        original = (self.root / "tests" / "test_app.py").read_text(encoding="utf-8")

        def tamper(*_args: object, **_kwargs: object) -> CommandResult:
            (self.root / "tests" / "test_app.py").write_text(
                "def test_value():\n    assert 1 == 1\n", encoding="utf-8"
            )
            return CommandResult(returncode=0, output="1 passed")

        with patch.object(DockerSandbox, "run_shell", side_effect=tamper):
            result = self.executor.execute(
                self.state(),
                RunShellAction(
                    action="run_shell", command="python -m pytest tests"
                ),
            )

        self.assertEqual(
            (self.root / "tests" / "test_app.py").read_text(encoding="utf-8"),
            original,
        )
        self.assertFalse(result.test_passed)
        self.assertIn("Protected benchmark tests", result.last_error or "")
        self.assertEqual(result.context[-1].kind, "policy_rejection")
        self.assertEqual(result.test_oracle_tamper_attempts, 1)

    def test_host_file_actions_reject_symlink_and_hardlink_aliases(self) -> None:
        original = (self.root / "tests" / "test_app.py").read_text(
            encoding="utf-8"
        )

        def create_symlink(*_args: object, **_kwargs: object) -> CommandResult:
            (self.root / "src" / "alias.py").symlink_to("../tests/test_app.py")
            return CommandResult(returncode=0, output="linked")

        with patch.object(DockerSandbox, "run_shell", side_effect=create_symlink):
            linked_state = self.executor.execute(
                self.state(),
                RunShellAction(
                    action="run_shell",
                    command="ln -s ../tests/test_app.py src/alias.py",
                ),
            )
        symlink_result = self.executor.execute(
            linked_state,
            WriteFileAction(
                action="write_file",
                path="src/alias.py",
                content="def test_value():\n    assert False\n",
            ),
        )

        self.assertIn("symlinks", symlink_result.last_error or "")
        self.assertEqual(
            (self.root / "tests" / "test_app.py").read_text(encoding="utf-8"),
            original,
        )

        os.link(
            self.root / "tests" / "test_app.py",
            self.root / "src" / "hardlink.py",
        )
        hardlink_result = self.executor.execute(
            self.state(),
            EditFileAction(
                action="edit_file",
                path="src/hardlink.py",
                old_string="assert True",
                new_string="assert False",
            ),
        )

        self.assertIn("hardlinked", hardlink_result.last_error or "")
        self.assertEqual(
            (self.root / "tests" / "test_app.py").read_text(encoding="utf-8"),
            original,
        )

    def test_workspace_never_exposes_vcs_metadata_or_symlink_alias(self) -> None:
        (self.root / "git-alias").symlink_to(".git", target_is_directory=True)

        with self.assertRaisesRegex(ValueError, "VCS metadata"):
            self.workspace.read_file(".git/config")
        with self.assertRaisesRegex(ValueError, "VCS metadata"):
            self.workspace.write_file(".git/config", "malicious = true\n")
        with self.assertRaisesRegex(ValueError, "VCS metadata"):
            self.workspace.edit_file(".git/config", "[core]", "[unsafe]")

        for operation in (
            lambda: self.workspace.read_file("git-alias/config"),
            lambda: self.workspace.write_file(
                "git-alias/config", "malicious = true\n"
            ),
            lambda: self.workspace.edit_file(
                "git-alias/config", "[core]", "[unsafe]"
            ),
        ):
            with self.assertRaises(ValueError):
                operation()

    def test_executor_configures_scalable_readonly_test_roots(self) -> None:
        self.assertEqual(self.sandbox.protected_test_paths, ())
        self.assertEqual(self.sandbox.protected_test_directories, ("tests",))

    def test_snapshot_tracks_versioned_files_inside_ignored_directory(self) -> None:
        before = self.workspace.worktree_snapshot()
        (self.root / "vendor" / "tracked.txt").write_text(
            "changed\n", encoding="utf-8"
        )

        after = self.workspace.worktree_snapshot()

        self.assertIn("vendor/tracked.txt", before.changed_paths(after))
        # A tracked symlink outside the repository is hashed, never followed.
        self.assertIn("external-link", before.files)

    def test_snapshot_rejects_file_larger_than_remaining_byte_budget(self) -> None:
        (self.root / "src" / "large.bin").write_bytes(b"x" * 64)

        with self.assertRaisesRegex(ValueError, "byte safety limit"):
            self.workspace.worktree_snapshot(max_bytes=32)

    def test_search_is_relative_and_cache_key_covers_result_shape(self) -> None:
        output = self.workspace.search("value", ".", 10, "literal")

        self.assertIn("src/app.py:1:value = 1", output)
        self.assertNotIn(str(self.root), output)
        base = self.workspace.search_key("value", ".", 10, "literal")
        self.assertNotEqual(
            base, self.workspace.search_key("value", ".", 11, "literal")
        )
        self.assertNotEqual(
            base, self.workspace.search_key("value", ".", 10, "regex")
        )
        self.assertEqual(
            self.workspace.search_key("value", "./src/../src", 10, "literal"),
            self.workspace.search_key("value", "src", 10, "literal"),
        )

    def test_search_cache_reuses_canonical_repository_path(self) -> None:
        first = self.executor.execute(
            self.state(),
            SearchAction(
                action="search",
                query="value",
                path="./src/../src",
                max_results=10,
                mode="literal",
            ),
        )

        second = self.executor.execute(
            first,
            SearchAction(
                action="search",
                query="value",
                path="src",
                max_results=10,
                mode="literal",
            ),
        )

        self.assertEqual(second.context[-1].kind, "search_reused")
        self.assertEqual(second.context[-1].path, "src")

    def test_search_cache_key_preserves_exact_query_semantics(self) -> None:
        self.assertNotEqual(
            self.workspace.search_key(r"a\|b", ".", 10, "literal"),
            self.workspace.search_key("a|b", ".", 10, "literal"),
        )
        self.assertNotEqual(
            self.workspace.search_key("two  spaces", ".", 10, "literal"),
            self.workspace.search_key("two spaces", ".", 10, "literal"),
        )

    def test_explicit_multiline_regex_is_not_downgraded_to_literal(self) -> None:
        (self.root / "src" / "multiline.txt").write_text(
            "cat\nvalue\n", encoding="utf-8"
        )

        query = "cat\nv.lue"
        result = self.workspace.search(query, ".", 10, "regex")
        fallback = self.workspace._python_search(
            self.root, query, 10, "regex"
        )

        self.assertIn("src/multiline.txt:1:cat", result)
        self.assertIn("src/multiline.txt:1:cat", fallback)

    def test_multiline_fallback_bounds_empty_eof_matches(self) -> None:
        (self.root / "src" / "tail.txt").write_text("value\n", encoding="utf-8")

        result = self.workspace._python_search(
            self.root,
            "\n?",
            2,
            "regex",
        )

        self.assertLessEqual(len(result.splitlines()), 2)

    def test_python_search_auto_mode_matches_regex_and_invalid_literal(self) -> None:
        (self.root / "src" / "fallback.txt").write_text(
            "cat\nvalue(\n", encoding="utf-8"
        )

        regex = self.workspace._python_search(
            self.root, "cat|dog", 10, "auto"
        )
        malformed = self.workspace._python_search(
            self.root, "value(", 10, "auto"
        )

        self.assertIn("src/fallback.txt:1:cat", regex)
        self.assertIn("src/fallback.txt:2:value(", malformed)

    def test_python_search_does_not_follow_external_file_symlinks(self) -> None:
        external = Path(self.temp.name).parent / f"{self.root.name}-secret.txt"
        external.write_text("OUTSIDE_SECRET\n", encoding="utf-8")
        (self.root / "src" / "secret-link.txt").symlink_to(external)
        try:
            result = self.workspace._python_search(
                self.root, "OUTSIDE_SECRET", 10, "literal"
            )
        finally:
            external.unlink(missing_ok=True)

        self.assertEqual(result, "No matches.")

    def test_search_ignores_dependency_and_cache_directories(self) -> None:
        (self.root / "node_modules" / "pkg").mkdir(parents=True)
        (self.root / "node_modules" / "pkg" / "index.js").write_text(
            "hidden", encoding="utf-8"
        )
        (self.root / ".ruff_cache").mkdir()
        (self.root / ".ruff_cache" / "entry").write_text("hidden", encoding="utf-8")
        (self.root / "generated.egg-info").mkdir()
        (self.root / "generated.egg-info" / "PKG-INFO").write_text(
            "hidden", encoding="utf-8"
        )

        listing = self.workspace.list_files()

        self.assertNotIn("node_modules", listing)
        self.assertNotIn(".ruff_cache", listing)
        self.assertNotIn("generated.egg-info", listing)

    def test_action_timeouts_are_bounded_at_schema_boundary(self) -> None:
        with self.assertRaises(ValidationError):
            SearchAction(action="search", query="")
        with self.assertRaises(ValidationError):
            RunShellAction(
                action="run_shell", command="sleep 1", timeout_seconds=601
            )
        with self.assertRaises(ValidationError):
            RunTestsAction(action="run_tests", timeout_seconds=1201)

    def test_observation_truncation_keeps_relevant_end_for_commands(self) -> None:
        executor = self.executor.model_copy(update={"max_observation_chars": 4})

        self.assertTrue(executor._truncate("012345").startswith("0123"))
        self.assertTrue(
            executor._truncate("012345", keep_tail=True).endswith("2345")
        )


class DockerHardeningTests(unittest.TestCase):
    def test_timed_out_exec_reaps_all_unexpected_container_processes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            sandbox = DockerSandbox(
                workdir=Path(directory),
                container_id="sandbox-1",
                shell_timeout_seconds=30,
            )
            sandbox._keepalive_pid = 7
            timed_out = subprocess.TimeoutExpired(
                ["docker", "exec"],
                30,
                output="partial",
                stderr="",
            )
            cleaned = CompletedProcess([], 0, stdout="", stderr="")
            with patch(
                "src.agents.sandbox.subprocess.run",
                side_effect=[timed_out, cleaned],
            ) as run:
                result = sandbox.run_shell("sleep 999", timeout_seconds=30)

        self.assertEqual(result.returncode, 124)
        self.assertIn("partial", result.output)
        self.assertEqual(len(run.call_args_list), 2)
        cleanup_args = run.call_args_list[1].args[0]
        self.assertEqual(cleanup_args[:4], ["docker", "exec", "-i", "sandbox-1"])
        self.assertIn("signal_unexpected", cleanup_args[6])
        self.assertNotIn("/proc/[0-9]*/environ", cleanup_args[6])

    def test_cleanup_failure_invalidates_container_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            sandbox = DockerSandbox(
                workdir=Path(directory),
                container_id="sandbox-1",
            )
            sandbox._keepalive_pid = 7
            completed = CompletedProcess([], 0, stdout="ok", stderr="")
            cleanup_failed = CompletedProcess(
                [], 70, stdout="", stderr="missing keepalive"
            )
            stopped = CompletedProcess([], 0, stdout="", stderr="")
            removed = CompletedProcess(
                [], 1, stdout="", stderr="Error: No such object: sandbox-1"
            )
            with patch(
                "src.agents.sandbox.subprocess.run",
                side_effect=[completed, cleanup_failed, stopped, removed],
            ):
                result = sandbox.run_shell("true")

        self.assertEqual(result.returncode, 125)
        self.assertIn("cleanup failed", result.output)
        self.assertIsNone(sandbox.container_id)
        with self.assertRaisesRegex(RuntimeError, "cleanup failed"):
            sandbox.start()

    def test_oracle_mounts_pin_parents_and_use_immutable_staged_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "tests").mkdir()
            protected = root / "tests" / "test_app.py"
            protected.write_text("assert True\n", encoding="utf-8")
            sandbox = DockerSandbox(workdir=root)
            sandbox.configure_protected_test_paths(["tests/test_app.py"])

            started = CompletedProcess(
                [], 0, stdout="sandbox-1\n", stderr=""
            )
            captured = CompletedProcess([], 0, stdout="7\n", stderr="")
            stopped = CompletedProcess([], 0, stdout="", stderr="")
            removed = CompletedProcess(
                [], 1, stdout="", stderr="Error: No such object: sandbox-1"
            )
            with patch(
                "src.agents.sandbox.subprocess.run",
                side_effect=[started, captured, stopped, removed],
            ) as run:
                sandbox.start()
                snapshot = sandbox._oracle_snapshot_dir
                start_args = run.call_args_list[0].args[0]
                sandbox.stop()

        self.assertIsNotNone(snapshot)
        self.assertFalse(snapshot.exists())
        self.assertIn(
            f"type=bind,src={root.resolve() / 'tests'},dst=/workspace/tests",
            start_args,
        )
        file_mounts = [
            item
            for item in start_args
            if item.endswith("dst=/workspace/tests/test_app.py,readonly")
        ]
        self.assertEqual(len(file_mounts), 1)
        self.assertNotIn(f"src={protected.resolve()},", file_mounts[0])

    def test_container_stop_waits_for_auto_remove_before_releasing_mounts(
        self,
    ) -> None:
        stopped = CompletedProcess([], 0, stdout="", stderr="")
        still_present = CompletedProcess([], 0, stdout="{}", stderr="")
        daemon_error = CompletedProcess(
            [], 1, stdout="", stderr="cannot connect to Docker daemon"
        )
        removed = CompletedProcess(
            [], 1, stdout="", stderr="Error: No such object: sandbox-1"
        )
        with (
            patch(
                "src.agents.sandbox.subprocess.run",
                side_effect=[stopped, still_present, daemon_error, removed],
            ) as run,
            patch("src.agents.sandbox.time.sleep") as sleep,
        ):
            self.assertTrue(DockerSandbox._force_stop_container("sandbox-1"))

        self.assertEqual(
            [call.args[0][:2] for call in run.call_args_list],
            [
                ["docker", "stop"],
                ["docker", "inspect"],
                ["docker", "inspect"],
                ["docker", "inspect"],
            ],
        )
        self.assertEqual(sleep.call_count, 2)
        sleep.assert_called_with(0.05)

    def test_oracle_mount_limit_fails_closed_before_docker_start(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "tests").mkdir()
            (root / "tests" / "test_app.py").write_text(
                "assert True\n", encoding="utf-8"
            )
            sandbox = DockerSandbox(
                workdir=root,
                max_protected_oracle_mounts=1,
            )
            sandbox.configure_protected_test_paths(["tests/test_app.py"])

            with patch("src.agents.sandbox.subprocess.run") as run:
                with self.assertRaisesRegex(RuntimeError, "fail-closed limit"):
                    sandbox.start()

        run.assert_not_called()
        self.assertIsNone(sandbox._oracle_snapshot_dir)

    def test_scoring_can_mount_test_directory_read_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "tests").mkdir()
            (root / "tests" / "test_app.py").write_text(
                "assert True\n", encoding="utf-8"
            )
            sandbox = DockerSandbox(workdir=root)
            sandbox.configure_protected_test_paths(
                ["tests/test_app.py"],
                readonly_directories=["tests"],
            )

            args = sandbox._prepare_oracle_mount_args()
            sandbox._cleanup_oracle_snapshot()

        self.assertIn(
            f"type=bind,src={root.resolve() / 'tests'},"
            "dst=/workspace/tests,readonly",
            args,
        )

    def test_readonly_test_root_bounds_mounts_for_large_oracle(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "tests").mkdir()
            sandbox = DockerSandbox(
                workdir=root,
                max_protected_oracle_mounts=4,
            )
            sandbox.configure_protected_test_paths(
                [f"tests/test_{index}.py" for index in range(9_000)],
                readonly_directories=["tests"],
            )

            args = sandbox._prepare_oracle_mount_args()
            sandbox._cleanup_oracle_snapshot()

        mounts = [
            args[index + 1]
            for index, value in enumerate(args[:-1])
            if value == "--mount"
        ]
        self.assertEqual(len(mounts), 1)
        self.assertTrue(mounts[0].endswith("dst=/workspace/tests,readonly"))

    @unittest.skipUnless(
        os.environ.get("REPO_EVAL_RUN_DOCKER_TESTS") == "1",
        "set REPO_EVAL_RUN_DOCKER_TESTS=1 for Docker mount/process smoke tests",
    )
    def test_real_container_blocks_oracle_replacement_and_env_cleanup_bypass(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as directory:
            root = Path(directory)
            (root / "tests").mkdir()
            protected = root / "tests" / "test_app.py"
            protected.write_text("assert True\n", encoding="utf-8")
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            subprocess.run(["git", "-C", str(root), "add", "-A"], check=True)
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(root),
                    "-c",
                    "user.name=Hardening Test",
                    "-c",
                    "user.email=hardening@example.invalid",
                    "commit",
                    "-qm",
                    "base",
                ],
                check=True,
            )
            sandbox = DockerSandbox(workdir=root)
            executor = ActionExecutor(
                workspace=Workspace(root=root),
                sandbox=sandbox,
            )
            try:
                attack = executor.execute(
                    State(
                        task="protect oracle",
                        workdir=root,
                        test_command="python -m pytest tests",
                        max_steps=20,
                    ),
                    RunShellAction(
                        action="run_shell",
                        command=(
                            "set +e; "
                            "cp tests/test_app.py /tmp/original; "
                            "printf 'assert False\\n' > tests/test_app.py "
                            "2>/dev/null; rm -f tests/test_app.py 2>/dev/null; "
                            "ln -sf /tmp/original tests/test_app.py 2>/dev/null; "
                            "mv tests tests.old 2>/dev/null; mkdir -p tests; "
                            "printf 'assert False\\n' > tests/test_app.py "
                            "2>/dev/null; ln tests/test_app.py oracle-hardlink "
                            "2>/dev/null; printf 'assert False\\n' > "
                            "oracle-hardlink 2>/dev/null; "
                            "if grep -Fqx 'assert True' tests/test_app.py; then "
                            "echo ORACLE_INTACT; exit 0; else "
                            "echo ORACLE_REPLACED; exit 42; fi"
                        ),
                    ),
                )
                self.assertEqual(
                    protected.read_text(encoding="utf-8"),
                    "assert True\n",
                )
                self.assertIn("\nORACLE_INTACT\n", attack.context[-1].text)
                self.assertNotIn("\nORACLE_REPLACED\n", attack.context[-1].text)

                background = sandbox.run_shell(
                    "env -u REPO_EVAL_ACTION_ID sh -c "
                    "'sleep 300 </dev/null >/dev/null 2>&1 & child=$!; "
                    "echo $child > /tmp/repo-eval-keepalive.pid; echo $child'"
                )
                self.assertTrue(background.success, background.output)
                background_pid = int(background.output.strip())
                survived = sandbox.run_shell(f"kill -0 {background_pid}")
                self.assertFalse(survived.success, survived.output)
            finally:
                sandbox.stop()

    def test_requested_timeouts_cannot_exceed_experiment_ceilings(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            sandbox = DockerSandbox(
                workdir=Path(directory),
                container_id="sandbox-1",
                shell_timeout_seconds=30,
                test_timeout_seconds=90,
            )
            sandbox._keepalive_pid = 7
            completed = CompletedProcess([], 0, stdout="ok\n", stderr="")
            with patch(
                "src.agents.sandbox.subprocess.run", return_value=completed
            ) as run:
                sandbox.run_shell("slow shell", timeout_seconds=600)
                sandbox.run_tests("slow tests", timeout_seconds=1200)
        command_calls = [
            call
            for call in run.call_args_list
            if "--env" in call.args[0]
        ]
        shell_timeout = command_calls[0].kwargs["timeout"]
        test_timeout = command_calls[1].kwargs["timeout"]

        self.assertEqual(shell_timeout, 30)
        self.assertEqual(test_timeout, 90)

    def test_evaluator_evidence_is_read_from_container_tmpfs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            sandbox = DockerSandbox(
                workdir=Path(directory), container_id="sandbox-1"
            )
            completed = CompletedProcess(
                [], 0, stdout=b"<testsuites />", stderr=b""
            )
            with patch(
                "src.agents.sandbox.subprocess.run", return_value=completed
            ) as run:
                payload = sandbox.pop_isolated_file(
                    "/tmp/repo-eval-hidden-token.xml"
                )

            self.assertEqual(payload, b"<testsuites />")
            command = run.call_args.args[0]
            self.assertEqual(command[:4], ["docker", "exec", "-i", "sandbox-1"])
            self.assertEqual(command[-1], "/tmp/repo-eval-hidden-token.xml")
            with self.assertRaisesRegex(
                ValueError, "Unsafe isolated evaluator path"
            ):
                sandbox.pop_isolated_file("/workspace/forged.xml")

    def test_setup_and_agent_share_one_hardened_non_root_container(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / ".git").mkdir()
            sandbox = DockerSandbox(workdir=root)

            def docker_run(args: list[str], **_kwargs: object) -> CompletedProcess[str]:
                if args[:3] == ["docker", "run", "-d"]:
                    return CompletedProcess(args, 0, stdout="sandbox-1\n", stderr="")
                if "read -r pid" in " ".join(args):
                    return CompletedProcess(args, 0, stdout="7\n", stderr="")
                return CompletedProcess(args, 0, stdout="ok\n", stderr="")

            with patch(
                "src.agents.sandbox.subprocess.run", side_effect=docker_run
            ) as run:
                sandbox.run_shell("python -m pip install -e .")
                sandbox.run_shell("python -m pytest")

        calls = [call.args[0] for call in run.call_args_list]
        starts = [args for args in calls if args[:3] == ["docker", "run", "-d"]]
        execs = [args for args in calls if args[:3] == ["docker", "exec", "-i"]]
        command_execs = [args for args in execs if "--env" in args]
        cleanup_execs = [args for args in execs if "--env" not in args]
        self.assertEqual(len(starts), 1)
        self.assertEqual(len(command_execs), 2)
        # One startup exec captures the trusted keep-alive pid; two more reap
        # processes after the model-controlled actions.
        self.assertEqual(len(cleanup_execs), 3)
        self.assertTrue(all(args[5] == "sandbox-1" for args in command_execs))
        self.assertTrue(all(args[3] == "sandbox-1" for args in cleanup_execs))

        args = starts[0]
        for flag in (
            "--init",
            "--label",
            "--cap-drop",
            "--security-opt",
            "--pids-limit",
            "--memory",
            "--cpus",
            "--user",
            "--read-only",
        ):
            self.assertIn(flag, args)
        self.assertIn("ALL", args)
        self.assertIn("no-new-privileges:true", args)
        self.assertIn(f"{os.getuid()}:{os.getgid()}", args)
        self.assertIn("HOME=/home/agent", args)
        self.assertIn("PIP_USER=1", args)
        self.assertTrue(any("dst=/workspace/.git,readonly" in item for item in args))


if __name__ == "__main__":
    unittest.main()
