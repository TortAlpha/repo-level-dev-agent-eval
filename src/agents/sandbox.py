from __future__ import annotations

import os
import re
import shutil
import stat
import subprocess
import tempfile
import uuid
from pathlib import Path, PurePosixPath

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from .tracing import trace_shell_command

SANDBOX_LABEL = "repo-level-dev-agent-eval.sandbox=1"
ACTION_PROCESS_MARKER = "REPO_EVAL_ACTION_ID"
DOCKER_CPU_RANGE_PATTERN = re.compile(
    r"range of CPUs is from [0-9.]+ to (?P<maximum>[0-9.]+)",
    re.IGNORECASE,
)
MAX_PROTECTED_ORACLE_MOUNTS = 8192
DOCKER_ARGV_SAFETY_MARGIN = 32 * 1024
KEEPALIVE_PID_FILE = "/tmp/repo-eval-keepalive.pid"
EVALUATOR_HELPER_CONTAINER_PATH = "/tmp/repo-eval-trusted-pytest.py"
EVALUATOR_BIN_CONTAINER_PATH = "/tmp/repo-eval-bin"


class CommandResult(BaseModel):
    returncode: int
    output: str

    @property
    def success(self) -> bool:
        return self.returncode == 0


class DockerSandbox(BaseModel):
    model_config = ConfigDict(validate_assignment=True)

    workdir: Path
    image: str = "python:3.11-slim"
    network_disabled: bool = True
    shell_timeout_seconds: int = Field(default=300, gt=0)
    test_timeout_seconds: int = Field(default=600, gt=0)
    cpu_limit: float = Field(default=4.0, gt=0)
    memory_limit: str = "4g"
    pids_limit: int = Field(default=512, gt=0)
    read_only_rootfs: bool = True
    container_workdir: str = "/workspace"
    container_id: str | None = None
    protected_test_paths: tuple[str, ...] = ()
    protected_test_directories: tuple[str, ...] = ()
    dependency_environment: Path | None = None
    dependency_environment_readonly: bool = False
    evaluator_helper_path: Path | None = None
    evaluator_bin_path: Path | None = None
    max_protected_oracle_mounts: int = Field(
        default=MAX_PROTECTED_ORACLE_MOUNTS, gt=0
    )
    _oracle_snapshot_dir: Path | None = PrivateAttr(default=None)
    _invalidated_reason: str | None = PrivateAttr(default=None)
    # Captured on the host immediately after startup. The agent can write to
    # container tmpfs, so cleanup must never trust an in-container pidfile
    # after a model-controlled command has run.
    _keepalive_pid: int | None = PrivateAttr(default=None)

    def configure_protected_test_paths(
        self,
        paths: list[str],
        *,
        readonly_directories: list[str] | None = None,
    ) -> None:
        """Configure tracked tests before the container is started."""
        if self.container_id is not None:
            raise RuntimeError(
                "Test-oracle protection must be configured before Docker starts."
            )
        normalized: set[str] = set()
        for path in paths:
            candidate = Path(path)
            if candidate.is_absolute() or ".." in candidate.parts:
                raise ValueError(f"Unsafe protected test path: {path}")
            normalized.add(candidate.as_posix())
        self.protected_test_paths = tuple(sorted(normalized))
        normalized_directories: set[str] = set()
        for path in readonly_directories or []:
            candidate = Path(path)
            if (
                candidate in {Path(""), Path(".")}
                or candidate.is_absolute()
                or ".." in candidate.parts
            ):
                raise ValueError(f"Unsafe protected test directory: {path}")
            normalized_directories.add(candidate.as_posix())
        self.protected_test_directories = tuple(sorted(normalized_directories))

    def _prepare_oracle_mount_args(self) -> list[str]:
        """Stage immutable oracle copies and pin every ancestor directory.

        A read-only file mount protects bytes, while writable bind mounts on
        each ancestor keep the protected path from being moved aside. The
        directories themselves remain writable, so agents can still add new
        regression-test files next to the tracked oracle.
        """
        if not self.protected_test_paths and not self.protected_test_directories:
            return []

        root = self.workdir.resolve()
        snapshot = Path(
            tempfile.mkdtemp(
                prefix=".repo-eval-oracle-",
                dir=root.parent,
            )
        )
        self._oracle_snapshot_dir = snapshot
        pinned_directories: set[Path] = set()
        staged_files: list[tuple[Path, PurePosixPath]] = []
        readonly_roots = {
            PurePosixPath(path) for path in self.protected_test_directories
        }

        try:
            for relative_name in self.protected_test_paths:
                relative = Path(relative_name)
                container_relative = PurePosixPath(relative.as_posix())
                if any(
                    container_relative == root or root in container_relative.parents
                    for root in readonly_roots
                ):
                    continue
                source = root / relative
                try:
                    metadata = source.lstat()
                except OSError as exc:
                    raise RuntimeError(
                        "Tracked test-oracle file is unavailable: "
                        f"{relative_name}: {exc}"
                    ) from exc
                if not stat.S_ISREG(metadata.st_mode):
                    raise RuntimeError(
                        "Tracked test-oracle entries must be regular files; "
                        f"refusing an unsafe mount for {relative_name}."
                    )

                parent = relative.parent
                while parent != Path("."):
                    host_parent = root / parent
                    try:
                        parent_metadata = host_parent.lstat()
                    except OSError as exc:
                        raise RuntimeError(
                            f"Test-oracle parent is unavailable: {parent}: {exc}"
                        ) from exc
                    if not stat.S_ISDIR(parent_metadata.st_mode):
                        raise RuntimeError(
                            "Test-oracle paths cannot traverse symlinked or "
                            f"non-directory parents: {relative_name}."
                        )
                    pinned_directories.add(parent)
                    parent = parent.parent

                staged = snapshot / relative
                staged.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, staged, follow_symlinks=False)
                shutil.copymode(source, staged, follow_symlinks=False)
                staged_files.append((staged, container_relative))

            for relative_name in self.protected_test_directories:
                relative_dir = Path(relative_name)
                source_dir = root / relative_dir
                try:
                    metadata = source_dir.lstat()
                except OSError as exc:
                    raise RuntimeError(
                        "Protected test directory is unavailable: "
                        f"{relative_name}: {exc}"
                    ) from exc
                if not stat.S_ISDIR(metadata.st_mode):
                    raise RuntimeError(
                        "Protected test directories must be real directories: "
                        f"{relative_name}."
                    )
                pinned_directories.add(relative_dir)
                parent = relative_dir.parent
                while parent != Path("."):
                    host_parent = root / parent
                    try:
                        parent_metadata = host_parent.lstat()
                    except OSError as exc:
                        raise RuntimeError(
                            "Protected test-directory parent is unavailable: "
                            f"{parent}: {exc}"
                        ) from exc
                    if not stat.S_ISDIR(parent_metadata.st_mode):
                        raise RuntimeError(
                            "Protected test directories cannot traverse "
                            f"symlinked or non-directory parents: {relative_name}."
                        )
                    pinned_directories.add(parent)
                    parent = parent.parent

            mount_count = len(pinned_directories) + len(staged_files)
            if mount_count > self.max_protected_oracle_mounts:
                raise RuntimeError(
                    "Tracked test oracle needs "
                    f"{mount_count} bind mounts, exceeding the fail-closed limit "
                    f"of {self.max_protected_oracle_mounts}."
                )

            args: list[str] = []
            container_root = PurePosixPath(self.container_workdir)
            for relative_dir in sorted(
                pinned_directories,
                key=lambda item: (len(item.parts), item.as_posix()),
            ):
                source_dir = root / relative_dir
                target_dir = container_root / PurePosixPath(relative_dir.as_posix())
                args.extend(
                    [
                        "--mount",
                        "type=bind,"
                        f"src={source_dir},dst={target_dir}"
                        + (
                            ",readonly"
                            if relative_dir.as_posix()
                            in self.protected_test_directories
                            else ""
                        ),
                    ]
                )
            for staged, container_relative in staged_files:
                target = container_root / container_relative
                args.extend(
                    [
                        "--mount",
                        f"type=bind,src={staged},dst={target},readonly",
                    ]
                )
            return args
        except Exception:
            self._cleanup_oracle_snapshot()
            raise

    def _cleanup_oracle_snapshot(self) -> None:
        snapshot = self._oracle_snapshot_dir
        self._oracle_snapshot_dir = None
        if snapshot is not None:
            shutil.rmtree(snapshot, ignore_errors=True)

    @staticmethod
    def _validate_docker_argv(args: list[str]) -> None:
        try:
            argument_limit = os.sysconf("SC_ARG_MAX")
        except (AttributeError, OSError, ValueError):
            argument_limit = 256 * 1024
        encoded_size = sum(len(item.encode("utf-8")) + 1 for item in args)
        safe_limit = max(argument_limit - DOCKER_ARGV_SAFETY_MARGIN, 1)
        if encoded_size > safe_limit:
            raise RuntimeError(
                "Docker sandbox arguments exceed the safe OS argv limit while "
                "mounting the tracked test oracle; refusing to run unprotected."
            )

    def start(self) -> str:
        if self._invalidated_reason is not None:
            raise RuntimeError(self._invalidated_reason)
        if self.container_id:
            return self.container_id

        self.workdir.mkdir(parents=True, exist_ok=True)
        args = [
            "docker",
            "run",
            "-d",
            "--rm",
            # Neutralize image ENTRYPOINTs (e.g. SWE-bench Pro envs ship
            # ENTRYPOINT=/bin/bash, which would swallow our keep-alive loop).
            "--entrypoint",
            "/bin/sh",
            "--init",
            "--label",
            SANDBOX_LABEL,
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges:true",
            "--pids-limit",
            str(self.pids_limit),
            "--memory",
            self.memory_limit,
            "--cpus",
            str(self.cpu_limit),
            "--user",
            f"{os.getuid()}:{os.getgid()}",
            "--mount",
            f"type=bind,src={self.workdir.resolve()},dst={self.container_workdir}",
            "--tmpfs",
            "/tmp:rw,nosuid,nodev,size=512m,mode=1777",
            "--tmpfs",
            "/home/agent:rw,nosuid,nodev,size=512m,mode=1777",
            "--env",
            "HOME=/home/agent",
            "--env",
            "PYTHONUSERBASE=/home/agent/.local",
            "--env",
            "PYTHONSAFEPATH=1",
            "--env",
            "PIP_USER=1",
            "--env",
            "PIP_CACHE_DIR=/tmp/.cache/pip",
            "--env",
            "XDG_CACHE_HOME=/tmp/.cache",
            "--env",
            "UV_CACHE_DIR=/tmp/.cache/uv",
            "--env",
            "NPM_CONFIG_CACHE=/tmp/.cache/npm",
            "--env",
            "CARGO_HOME=/tmp/.cache/cargo",
            "--env",
            "PATH=/usr/local/bin:/usr/bin:/bin:/home/agent/.local/bin",
            "-w",
            self.container_workdir,
        ]
        if self.dependency_environment is not None:
            dependency_environment = self.dependency_environment.resolve()
            dependency_environment.mkdir(parents=True, exist_ok=True)
            args.extend(
                [
                    "--mount",
                    "type=bind,"
                    f"src={dependency_environment},dst=/home/agent/.local"
                    + (",readonly" if self.dependency_environment_readonly else ""),
                ]
            )
        if self.evaluator_helper_path is not None:
            helper = self.evaluator_helper_path.resolve()
            try:
                helper_metadata = helper.lstat()
            except OSError as exc:
                raise RuntimeError(
                    f"Evaluator helper is unavailable: {helper}: {exc}"
                ) from exc
            if not stat.S_ISREG(helper_metadata.st_mode):
                raise RuntimeError(
                    f"Evaluator helper must be a regular file: {helper}"
                )
            args.extend(
                [
                    "--mount",
                    "type=bind,"
                    f"src={helper},dst={EVALUATOR_HELPER_CONTAINER_PATH},readonly",
                ]
            )
        if self.evaluator_bin_path is not None:
            evaluator_bin = self.evaluator_bin_path.resolve()
            try:
                bin_metadata = evaluator_bin.lstat()
            except OSError as exc:
                raise RuntimeError(
                    f"Evaluator launcher directory is unavailable: "
                    f"{evaluator_bin}: {exc}"
                ) from exc
            if not stat.S_ISDIR(bin_metadata.st_mode) or evaluator_bin.is_symlink():
                raise RuntimeError(
                    "Evaluator launcher path must be a real directory: "
                    f"{evaluator_bin}"
                )
            required_launchers = {
                "_python_shim.py",
                "py.test",
                "pytest",
                "python",
                "python3",
            }
            for name in required_launchers:
                launcher = evaluator_bin / name
                try:
                    launcher_metadata = launcher.lstat()
                except OSError as exc:
                    raise RuntimeError(
                        f"Evaluator launcher is unavailable: {launcher}: {exc}"
                    ) from exc
                if not stat.S_ISREG(launcher_metadata.st_mode):
                    raise RuntimeError(
                        f"Evaluator launcher must be a regular file: {launcher}"
                    )
            args.extend(
                [
                    "--mount",
                    "type=bind,"
                    f"src={evaluator_bin},dst={EVALUATOR_BIN_CONTAINER_PATH},readonly",
                ]
            )
        args.extend(self._prepare_oracle_mount_args())
        git_metadata = self.workdir.resolve() / ".git"
        if git_metadata.exists():
            # The repository itself remains writable for code edits, but Git
            # metadata is not an agent tool. This blocks commit/reset tricks
            # and protects both the baseline and post-run patch capture. The
            # source may be either a directory or a worktree `.git` file.
            args.extend(
                [
                    "--mount",
                    "type=bind,"
                    f"src={git_metadata},dst={self.container_workdir}/.git,readonly",
                ]
            )
        if self.read_only_rootfs:
            args.append("--read-only")
        if self.network_disabled:
            args.extend(["--network", "none"])
        args.extend(
            [
                self.image,
                "-lc",
                f"printf '%s\\n' \"$$\" > {KEEPALIVE_PID_FILE}; "
                # Replace the shell in-place with one stable process. A shell
                # loop around `sleep` would continuously create a legitimate
                # child that is indistinguishable from an action survivor.
                "exec tail -f /dev/null",
            ]
        )
        try:
            self._validate_docker_argv(args)
        except Exception:
            self._cleanup_oracle_snapshot()
            raise

        for attempt in range(2):
            try:
                result = subprocess.run(
                    args,
                    capture_output=True,
                    check=False,
                    text=True,
                    timeout=60,
                )
            except FileNotFoundError as exc:
                self._cleanup_oracle_snapshot()
                raise RuntimeError(
                    "Docker CLI not found. Is Docker installed?"
                ) from exc
            except subprocess.TimeoutExpired as exc:
                output = combine_output(_to_text(exc.stdout), _to_text(exc.stderr))
                self._cleanup_oracle_snapshot()
                raise RuntimeError(
                    f"Timed out while starting Docker sandbox:\n{output}"
                ) from exc

            output = combine_output(result.stdout, result.stderr)
            if result.returncode == 0:
                break
            available_cpus = _docker_available_cpu_limit(output)
            if (
                attempt == 0
                and available_cpus is not None
                and available_cpus < self.cpu_limit
            ):
                args = list(args)
                args[args.index("--cpus") + 1] = str(available_cpus)
                continue
            self._cleanup_oracle_snapshot()
            raise RuntimeError(f"Failed to start Docker sandbox:\n{output}")

        self.container_id = result.stdout.strip()
        try:
            self._keepalive_pid = self._capture_keepalive_pid(self.container_id)
        except Exception:
            container_id = self.container_id
            self.container_id = None
            self._keepalive_pid = None
            if container_id:
                self._force_stop_container(container_id)
            self._cleanup_oracle_snapshot()
            raise
        return self.container_id

    def resolve_trusted_python(self) -> str:
        """Resolve Python from the evaluator's fixed, non-workspace PATH."""
        container_id = self.start()
        for executable in ("python", "python3"):
            try:
                result = subprocess.run(
                    [
                        "docker",
                        "exec",
                        "-i",
                        container_id,
                        "/bin/sh",
                        "-c",
                        'command -v "$1"',
                        "evaluator-python",
                        executable,
                    ],
                    capture_output=True,
                    check=False,
                    text=True,
                    timeout=15,
                )
            except FileNotFoundError as exc:
                raise RuntimeError(
                    "Docker CLI disappeared while resolving evaluator Python."
                ) from exc
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError(
                    "Timed out while resolving evaluator Python."
                ) from exc
            raw = result.stdout.strip()
            candidate = PurePosixPath(raw)
            if (
                result.returncode == 0
                and raw.startswith("/")
                and "\n" not in raw
                and candidate.parent
                in {
                    PurePosixPath("/bin"),
                    PurePosixPath("/usr/bin"),
                    PurePosixPath("/usr/local/bin"),
                }
            ):
                return raw
        raise RuntimeError(
            "Evaluator image has no Python interpreter on its trusted system PATH."
        )

    @staticmethod
    def _capture_keepalive_pid(container_id: str) -> int:
        """Read the main-process pid once, then remove the writable marker."""
        try:
            result = subprocess.run(
                [
                    "docker",
                    "exec",
                    "-i",
                    container_id,
                    "/bin/sh",
                    "-lc",
                    (
                        f"read -r pid < {KEEPALIVE_PID_FILE} && "
                        f"rm -f {KEEPALIVE_PID_FILE} && printf '%s\\n' \"$pid\""
                    ),
                ],
                capture_output=True,
                check=False,
                text=True,
                timeout=15,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                "Docker CLI disappeared while securing the sandbox process id."
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                "Timed out while securing the sandbox process id."
            ) from exc
        raw = result.stdout.strip()
        if result.returncode != 0 or not raw.isdigit() or int(raw) <= 1:
            output = combine_output(result.stdout, result.stderr)
            raise RuntimeError(
                "Could not capture the sandbox keep-alive process id before "
                "agent execution."
                + (f"\n{output}" if output else "")
            )
        return int(raw)

    @trace_shell_command
    def run_shell(
        self,
        command: str,
        timeout_seconds: int | None = None,
    ) -> CommandResult:
        timeout = (
            self.shell_timeout_seconds
            if timeout_seconds is None
            else min(timeout_seconds, self.shell_timeout_seconds)
        )
        return self._run_in_container(command, timeout)

    def _run_in_container(self, command: str, timeout: int) -> CommandResult:
        container_id = self.start()
        action_id = uuid.uuid4().hex

        try:
            result = subprocess.run(
                [
                    "docker",
                    "exec",
                    "-i",
                    "--env",
                    f"{ACTION_PROCESS_MARKER}={action_id}",
                    container_id,
                    "/bin/sh",
                    "-lc",
                    command,
                ],
                capture_output=True,
                check=False,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as exc:
            output = combine_output(_to_text(exc.stdout), _to_text(exc.stderr))
            command_result = CommandResult(
                returncode=124,
                output=f"Command timed out after {timeout} seconds.\n{output}".strip(),
            )
        else:
            command_result = CommandResult(
                returncode=result.returncode,
                output=combine_output(result.stdout, result.stderr),
            )

        # Killing a timed-out `docker exec` client does not guarantee that its
        # in-container process died. Successful commands can also leave `&`
        # children behind. Reap every process except the dedicated PID 1 and
        # keep-alive shell, including children that deliberately unset the
        # per-action environment marker.
        keepalive_pid = self._keepalive_pid
        cleanup_error: str | None
        if keepalive_pid is None:
            cleanup_error = (
                "Sandbox process cleanup has no trusted keep-alive process id."
            )
        else:
            cleanup_error = self._cleanup_action_processes(
                container_id,
                action_id,
                keepalive_pid,
            )
        if cleanup_error is not None:
            self._invalidate_container(cleanup_error)
            return CommandResult(
                returncode=125,
                output=combine_output(command_result.output, cleanup_error),
            )
        return command_result

    def pop_isolated_file(
        self,
        path: str,
        *,
        max_bytes: int = 16 * 1024 * 1024,
    ) -> bytes | None:
        """Read and remove an evaluator artifact from container-local tmpfs.

        The path is never part of the writable repository bind mount.  This is
        intended for scorer evidence such as JUnit XML, not agent-visible file
        operations.
        """
        candidate = PurePosixPath(path)
        if (
            not candidate.is_absolute()
            or candidate.parts[:2] != ("/", "tmp")
            or not candidate.name.startswith("repo-eval-")
            or any(part in {"", ".", ".."} for part in candidate.parts[1:])
        ):
            raise ValueError(f"Unsafe isolated evaluator path: {path}")
        if max_bytes <= 0:
            raise ValueError("max_bytes must be positive")
        container_id = self.container_id
        if container_id is None:
            raise RuntimeError("Evaluator container is not running")
        script = (
            'path="$1"; [ -f "$path" ] && [ ! -L "$path" ] || exit 44; '
            '/bin/cat "$path"; status=$?; /bin/rm -f "$path"; exit "$status"'
        )
        try:
            result = subprocess.run(
                [
                    "docker",
                    "exec",
                    "-i",
                    container_id,
                    "/bin/sh",
                    "-c",
                    script,
                    "evaluator-artifact",
                    path,
                ],
                capture_output=True,
                check=False,
                timeout=30,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                "Docker CLI disappeared while collecting evaluator evidence."
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                "Timed out while collecting evaluator evidence."
            ) from exc
        if result.returncode == 44:
            return None
        if result.returncode != 0:
            detail = _to_text(result.stderr)[-1000:]
            raise RuntimeError(
                "Could not collect isolated evaluator evidence"
                + (f":\n{detail}" if detail else "")
            )
        if len(result.stdout) > max_bytes:
            raise RuntimeError(
                "Isolated evaluator evidence exceeds the byte safety limit"
            )
        return result.stdout

    @staticmethod
    def _cleanup_action_processes(
        container_id: str,
        action_id: str,
        keepalive_pid: int,
    ) -> str | None:
        cleanup = r'''
keepalive_pid="$1"
self_pid="$$"
signal_unexpected() {
  requested_signal="$1"
  for process in /proc/[0-9]*; do
    pid="${process#/proc/}"
    case "$pid" in
      1|"$self_pid"|"$keepalive_pid") continue ;;
    esac
    kill "-$requested_signal" "$pid" 2>/dev/null || true
  done
}
signal_unexpected TERM
sleep 0.2 2>/dev/null || sleep 1
# Repeat a bounded scan so a process killed in the first pass cannot leave a
# just-forked, re-parented child behind. The pids limit bounds adversarial churn.
signal_unexpected KILL
signal_unexpected KILL
sleep 0.2 2>/dev/null || sleep 1
survivors=""
for process in /proc/[0-9]*; do
  pid="${process#/proc/}"
  case "$pid" in
    1|"$self_pid"|"$keepalive_pid") continue ;;
  esac
  if kill -0 "$pid" 2>/dev/null; then
    stat_line=""
    IFS= read -r stat_line < "$process/stat" 2>/dev/null || true
    after_name="${stat_line##*) }"
    state="${after_name%% *}"
    # A zombie cannot mutate the workspace and Docker's PID 1 will reap it.
    [ "$state" = "Z" ] && continue
    survivors="$survivors $pid"
  fi
done
if [ -n "$survivors" ]; then
  echo "sandbox cleanup left unexpected process(es):$survivors" >&2
  exit 71
fi
'''.strip()
        try:
            result = subprocess.run(
                [
                    "docker",
                    "exec",
                    "-i",
                    container_id,
                    "/bin/sh",
                    "-lc",
                    cleanup,
                    "sandbox-cleanup",
                    str(keepalive_pid),
                    action_id,
                ],
                capture_output=True,
                check=False,
                text=True,
                timeout=15,
            )
        except FileNotFoundError:
            return "Sandbox process cleanup failed: Docker CLI disappeared."
        except subprocess.TimeoutExpired as exc:
            output = combine_output(_to_text(exc.stdout), _to_text(exc.stderr))
            return (
                "Sandbox process cleanup timed out; the container was invalidated."
                + (f"\n{output}" if output else "")
            )
        if result.returncode != 0:
            output = combine_output(result.stdout, result.stderr)
            return (
                "Sandbox process cleanup failed; the container was invalidated."
                + (f"\n{output}" if output else "")
            )
        return None

    def _invalidate_container(self, reason: str) -> None:
        container_id = self.container_id
        self._invalidated_reason = reason
        if container_id is None or self._force_stop_container(container_id):
            self.container_id = None
            self._keepalive_pid = None
            self._cleanup_oracle_snapshot()
            return
        # Retain the id and immutable mount sources so the agent's outer
        # finally block can retry. Continuing while an un-reaped container may
        # still write into the workspace would violate the snapshot boundary.
        raise RuntimeError(
            f"{reason}\nCould not stop the invalidated Docker container "
            f"{container_id}; aborting the agent run."
        )

    @staticmethod
    def _force_stop_container(container_id: str) -> bool:
        for command in (
            ["docker", "stop", "--time", "1", container_id],
            ["docker", "rm", "-f", container_id],
        ):
            try:
                result = subprocess.run(
                    command,
                    capture_output=True,
                    check=False,
                    text=True,
                    timeout=15,
                )
            except (FileNotFoundError, subprocess.TimeoutExpired):
                return False
            if result.returncode == 0:
                return True
        try:
            inspected = subprocess.run(
                ["docker", "inspect", container_id],
                capture_output=True,
                check=False,
                text=True,
                timeout=15,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False
        return inspected.returncode != 0

    @trace_shell_command
    def run_tests(
        self,
        command: str,
        timeout_seconds: int | None = None,
    ) -> CommandResult:
        # Test runs have their own (usually larger) experiment ceiling and
        # must not be accidentally clamped to the interactive shell limit.
        timeout = (
            self.test_timeout_seconds
            if timeout_seconds is None
            else min(timeout_seconds, self.test_timeout_seconds)
        )
        return self._run_in_container(command, timeout)

    def disable_network(self) -> None:
        """Disconnect a running setup container before the agent can act.

        Dependency installation may need the network, while benchmark agents
        must not be able to fetch a public upstream patch. Containers started
        directly with ``--network none`` are already isolated.
        """
        if self.network_disabled:
            return
        container_id = self.start()
        result = subprocess.run(
            ["docker", "network", "disconnect", "bridge", container_id],
            capture_output=True,
            check=False,
            text=True,
            timeout=60,
        )
        if result.returncode != 0:
            output = combine_output(result.stdout, result.stderr)
            raise RuntimeError(
                f"Failed to isolate benchmark container network:\n{output}"
            )
        self.network_disabled = True

    def stop(self) -> None:
        if not self.container_id:
            self._keepalive_pid = None
            self._cleanup_oracle_snapshot()
            return

        container_id = self.container_id
        if not self._force_stop_container(container_id):
            raise RuntimeError(f"Failed to stop Docker sandbox {container_id}.")
        self.container_id = None
        self._keepalive_pid = None
        self._cleanup_oracle_snapshot()


def combine_output(stdout: str, stderr: str) -> str:
    if stdout and stderr:
        return f"{stdout.rstrip()}\n{stderr.rstrip()}".strip()
    return (stdout or stderr).strip()


def _docker_available_cpu_limit(output: str) -> float | None:
    match = DOCKER_CPU_RANGE_PATTERN.search(output)
    if match is None:
        return None
    maximum = float(match.group("maximum"))
    return maximum if maximum > 0 else None


def _to_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value
