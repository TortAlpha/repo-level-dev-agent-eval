from __future__ import annotations

import subprocess
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from .tracing import trace_shell_command


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
    container_workdir: str = "/workspace"
    container_id: str | None = None

    def start(self) -> str:
        if self.container_id:
            return self.container_id

        self.workdir.mkdir(parents=True, exist_ok=True)
        args = [
            "docker",
            "run",
            "-d",
            "--rm",
            "--mount",
            f"type=bind,src={self.workdir.resolve()},dst={self.container_workdir}",
            "-w",
            self.container_workdir,
        ]
        if self.network_disabled:
            args.extend(["--network", "none"])
        args.extend([self.image, "sh", "-lc", "while true; do sleep 1; done"])

        try:
            result = subprocess.run(
                args,
                capture_output=True,
                check=False,
                text=True,
                timeout=60,
            )
        except FileNotFoundError as exc:
            raise RuntimeError("Docker CLI not found. Is Docker installed?") from exc
        except subprocess.TimeoutExpired as exc:
            output = combine_output(_to_text(exc.stdout), _to_text(exc.stderr))
            raise RuntimeError(
                f"Timed out while starting Docker sandbox:\n{output}"
            ) from exc
        output = combine_output(result.stdout, result.stderr)
        if result.returncode != 0:
            raise RuntimeError(f"Failed to start Docker sandbox:\n{output}")

        self.container_id = result.stdout.strip()
        return self.container_id

    @trace_shell_command
    def run_shell(
        self,
        command: str,
        timeout_seconds: int | None = None,
    ) -> CommandResult:
        container_id = self.start()
        timeout = timeout_seconds or self.shell_timeout_seconds

        try:
            result = subprocess.run(
                ["docker", "exec", "-i", container_id, "sh", "-lc", command],
                capture_output=True,
                check=False,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as exc:
            output = combine_output(_to_text(exc.stdout), _to_text(exc.stderr))
            return CommandResult(
                returncode=124,
                output=f"Command timed out after {timeout} seconds.\n{output}".strip(),
            )

        return CommandResult(
            returncode=result.returncode,
            output=combine_output(result.stdout, result.stderr),
        )

    def run_tests(
        self,
        command: str,
        timeout_seconds: int | None = None,
    ) -> CommandResult:
        timeout = timeout_seconds or self.test_timeout_seconds
        return self.run_shell(command, timeout_seconds=timeout)

    def stop(self) -> None:
        if not self.container_id:
            return

        subprocess.run(
            ["docker", "stop", self.container_id],
            capture_output=True,
            check=False,
            text=True,
            timeout=60,
        )
        self.container_id = None


def combine_output(stdout: str, stderr: str) -> str:
    if stdout and stderr:
        return f"{stdout.rstrip()}\n{stderr.rstrip()}".strip()
    return (stdout or stderr).strip()


def _to_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value
