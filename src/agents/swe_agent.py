"""Run SWE-agent as a drop-in alternative agent.

The benchmark scores a *patch*, not an agent, so any agent that turns
``repo@base + task.md`` into edits on the workspace can be evaluated by the same
downstream pipeline (visible/hidden tests, regressions, quality, cost). This
adapter shells out to `SWE-agent <https://swe-agent.com>`_, applies the patch it
produces onto the prepared workspace, and maps its trajectory onto a ``State``
so the runner and metrics work unchanged.

Requirements to actually run it: ``pip install sweagent`` and a working Docker.
The comparable adapter accepts only typed, fingerprinted invocation fields;
release-specific arbitrary CLI passthrough is intentionally rejected.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import signal
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ..git_safety import safe_git_command, safe_git_env
from .model import LangChainModel
from .state import State
from .tracing import trace_agent_run


class SweAgentError(RuntimeError):
    """Raised when SWE-agent is missing or its run fails."""


# The base image the fast path auto-builds/uses: python + git + swe-rex, so the
# standalone-python build can be skipped. Contains no repo or task specifics.
MANAGED_BASE_IMAGE = "swe-adapter-base:py311"

SWE_CONTAINER_LABEL = "repo-level-dev-agent-eval.swe-run"

_UNTRUSTED_HOST_ENV_PREFIXES = (
    "SWE_AGENT_",
    "SWEAGENT_",
    "SWE_REX_",
    "SWEREX_",
    "PYTHON",
    "LITELLM_",
    "LD_",
    "DYLD_",
)
_UNTRUSTED_HOST_ENV_KEYS = {
    "BASH_ENV",
    "ENV",
    "COVERAGE_PROCESS_START",
    "COVERAGE_RCFILE",
}


@dataclass(frozen=True)
class SweAgentUsage:
    """Strictly validated usage extracted from one external trajectory."""

    steps: int = 0
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    accounting_complete: bool = False


def ensure_managed_base_image() -> None:
    """Build the harness-managed SWE-ReX image once if it is absent."""
    if subprocess.run(
        ["docker", "image", "inspect", MANAGED_BASE_IMAGE],
        capture_output=True,
        check=False,
    ).returncode == 0:
        return
    try:
        import swerex  # noqa: PLC0415

        rex_version = getattr(swerex, "__version__", "")
    except Exception:
        rex_version = ""
    rex_pin = f"swe-rex=={rex_version}" if rex_version else "swe-rex"
    dockerfile = (
        "FROM python:3.11-slim\n"
        "RUN apt-get update && apt-get install -y --no-install-recommends git "
        "&& rm -rf /var/lib/apt/lists/*\n"
        f"RUN pip install --no-cache-dir {rex_pin}\n"
    )
    proc = subprocess.run(
        ["docker", "build", "-t", MANAGED_BASE_IMAGE, "-"],
        input=dockerfile,
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        raise SweAgentError(
            f"failed to build base image {MANAGED_BASE_IMAGE}:\n"
            f"{proc.stderr[-800:]}"
        )


class SweAgentAdapter(BaseModel):
    """Runs SWE-agent and returns a ``State`` describing the resulting patch.

    Exposes the same attributes the runner reads from the built-in agents
    (``docker_image``, ``docker_network_disabled``, ``context_window_tokens``,
    ``max_response_tokens``) plus ``run(state)``.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    model: LangChainModel
    api_key: str
    base_url: str | None = None
    # Runner-interface parity (SWE-agent manages its own container/env, so these
    # only affect the downstream evaluation sandbox, not the SWE-agent run).
    docker_image: str = "python:3.11-slim"
    docker_network_disabled: bool = False
    context_window_tokens: int = Field(default=256_000, gt=0)
    max_response_tokens: int = Field(default=4096, gt=0)
    setup_commands: list[str] = Field(default_factory=list)
    # SWE-agent invocation knobs.
    sweagent_bin: str = "sweagent"
    run_subcommand: str = "run"
    config_path: str | None = None  # defaults to SWE-agent's bundled default.yaml
    # litellm can't price many OpenRouter models, which makes SWE-agent's cost
    # guard abort the run; disable it and derive cost ourselves from token
    # counts via pricing.csv. Set False + per_instance_cost_limit for a cap.
    disable_cost_limit: bool = True
    per_instance_cost_limit: float = 2.0
    # Hard cap on model calls per instance (SWE-agent raises and ends the run
    # when exceeded). With the cost limit disabled this is the safety net against
    # a weak model degenerating into a loop (a legit solve here finishes in
    # ~20-25 calls); 0 would mean unlimited, so keep it positive.
    call_limit: int = Field(default=75, ge=1)
    # ACI parsing mode. None => SWE-agent's default (function_calling), which
    # works for tool-capable models via OpenRouter (gpt-4o(-mini), sonnet, ...)
    # despite litellm's "no function calling" warning and yields clean tool
    # calls. Set "thought_action" only for models that truly lack tool-calling
    # (e.g. glm) — but they tend to mangle it (emit raw code, which SWE-ReX's
    # bashlex check rejects), so prefer a tool-capable model for swe-agent.
    parse_function: str | None = None
    # Fast path: run on a prebuilt base image that already contains swe-rex, so
    # `python_standalone_dir=""` skips SWE-ReX's slow (minutes) standalone-python
    # build. Repo deps are installed via SWE-agent's own post_startup_commands
    # (the same commands single-agent uses -> fair; nothing task-specific baked
    # into the image, so no answer leak). env_image=None keeps the slow default.
    env_image: str | None = None
    # Filled by the benchmark runner before an actual run. The overlay uses
    # the immutable ID while provenance retains the human-readable reference.
    env_image_reference: str | None = None
    env_image_identity: dict[str, Any] | None = None
    container_label_value: str | None = None
    post_startup_commands: list[str] = Field(default_factory=list)
    extra_args: list[str] = Field(default_factory=list)
    wall_timeout_seconds: int = Field(default=7200, gt=0)

    @field_validator("extra_args")
    @classmethod
    def _extra_args_cannot_override_policy(cls, values: list[str]) -> list[str]:
        """Keep the fingerprinted invocation contract equal to the command.

        Extra release-compatibility flags remain possible, but model, image,
        repository, task, output, parser, and budget controls have dedicated
        typed fields and cannot be replaced by a later command-line argument.
        """
        if values:
            raise ValueError(
                "arbitrary sweagent extra arguments are not accepted by the "
                "comparable adapter; add a typed, fingerprinted field instead"
            )
        return values

    @trace_agent_run
    def run(self, state: State) -> State:
        self._ensure_available()
        self._ensure_base_image()
        workdir = Path(state.workdir)
        # Persist SWE-agent's own trajectory next to the run so it can be
        # inspected with `sweagent inspector <dir>` (LangSmith only sees this
        # adapter's outer span, not SWE-agent's internal steps).
        artifact_root = workdir.parent / "swe_agent"
        artifact_root.mkdir(parents=True, exist_ok=True)
        # A multi-swe repair episode must never discover the previous
        # episode's patch or trajectory. State.step is monotonic in that
        # architecture; an accidental duplicate fails instead of reusing data.
        out_dir = artifact_root / f"episode-{state.step:04d}"
        try:
            out_dir.mkdir()
        except FileExistsError as exc:
            raise SweAgentError(
                f"refusing to reuse SWE-agent episode directory: {out_dir}"
            ) from exc

        problem_path = out_dir / "problem.md"
        problem_path.write_text(state.task, encoding="utf-8")

        proc = self._invoke(workdir, problem_path, out_dir)
        patch = self._find_patch(out_dir)
        changed: list[str] = []
        if patch:
            self._apply_patch(workdir, patch)
            changed = self._changed_files(workdir)

        usage = self._read_trajectory(out_dir)
        self._record_usage(usage)
        if not usage.accounting_complete:
            raise SweAgentError(
                "SWE-agent completed without a valid token-usage trajectory; "
                "cost and call accounting are unknown, so this run cannot be "
                "recorded as a comparable benchmark result"
            )
        return self._to_state(state, changed, usage.steps, proc)

    # -- SWE-agent invocation -------------------------------------------- #

    def _ensure_available(self) -> None:
        if self._resolved_binary() is None:
            raise SweAgentError(
                f"'{self.sweagent_bin}' not found on PATH. Install it with "
                "`pip install sweagent` (and ensure Docker is running) to use "
                "--agent swe-agent."
            )

    def _resolved_binary(self) -> str | None:
        """Resolve the snapshotted entrypoint instead of trusting PATH at exec."""
        return shutil.which(self.sweagent_bin)

    def _ensure_base_image(self) -> None:
        """Build the managed base image (python + git + swe-rex) once if missing.

        Only auto-builds the tag this adapter manages; a user-supplied
        ``env_image`` is assumed to exist. The image carries swe-rex so
        ``python_standalone_dir=""`` can skip the slow standalone build; it has
        no repo/task contents, so it never leaks a solution.
        """
        if self.env_image != MANAGED_BASE_IMAGE:
            return
        ensure_managed_base_image()

    def _overlay_yaml(self) -> str:
        """SWE-agent config overlay for the fast path (merged after default)."""
        lines = [
            "env:",
            "  deployment:",
            f"    image: {self.env_image}",
            '    python_standalone_dir: ""',
        ]
        if self.container_label_value:
            lines.extend(
                [
                    "    docker_args:",
                    "      - --label="
                    f"{SWE_CONTAINER_LABEL}={self.container_label_value}",
                ]
            )
        if self.post_startup_commands:
            lines.append("  post_startup_commands:")
            lines += [f"    - {cmd}" for cmd in self.post_startup_commands]
        return "\n".join(lines) + "\n"

    def _default_config(self) -> str | None:
        if self.config_path:
            return self.config_path
        try:
            from sweagent import CONFIG_DIR  # type: ignore[attr-defined]

            return str(Path(CONFIG_DIR) / "default.yaml")
        except Exception:
            return None

    def _command(self, workdir: Path, problem_path: Path, out_dir: Path) -> list[str]:
        """SWE-agent 1.x invocation (validated against 1.1.0). Flags are
        release-sensitive and therefore fixed by typed policy. The
        ``--problem_statement.path`` / ``--env.repo.path`` flags select the
        file and local-repo modes; no explicit ``type`` is needed."""
        cmd = [self.sweagent_bin, self.run_subcommand]
        config = self._default_config()
        if config:
            cmd.append(f"--config={config}")
        # Fast path: an overlay config (merged after the default) selects the
        # prebuilt image, skips the standalone-python build, and installs the
        # repo's deps via SWE-agent's own post_startup_commands.
        if self.env_image:
            overlay = out_dir / "fast_overlay.yaml"
            overlay.write_text(self._overlay_yaml(), encoding="utf-8")
            cmd.append(f"--config={overlay}")
        if self.disable_cost_limit:
            cost_flags = [
                "--agent.model.per_instance_cost_limit=0",
                "--agent.model.total_cost_limit=0",
            ]
        else:
            cost_flags = [
                f"--agent.model.per_instance_cost_limit={self.per_instance_cost_limit}"
            ]
        # Always bound calls per instance so a looping model can't run away.
        cost_flags.append(f"--agent.model.per_instance_call_limit={self.call_limit}")
        parse_flags = (
            [f"--agent.tools.parse_function.type={self.parse_function}"]
            if self.parse_function else []
        )
        cmd += [
            f"--agent.model.name=openrouter/{self.model.model}",
            *cost_flags,
            *parse_flags,
            f"--env.repo.path={workdir}",
            f"--problem_statement.path={problem_path}",
            f"--output_dir={out_dir}",
            *self.extra_args,
        ]
        return cmd

    def _invoke(
        self, workdir: Path, problem_path: Path, out_dir: Path
    ) -> subprocess.CompletedProcess[str]:
        env = self._subprocess_environment()
        # Stream SWE-agent's output to a log in the workspace (not captured) so a
        # long run is observable live (`tail -f .../swe_agent/sweagent.log`)
        # instead of looking hung.
        log_path = out_dir / "sweagent.log"
        command = self._command(workdir, problem_path, out_dir)
        resolved_binary = self._resolved_binary()
        if resolved_binary is None:
            raise SweAgentError(f"'{self.sweagent_bin}' disappeared before launch")
        command[0] = resolved_binary
        with log_path.open("w", encoding="utf-8") as log:
            process: subprocess.Popen[str] = subprocess.Popen(
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=env,
                cwd=out_dir,
                start_new_session=True,
                text=True,
            )
            try:
                returncode = process.wait(timeout=self.wall_timeout_seconds)
            except subprocess.TimeoutExpired as exc:
                self._terminate_process_group(process)
                self._cleanup_execution_containers()
                raise SweAgentError(
                    "SWE-agent exceeded the fingerprinted wall-clock limit "
                    f"of {self.wall_timeout_seconds} seconds"
                ) from exc
            except BaseException:
                self._terminate_process_group(process)
                self._cleanup_execution_containers()
                raise
        self._cleanup_execution_containers()
        proc: subprocess.CompletedProcess[str] = subprocess.CompletedProcess(
            command, returncode
        )
        if proc.returncode != 0 and self._find_patch(out_dir) is None:
            tail = log_path.read_text(encoding="utf-8", errors="replace")[-1500:]
            raise SweAgentError(
                f"SWE-agent run failed (exit {proc.returncode}):\n{tail}"
            )
        return proc

    def _subprocess_environment(self) -> dict[str, str]:
        """Return a provider environment without host-side code injection.

        SWE-agent remains an explicitly online ablation, but its Python import
        graph and LiteLLM behavior must come from the fingerprinted installation,
        not inherited ``PYTHONPATH``, dotenv, callbacks, or loader overrides.
        """
        env = os.environ.copy()
        for key in list(env):
            if key in _UNTRUSTED_HOST_ENV_KEYS or key.startswith(
                _UNTRUSTED_HOST_ENV_PREFIXES
            ):
                env.pop(key, None)
        env["OPENROUTER_API_KEY"] = self.api_key
        env.update(
            {
                "PYTHONUNBUFFERED": "1",
                "PYTHONNOUSERSITE": "1",
                "PYTHONSAFEPATH": "1",
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONHASHSEED": "0",
                "PYTHON_DOTENV_DISABLED": "1",
            }
        )
        if self.base_url:
            env["OPENROUTER_API_BASE"] = self.base_url
        else:
            env.pop("OPENROUTER_API_BASE", None)
        return env

    @staticmethod
    def _terminate_process_group(process: subprocess.Popen[str]) -> None:
        """Bounded TERM→KILL cleanup for the external process tree."""
        if process.poll() is not None:
            return
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except (OSError, ProcessLookupError):
            process.terminate()
        try:
            process.wait(timeout=10)
            return
        except subprocess.TimeoutExpired:
            pass
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except (OSError, ProcessLookupError):
            process.kill()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired as exc:
            raise SweAgentError(
                "could not terminate timed-out SWE-agent process group"
            ) from exc

    def _cleanup_execution_containers(self) -> None:
        """Remove only SWE-ReX containers carrying this run's unique label."""
        if not self.container_label_value:
            return
        label = f"{SWE_CONTAINER_LABEL}={self.container_label_value}"
        listed = subprocess.run(
            ["docker", "ps", "-aq", "--filter", f"label={label}"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        if listed.returncode != 0:
            raise SweAgentError(
                "failed to inventory SWE-ReX containers during cleanup:\n"
                + listed.stderr[-800:]
            )
        container_ids = [item for item in listed.stdout.splitlines() if item]
        if not container_ids:
            return
        removed = subprocess.run(
            ["docker", "rm", "-f", *container_ids],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        if removed.returncode != 0:
            raise SweAgentError(
                "failed to remove run-scoped SWE-ReX containers:\n"
                + removed.stderr[-800:]
            )

    # -- Patch discovery / application ----------------------------------- #

    def _find_patch(self, out_dir: Path) -> str | None:
        """Return the model's unified diff, from a ``*.patch`` file or the
        ``model_patch`` field of a predictions JSON."""
        for patch_file in sorted(out_dir.rglob("*.patch")):
            text = patch_file.read_text(encoding="utf-8", errors="replace").strip()
            if text:
                return _strip_artifact_hunks(text)
        prediction_files = sorted(out_dir.rglob("*.pred")) + sorted(
            out_dir.rglob("*preds*.json")
        )
        for pred_file in prediction_files:
            try:
                data = json.loads(pred_file.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            patch = _extract_model_patch(data)
            if patch:
                return _strip_artifact_hunks(patch)
        return None

    def _apply_patch(self, workdir: Path, patch: str) -> None:
        patch_file = workdir / ".sweagent.patch"
        patch_file.write_text(
            patch if patch.endswith("\n") else patch + "\n",
            encoding="utf-8",
        )
        try:
            for extra in ([], ["--3way"]):
                proc = subprocess.run(
                    safe_git_command(workdir, "apply", *extra, str(patch_file)),
                    capture_output=True,
                    text=True,
                    check=False,
                    env=safe_git_env(),
                )
                if proc.returncode == 0:
                    return
            raise SweAgentError(
                f"git apply failed for SWE-agent patch:\n{proc.stderr[-800:]}"
            )
        finally:
            patch_file.unlink(missing_ok=True)

    def _changed_files(self, workdir: Path) -> list[str]:
        changed = subprocess.run(
            safe_git_command(
                workdir,
                "diff",
                "--name-only",
                "--no-ext-diff",
                "HEAD",
                "--",
                ".",
             # ``glob`` magic so ``**/`` also matches root-level artifacts.
             ":(exclude,glob)**/__pycache__/**", ":(exclude,glob)**/*.pyc",
             ":(exclude,glob)**/*.egg-info/**", ":(exclude,glob)**/*.egg-info",
             ":(exclude,glob)**/.pytest_cache/**", ":(exclude,glob)**/build/**",
                ":(exclude,glob)**/.eggs/**",
            ),
            capture_output=True,
            text=True,
            check=False,
            env=safe_git_env(),
        )
        untracked = subprocess.run(
            safe_git_command(
                workdir,
                "ls-files",
                "--others",
                "--exclude-standard",
            ),
            capture_output=True,
            text=True,
            check=False,
            env=safe_git_env(),
        )
        if changed.returncode != 0 or untracked.returncode != 0:
            raise SweAgentError("failed to inventory the applied SWE-agent patch")
        return sorted(
            {
                line
                for output in (changed.stdout, untracked.stdout)
                for line in output.splitlines()
                if line.strip()
                and not any(
                    part in {"__pycache__", ".pytest_cache", "build", ".eggs"}
                    or part.endswith((".pyc", ".egg-info"))
                    for part in Path(line).parts
                )
            }
        )

    # -- Trajectory -> metrics ------------------------------------------- #

    def _read_trajectory(self, out_dir: Path) -> SweAgentUsage:
        """Read one complete usage record; malformed/partial data is unknown."""
        trajectories = sorted(out_dir.rglob("*.traj")) + sorted(
            out_dir.rglob("*.traj.json")
        )
        for traj in trajectories:
            try:
                data = json.loads(traj.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            if not isinstance(data, dict):
                continue
            history = data.get("trajectory") or data.get("history") or []
            if not isinstance(history, list):
                continue
            steps = len(history)
            info = data.get("info", {})
            if not isinstance(info, dict):
                continue
            raw_usage = info.get("model_stats") or data.get("model_stats")
            if not isinstance(raw_usage, dict):
                continue
            raw_calls = raw_usage.get("api_calls", raw_usage.get("calls"))
            raw_input = raw_usage.get(
                "tokens_sent", raw_usage.get("input_tokens")
            )
            raw_output = raw_usage.get(
                "tokens_received", raw_usage.get("output_tokens")
            )
            if raw_calls is None or raw_input is None or raw_output is None:
                continue
            try:
                calls = int(raw_calls)
                in_tok = int(raw_input)
                out_tok = int(raw_output)
            except (TypeError, ValueError):
                continue
            if calls < 1 or in_tok < 0 or out_tok < 0 or in_tok + out_tok < 1:
                continue
            return SweAgentUsage(
                steps=steps,
                calls=calls,
                input_tokens=in_tok,
                output_tokens=out_tok,
                accounting_complete=True,
            )
        return SweAgentUsage()

    def _record_usage(self, usage: SweAgentUsage) -> None:
        if not usage.accounting_complete:
            # The process did invoke an external model boundary, even though
            # its exact call count is unavailable. One sentinel call prevents
            # the adapter from looking like an unused $0 role in diagnostics.
            self.model.calls += 1
            self.model.usage_accounting_complete = False
            return
        self.model.calls += usage.calls
        self.model.input_tokens += usage.input_tokens
        self.model.output_tokens += usage.output_tokens

    def _to_state(
        self,
        state: State,
        changed: list[str],
        steps: int,
        proc: subprocess.CompletedProcess[str],
    ) -> State:
        if not changed:
            return state.mark_handoff("SWE-agent produced no applicable changes.")
        remaining = max(state.max_steps - state.step, 0)
        for _ in range(min(steps, remaining)):
            state = state.advance_step()
        summary = f"SWE-agent produced a patch touching {len(changed)} file(s)."
        return (
            state.with_changed_files([Path(path) for path in changed])
            # A produced patch is a submission, not proof of correctness.
            # The benchmark evaluator decides task_success afterwards.
            .with_status("submitted")
            .with_observation(summary)
            .with_context(kind="finish", text=summary)
        )


# Build artifacts that `pip install -e .` (run as post_startup) creates in the
# SWE-agent container and that leak into its submit diff; they don't exist in the
# clean eval workspace, so their hunks break `git apply`.
_ARTIFACT_PATH_RE = re.compile(
    r"\.egg-info(/|$)|(^|/)__pycache__/|\.pyc$|(^|/)(build|\.eggs)/"
)


def _strip_artifact_hunks(patch: str) -> str:
    """Drop per-file diff sections whose target path is a build artifact,
    keeping the real source hunks so ``git apply`` succeeds on a clean repo."""
    sections = re.split(r"(?m)^(?=diff --git )", patch)
    kept = [
        sec
        for sec in sections
        if sec.strip()
        and not (
            (m := re.match(r"diff --git a/\S+ b/(\S+)", sec))
            and _ARTIFACT_PATH_RE.search(m.group(1))
        )
    ]
    result = "".join(kept)
    return result + "\n" if result and not result.endswith("\n") else result


def _extract_model_patch(data: object) -> str | None:
    """Pull a ``model_patch`` out of the varied predictions JSON shapes."""
    if isinstance(data, dict):
        if isinstance(data.get("model_patch"), str) and data["model_patch"].strip():
            return data["model_patch"]
        for value in data.values():
            found = _extract_model_patch(value)
            if found:
                return found
    elif isinstance(data, list):
        for item in data:
            found = _extract_model_patch(item)
            if found:
                return found
    return None
