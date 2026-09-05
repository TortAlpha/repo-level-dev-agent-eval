"""Run SWE-agent as a drop-in alternative agent.

The benchmark scores a *patch*, not an agent, so any agent that turns
``repo@base + task.md`` into edits on the workspace can be evaluated by the same
downstream pipeline (visible/hidden tests, regressions, quality, cost). This
adapter shells out to `SWE-agent <https://swe-agent.com>`_, applies the patch it
produces onto the prepared workspace, and maps its trajectory onto a ``State``
so the runner and metrics work unchanged.

Requirements to actually run it: ``pip install sweagent`` and a working Docker.
The exact ``sweagent run`` flags vary across SWE-agent releases — override them
with ``extra_args`` / ``run_subcommand`` if your installed version differs.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from .model import LangChainModel
from .state import State
from .tracing import trace_agent_run


class SweAgentError(RuntimeError):
    """Raised when SWE-agent is missing or its run fails."""


# The base image the fast path auto-builds/uses: python + git + swe-rex, so the
# standalone-python build can be skipped. Contains no repo or task specifics.
MANAGED_BASE_IMAGE = "swe-adapter-base:py311"


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
    post_startup_commands: list[str] = Field(default_factory=list)
    extra_args: list[str] = Field(default_factory=list)

    @trace_agent_run
    def run(self, state: State) -> State:
        self._ensure_available()
        self._ensure_base_image()
        workdir = Path(state.workdir)
        # Persist SWE-agent's own trajectory next to the run so it can be
        # inspected with `sweagent inspector <dir>` (LangSmith only sees this
        # adapter's outer span, not SWE-agent's internal steps).
        out_dir = workdir.parent / "swe_agent"
        out_dir.mkdir(parents=True, exist_ok=True)

        problem_path = out_dir / "problem.md"
        problem_path.write_text(state.task, encoding="utf-8")

        proc = self._invoke(workdir, problem_path, out_dir)
        patch = self._find_patch(out_dir)
        changed: list[str] = []
        if patch:
            self._apply_patch(workdir, patch)
            changed = self._changed_files(workdir)

        steps, calls, in_tok, out_tok = self._read_trajectory(out_dir)
        self._record_usage(calls, in_tok, out_tok)
        return self._to_state(state, changed, steps, proc)

    # -- SWE-agent invocation -------------------------------------------- #

    def _ensure_available(self) -> None:
        if shutil.which(self.sweagent_bin) is None:
            raise SweAgentError(
                f"'{self.sweagent_bin}' not found on PATH. Install it with "
                "`pip install sweagent` (and ensure Docker is running) to use "
                "--agent swe-agent."
            )

    def _ensure_base_image(self) -> None:
        """Build the managed base image (python + git + swe-rex) once if missing.

        Only auto-builds the tag this adapter manages; a user-supplied
        ``env_image`` is assumed to exist. The image carries swe-rex so
        ``python_standalone_dir=""`` can skip the slow standalone build; it has
        no repo/task contents, so it never leaks a solution.
        """
        if self.env_image != MANAGED_BASE_IMAGE:
            return
        if subprocess.run(
            ["docker", "image", "inspect", self.env_image],
            capture_output=True, check=False,
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
            ["docker", "build", "-t", self.env_image, "-"],
            input=dockerfile, text=True, capture_output=True, check=False,
        )
        if proc.returncode != 0:
            raise SweAgentError(
                f"failed to build base image {self.env_image}:\n{proc.stderr[-800:]}"
            )

    def _overlay_yaml(self) -> str:
        """SWE-agent config overlay for the fast path (merged after default)."""
        lines = [
            "env:",
            "  deployment:",
            f"    image: {self.env_image}",
            '    python_standalone_dir: ""',
        ]
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
        release-sensitive; pass ``extra_args`` to override for your version.
        ``--problem_statement.path`` / ``--env.repo.path`` select the file and
        local-repo modes; no explicit ``type`` needed."""
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
        env = os.environ.copy()
        env["OPENROUTER_API_KEY"] = self.api_key
        env["PYTHONUNBUFFERED"] = "1"  # line-flush so the streamed log is live
        if self.base_url:
            env.setdefault("OPENROUTER_API_BASE", self.base_url)
        # Stream SWE-agent's output to a log in the workspace (not captured) so a
        # long run is observable live (`tail -f .../swe_agent/sweagent.log`)
        # instead of looking hung.
        log_path = out_dir / "sweagent.log"
        with log_path.open("w", encoding="utf-8") as log:
            proc = subprocess.run(
                self._command(workdir, problem_path, out_dir),
                stdout=log,
                stderr=subprocess.STDOUT,
                env=env,
                check=False,
            )
        if proc.returncode != 0 and self._find_patch(out_dir) is None:
            tail = log_path.read_text(encoding="utf-8", errors="replace")[-1500:]
            raise SweAgentError(f"SWE-agent run failed (exit {proc.returncode}):\n{tail}")
        return proc

    # -- Patch discovery / application ----------------------------------- #

    def _find_patch(self, out_dir: Path) -> str | None:
        """Return the model's unified diff, from a ``*.patch`` file or the
        ``model_patch`` field of a predictions JSON."""
        for patch_file in sorted(out_dir.rglob("*.patch")):
            text = patch_file.read_text(encoding="utf-8", errors="replace").strip()
            if text:
                return _strip_artifact_hunks(text)
        for pred_file in sorted(out_dir.rglob("*.pred")) + sorted(out_dir.rglob("*preds*.json")):
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
        patch_file.write_text(patch if patch.endswith("\n") else patch + "\n", encoding="utf-8")
        try:
            for extra in ([], ["--3way"]):
                proc = subprocess.run(
                    ["git", "-C", str(workdir), "apply", *extra, str(patch_file)],
                    capture_output=True, text=True, check=False,
                )
                if proc.returncode == 0:
                    return
            raise SweAgentError(f"git apply failed for SWE-agent patch:\n{proc.stderr[-800:]}")
        finally:
            patch_file.unlink(missing_ok=True)

    def _changed_files(self, workdir: Path) -> list[str]:
        subprocess.run(["git", "-C", str(workdir), "add", "-A"], capture_output=True, check=False)
        proc = subprocess.run(
            ["git", "-C", str(workdir), "diff", "--cached", "--name-only", "--", ".",
             # ``glob`` magic so ``**/`` also matches root-level artifacts.
             ":(exclude,glob)**/__pycache__/**", ":(exclude,glob)**/*.pyc",
             ":(exclude,glob)**/*.egg-info/**", ":(exclude,glob)**/*.egg-info",
             ":(exclude,glob)**/.pytest_cache/**", ":(exclude,glob)**/build/**",
             ":(exclude,glob)**/.eggs/**"],
            capture_output=True, text=True, check=False,
        )
        return [line for line in proc.stdout.splitlines() if line.strip()]

    # -- Trajectory -> metrics ------------------------------------------- #

    def _read_trajectory(self, out_dir: Path) -> tuple[int, int, int, int]:
        """Best-effort (steps, calls, input_tokens, output_tokens) from the
        SWE-agent trajectory; zeros when it can't be parsed."""
        for traj in sorted(out_dir.rglob("*.traj")) + sorted(out_dir.rglob("*.traj.json")):
            try:
                data = json.loads(traj.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            history = data.get("trajectory") or data.get("history") or []
            steps = len(history)
            info = data.get("info", {})
            usage = info.get("model_stats") or data.get("model_stats") or {}
            calls = int(usage.get("api_calls") or usage.get("calls") or steps or 0)
            in_tok = int(usage.get("tokens_sent") or usage.get("input_tokens") or 0)
            out_tok = int(usage.get("tokens_received") or usage.get("output_tokens") or 0)
            return steps, calls, in_tok, out_tok
        return 0, 0, 0, 0

    def _record_usage(self, calls: int, in_tok: int, out_tok: int) -> None:
        self.model.calls += calls
        self.model.input_tokens += in_tok
        self.model.output_tokens += out_tok

    def _to_state(
        self, state: State, changed: list[str], steps: int, proc: subprocess.CompletedProcess[str]
    ) -> State:
        if not changed:
            return state.mark_handoff("SWE-agent produced no applicable changes.")
        remaining = max(state.max_steps - state.step, 0)
        for _ in range(min(steps, remaining)):
            state = state.advance_step()
        summary = f"SWE-agent produced a patch touching {len(changed)} file(s)."
        return (
            state.with_changed_files(changed)
            # A produced patch is a submission, not proof of correctness.
            # The benchmark evaluator decides task_success afterwards.
            .with_status("submitted")
            .with_observation(summary)
            .with_context(kind="finish", text=summary)
        )


# Build artifacts that `pip install -e .` (run as post_startup) creates in the
# SWE-agent container and that leak into its submit diff; they don't exist in the
# clean eval workspace, so their hunks break `git apply`.
_ARTIFACT_PATH_RE = re.compile(r"\.egg-info(/|$)|(^|/)__pycache__/|\.pyc$|(^|/)(build|\.eggs)/")


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
