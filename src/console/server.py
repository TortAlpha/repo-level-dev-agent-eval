from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

from ..metrics.compute import compute_metrics, group_by
from ..metrics.difficulty import attach_difficulty, empirical_difficulty
from ..metrics.pricing import estimate_cost
from ..metrics.records import (
    DEFAULT_COLLECTION_PATH,
    DEFAULT_QUALITY_PATH,
    DEFAULT_RUNS_PATH,
    attach_hidden_suite_fallbacks,
    RunRecord,
    attach_quality,
    attach_sizes,
    attach_task_types,
    load_quality,
    load_runs,
    load_hidden_suite_specs,
    load_sizes,
    load_task_types,
)

ROOT = Path(__file__).resolve().parents[2]
WEB_DIST = ROOT / "web-console" / "dist"
JOBS_DIR = ROOT / "experiments" / "results" / "jobs"
MAX_BODY_BYTES = 64_000
MAX_LOG_TAIL_CHARS = 20_000
MAX_PATCH_CHARS = 200_000
PROVIDERS = {"local", "openrouter"}
AGENTS = ("single", "swe-agent", "multi")
ACTION_TRANSPORTS = ("text_json", "tools", "auto")
# Suggested models for the launcher (free text still allowed).
MODEL_PRESETS = (
    "openai/gpt-4o-mini",
    "openai/gpt-4o",
    "anthropic/claude-sonnet-4.5",
    "deepseek/deepseek-chat",
    "qwen/qwen3-coder",
    "z-ai/glm-5.2",
)


def _split_csv(value: str) -> list[str]:
    return [item.strip() for item in (value or "").split(",") if item.strip()]


def _terminate_tree(process: subprocess.Popen) -> None:
    """Stop a job and its children — a sweep spawns runner subprocesses (which in
    turn start Docker containers), so signalling only the top process would
    orphan the active run. Relies on ``start_new_session=True`` at launch: signal
    the whole process group, escalating SIGTERM -> SIGKILL. (SWE-ReX containers
    are owned by the Docker daemon, not the process tree, so a mid-run stop can
    still leave one container to clean up separately.)"""
    try:
        pgid = os.getpgid(process.pid)
    except (ProcessLookupError, OSError):
        pgid = None
    for sig in (signal.SIGTERM, signal.SIGKILL):
        if process.poll() is not None:
            return
        try:
            if pgid is not None:
                os.killpg(pgid, sig)
            else:
                process.send_signal(sig)
        except (ProcessLookupError, OSError):
            return
        try:
            process.wait(timeout=8)
            return
        except subprocess.TimeoutExpired:
            continue


def main() -> int:
    args = parse_args()
    server = ConsoleServer(
        (args.host, args.port),
        ConsoleHandler,
        root=ROOT,
        static_dir=args.static_dir,
    )
    print(f"Console API listening on http://{args.host}:{args.port}")
    print(f"Static assets: {server.static_dir}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping console API.")
    finally:
        server.server_close()
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the local benchmark console API.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--static-dir", type=Path, default=WEB_DIST)
    return parser.parse_args()


class ConsoleServer(ThreadingHTTPServer):
    def __init__(
        self,
        server_address: tuple[str, int],
        handler_class: type[BaseHTTPRequestHandler],
        *,
        root: Path,
        static_dir: Path,
    ) -> None:
        super().__init__(server_address, handler_class)
        self.root = root
        self.static_dir = static_dir
        self.jobs = JobStore(root)


class ConsoleHandler(BaseHTTPRequestHandler):
    server: ConsoleServer

    def do_OPTIONS(self) -> None:  # noqa: N802 - stdlib handler API
        self._send_empty(HTTPStatus.NO_CONTENT)

    def do_HEAD(self) -> None:  # noqa: N802 - stdlib handler API
        parsed = urlparse(self.path)
        if parsed.path.startswith("/api/"):
            self._send_empty(HTTPStatus.OK)
            return
        try:
            self._serve_static(parsed.path, write_body=False)
        except ApiError as exc:
            self._send_empty(exc.status)

    def do_GET(self) -> None:  # noqa: N802 - stdlib handler API
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"

        try:
            if path == "/api/health":
                self._send_json({"ok": True, "generated_at": utc_now()})
            elif path == "/api/overview":
                self._send_json(build_overview(self.server.root))
            elif path == "/api/runs":
                self._send_json({"runs": load_console_runs(self.server.root)})
            elif path.startswith("/api/runs/"):
                run_id = unquote(path.removeprefix("/api/runs/"))
                self._send_json(load_run_detail(self.server.root, run_id))
            elif path == "/api/tasks":
                self._send_json({"tasks": load_tasks(self.server.root)})
            elif path == "/api/meta":
                self._send_json(build_meta(self.server.root))
            elif path == "/api/jobs":
                self._send_json({"jobs": self.server.jobs.list_jobs()})
            elif path.startswith("/api/jobs/"):
                job_id = unquote(path.removeprefix("/api/jobs/"))
                self._send_json(self.server.jobs.get_job(job_id))
            elif path.startswith("/api/"):
                self._send_json({"error": "unknown endpoint"}, HTTPStatus.NOT_FOUND)
            else:
                self._serve_static(parsed.path)
        except ApiError as exc:
            self._send_json({"error": exc.message}, exc.status)
        except Exception as exc:  # noqa: BLE001 - keep the local API alive
            self._send_json(
                {"error": f"{type(exc).__name__}: {exc}"},
                HTTPStatus.INTERNAL_SERVER_ERROR,
            )

    def do_POST(self) -> None:  # noqa: N802 - stdlib handler API
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"
        try:
            if path == "/api/jobs":
                payload = self._read_json()
                self._send_json(
                    self.server.jobs.start_job(payload), HTTPStatus.CREATED
                )
            elif path == "/api/sweeps":
                payload = self._read_json()
                self._send_json(
                    self.server.jobs.start_sweep(payload), HTTPStatus.CREATED
                )
            elif path.startswith("/api/jobs/") and path.endswith("/stop"):
                job_id = unquote(path.removeprefix("/api/jobs/").removesuffix("/stop"))
                self._send_json(self.server.jobs.stop_job(job_id))
            else:
                self._send_json({"error": "unknown endpoint"}, HTTPStatus.NOT_FOUND)
        except ApiError as exc:
            self._send_json({"error": exc.message}, exc.status)
        except Exception as exc:  # noqa: BLE001
            self._send_json(
                {"error": f"{type(exc).__name__}: {exc}"},
                HTTPStatus.INTERNAL_SERVER_ERROR,
            )

    def log_message(self, fmt: str, *args: object) -> None:
        sys.stderr.write(f"[console] {self.address_string()} - {fmt % args}\n")

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length") or "0")
        if length > MAX_BODY_BYTES:
            raise ApiError("request body is too large", HTTPStatus.REQUEST_ENTITY_TOO_LARGE)
        body = self.rfile.read(length).decode("utf-8") if length else "{}"
        try:
            data = json.loads(body)
        except json.JSONDecodeError as exc:
            raise ApiError(f"invalid JSON: {exc}", HTTPStatus.BAD_REQUEST) from exc
        if not isinstance(data, dict):
            raise ApiError("JSON body must be an object", HTTPStatus.BAD_REQUEST)
        return data

    def _send_empty(self, status: HTTPStatus) -> None:
        self.send_response(status)
        self._cors()
        self.end_headers()

    def _send_json(
        self,
        payload: object,
        status: HTTPStatus = HTTPStatus.OK,
    ) -> None:
        raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self._cors()
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _serve_static(self, raw_path: str, *, write_body: bool = True) -> None:
        static_dir = self.server.static_dir
        if not static_dir.is_dir():
            self._send_json(
                {
                    "error": (
                        "frontend build not found; run `npm install` and "
                        "`npm run build` in web-console, or use the Vite dev server"
                    )
                },
                HTTPStatus.NOT_FOUND,
            )
            return

        rel = unquote(raw_path).lstrip("/") or "index.html"
        candidate = (static_dir / rel).resolve()
        if not candidate.is_relative_to(static_dir.resolve()) or not candidate.is_file():
            candidate = static_dir / "index.html"

        content = candidate.read_bytes()
        self.send_response(HTTPStatus.OK)
        self._cors()
        self.send_header("Content-Type", content_type(candidate))
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        if write_body:
            self.wfile.write(content)

    def _cors(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")


class ApiError(Exception):
    def __init__(
        self,
        message: str,
        status: HTTPStatus = HTTPStatus.BAD_REQUEST,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.status = status


class JobStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.jobs_dir = JOBS_DIR
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._processes: dict[str, subprocess.Popen[str]] = {}

    def list_jobs(self) -> list[dict]:
        with self._lock:
            jobs = [self._refresh(self._read_job(path)) for path in self._job_files()]
        return sorted(jobs, key=lambda item: item.get("started_at", ""), reverse=True)

    def get_job(self, job_id: str) -> dict:
        with self._lock:
            job = self._refresh(self._read_job(self._job_path(job_id)))
        job["log_tail"] = tail(Path(job["log_path"]), MAX_LOG_TAIL_CHARS)
        return job

    def start_job(self, payload: dict) -> dict:
        command = self._command_from_payload(payload)
        return self._spawn(command, {
            "kind": "run",
            "task_id": clean_required_str(payload.get("task_id"), "task_id"),
            "provider": payload.get("provider") or "openrouter",
            "model": clean_optional_str(payload.get("model")),
            "agent": clean_optional_str(payload.get("agent")),
            "action_transport": clean_optional_str(payload.get("action_transport")),
            "session": clean_optional_str(payload.get("session")),
            "dry_run": bool(payload.get("dry_run", False)),
        })

    def start_sweep(self, payload: dict) -> dict:
        """One job that sweeps tasks x models x agents into a session."""
        command = self._command_from_sweep(payload)
        return self._spawn(command, {
            "kind": "sweep",
            "task_id": clean_optional_str(payload.get("tasks")) or "all",
            "provider": payload.get("provider") or "openrouter",
            "model": clean_optional_str(payload.get("models")),
            "agent": clean_optional_str(payload.get("agents")) or "single",
            "action_transport": clean_optional_str(payload.get("action_transport")),
            "session": clean_optional_str(payload.get("session")),
            "dry_run": False,
        })

    def _spawn(self, command: list[str], meta: dict) -> dict:
        job_id = str(uuid.uuid4())
        log_path = self.jobs_dir / f"{job_id}.log"
        job = {
            "job_id": job_id,
            "status": "running",
            "returncode": None,
            **meta,
            "command": command,
            "pid": None,
            "log_path": str(log_path),
            "started_at": utc_now(),
            "finished_at": None,
        }
        log_handle = log_path.open("w", encoding="utf-8")
        try:
            process = subprocess.Popen(
                command, cwd=self.root, stdout=log_handle,
                stderr=subprocess.STDOUT, text=True,
                # Own session/process group so Stop can kill the whole tree
                # (a sweep spawns runner subprocesses that spawn containers).
                start_new_session=True,
            )
        except Exception:
            log_handle.close()
            raise

        job["pid"] = process.pid
        with self._lock:
            self._processes[job_id] = process
            self._write_job(job)

        thread = threading.Thread(
            target=self._wait_for_job, args=(job_id, process, log_handle), daemon=True,
        )
        thread.start()
        return self.get_job(job_id)

    def stop_job(self, job_id: str) -> dict:
        with self._lock:
            job = self._read_job(self._job_path(job_id))
            process = self._processes.get(job_id)
        if not process or process.poll() is not None:
            return self.get_job(job_id)

        _terminate_tree(process)

        with self._lock:
            job["status"] = "stopped"
            job["returncode"] = process.returncode
            job["finished_at"] = utc_now()
            self._write_job(job)
            self._processes.pop(job_id, None)
        return self.get_job(job_id)

    def _wait_for_job(
        self,
        job_id: str,
        process: subprocess.Popen[str],
        log_handle,
    ) -> None:
        try:
            returncode = process.wait()
        finally:
            log_handle.close()

        with self._lock:
            try:
                job = self._read_job(self._job_path(job_id))
            except ApiError:
                self._processes.pop(job_id, None)
                return
            if job.get("status") == "running":
                job["status"] = "completed" if returncode == 0 else "failed"
            job["returncode"] = returncode
            job["finished_at"] = utc_now()
            self._write_job(job)
            self._processes.pop(job_id, None)

    def _command_from_payload(self, payload: dict) -> list[str]:
        tasks = {task["task_id"] for task in load_tasks(self.root)}
        task_id = clean_required_str(payload.get("task_id"), "task_id")
        if task_id not in tasks:
            raise ApiError(f"unknown task_id: {task_id}", HTTPStatus.BAD_REQUEST)

        provider = clean_optional_str(payload.get("provider")) or "openrouter"
        if provider not in PROVIDERS:
            raise ApiError(f"unsupported provider: {provider}", HTTPStatus.BAD_REQUEST)

        command = [
            sys.executable,
            "-m",
            "src.benchmark.runner",
            "--task-id",
            task_id,
            "--provider",
            provider,
        ]

        model = clean_optional_str(payload.get("model"))
        if model:
            command.extend(["--model", model])

        agent = clean_optional_str(payload.get("agent"))
        if agent:
            if agent not in AGENTS:
                raise ApiError(f"unsupported agent: {agent}", HTTPStatus.BAD_REQUEST)
            command.extend(["--agent", agent])

        session = clean_optional_str(payload.get("session"))
        if session:
            command.extend(["--session", session])

        transport = clean_optional_str(payload.get("action_transport"))
        if transport:
            if transport not in ACTION_TRANSPORTS:
                raise ApiError(
                    f"unsupported action_transport: {transport}",
                    HTTPStatus.BAD_REQUEST,
                )
            command.extend(["--action-transport", transport])

        for key, flag, lower, upper in (
            ("max_steps", "--max-steps", 1, 200),
            ("max_iterations", "--max-iterations", 1, 50),
        ):
            value = payload.get(key)
            if value not in (None, ""):
                command.extend([flag, str(clean_int(value, key, lower, upper))])

        bool_flags = {
            "dry_run": "--dry-run",
            "no_score": "--no-score",
            "no_regression": "--no-regression",
            "enable_review": "--enable-review",
            "no_network": "--no-network",
            "no_setup": "--no-setup",
        }
        for key, flag in bool_flags.items():
            if bool(payload.get(key, False)):
                command.append(flag)
        return command

    def _command_from_sweep(self, payload: dict) -> list[str]:
        provider = clean_optional_str(payload.get("provider")) or "openrouter"
        if provider not in PROVIDERS:
            raise ApiError(f"unsupported provider: {provider}", HTTPStatus.BAD_REQUEST)
        agents = clean_optional_str(payload.get("agents")) or "single"
        for agent in _split_csv(agents):
            if agent not in AGENTS:
                raise ApiError(f"unsupported agent: {agent}", HTTPStatus.BAD_REQUEST)
        tasks = clean_optional_str(payload.get("tasks")) or "all"
        if tasks != "all":
            known = {task["task_id"] for task in load_tasks(self.root)}
            for task_id in _split_csv(tasks):
                if task_id not in known:
                    raise ApiError(f"unknown task_id: {task_id}", HTTPStatus.BAD_REQUEST)

        command = [
            sys.executable, "-m", "src.benchmark.sweep",
            "--provider", provider, "--tasks", tasks, "--agents", agents,
        ]
        models = clean_optional_str(payload.get("models"))
        if models:
            command.extend(["--models", models])
        session = clean_optional_str(payload.get("session"))
        if session:
            command.extend(["--session", session])
        transport = clean_optional_str(payload.get("action_transport"))
        if transport:
            if transport not in ACTION_TRANSPORTS:
                raise ApiError(
                    f"unsupported action_transport: {transport}",
                    HTTPStatus.BAD_REQUEST,
                )
            command.extend(["--action-transport", transport])
        for key, flag, lower, upper in (
            ("max_steps", "--max-steps", 1, 200),
            ("max_iterations", "--max-iterations", 1, 50),
            ("concurrency", "--concurrency", 1, 8),
        ):
            value = payload.get(key)
            if value not in (None, ""):
                command.extend([flag, str(clean_int(value, key, lower, upper))])
        for key, flag in (
            ("no_score", "--no-score"),
            ("no_regression", "--no-regression"),
            ("enable_review", "--enable-review"),
            ("no_network", "--no-network"),
            ("no_setup", "--no-setup"),
        ):
            if bool(payload.get(key, False)):
                command.append(flag)
        return command

    def _job_files(self) -> list[Path]:
        return sorted(self.jobs_dir.glob("*.json"))

    def _job_path(self, job_id: str) -> Path:
        if not job_id or "/" in job_id or "\\" in job_id:
            raise ApiError("invalid job id", HTTPStatus.BAD_REQUEST)
        return self.jobs_dir / f"{job_id}.json"

    def _read_job(self, path: Path) -> dict:
        if not path.is_file():
            raise ApiError("job not found", HTTPStatus.NOT_FOUND)
        return json.loads(path.read_text(encoding="utf-8"))

    def _write_job(self, job: dict) -> None:
        self._job_path(job["job_id"]).write_text(
            json.dumps(job, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def _refresh(self, job: dict) -> dict:
        process = self._processes.get(job["job_id"])
        if not process or job.get("status") != "running":
            return job
        returncode = process.poll()
        if returncode is None:
            return job
        job["status"] = "completed" if returncode == 0 else "failed"
        job["returncode"] = returncode
        job["finished_at"] = job.get("finished_at") or utc_now()
        self._write_job(job)
        self._processes.pop(job["job_id"], None)
        return job


def build_meta(root: Path) -> dict:
    """Launcher metadata: agents, providers, model presets, and known sessions."""
    records = load_records(root)
    sessions: dict[str, dict] = {}
    for record in records:
        name = record.session_id or "default"
        bucket = sessions.setdefault(
            name, {"session_id": name, "runs": 0, "last_finished_at": None}
        )
        bucket["runs"] += 1
        if record.finished_at and (
            bucket["last_finished_at"] is None
            or record.finished_at > bucket["last_finished_at"]
        ):
            bucket["last_finished_at"] = record.finished_at
    return {
        "generated_at": utc_now(),
        "agents": list(AGENTS),
        "action_transports": list(ACTION_TRANSPORTS),
        "providers": sorted(PROVIDERS),
        "model_presets": list(MODEL_PRESETS),
        "models_seen": sorted({r.model for r in records if r.model}),
        "sessions": sorted(
            sessions.values(), key=lambda s: s["last_finished_at"] or "", reverse=True
        ),
    }


def build_overview(root: Path) -> dict:
    records = load_records(root)
    tasks = load_tasks(root)
    quality = load_quality_rows(root)
    runs = [serialize_run(record, root, quality) for record in records]
    return {
        "generated_at": utc_now(),
        "summary": {
            "runs": len(records),
            "run_tasks": len({run.task_id for run in records}),
            "collection_tasks": len(tasks),
            "verified_tasks": sum(1 for task in tasks if task["task_status"] == "pr_task_verified"),
            "unverified_tasks": sum(1 for task in tasks if task["task_status"] != "pr_task_verified"),
            "unknown_run_tasks": sorted(
                {run.task_id for run in records}
                - {task["task_id"] for task in tasks}
            ),
            "last_finished_at": max(
                (run.finished_at for run in records if run.finished_at),
                default=None,
            ),
        },
        "sessions": serialize_sessions(records, runs),
        "metrics": {
            "overall": compute_metrics(records).as_dict() if records else None,
            "by_agent_mode": metric_groups(records, "agent_mode"),
            "by_size": metric_groups(records, "size"),
            "by_task_type": metric_groups(records, "task_type"),
            "by_difficulty": metric_groups(records, "difficulty"),
            "by_difficulty_estimate": metric_groups(records, "difficulty_estimate"),
            "by_model": metric_groups(records, "model"),
            "by_action_transport": metric_groups(records, "action_transport"),
            "by_session": metric_groups(records, "session_id"),
        },
        "task_difficulty": serialize_task_difficulty(records),
        "calls": call_summary(runs),
        "actions": aggregate_actions(records),
        "runs": runs,
        "tasks": tasks,
    }


def serialize_sessions(records: list[RunRecord], runs: list[dict]) -> list[dict]:
    run_by_id = {run["run_id"]: run for run in runs}
    rows = []
    for session_id, group in group_by(records, "session_id").items():
        session_runs = [
            run_by_id[record.run_id]
            for record in group
            if record.run_id in run_by_id
        ]
        finished = [record.finished_at for record in group if record.finished_at]
        rows.append(
            {
                "session_id": session_id,
                "runs": len(group),
                "tasks": len({record.task_id for record in group}),
                "agents": sorted({record.agent_mode for record in group if record.agent_mode}),
                "models": sorted({record.model for record in group if record.model}),
                "first_finished_at": min(finished, default=None),
                "last_finished_at": max(finished, default=None),
                "metrics": compute_metrics(group).as_dict(),
                "groups": {
                    "by_agent_mode": metric_groups(group, "agent_mode"),
                    "by_size": metric_groups(group, "size"),
                    "by_task_type": metric_groups(group, "task_type"),
                    "by_difficulty": metric_groups(group, "difficulty"),
                    "by_difficulty_estimate": metric_groups(
                        group, "difficulty_estimate"
                    ),
                    "by_model": metric_groups(group, "model"),
                    "by_action_transport": metric_groups(group, "action_transport"),
                },
                "task_difficulty": serialize_task_difficulty(group),
                "calls": call_summary(session_runs),
                "actions": aggregate_actions(group),
            }
        )
    return sorted(rows, key=lambda row: row["last_finished_at"] or "", reverse=True)


def load_run_detail(root: Path, run_id: str) -> dict:
    records = [run for run in load_records(root) if run.run_id == run_id]
    if not records:
        raise ApiError("run not found", HTTPStatus.NOT_FOUND)

    run = serialize_run(records[0], root)
    metrics_path = root / "experiments" / "results" / "metrics" / f"{run_id}.json"
    metrics = read_json(metrics_path) if metrics_path.is_file() else None
    patch = load_patch(root, records[0])
    return {"run": run, "metrics_file": metrics, "patch": patch}


def load_console_runs(root: Path) -> list[dict]:
    return serialize_runs(load_records(root), root)


def load_records(root: Path) -> list[RunRecord]:
    runs_path = root / DEFAULT_RUNS_PATH
    collection_path = root / DEFAULT_COLLECTION_PATH
    quality_path = root / DEFAULT_QUALITY_PATH
    records = load_runs(runs_path)
    attach_hidden_suite_fallbacks(records, load_hidden_suite_specs(collection_path))
    attach_sizes(records, load_sizes(collection_path))
    attach_task_types(records, load_task_types(collection_path))
    attach_difficulty(records, collection_path)
    attach_quality(records, load_quality(quality_path))
    return records


def load_tasks(root: Path) -> list[dict]:
    path = root / DEFAULT_COLLECTION_PATH
    if not path.is_file():
        return []
    static_difficulty = {}
    try:
        from ..metrics.difficulty import load_static_difficulty

        static_difficulty = load_static_difficulty(path)
    except Exception:  # noqa: BLE001 - optional enrichment
        static_difficulty = {}
    with path.open(encoding="utf-8") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    for row in rows:
        for key in (
            "source_files",
            "source_loc",
            "source_loc_nonblank",
            "patch_files",
            "patch_loc",
            "test_files",
        ):
            row[key] = to_int(row.get(key))
        row["large_by_spec"] = bool(row.get("source_loc") and row["source_loc"] > 15_000)
        row["difficulty_estimate"] = static_difficulty.get(row["task_id"])
    return rows


def serialize_runs(records: list[RunRecord], root: Path) -> list[dict]:
    quality = load_quality_rows(root)
    return [serialize_run(record, root, quality) for record in records]


def serialize_run(
    record: RunRecord,
    root: Path,
    quality_rows: dict[str, dict] | None = None,
) -> dict:
    quality_rows = quality_rows if quality_rows is not None else load_quality_rows(root)
    quality_row = quality_rows.get(record.run_id, {})
    cost = None
    if record.input_tokens is not None and record.output_tokens is not None:
        cost = estimate_cost(record.model, record.input_tokens, record.output_tokens)
    return {
        "task_id": record.task_id,
        "run_id": record.run_id,
        "session_id": record.session_id,
        "agent_mode": record.agent_mode,
        "provider": record.provider,
        "action_transport": record.action_transport,
        "model": record.model,
        "status": record.status,
        "steps": record.steps,
        "iterations": record.iterations,
        "test_passed": record.test_passed,
        "task_success": record.task_success,
        "hidden_tests_passed": record.hidden_tests_passed,
        "hidden_required_tests_passed": record.hidden_required_tests_passed,
        "hidden_semantic_tests_passed": record.hidden_semantic_tests_passed,
        "hidden_compat_tests_passed": record.hidden_compat_tests_passed,
        "hidden_pr_parity_tests_passed": record.hidden_pr_parity_tests_passed,
        "hidden_suite_results": record.hidden_suite_results,
        "changed_files": record.changed_files,
        "finished_at": record.finished_at,
        "duration_s": record.duration_s,
        "llm_calls": record.llm_calls,
        "input_tokens": record.input_tokens,
        "output_tokens": record.output_tokens,
        "total_tokens": record.total_tokens,
        "cost_usd": cost,
        "action_counts": record.action_counts,
        "regressions": record.regressions,
        "workspace": record.workspace,
        "task_type": record.task_type,
        "size": record.size,
        "difficulty": record.difficulty,
        "difficulty_estimate": record.difficulty_estimate,
        "quality_score": record.quality_score,
        "quality_rationale": quality_row.get("rationale"),
        "reviewer_model": quality_row.get("reviewer_model"),
    }


def serialize_task_difficulty(records: list[RunRecord]) -> list[dict]:
    rows = [
        {
            "task_id": item.task_id,
            "n_runs": item.n_runs,
            "n_models": item.n_models,
            "solve_rate": item.solve_rate,
            "difficulty": item.difficulty,
            "discrimination": item.discrimination,
        }
        for item in empirical_difficulty(records).values()
    ]
    return sorted(
        rows,
        key=lambda row: (
            -float(row["difficulty"]),
            -float(row["discrimination"] or 0),
            row["task_id"],
        ),
    )


def metric_groups(records: list[RunRecord], key: str) -> dict[str, dict]:
    if not records:
        return {}
    return {
        name: compute_metrics(group).as_dict()
        for name, group in group_by(records, key).items()
    }


def call_summary(runs: list[dict]) -> dict:
    cost_values = [run["cost_usd"] for run in runs if run["cost_usd"] is not None]
    return {
        "llm_calls": sum(run["llm_calls"] or 0 for run in runs),
        "input_tokens": sum(run["input_tokens"] or 0 for run in runs),
        "output_tokens": sum(run["output_tokens"] or 0 for run in runs),
        "total_tokens": sum(run["total_tokens"] or 0 for run in runs),
        "total_cost_usd": sum(cost_values) if cost_values else None,
        "runs_with_usage": sum(1 for run in runs if run["total_tokens"] is not None),
    }


def aggregate_actions(records: list[RunRecord]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        for key, value in record.action_counts.items():
            counts[key] = counts.get(key, 0) + value
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def load_quality_rows(root: Path) -> dict[str, dict]:
    path = root / DEFAULT_QUALITY_PATH
    if not path.is_file():
        return {}
    rows: dict[str, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            rows[row["run_id"]] = row
    return rows


def load_patch(root: Path, record: RunRecord) -> str:
    candidates: list[Path] = []
    if record.workspace:
        candidates.append(Path(record.workspace).parent / "agent.patch")
    candidates.append(
        root
        / "experiments"
        / "workspaces"
        / f"{record.task_id}-{record.run_id}"
        / "agent.patch"
    )
    for path in candidates:
        if path.is_file():
            text = path.read_text(encoding="utf-8", errors="replace")
            if len(text) > MAX_PATCH_CHARS:
                return text[:MAX_PATCH_CHARS] + "\n... truncated ..."
            return text
    return ""


def clean_required_str(value: object, key: str) -> str:
    text = clean_optional_str(value)
    if not text:
        raise ApiError(f"missing required field: {key}", HTTPStatus.BAD_REQUEST)
    return text


def clean_optional_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if "\n" in text or "\r" in text or "\x00" in text:
        raise ApiError("string fields must be single-line", HTTPStatus.BAD_REQUEST)
    return text


def clean_int(value: object, key: str, lower: int, upper: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ApiError(f"{key} must be an integer", HTTPStatus.BAD_REQUEST) from exc
    if not lower <= parsed <= upper:
        raise ApiError(
            f"{key} must be between {lower} and {upper}",
            HTTPStatus.BAD_REQUEST,
        )
    return parsed


def read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def tail(path: Path, max_chars: int) -> str:
    if not path.is_file():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace")
    return text[-max_chars:]


def to_int(value: object) -> int | None:
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return None


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def content_type(path: Path) -> str:
    suffix = path.suffix.lower()
    return {
        ".html": "text/html; charset=utf-8",
        ".js": "application/javascript; charset=utf-8",
        ".css": "text/css; charset=utf-8",
        ".svg": "image/svg+xml",
        ".json": "application/json; charset=utf-8",
    }.get(suffix, "application/octet-stream")


if __name__ == "__main__":
    raise SystemExit(main())
