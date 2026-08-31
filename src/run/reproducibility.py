"""Deterministic provenance for model prompts, tools, routes, and policy.

Hashes are deliberately content based: filenames and model aliases are useful
labels, but only the bytes/normalized schemas prove two runs received the same
agent kernel.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import re
import shlex
import shutil
import stat
import subprocess
import sys
import sysconfig
from collections import deque
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from packaging.markers import default_environment
from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import canonicalize_name

from ..agents.actions import ACTION_SPACES, action_tool_schemas
from ..agents.model import TOOL_MODE_PROMPT
from ..agents.prompts import CONTEXT_SUMMARY_PROMPT
from ..git_safety import safe_git_command, safe_git_env, validate_local_git_config

RUN_REPRODUCIBILITY_SCHEMA_VERSION = "run-reproducibility-v2"
PROMPT_MANIFEST_VERSION = "prompt-manifest-v1"
TOOL_SCHEMA_MANIFEST_VERSION = "tool-schema-manifest-v1"
POLICY_KERNEL_VERSION = "agent-policy-kernel-v2"
SOURCE_REPOSITORY_MANIFEST_VERSION = "source-repository-v1"
POLICY_KERNEL_FILES = (
    "src/agents/actions.py",
    "src/agents/context.py",
    "src/agents/decomposition.py",
    "src/agents/executor.py",
    "src/agents/model.py",
    "src/agents/multi_agent.py",
    "src/agents/prompts.py",
    "src/agents/roles.py",
    "src/agents/sandbox.py",
    "src/agents/single_agent.py",
    "src/agents/swe_agent.py",
    "src/agents/state/budget.py",
    "src/agents/state/core.py",
    "src/agents/state/entry.py",
    "src/agents/state/rendering.py",
    "src/agents/state/status.py",
    "src/agents/state/subtask.py",
    "src/agents/transport.py",
    "src/agents/workspace.py",
    "src/benchmark/_trusted_pytest.py",
    "src/benchmark/evaluator_bin/__init__.py",
    "src/benchmark/evaluator_bin/_python_shim.py",
    "src/benchmark/evaluator_bin/py.test",
    "src/benchmark/evaluator_bin/pytest",
    "src/benchmark/evaluator_bin/python",
    "src/benchmark/evaluator_bin/python3",
    "src/benchmark/collection.py",
    "src/benchmark/evaluation.py",
    "src/benchmark/review.py",
    "src/benchmark/runner.py",
    "src/benchmark/sweep.py",
    "src/benchmark/task_sets.py",
    "src/benchmark/test_launch.py",
    "src/benchmark/workspace.py",
    "src/config.py",
    "src/git_safety.py",
    "src/metrics/pricing.py",
    "src/metrics/compute.py",
    "src/metrics/records.py",
    "src/run/cli.py",
    "src/run/models.py",
    "src/run/reproducibility.py",
    "src/run/results.py",
    "static/pricing.csv",
)
HARNESS_SOURCE_PATHS = (
    "src/agents",
    "src/benchmark",
    "src/run",
    "src/metrics",
    "src/config.py",
    "src/git_safety.py",
    "static/prompts",
    "static/pricing.csv",
    "eval/task_sets",
    "pyproject.toml",
    "docs/final_benchmark_task_set.md",
)
RUNTIME_DISTRIBUTIONS = (
    "langchain-core",
    "langchain-openai",
    "openai",
    "packaging",
    "pydantic",
    "sweagent",
)
EXTERNAL_SWE_ROOT_DISTRIBUTIONS = ("sweagent", "swe-rex")


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def harness_source_manifest(root: Path | None = None) -> dict[str, Any]:
    """Content identity for every source/input path locked by a sweep."""
    workspace_root = (
        root.resolve() if root is not None else Path(__file__).resolve().parents[2]
    )
    files: list[tuple[str, str]] = []
    for relative in HARNESS_SOURCE_PATHS:
        candidate = workspace_root / relative
        paths = candidate.rglob("*") if candidate.is_dir() else [candidate]
        for path in paths:
            if path.is_file() and "__pycache__" not in path.parts:
                files.append(
                    (
                        path.relative_to(workspace_root).as_posix(),
                        hashlib.sha256(path.read_bytes()).hexdigest(),
                    )
                )
    tree_payload = "\n".join(
        f"{path}\0{digest}" for path, digest in sorted(files)
    )
    return {
        "tree_sha256": hashlib.sha256(tree_payload.encode()).hexdigest(),
        "file_count": len(files),
    }


def _git_bytes(repo: Path, *args: str) -> bytes:
    try:
        result = subprocess.run(
            safe_git_command(repo, *args),
            capture_output=True,
            check=False,
            env=safe_git_env(),
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError(f"cannot inspect source repository {repo}: {exc}") from exc
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(
            f"cannot inspect source repository {repo} with git {' '.join(args)}"
            + (f": {detail}" if detail else "")
        )
    return result.stdout


def _hash_untracked_files(repo: Path, paths: list[bytes]) -> str | None:
    if not paths:
        return None
    digest = hashlib.sha256()
    for relative_bytes in sorted(paths):
        relative = os.fsdecode(relative_bytes)
        candidate = repo / relative
        metadata = candidate.lstat()
        digest.update(len(relative_bytes).to_bytes(8, "big"))
        digest.update(relative_bytes)
        portable_mode = stat.S_IFMT(metadata.st_mode) | stat.S_IMODE(
            metadata.st_mode
        )
        digest.update(portable_mode.to_bytes(8, "big"))
        if stat.S_ISLNK(metadata.st_mode):
            content = os.fsencode(os.readlink(candidate))
            digest.update(len(content).to_bytes(8, "big"))
            digest.update(content)
            continue
        if not stat.S_ISREG(metadata.st_mode):
            # Git's untracked listing normally contains files/symlinks only.
            # Preserve the type deterministically if a special entry appears.
            digest.update((0).to_bytes(8, "big"))
            continue
        digest.update(metadata.st_size.to_bytes(8, "big"))
        with candidate.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


def source_repository_identity(
    repo_dir: Path,
    *,
    declared_base_commit: str | None = None,
    require_clean: bool = False,
    include_ignored_files: bool = False,
) -> dict[str, Any]:
    """Describe the repository bytes visible to the agent before a run.

    Clean benchmark inputs are locked to their declared commit and Git tree.
    The generic CLI may intentionally start from a dirty checkout, in which
    case tracked diffs, status state, and untracked file bytes are included.
    Absolute paths are deliberately excluded so moving the same checkout does
    not alter the fingerprint.
    """
    requested_repo = repo_dir.resolve()
    validate_local_git_config(requested_repo)
    repo = Path(
        _git_bytes(requested_repo, "rev-parse", "--show-toplevel")
        .decode("utf-8", errors="surrogateescape")
        .strip()
    ).resolve()
    head_commit = _git_bytes(repo, "rev-parse", "--verify", "HEAD^{commit}").decode(
        "ascii"
    ).strip()
    head_tree = _git_bytes(repo, "rev-parse", "--verify", "HEAD^{tree}").decode(
        "ascii"
    ).strip()

    declared = (declared_base_commit or "").strip() or None
    resolved_base: str | None = None
    if declared is not None:
        # Collection manifests use full object IDs. Restricting this value
        # prevents a revision beginning with '-' from being parsed as an
        # option and makes the validation contract unambiguous.
        if re.fullmatch(r"[0-9a-fA-F]{7,64}", declared) is None:
            raise ValueError(f"invalid declared base commit: {declared!r}")
        resolved_base = _git_bytes(
            repo, "rev-parse", "--verify", f"{declared}^{{commit}}"
        ).decode("ascii").strip()
        if resolved_base != head_commit:
            raise ValueError(
                "source repository HEAD mismatch: "
                f"{head_commit} != declared base {resolved_base}"
            )

    status_bytes = _git_bytes(
        repo, "status", "--porcelain=v1", "-z", "--untracked-files=all"
    )
    dirty = bool(status_bytes)
    if require_clean and dirty:
        raise ValueError(f"source repository is dirty: {repo}")

    untracked_raw = _git_bytes(
        repo, "ls-files", "--others", "--exclude-standard", "-z"
    )
    untracked = [item for item in untracked_raw.split(b"\0") if item]
    ignored: list[bytes] = []
    if include_ignored_files:
        ignored_raw = _git_bytes(
            repo,
            "ls-files",
            "--others",
            "--ignored",
            "--exclude-standard",
            "-z",
        )
        ignored = [item for item in ignored_raw.split(b"\0") if item]
    tracked_diff = (
        _git_bytes(
            repo,
            "diff",
            "--binary",
            "--full-index",
            "--no-ext-diff",
            "HEAD",
            "--",
        )
        if dirty
        else b""
    )
    tracked_diff_sha256 = (
        hashlib.sha256(tracked_diff).hexdigest() if tracked_diff else None
    )
    untracked_tree_sha256 = _hash_untracked_files(repo, untracked)
    ignored_tree_sha256 = _hash_untracked_files(repo, ignored)
    worktree_payload = {
        "head_tree": head_tree,
        "status_sha256": hashlib.sha256(status_bytes).hexdigest(),
        "tracked_diff_sha256": tracked_diff_sha256,
        "untracked_tree_sha256": untracked_tree_sha256,
        "ignored_tree_sha256": ignored_tree_sha256,
    }
    return {
        "version": SOURCE_REPOSITORY_MANIFEST_VERSION,
        "vcs": "git",
        "head_commit": head_commit,
        "head_tree": head_tree,
        "declared_base_commit": declared,
        "declared_base_commit_resolved": resolved_base,
        "base_commit_validated": declared is not None,
        "dirty": dirty,
        "dirty_entry_count": len(
            [item for item in status_bytes.split(b"\0") if item]
        ),
        "status_sha256": worktree_payload["status_sha256"],
        "tracked_diff_sha256": tracked_diff_sha256,
        "untracked_file_count": len(untracked),
        "untracked_tree_sha256": untracked_tree_sha256,
        "ignored_file_count": len(ignored),
        "ignored_tree_sha256": ignored_tree_sha256,
        "worktree_sha256": canonical_sha256(worktree_payload),
        "ignored_files_included": include_ignored_files,
    }


def filesystem_tree_identity(path: Path) -> dict[str, Any] | None:
    """Hash evaluator fixture bytes without depending on their absolute path."""
    if not path.parts or not path.exists():
        return None
    root = path.resolve()
    candidates = [root] if root.is_file() else sorted(root.rglob("*"))
    digest = hashlib.sha256()
    count = 0
    for candidate in candidates:
        if "__pycache__" in candidate.parts:
            continue
        metadata = candidate.lstat()
        if stat.S_ISDIR(metadata.st_mode):
            continue
        relative = (
            candidate.name
            if root.is_file()
            else candidate.relative_to(root).as_posix()
        )
        relative_bytes = os.fsencode(relative)
        digest.update(len(relative_bytes).to_bytes(8, "big"))
        digest.update(relative_bytes)
        portable_mode = stat.S_IFMT(metadata.st_mode) | stat.S_IMODE(
            metadata.st_mode
        )
        digest.update(portable_mode.to_bytes(8, "big"))
        if stat.S_ISLNK(metadata.st_mode):
            content = os.fsencode(os.readlink(candidate))
            digest.update(len(content).to_bytes(8, "big"))
            digest.update(content)
        elif stat.S_ISREG(metadata.st_mode):
            digest.update(metadata.st_size.to_bytes(8, "big"))
            with candidate.open("rb") as handle:
                while chunk := handle.read(1024 * 1024):
                    digest.update(chunk)
        else:
            digest.update((0).to_bytes(8, "big"))
        count += 1
    return {"sha256": digest.hexdigest(), "file_count": count}


def distribution_content_snapshot(name: str) -> dict[str, Any]:
    """Hash installed package bytes, including an editable source checkout."""
    try:
        distribution = importlib.metadata.distribution(name)
    except importlib.metadata.PackageNotFoundError:
        return {
            "name": name,
            "version": None,
            "aggregate_sha256": None,
            "file_count": 0,
            "missing_files": [],
            "editable_source": None,
            "verifiable": False,
        }

    files: dict[str, str] = {}
    missing: list[str] = []
    for item in sorted(distribution.files or [], key=str):
        label = str(item)
        if "__pycache__" in item.parts or label.endswith((".pyc", ".pyo")):
            continue
        candidate = Path(str(distribution.locate_file(item)))
        if candidate.is_file():
            files[label] = hashlib.sha256(candidate.read_bytes()).hexdigest()
        else:
            missing.append(label)

    editable_source: dict[str, Any] | None = None
    direct_url_candidates = [
        item
        for item in distribution.files or []
        if str(item).endswith(".dist-info/direct_url.json")
    ]
    if direct_url_candidates:
        direct_url_path = Path(
            str(distribution.locate_file(direct_url_candidates[0]))
        )
        try:
            direct_url = json.loads(direct_url_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            direct_url = {}
        if (direct_url.get("dir_info") or {}).get("editable"):
            parsed = urlparse(str(direct_url.get("url") or ""))
            source_path = (
                Path(unquote(parsed.path)) if parsed.scheme == "file" else None
            )
            if source_path is not None and source_path.is_dir():
                try:
                    source_identity = source_repository_identity(
                        source_path,
                        require_clean=True,
                        include_ignored_files=True,
                    )
                except (RuntimeError, ValueError):
                    editable_source = {
                        "path_kind": "non_git_or_dirty",
                        "tree": filesystem_tree_identity(source_path),
                        "verifiable": False,
                    }
                else:
                    editable_source = {
                        "path_kind": "clean_git_checkout",
                        "source_repository": source_identity,
                        "verifiable": True,
                    }
            else:
                editable_source = {
                    "path_kind": "missing_or_non_file_url",
                    "verifiable": False,
                }
    payload = {"files": files, "editable_source": editable_source}
    editable_verifiable = editable_source is None or bool(
        editable_source.get("verifiable")
    )
    return {
        "name": name,
        "version": distribution.version,
        "aggregate_sha256": canonical_sha256(payload),
        "file_count": len(files),
        "missing_files": missing,
        "editable_source": editable_source,
        "verifiable": not missing and editable_verifiable and bool(files),
    }


def distribution_dependency_closure_snapshot(
    roots: tuple[str, ...] | list[str],
) -> dict[str, Any]:
    """Hash the installed, marker-resolved dependency closure of ``roots``.

    Version labels alone are insufficient provenance: an installed wheel can be
    modified in place and SWE-agent delegates provider behavior to dependencies
    such as LiteLLM.  Each reachable distribution is therefore content-hashed.
    Environment markers are evaluated for this Python runtime; optional extras
    are followed only when a parent requirement explicitly requests them.
    """
    normalized_roots = tuple(canonicalize_name(name) for name in roots)
    requested_extras: dict[str, set[str]] = {
        name: set() for name in normalized_roots
    }
    processed_extras: dict[str, set[str]] = {}
    pending = deque(normalized_roots)
    distributions: dict[str, dict[str, Any]] = {}
    missing: set[str] = set()
    invalid_requirements: list[dict[str, str]] = []
    unsatisfied_requirements: list[dict[str, str]] = []
    edges: set[tuple[str, str, str]] = set()
    marker_base = {
        str(key): str(value) for key, value in default_environment().items()
    }

    while pending:
        requested_name = pending.popleft()
        extras = requested_extras.setdefault(requested_name, set())
        if extras.issubset(processed_extras.get(requested_name, set())) and (
            requested_name in distributions or requested_name in missing
        ):
            continue
        processed_extras.setdefault(requested_name, set()).update(extras)
        try:
            distribution = importlib.metadata.distribution(requested_name)
        except importlib.metadata.PackageNotFoundError:
            missing.add(requested_name)
            continue

        resolved_name = canonicalize_name(
            distribution.metadata["Name"] or requested_name
        )
        content = distribution_content_snapshot(requested_name)
        requirements = sorted(distribution.requires or [])
        distributions[resolved_name] = {
            **content,
            "resolved_name": resolved_name,
            "requested_extras": sorted(extras),
            "requirements": requirements,
        }

        marker_environments: list[dict[str, str]] = []
        for extra in ["", *sorted(extras)]:
            marker_environments.append({**marker_base, "extra": extra})
        for raw_requirement in requirements:
            try:
                requirement = Requirement(raw_requirement)
            except InvalidRequirement:
                invalid_requirements.append(
                    {"distribution": resolved_name, "requirement": raw_requirement}
                )
                continue
            if requirement.marker is not None and not any(
                requirement.marker.evaluate(environment)
                for environment in marker_environments
            ):
                continue
            dependency_name = canonicalize_name(requirement.name)
            edges.add((resolved_name, dependency_name, str(requirement)))
            dependency_extras = requested_extras.setdefault(dependency_name, set())
            previous_extras = set(dependency_extras)
            dependency_extras.update(requirement.extras)
            if (
                dependency_name not in distributions
                and dependency_name not in missing
            ) or dependency_extras != previous_extras:
                pending.append(dependency_name)

    for parent, resolved_dependency, raw_requirement in sorted(edges):
        dependency = distributions.get(resolved_dependency)
        if dependency is None:
            continue
        requirement = Requirement(raw_requirement)
        version = dependency.get("version")
        version_satisfies = (
            version is None
            or not requirement.specifier
            or requirement.specifier.contains(version, prereleases=True)
        )
        if not version_satisfies:
            unsatisfied_requirements.append(
                {
                    "distribution": parent,
                    "requirement": raw_requirement,
                    "resolved_version": str(version),
                }
            )

    ordered_distributions = {
        name: distributions[name] for name in sorted(distributions)
    }
    payload: dict[str, Any] = {
        "roots": list(normalized_roots),
        "distributions": ordered_distributions,
        "edges": [list(edge) for edge in sorted(edges)],
        "missing_distributions": sorted(missing),
        "invalid_requirements": sorted(
            invalid_requirements,
            key=lambda item: (item["distribution"], item["requirement"]),
        ),
        "unsatisfied_requirements": sorted(
            unsatisfied_requirements,
            key=lambda item: (item["distribution"], item["requirement"]),
        ),
    }
    payload["aggregate_sha256"] = canonical_sha256(payload)
    payload["verifiable"] = (
        not missing
        and not invalid_requirements
        and not unsatisfied_requirements
        and all(item["verifiable"] for item in ordered_distributions.values())
        and all(root in ordered_distributions for root in normalized_roots)
    )
    return payload


def _file_content_snapshot(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {
            "path": None,
            "resolved_path": None,
            "sha256": None,
            "bytes": None,
            "available": False,
        }
    expanded = path.expanduser()
    try:
        resolved = expanded.resolve(strict=True)
    except OSError:
        resolved = expanded.resolve()
    available = resolved.is_file()
    return {
        "path": str(expanded),
        "resolved_path": str(resolved),
        "sha256": (
            hashlib.sha256(resolved.read_bytes()).hexdigest() if available else None
        ),
        "bytes": resolved.stat().st_size if available else None,
        "available": available,
    }


def _stdlib_tree_identity(path: Path) -> dict[str, Any] | None:
    """Hash stdlib bytes while excluding third-party and bytecode trees."""
    if not path.is_dir():
        return None
    digest = hashlib.sha256()
    count = 0
    excluded = {"__pycache__", "site-packages", "dist-packages"}
    for candidate in sorted(path.rglob("*")):
        relative_path = candidate.relative_to(path)
        if any(part in excluded for part in relative_path.parts):
            continue
        metadata = candidate.lstat()
        if stat.S_ISDIR(metadata.st_mode) or candidate.suffix in {".pyc", ".pyo"}:
            continue
        relative = relative_path.as_posix()
        relative_bytes = os.fsencode(relative)
        digest.update(len(relative_bytes).to_bytes(8, "big"))
        digest.update(relative_bytes)
        portable_mode = stat.S_IFMT(metadata.st_mode) | stat.S_IMODE(
            metadata.st_mode
        )
        digest.update(portable_mode.to_bytes(8, "big"))
        if stat.S_ISLNK(metadata.st_mode):
            content = os.fsencode(os.readlink(candidate))
            digest.update(len(content).to_bytes(8, "big"))
            digest.update(content)
        elif stat.S_ISREG(metadata.st_mode):
            digest.update(metadata.st_size.to_bytes(8, "big"))
            with candidate.open("rb") as handle:
                while chunk := handle.read(1024 * 1024):
                    digest.update(chunk)
        else:
            digest.update((0).to_bytes(8, "big"))
        count += 1
    return {"sha256": digest.hexdigest(), "file_count": count}


def python_runtime_snapshot() -> dict[str, Any]:
    """Content identity of the Python runtime that resolves SWE dependencies."""
    executable = _file_content_snapshot(Path(sys.executable))
    stdlib_path = Path(sysconfig.get_path("stdlib"))
    stdlib = _stdlib_tree_identity(stdlib_path)
    library_name = sysconfig.get_config_var("LDLIBRARY") or sysconfig.get_config_var(
        "LIBRARY"
    )
    library_dir = sysconfig.get_config_var("LIBDIR")
    runtime_library = _file_content_snapshot(
        Path(str(library_dir)) / str(library_name)
        if library_dir and library_name
        else None
    )
    payload = {
        "implementation": sys.implementation.name,
        "version": list(sys.version_info[:5]),
        "cache_tag": sys.implementation.cache_tag,
        "soabi": sysconfig.get_config_var("SOABI"),
        "platform": sysconfig.get_platform(),
        "executable": executable,
        "runtime_library": runtime_library,
        "stdlib_path": str(stdlib_path.resolve()),
        "stdlib_tree": stdlib,
    }
    payload["aggregate_sha256"] = canonical_sha256(payload)
    payload["verifiable"] = bool(
        executable["available"]
        and stdlib is not None
    )
    return payload


def _entrypoint_interpreter(path: Path) -> tuple[str | None, list[str]]:
    try:
        first_line = path.open("rb").readline(4096).decode(
            "utf-8", errors="replace"
        )
    except OSError:
        return None, []
    if not first_line.startswith("#!"):
        return None, []
    try:
        tokens = shlex.split(first_line[2:].strip())
    except ValueError:
        return None, []
    if not tokens:
        return None, []
    interpreter_token = tokens[0]
    arguments = tokens[1:]
    interpreter: str | None
    if Path(interpreter_token).name == "env":
        command = next((item for item in arguments if not item.startswith("-")), None)
        interpreter = shutil.which(command) if command is not None else None
    elif not Path(interpreter_token).is_absolute():
        interpreter = shutil.which(interpreter_token)
    else:
        interpreter = interpreter_token
    return interpreter, arguments


def external_swe_runtime_snapshot(sweagent_bin: str = "sweagent") -> dict[str, Any]:
    """Freeze the host runtime that executes the explicit online SWE ablation."""
    resolved_binary = shutil.which(sweagent_bin)
    entrypoint_path = Path(resolved_binary) if resolved_binary is not None else None
    entrypoint = _file_content_snapshot(entrypoint_path)
    interpreter_value, interpreter_arguments = (
        _entrypoint_interpreter(entrypoint_path)
        if entrypoint_path is not None
        else (None, [])
    )
    interpreter = _file_content_snapshot(
        Path(interpreter_value) if interpreter_value is not None else None
    )
    python_runtime = python_runtime_snapshot()
    runtime_executable = python_runtime["executable"]
    interpreter_matches_runtime = bool(
        interpreter["available"]
        and runtime_executable["available"]
        and interpreter["resolved_path"] == runtime_executable["resolved_path"]
    )
    dependency_closure = distribution_dependency_closure_snapshot(
        list(EXTERNAL_SWE_ROOT_DISTRIBUTIONS)
    )
    payload = {
        "entrypoint": entrypoint,
        "entrypoint_interpreter": interpreter,
        "entrypoint_interpreter_arguments": interpreter_arguments,
        "entrypoint_interpreter_matches_runtime": interpreter_matches_runtime,
        "python_runtime": python_runtime,
        "dependency_closure": dependency_closure,
        "network_policy": "explicit_online_ablation",
    }
    payload["aggregate_sha256"] = canonical_sha256(payload)
    payload["verifiable"] = bool(
        entrypoint["available"]
        and interpreter_matches_runtime
        and python_runtime["verifiable"]
        and dependency_closure["verifiable"]
    )
    return payload


def docker_image_identity(
    image: str,
    *,
    require_resolved: bool = False,
) -> dict[str, Any]:
    """Resolve a mutable Docker reference to the exact local image object."""
    error: str | None = None
    raw: dict[str, Any] | None = None
    try:
        result = subprocess.run(
            ["docker", "image", "inspect", image],
            capture_output=True,
            text=True,
            check=False,
            timeout=60,
        )
        if result.returncode == 0:
            decoded = json.loads(result.stdout)
            if isinstance(decoded, list) and decoded and isinstance(decoded[0], dict):
                raw = decoded[0]
        if raw is None:
            error = result.stderr.strip() or "image is not present locally"
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError) as exc:
        error = str(exc)

    identity = {
        "reference": image,
        "resolved": raw is not None and bool(raw.get("Id")),
        "image_id": raw.get("Id") if raw is not None else None,
        "repo_digests": (
            sorted(raw.get("RepoDigests") or []) if raw is not None else []
        ),
    }
    if require_resolved and not identity["resolved"]:
        raise RuntimeError(
            f"cannot resolve Docker image {image!r} before the run"
            + (f": {error}" if error else "")
            + "; pull/build it first so the benchmark can pin an immutable image ID"
        )
    return identity


def runtime_dependency_snapshot() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for distribution in RUNTIME_DISTRIBUTIONS:
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = None
    return versions


def _prompt_entry(text: str) -> dict[str, Any]:
    return {"sha256": text_sha256(text), "chars": len(text)}


def agent_prompt_manifest(agent: Any) -> dict[str, Any]:
    """Hash the exact built-in system prompts this agent can send."""
    class_name = type(agent).__name__
    if class_name == "SweAgentAdapter":
        # SWE-agent owns its prompt/config outside this harness. Do not claim a
        # false hash; its invocation configuration is covered by policy_config.
        return {
            "version": PROMPT_MANIFEST_VERSION,
            "external": True,
            "external_components": ["swe-agent:system-and-instance-prompts"],
            "prompts": {},
            "aggregate_sha256": None,
        }

    prompts: dict[str, str] = {}
    external_components: list[str] = []
    if class_name.startswith("Multi") or "OrchestratorAgent" in class_name:
        # Import registers the role action spaces as well as loading the exact
        # role prompts used by the multi-agent episodes.
        from ..agents.multi_agent import ORCHESTRATOR_PROMPT  # noqa: PLC0415
        from ..agents.roles import ROLES  # noqa: PLC0415

        for role, spec in sorted(ROLES.items()):
            if class_name == "MultiSweAgent" and role == "developer":
                external_components.append("developer:swe-agent-prompt")
                continue
            prompts[f"role:{role}"] = spec.prompt
        if "Orchestrator" in class_name:
            prompts["orchestrator"] = ORCHESTRATOR_PROMPT
    else:
        prompts["agent"] = str(agent.system_prompt)
    # Native-tools mode appends this exact suffix to every built-in role/system
    # prompt. Hash the complete derived strings, not just the source constant,
    # so the manifest describes every system prompt the configured kernel can
    # actually send after transport resolution.
    if getattr(agent, "action_transport", "text_json") in ("tools", "auto"):
        for name, text in list(prompts.items()):
            prompts[f"{name}:native_tools"] = f"{text}\n{TOOL_MODE_PROMPT}"
    if getattr(agent, "compaction_mode", None) == "summarize":
        prompts["context_summary_template"] = CONTEXT_SUMMARY_PROMPT

    entries = {name: _prompt_entry(text) for name, text in sorted(prompts.items())}
    return {
        "version": PROMPT_MANIFEST_VERSION,
        "external": False,
        "external_components": external_components,
        "prompts": entries,
        "aggregate_sha256": canonical_sha256(entries),
    }


def _ensure_action_spaces_registered() -> None:
    # Imports only declarative action-space registrations; no runtime routing.
    from ..agents import decomposition as _decomposition  # noqa: F401, PLC0415
    from ..agents import roles as _roles  # noqa: F401, PLC0415


def agent_tool_spaces(agent: Any) -> list[str]:
    class_name = type(agent).__name__
    if class_name == "SweAgentAdapter":
        return []
    _ensure_action_spaces_registered()
    if class_name == "DecomposedSingleAgent":
        return sorted(name for name in ACTION_SPACES if name.startswith("decomposed_"))
    if class_name.startswith("Multi") or "OrchestratorAgent" in class_name:
        spaces = ["planner", "tester", "reviewer"]
        if class_name != "MultiSweAgent":
            spaces.append("developer")
        if "Orchestrator" in class_name:
            spaces.append("orchestrator")
        return sorted(spaces)
    return [getattr(agent, "action_space", None) or "default"]


def tool_schema_manifest(spaces: list[str] | None = None) -> dict[str, Any]:
    _ensure_action_spaces_registered()
    selected = sorted(ACTION_SPACES if spaces is None else spaces)
    schemas = {
        name: action_tool_schemas(None if name == "default" else name)
        for name in selected
    }
    return {
        "version": TOOL_SCHEMA_MANIFEST_VERSION,
        "spaces": selected,
        "space_sha256": {
            name: canonical_sha256(schema) for name, schema in schemas.items()
        },
        "aggregate_sha256": canonical_sha256(schemas),
    }


def prompt_template_snapshot() -> dict[str, Any]:
    """All checked-in prompt templates for sweep-level fingerprints."""
    prompt_dir = Path(__file__).resolve().parents[2] / "static" / "prompts"
    entries: dict[str, dict[str, Any]] = {}
    for path in sorted(prompt_dir.rglob("*.txt")):
        text = path.read_text(encoding="utf-8").strip()
        entries[path.relative_to(prompt_dir).as_posix()] = _prompt_entry(text)
    return {
        "version": PROMPT_MANIFEST_VERSION,
        "prompts": entries,
        "aggregate_sha256": canonical_sha256(entries),
    }


def policy_kernel_source_manifest() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    files: dict[str, str | None] = {}
    for relative in POLICY_KERNEL_FILES:
        path = root / relative
        files[relative] = (
            hashlib.sha256(path.read_bytes()).hexdigest()
            if path.is_file()
            else None
        )
    return {
        "files": files,
        "aggregate_sha256": canonical_sha256(files),
    }


def policy_kernel_snapshot(configuration: dict[str, Any]) -> dict[str, Any]:
    sources = policy_kernel_source_manifest()
    payload = {
        "version": POLICY_KERNEL_VERSION,
        "configuration_sha256": canonical_sha256(configuration),
        "configuration": configuration,
        "sources": sources,
    }
    payload["aggregate_sha256"] = canonical_sha256(payload)
    return payload


def build_run_reproducibility(
    *,
    task_content: str,
    agent: Any,
    policy_configuration: dict[str, Any],
    model_routes: dict[str, Any],
    source_repository: dict[str, Any] | None = None,
) -> dict[str, Any]:
    prompts = agent_prompt_manifest(agent)
    tools = tool_schema_manifest(agent_tool_spaces(agent))
    policy = policy_kernel_snapshot(policy_configuration)
    payload = {
        "schema_version": RUN_REPRODUCIBILITY_SCHEMA_VERSION,
        "task_content_sha256": text_sha256(task_content),
        "task_content_chars": len(task_content),
        "source_repository": source_repository,
        "source_repository_sha256": (
            canonical_sha256(source_repository)
            if source_repository is not None
            else None
        ),
        "system_prompts": prompts,
        "tool_schemas": tools,
        "policy_kernel": policy,
        "model_routes": model_routes,
        "model_routes_sha256": canonical_sha256(model_routes),
        "runtime_dependencies": runtime_dependency_snapshot(),
    }
    payload["run_fingerprint"] = canonical_sha256(payload)
    return payload


def sweep_reproducibility_artifacts() -> dict[str, Any]:
    prompts = prompt_template_snapshot()
    tools = tool_schema_manifest()
    payload = {
        "schema_version": RUN_REPRODUCIBILITY_SCHEMA_VERSION,
        "prompt_templates": prompts,
        "tool_schemas": tools,
        "policy_kernel": policy_kernel_snapshot({}),
        "runtime_dependencies": runtime_dependency_snapshot(),
    }
    payload["aggregate_sha256"] = canonical_sha256(payload)
    return payload
