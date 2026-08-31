"""Post-run evaluation in a Docker sandbox: regressions and hidden tests.

Runs outside the agent's LangSmith trace (this is harness infrastructure, not
agent behavior), so setup/test spans do not pollute the project.
"""

from __future__ import annotations

import configparser
import hashlib
import os
import posixpath
import shlex
import shutil
import stat
import subprocess
import tomllib
import uuid
import xml.etree.ElementTree as ET
from collections.abc import Mapping
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path, PurePosixPath

from langsmith import tracing_context

from ..agents.sandbox import DockerSandbox
from ..agents.workspace import (
    TEST_ORACLE_CONTROL_NAMES,
    TEST_ORACLE_DIR_NAMES,
    is_test_oracle_path,
)
from ..git_safety import safe_git_command, safe_git_env, validate_local_git_config
from .collection import TaskSpec
from .test_launch import (
    pytest_selection_lower_bound,
    trusted_evaluator_test_command,
)

# Enough to import the package under test and run its pytest suite. Override
# per-repo with --setup-command when a task needs extra dependencies.
DEFAULT_SETUP_COMMANDS = [
    "python -m pip install -q -e .",
    "python -m pip install -q pytest",
]

_JUNIT_LABEL = "visible"
_HIDDEN_JUNIT_LABEL = "hidden"
_TRUSTED_PYTEST_HELPER = Path(__file__).with_name("_trusted_pytest.py")
_EVALUATOR_BIN = Path(__file__).with_name("evaluator_bin")
_PYTEST_CONFIG_NAMES = frozenset(
    {
        ".pytest.ini",
        ".pytest.toml",
        "pyproject.toml",
        "pytest.ini",
        "pytest.toml",
        "setup.cfg",
        "tox.ini",
    }
)


@dataclass
class EvalResult:
    visible_passed: bool | None = None
    hidden_passed: bool | None = None
    hidden_semantic_passed: bool | None = None
    hidden_compat_passed: bool | None = None
    hidden_pr_parity_passed: bool | None = None
    task_success: bool | None = None
    regressions: int | None = None
    test_oracle_tampered: bool = False
    setup_artifact_tampered: bool = False
    setup_artifact_conflicts: list[str] = field(default_factory=list)
    excluded_agent_test_files: list[str] = field(default_factory=list)
    hidden_suite_results: dict[str, bool] = field(default_factory=dict)


def task_success(
    visible_passed: bool,
    hidden_passed: bool,
    regressions: int | None,
    test_oracle_tampered: bool = False,
    setup_artifact_tampered: bool = False,
) -> bool:
    """Ground-truth success for a submitted patch.

    A green exit code is insufficient when the agent deleted or renamed tests:
    the before/after JUnit comparison must also retain every baseline pass.
    ``None`` keeps ``--no-regression`` usable when the check was explicitly
    disabled.
    """
    return (
        visible_passed
        and hidden_passed
        and regressions in (None, 0)
        and not test_oracle_tampered
        and not setup_artifact_tampered
    )


_SCORING_SCAN_IGNORED_DIRS = frozenset(
    {
        ".git",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".tox",
        ".venv",
        "__pycache__",
        "build",
        "dist",
        "node_modules",
        "target",
        "venv",
    }
)

# Hidden fixtures are evaluator-owned inputs, not a source checkout to prune.
# Names such as ``target``, ``build`` and ``dist`` can be legitimate package
# paths (SWE-bench Pro's Ansible fixtures contain one), so only deterministic
# interpreter/test caches and VCS metadata are excluded here.
_HIDDEN_ORACLE_IGNORED_DIRS = frozenset({".git", ".pytest_cache", "__pycache__"})
_HIDDEN_ORACLE_IGNORED_SUFFIXES = frozenset({".pyc", ".pyo"})

# Pristine setup may generate import-critical source-tree files which an
# editable install still reads during scoring (for example setuptools-scm's
# ``src/package/_version.py``).  They are evaluator inputs, not agent output.
# Cache/build trees remain disposable, and test-oracle paths are frozen by the
# stricter oracle machinery instead of this runtime-artifact channel.
_SETUP_ARTIFACT_IGNORED_DIRS = _SCORING_SCAN_IGNORED_DIRS
_SETUP_ARTIFACT_IGNORED_SUFFIXES = frozenset({".pyc", ".pyo"})
_SETUP_ARTIFACT_MAX_FILES = 4_096
_SETUP_ARTIFACT_MAX_BYTES = 512 * 1024 * 1024
_SETUP_ARTIFACT_PARENT_MODE = 0o755
_SETUP_ARTIFACT_MTIME_NS = 1_000_000_000


def _is_test_control_path(path: str) -> bool:
    return Path(path).name.lower() in TEST_ORACLE_CONTROL_NAMES


def _is_scoring_oracle_path(path: str) -> bool:
    return (
        is_test_oracle_path(path)
        and Path(path).name.lower() not in _PYTEST_CONFIG_NAMES
    )


def _pytest_config_candidates(root: Path) -> set[str]:
    result: set[str] = set()
    for current, directories, filenames in os.walk(root, followlinks=False):
        current_path = Path(current)
        directories[:] = [
            name
            for name in directories
            if name not in _SCORING_SCAN_IGNORED_DIRS
            and not (current_path / name).is_symlink()
        ]
        for name in filenames:
            if name.lower() in _PYTEST_CONFIG_NAMES:
                result.add((current_path / name).relative_to(root).as_posix())
    return result


def _pytest_config_signature(path: Path) -> object:
    """Security-relevant pytest configuration, excluding marker docs.

    Marker declarations are metadata and some benchmark gold patches add
    them. Selection, import, plugin, warning, and execution options remain
    protected. Presence/activation is part of the signature because adding a
    higher-precedence config file can suppress a pristine lower-precedence one.
    """
    name = path.name.lower()
    raw = path.read_bytes()
    try:
        if name in {"pyproject.toml", "pytest.toml", ".pytest.toml"}:
            document = tomllib.loads(raw.decode("utf-8"))
            if name == "pyproject.toml":
                pytest_table = document.get("tool", {}).get("pytest")
                active = isinstance(pytest_table, dict)
                options = (
                    pytest_table.get("ini_options", {}) if active else {}
                )
            else:
                pytest_table = document.get("pytest")
                active = isinstance(pytest_table, dict)
                options = pytest_table if active else {}
            if not isinstance(options, dict):
                options = {"__invalid_ini_options__": options}
            protected = {
                str(key).lower(): value
                for key, value in options.items()
                if str(key).lower() != "markers"
            }
            return {"active": active, "options": protected}

        parser = configparser.ConfigParser(
            interpolation=None,
            strict=False,
        )
        parser.read_string(raw.decode("utf-8"))
        section = "tool:pytest" if name == "setup.cfg" else "pytest"
        active = parser.has_section(section)
        options = {
            key.lower(): value
            for key, value in (parser.items(section) if active else [])
            if key.lower() != "markers"
        }
        return {"active": active, "options": options}
    except (UnicodeDecodeError, configparser.Error, tomllib.TOMLDecodeError):
        # Invalid config must not compare equal to a valid pristine one.
        return {"parse_error": raw.hex()}


def _restore_pytest_config_controls(repo: Path, pristine: Path) -> set[str]:
    """Restore only pytest-affecting config changes, preserving benign edits."""
    tampered: set[str] = set()
    candidates = _pytest_config_candidates(repo) | _pytest_config_candidates(pristine)
    for relative in sorted(candidates):
        source = pristine / relative
        destination = repo / relative
        tampered.update(_prepare_safe_destination(repo, relative, pristine=pristine))
        source_exists = source.exists() or source.is_symlink()
        destination_exists = destination.exists() or destination.is_symlink()
        unsafe = False
        for candidate in (source, destination):
            if not (candidate.exists() or candidate.is_symlink()):
                continue
            metadata = candidate.lstat()
            if (
                not stat.S_ISREG(metadata.st_mode)
                or metadata.st_nlink > 1
                or candidate.is_symlink()
            ):
                unsafe = True
        changed = source_exists != destination_exists or unsafe
        if source_exists and destination_exists and not unsafe:
            changed = (
                _pytest_config_signature(source)
                != _pytest_config_signature(destination)
            )
        if not changed:
            continue
        tampered.add(relative)
        if source_exists:
            _restore_pristine_entry(source, destination)
        else:
            _remove_scoring_entry(destination)
    return tampered


def _git_nul_paths(repo: Path, *arguments: str) -> set[str]:
    result = subprocess.run(
        safe_git_command(repo, *arguments),
        capture_output=True,
        check=False,
        env=safe_git_env(),
        timeout=60,
    )
    if result.returncode != 0:
        error = result.stderr.decode("utf-8", errors="replace")[-1000:]
        raise RuntimeError(f"failed to inventory scoring checkout:\n{error}")
    return {
        item.decode("utf-8", errors="surrogateescape")
        for item in result.stdout.split(b"\0")
        if item
    }


def _safe_setup_artifact_relative(relative: str) -> PurePosixPath:
    candidate = PurePosixPath(relative)
    if candidate.is_absolute() or ".." in candidate.parts or not candidate.parts:
        raise RuntimeError(f"unsafe pristine setup artifact path: {relative!r}")
    return candidate


def _setup_artifact_candidate(relative: str) -> bool:
    candidate = _safe_setup_artifact_relative(relative)
    return not (
        any(part in _SETUP_ARTIFACT_IGNORED_DIRS for part in candidate.parts)
        or candidate.suffix.lower() in _SETUP_ARTIFACT_IGNORED_SUFFIXES
        or _is_scoring_oracle_path(relative)
        or _is_test_control_path(relative)
        or Path(relative).name.lower() in _PYTEST_CONFIG_NAMES
    )


def _normalize_setup_artifact_parents(
    root: Path,
    relative: str,
    *,
    synthesized: set[str] | None = None,
) -> None:
    current = root
    parts = PurePosixPath(relative).parts[:-1]
    for index, part in enumerate(parts, start=1):
        current /= part
        current_relative = PurePosixPath(*parts[:index]).as_posix()
        if synthesized is not None and current_relative not in synthesized:
            continue
        metadata = current.lstat()
        if not stat.S_ISDIR(metadata.st_mode):
            raise RuntimeError(
                f"setup-artifact parent is not a directory: {relative}"
            )
        os.chmod(current, _SETUP_ARTIFACT_PARENT_MODE, follow_symlinks=False)


def setup_artifact_manifest(repo: Path) -> dict[str, str]:
    """Describe eligible untracked/ignored files without following aliases."""
    repo = repo.resolve()
    validate_local_git_config(repo)
    untracked = _git_nul_paths(repo, "ls-files", "--others", "--exclude-standard", "-z")
    ignored = _git_nul_paths(
        repo,
        "ls-files",
        "--others",
        "--ignored",
        "--exclude-standard",
        "-z",
    )
    candidates = sorted(
        relative
        for relative in untracked | ignored
        if _setup_artifact_candidate(relative)
    )
    if len(candidates) > _SETUP_ARTIFACT_MAX_FILES:
        raise RuntimeError(
            "setup-artifact inventory exceeds its file limit "
            f"({len(candidates)} > {_SETUP_ARTIFACT_MAX_FILES})"
        )
    manifest: dict[str, str] = {}
    total_bytes = 0
    for relative in candidates:
        source = repo / Path(relative)
        _validate_plain_parent_chain(repo, relative, label="setup-artifact source")
        try:
            metadata = source.lstat()
        except OSError as exc:
            raise RuntimeError(
                f"pristine setup artifact is unavailable: {relative}: {exc}"
            ) from exc
        mode = stat.S_IFMT(metadata.st_mode) | stat.S_IMODE(metadata.st_mode)
        total_bytes += metadata.st_size
        if total_bytes > _SETUP_ARTIFACT_MAX_BYTES:
            raise RuntimeError(
                "setup-artifact inventory exceeds its byte limit "
                f"({total_bytes} > {_SETUP_ARTIFACT_MAX_BYTES})"
            )
        if stat.S_ISREG(metadata.st_mode):
            digest = hashlib.sha256()
            try:
                with source.open("rb") as handle:
                    while chunk := handle.read(1024 * 1024):
                        digest.update(chunk)
            except OSError as exc:
                raise RuntimeError(
                    f"pristine setup artifact is unreadable: {relative}: {exc}"
                ) from exc
            manifest[relative] = (
                f"regular:{mode:o}:{metadata.st_nlink}:{metadata.st_size}:"
                f"{digest.hexdigest()}"
            )
        elif stat.S_ISLNK(metadata.st_mode):
            manifest[relative] = f"symlink:{mode:o}:{os.readlink(source)}"
        else:
            manifest[relative] = f"special:{mode:o}"
    return manifest


def freeze_setup_artifacts(
    repo: Path,
    destination: Path,
    *,
    before_setup: Mapping[str, str] | None = None,
) -> list[str]:
    """Freeze pristine setup-generated runtime files before model execution.

    Only Git-untracked/ignored regular files outside the test oracle are
    eligible.  The snapshot is stored outside the model-visible checkout and
    fingerprinted with the other evaluator inputs.  This preserves generated
    version modules and editable-install metadata without carrying arbitrary
    ignored files created later by the model into final scoring.
    """
    repo = repo.resolve()
    if destination.exists() or destination.is_symlink():
        raise RuntimeError(
            f"frozen setup-artifact destination already exists: {destination}"
        )
    validate_local_git_config(repo)
    tracked_changes = _git_nul_paths(repo, "diff", "--name-only", "-z", "HEAD", "--")
    if tracked_changes:
        raise RuntimeError(
            "pristine setup modified tracked repository files: "
            + ", ".join(sorted(tracked_changes)[:20])
        )
    after_setup = setup_artifact_manifest(repo)
    candidates = sorted(
        relative
        for relative, identity in after_setup.items()
        if before_setup is None or before_setup.get(relative) != identity
    )
    if len(candidates) > _SETUP_ARTIFACT_MAX_FILES:
        raise RuntimeError(
            "pristine setup generated too many runtime artifacts "
            f"({len(candidates)} > {_SETUP_ARTIFACT_MAX_FILES})"
        )

    destination.mkdir(parents=True)
    total_bytes = 0
    copied: list[str] = []
    for relative in candidates:
        source = repo / Path(relative)
        _validate_plain_parent_chain(repo, relative, label="setup-artifact source")
        try:
            metadata = source.lstat()
        except OSError as exc:
            raise RuntimeError(
                f"pristine setup artifact is unavailable: {relative}: {exc}"
            ) from exc
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink > 1:
            raise RuntimeError(
                f"pristine setup artifacts must be unaliased regular files: {relative}"
            )
        total_bytes += metadata.st_size
        if total_bytes > _SETUP_ARTIFACT_MAX_BYTES:
            raise RuntimeError(
                "pristine setup generated too many runtime-artifact bytes "
                f"({total_bytes} > {_SETUP_ARTIFACT_MAX_BYTES})"
            )
        target = destination / Path(relative)
        _prepare_safe_destination(destination, relative, pristine=None)
        _normalize_setup_artifact_parents(destination, relative)
        shutil.copy2(source, target, follow_symlinks=False)
        os.utime(
            target,
            ns=(_SETUP_ARTIFACT_MTIME_NS, _SETUP_ARTIFACT_MTIME_NS),
            follow_symlinks=False,
        )
        copied.append(relative)
    return copied


def frozen_setup_artifact_files(source: Path) -> list[str]:
    try:
        source_metadata = source.lstat()
    except OSError as exc:
        raise RuntimeError(
            f"frozen setup-artifact tree is unavailable: {source}: {exc}"
        ) from exc
    if not stat.S_ISDIR(source_metadata.st_mode) or source.is_symlink():
        raise RuntimeError(f"frozen setup-artifact tree is unavailable: {source}")
    source = source.resolve()
    files: list[str] = []
    total_bytes = 0
    for current, directory_names, file_names in os.walk(source, followlinks=False):
        current_path = Path(current)
        for name in sorted(directory_names):
            directory = current_path / name
            relative = directory.relative_to(source).as_posix()
            try:
                metadata = directory.lstat()
            except OSError as exc:
                raise RuntimeError(
                    f"frozen setup-artifact directory is unavailable: {relative}: {exc}"
                ) from exc
            if not stat.S_ISDIR(metadata.st_mode):
                raise RuntimeError(
                    f"frozen setup-artifact paths cannot traverse aliases: {relative}"
                )
        for name in sorted(file_names):
            path = current_path / name
            relative = path.relative_to(source).as_posix()
            _safe_setup_artifact_relative(relative)
            try:
                metadata = path.lstat()
            except OSError as exc:
                raise RuntimeError(
                    f"frozen setup artifact is unavailable: {relative}: {exc}"
                ) from exc
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink > 1:
                raise RuntimeError(
                    "frozen setup artifacts must remain unaliased regular files: "
                    f"{relative}"
                )
            total_bytes += metadata.st_size
            files.append(relative)
    if len(files) > _SETUP_ARTIFACT_MAX_FILES:
        raise RuntimeError(
            "frozen setup-artifact tree exceeds its file limit "
            f"({len(files)} > {_SETUP_ARTIFACT_MAX_FILES})"
        )
    if total_bytes > _SETUP_ARTIFACT_MAX_BYTES:
        raise RuntimeError(
            "frozen setup-artifact tree exceeds its byte limit "
            f"({total_bytes} > {_SETUP_ARTIFACT_MAX_BYTES})"
        )
    return sorted(files)


def restore_setup_artifacts(
    repo: Path,
    source: Path,
    *,
    conflicts: list[str] | None = None,
) -> list[str]:
    """Restore the frozen pristine runtime delta after agent patch replay."""
    repo = repo.resolve()
    validate_local_git_config(repo)
    restored: list[str] = []
    for relative in frozen_setup_artifact_files(source):
        artifact = source / Path(relative)
        synthesized_parents: set[str] = set()
        current = repo
        parent_missing_or_unsafe = False
        parts = PurePosixPath(relative).parts[:-1]
        for index, part in enumerate(parts, start=1):
            current /= part
            current_relative = PurePosixPath(*parts[:index]).as_posix()
            if parent_missing_or_unsafe:
                synthesized_parents.add(current_relative)
                continue
            try:
                metadata = current.lstat()
            except OSError:
                synthesized_parents.add(current_relative)
                parent_missing_or_unsafe = True
                continue
            if not stat.S_ISDIR(metadata.st_mode):
                synthesized_parents.add(current_relative)
                parent_missing_or_unsafe = True
        repaired = _prepare_safe_destination(repo, relative, pristine=None)
        _normalize_setup_artifact_parents(
            repo,
            relative,
            synthesized=synthesized_parents,
        )
        if conflicts is not None:
            conflicts.extend(sorted(repaired))
        destination = repo / Path(relative)
        if destination.exists() or destination.is_symlink():
            metadata = destination.lstat()
            if conflicts is not None and relative not in conflicts:
                conflicts.append(relative)
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink > 1:
                _remove_scoring_entry(destination)
        shutil.copy2(artifact, destination, follow_symlinks=False)
        os.utime(
            destination,
            ns=(_SETUP_ARTIFACT_MTIME_NS, _SETUP_ARTIFACT_MTIME_NS),
            follow_symlinks=False,
        )
        restored.append(relative)
    return restored


def _configured_test_control_paths(repo: Path, commands: list[str]) -> set[str]:
    """Return lexical repo-relative scripts explicitly used to launch tests.

    This deliberately does not resolve or stat a token in the agent checkout:
    a deleted pristine runner still needs restoring, and an agent-created
    symlink must not redirect discovery outside the repository.
    """
    del repo  # Kept in the signature for compatibility with existing callers.
    result: set[str] = set()

    def add_launcher(raw: str, *, suffixes: set[str]) -> None:
        normalized = posixpath.normpath(raw.replace("\\", "/"))
        candidate = PurePosixPath(normalized)
        if (
            normalized in {"", "."}
            or candidate.is_absolute()
            or ".." in candidate.parts
            or any(part in _SCORING_SCAN_IGNORED_DIRS for part in candidate.parts)
            or candidate.suffix.lower() not in suffixes
        ):
            return
        result.add(candidate.as_posix().removeprefix("./"))

    for command in commands:
        try:
            tokens = shlex.split(command)
        except ValueError:
            continue
        index = 0
        while index < len(tokens):
            name, separator, _value = tokens[index].partition("=")
            if separator and name.replace("_", "a").isalnum() and not name[0].isdigit():
                index += 1
                continue
            break
        remaining = tokens[index:]
        if not remaining:
            continue
        executable = PurePosixPath(remaining[0]).name
        if executable in {"bash", "sh"}:
            script_index = 1
            while (
                script_index < len(remaining)
                and remaining[script_index].startswith("-")
            ):
                script_index += 1
            if script_index < len(remaining):
                add_launcher(remaining[script_index], suffixes={".sh"})
            continue
        if executable in {"python", "python3"} or executable.startswith("python3."):
            script_index = 1
            while script_index < len(remaining):
                token = remaining[script_index]
                if token == "-m":
                    break
                if token in {"-W", "-X"}:
                    script_index += 2
                    continue
                if token.startswith("-"):
                    script_index += 1
                    continue
                add_launcher(token, suffixes={".py"})
                break
            continue
        add_launcher(remaining[0], suffixes={".py", ".sh"})
    return result


def _scoring_oracle_candidates(repo: Path, configured: set[str]) -> set[str]:
    """Find test/control paths even when Git ignores or already staged them."""
    root = repo.resolve()
    result: set[str] = set(configured)
    for current, directories, filenames in os.walk(root, followlinks=False):
        current_path = Path(current)
        kept_directories: list[str] = []
        for name in directories:
            path = current_path / name
            relative = path.relative_to(root).as_posix()
            if path.is_symlink():
                if _is_scoring_oracle_path(relative):
                    result.add(relative)
                continue
            if name in _SCORING_SCAN_IGNORED_DIRS or name.endswith(".egg-info"):
                continue
            kept_directories.append(name)
        directories[:] = kept_directories
        for name in filenames:
            relative = (current_path / name).relative_to(root).as_posix()
            if _is_scoring_oracle_path(relative) or relative in configured:
                result.add(relative)
    return result


def _same_filesystem_entry(left: Path, right: Path) -> bool:
    if left.is_symlink() or right.is_symlink():
        return (
            left.is_symlink()
            and right.is_symlink()
            and os.readlink(left) == os.readlink(right)
        )
    if not left.is_file() or not right.is_file():
        return False
    try:
        return left.read_bytes() == right.read_bytes()
    except OSError:
        return False


def _restore_pristine_entry(source: Path, destination: Path) -> None:
    if destination.is_symlink() or (destination.exists() and not destination.is_dir()):
        destination.unlink()
    elif destination.is_dir():
        shutil.rmtree(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.is_symlink():
        destination.symlink_to(os.readlink(source))
    elif source.is_dir():
        shutil.copytree(source, destination, symlinks=True)
    elif source.is_file():
        shutil.copy2(source, destination)
    else:
        raise RuntimeError(f"unsupported pristine oracle entry: {source}")


def _remove_scoring_entry(path: Path) -> None:
    if path.is_symlink() or (path.exists() and not path.is_dir()):
        path.unlink(missing_ok=True)
    elif path.is_dir():
        shutil.rmtree(path)


def _prepare_safe_destination(
    repo: Path,
    relative: str,
    *,
    pristine: Path | None,
) -> set[str]:
    """Make destination parents local directories before any restore/remove.

    Every component is inspected without following it. Replacing an
    agent-created symlink or non-directory parent prevents checkout/copy/unlink
    operations from escaping through that component. A pristine oracle whose
    own parent is a symlink or file is unsupported and fails closed.
    """
    candidate = PurePosixPath(relative)
    if candidate.is_absolute() or ".." in candidate.parts or not candidate.parts:
        raise RuntimeError(f"unsafe scoring-oracle path: {relative!r}")

    replaced: set[str] = set()
    current = repo
    pristine_current = pristine
    for index, part in enumerate(candidate.parts[:-1], start=1):
        current /= part
        if pristine_current is not None:
            pristine_current /= part
            if pristine_current.is_symlink() or (
                pristine_current.exists() and not pristine_current.is_dir()
            ):
                raise RuntimeError(
                    "unsupported pristine test-oracle parent: "
                    f"{PurePosixPath(*candidate.parts[:index])}"
                )
        if current.is_symlink() or (current.exists() and not current.is_dir()):
            _remove_scoring_entry(current)
            current.mkdir()
            replaced.add(PurePosixPath(*candidate.parts[:index]).as_posix())
        elif not current.exists():
            current.mkdir()
    return replaced


def _remove_aliased_destination(path: Path) -> bool:
    """Unlink entries that could redirect or alias a subsequent restore."""
    if path.is_symlink() or path.is_dir():
        _remove_scoring_entry(path)
        return True
    try:
        if path.exists() and path.stat().st_nlink > 1:
            path.unlink()
            return True
    except OSError as exc:
        raise RuntimeError(
            f"failed to inspect scoring destination {path}: {exc}"
        ) from exc
    return False


def restore_test_oracle(
    repo: Path,
    *,
    pristine_repo: Path | None = None,
    test_commands: list[str] | None = None,
    excluded_agent_test_files: list[str] | None = None,
) -> list[str]:
    """Restore the pristine oracle and exclude agent-added tests from scoring.

    The agent patch has already been captured when this runs. Restoring here
    makes the visible suite independent of attempts to delete, replace, or
    weaken its tracked tests. Agent-added regression tests remain in that saved
    patch, but do not execute during visible/hidden scoring. Ordinary new
    production files remain in the checkout.
    """
    repo = repo.resolve()
    try:
        validate_local_git_config(repo)
    except RuntimeError as exc:
        raise RuntimeError(f"failed to inventory scoring checkout:\n{exc}") from exc
    pristine = pristine_repo.resolve() if pristine_repo is not None else None
    configured = _configured_test_control_paths(repo, test_commands or [])
    tracked = _git_nul_paths(repo, "ls-tree", "-r", "-z", "--name-only", "HEAD")
    changed = _git_nul_paths(repo, "diff", "--name-only", "-z", "HEAD", "--")
    tracked_tests = {
        path for path in tracked if _is_scoring_oracle_path(path) or path in configured
    }
    tampered = set(changed & tracked_tests)
    tracked_to_restore = sorted(changed & tracked_tests)
    for relative in tracked_to_restore:
        tampered.update(_prepare_safe_destination(repo, relative, pristine=pristine))
        if _remove_aliased_destination(repo / relative):
            tampered.add(relative)
    for start in range(0, len(tracked_to_restore), 200):
        chunk = tracked_to_restore[start : start + 200]
        restored = subprocess.run(
            safe_git_command(repo, "checkout", "HEAD", "--", *chunk),
            capture_output=True,
            text=True,
            check=False,
            env=safe_git_env(),
            timeout=60,
        )
        if restored.returncode != 0:
            raise RuntimeError(
                "failed to restore protected visible tests:\n" + restored.stderr[-1000:]
            )

    candidates = _scoring_oracle_candidates(repo, configured)
    if pristine is not None:
        candidates |= _scoring_oracle_candidates(pristine, configured)
    for relative in sorted(candidates - tracked_tests):
        destination = repo / relative
        source = pristine / relative if pristine is not None else None
        tampered.update(_prepare_safe_destination(repo, relative, pristine=pristine))
        if source is not None and (source.exists() or source.is_symlink()):
            if not _same_filesystem_entry(source, destination):
                tampered.add(relative)
            _restore_pristine_entry(source, destination)
            continue
        if not (destination.exists() or destination.is_symlink()):
            continue
        # Added control files are policy violations. Ordinary regression tests
        # are allowed in the archived patch, but are never part of the oracle.
        if relative in configured or _is_test_control_path(relative):
            tampered.add(relative)
        if (
            excluded_agent_test_files is not None
            and relative not in excluded_agent_test_files
        ):
            excluded_agent_test_files.append(relative)
        _remove_scoring_entry(destination)
    if pristine is not None:
        tampered.update(_restore_pytest_config_controls(repo, pristine))
    elif _pytest_config_candidates(repo):
        raise RuntimeError(
            "a pristine oracle is required to validate pytest configuration"
        )
    return sorted(tampered)


def freeze_test_oracle(
    repo: Path,
    destination: Path,
    *,
    test_commands: list[str],
) -> None:
    """Copy visible oracle/control bytes before any model-controlled action."""
    repo = repo.resolve()
    if destination.exists() or destination.is_symlink():
        raise RuntimeError(f"frozen oracle destination already exists: {destination}")
    destination.mkdir(parents=True)
    configured = _configured_test_control_paths(repo, test_commands)
    candidates = _scoring_oracle_candidates(repo, configured)
    candidates |= _pytest_config_candidates(repo)
    for relative in sorted(candidates):
        source = repo / relative
        if not (source.exists() or source.is_symlink()):
            continue
        _prepare_safe_destination(destination, relative, pristine=None)
        _restore_pristine_entry(source, destination / relative)


def scoring_oracle_manifest(
    repo: Path,
    *,
    pristine_repo: Path,
    test_commands: list[str],
) -> tuple[list[str], list[str]]:
    """Files and test-directory roots mounted read-only during scoring."""
    repo = repo.resolve()
    pristine = pristine_repo.resolve()
    configured = _configured_test_control_paths(repo, test_commands)
    tracked = _git_nul_paths(repo, "ls-tree", "-r", "-z", "--name-only", "HEAD")
    candidates = {
        path for path in tracked if _is_scoring_oracle_path(path) or path in configured
    }
    candidates |= _scoring_oracle_candidates(pristine, configured)

    files: list[str] = []
    directories: set[str] = set()
    for relative in sorted(candidates):
        destination = repo / relative
        if not destination.is_file() or destination.is_symlink():
            raise RuntimeError(
                f"scoring oracle must contain regular files after restore: {relative}"
            )
        files.append(relative)
        parts = PurePosixPath(relative).parts[:-1]
        for index, part in enumerate(parts, start=1):
            if part.lower() in TEST_ORACLE_DIR_NAMES:
                directories.add(PurePosixPath(*parts[:index]).as_posix())
                break
    return files, sorted(directories)


def prepare_hidden_oracle_directories(
    repo: Path,
    hidden: Path,
    *,
    pristine_repo: Path,
) -> tuple[list[str], list[str], list[str]]:
    """Prepare withheld destinations for read-only mounts without copying data.

    Hidden contents must remain outside the checkout until the visible suite
    has finished. Docker mounts, however, are fixed at container start. We
    therefore create only the empty destination directory roots, mount them
    read-only in-container, and copy the withheld bytes into the host bind
    source after visible scoring. No hidden content is exposed to the agent or
    dependency setup.

    Returns ``(fixture_files, readonly_directories, tampered_paths)``.
    Root-level hidden files are rejected: protecting them would require an
    observable placeholder or a second evaluator environment.
    """
    if not hidden.exists():
        return [], [], []
    repo = repo.resolve()
    hidden = hidden.resolve()
    pristine = pristine_repo.resolve()
    files: list[str] = []
    directories: set[str] = set()
    tampered: set[str] = set()
    for relative in _hidden_oracle_file_manifest(hidden):
        relative_path = PurePosixPath(relative)
        files.append(relative)
        tampered.update(_prepare_safe_destination(repo, relative, pristine=pristine))
        destination = repo / relative_path
        if destination.exists() or destination.is_symlink():
            try:
                destination_metadata = destination.lstat()
            except OSError as exc:
                raise RuntimeError(
                    f"hidden oracle destination is unavailable: {relative}: {exc}"
                ) from exc
            # Existing regular files are expected when a hidden patch augments
            # a visible test/support file. Keep their pristine/candidate bytes
            # for the visible run; the read-only directory mount freezes them
            # until the host overlays hidden bytes afterwards. Aliases and
            # non-files are unsafe and are removed before the mount is built.
            if (
                not stat.S_ISREG(destination_metadata.st_mode)
                or destination_metadata.st_nlink > 1
            ):
                tampered.add(relative)
                _remove_scoring_entry(destination)

        parent_parts = relative_path.parts[:-1]
        root_parts: tuple[str, ...] | None = None
        for index, part in enumerate(parent_parts, start=1):
            if part.lower() in TEST_ORACLE_DIR_NAMES:
                root_parts = parent_parts[:index]
                break
        if root_parts is None:
            root_parts = parent_parts
        if not root_parts:
            raise RuntimeError(
                f"root-level hidden oracle files cannot be mounted safely: {relative}"
            )
        directory = PurePosixPath(*root_parts).as_posix()
        destination_dir = repo / directory
        if not destination_dir.is_dir() or destination_dir.is_symlink():
            raise RuntimeError(
                f"hidden oracle destination is not a safe directory: {directory}"
            )
        directories.add(directory)
    return files, sorted(directories), sorted(tampered)


def _hidden_oracle_file_manifest(hidden: Path) -> list[str]:
    """Return the exact, canonical file list eligible for hidden overlay."""
    hidden = hidden.resolve()
    files: list[str] = []
    for current, directory_names, file_names in os.walk(hidden, followlinks=False):
        current_path = Path(current)
        kept_directories: list[str] = []
        for name in sorted(directory_names):
            source = current_path / name
            relative = source.relative_to(hidden).as_posix()
            if name.lower() in _HIDDEN_ORACLE_IGNORED_DIRS:
                continue
            try:
                metadata = source.lstat()
            except OSError as exc:
                raise RuntimeError(
                    f"hidden oracle fixture is unavailable: {relative}: {exc}"
                ) from exc
            if not stat.S_ISDIR(metadata.st_mode):
                raise RuntimeError(
                    "hidden oracle paths cannot traverse symlinked or "
                    f"non-directory parents: {relative}"
                )
            kept_directories.append(name)
        directory_names[:] = kept_directories

        for name in sorted(file_names):
            source = current_path / name
            relative = source.relative_to(hidden).as_posix()
            if source.suffix.lower() in _HIDDEN_ORACLE_IGNORED_SUFFIXES:
                continue
            try:
                metadata = source.lstat()
            except OSError as exc:
                raise RuntimeError(
                    f"hidden oracle fixture is unavailable: {relative}: {exc}"
                ) from exc
            if not stat.S_ISREG(metadata.st_mode):
                raise RuntimeError(
                    f"hidden oracle fixtures must be regular files: {relative}"
                )
            files.append(relative)
    return sorted(files)


def _validate_plain_parent_chain(root: Path, relative: str, *, label: str) -> None:
    """Reject symlink/non-directory parents without repairing or following them."""
    candidate = PurePosixPath(relative)
    if candidate.is_absolute() or ".." in candidate.parts or not candidate.parts:
        raise RuntimeError(f"unsafe hidden-oracle path: {relative!r}")
    current = root
    for index, part in enumerate(candidate.parts[:-1], start=1):
        current /= part
        try:
            metadata = current.lstat()
        except OSError as exc:
            raise RuntimeError(
                f"hidden oracle {label} parent is unavailable at "
                f"{PurePosixPath(*candidate.parts[:index])}: {exc}"
            ) from exc
        if not stat.S_ISDIR(metadata.st_mode):
            raise RuntimeError(
                f"hidden oracle {label} path traverses a symlink or "
                f"non-directory parent: {relative}"
            )


def copy_hidden_oracle(
    hidden: Path,
    repo: Path,
    expected_files: list[str],
) -> None:
    """Overlay and verify exactly the pre-inventoried withheld fixture bytes."""
    if not expected_files:
        return
    hidden = hidden.resolve()
    repo = repo.resolve()
    actual_files = _hidden_oracle_file_manifest(hidden)
    if actual_files != expected_files:
        raise RuntimeError(
            "hidden oracle contents changed after inventory; refusing an "
            "incomplete or expanded overlay"
        )
    for relative in expected_files:
        source = hidden / relative
        destination = repo / relative
        _validate_plain_parent_chain(hidden, relative, label="source")
        _validate_plain_parent_chain(repo, relative, label="destination")
        try:
            source_metadata = source.lstat()
        except OSError as exc:
            raise RuntimeError(
                f"hidden oracle overlay is incomplete at {relative}: {exc}"
            ) from exc
        if not stat.S_ISREG(source_metadata.st_mode):
            raise RuntimeError(
                f"hidden oracle source must remain a regular file: {relative}"
            )
        if destination.exists() or destination.is_symlink():
            destination_metadata = destination.lstat()
            if (
                not stat.S_ISREG(destination_metadata.st_mode)
                or destination_metadata.st_nlink > 1
            ):
                raise RuntimeError(
                    "hidden oracle destination became an unsafe alias or "
                    f"non-file after inventory: {relative}"
                )
        shutil.copy2(source, destination, follow_symlinks=False)
        destination_metadata = destination.lstat()
        if not stat.S_ISREG(destination_metadata.st_mode):
            raise RuntimeError(
                f"hidden oracle overlay must remain regular files: {relative}"
            )
        if source.read_bytes() != destination.read_bytes():
            raise RuntimeError(f"hidden oracle overlay verification failed: {relative}")


def _exact_visible_oracle_mounts(
    protected_files: list[str],
    hidden_fixture_files: list[str],
) -> list[str]:
    """Leave hidden-overlap paths visible through their read-only root mount."""
    hidden = set(hidden_fixture_files)
    return [path for path in protected_files if path not in hidden]


def _sandbox(
    repo: Path,
    image: str,
    network_disabled: bool,
    timeout: int,
    *,
    dependency_environment: Path | None = None,
    dependency_environment_readonly: bool = False,
) -> DockerSandbox:
    return DockerSandbox(
        workdir=repo,
        image=image,
        network_disabled=network_disabled,
        shell_timeout_seconds=timeout,
        test_timeout_seconds=timeout,
        dependency_environment=dependency_environment,
        dependency_environment_readonly=dependency_environment_readonly,
        evaluator_helper_path=_TRUSTED_PYTEST_HELPER,
        evaluator_bin_path=_EVALUATOR_BIN,
    )


def _setup_and_isolate(
    sandbox: DockerSandbox,
    setup_commands: list[str],
    shell_timeout: int,
    *,
    isolate_after_setup: bool,
) -> None:
    for setup in setup_commands:
        result = sandbox.run_shell(setup, shell_timeout)
        if not result.success:
            raise RuntimeError(
                f"Benchmark environment setup failed ({setup}):\n"
                f"{result.output[-1500:]}"
            )
    if isolate_after_setup:
        sandbox.disable_network()


@dataclass(frozen=True)
class _JUnitEvidence:
    passing: frozenset[str] = frozenset()
    collected: int = 0
    complete: bool = False
    failures: int = 0
    errors: int = 0
    skipped: int = 0
    error: str | None = None


def _junit_count(suite: ET.Element, field: str) -> int:
    raw = suite.get(field)
    if raw is None:
        raise ValueError(f"testsuite is missing the {field!r} count")
    value = int(raw)
    if value < 0:
        raise ValueError(f"testsuite has a negative {field!r} count")
    return value


def _parse_junit_evidence(source: Path | bytes | None) -> _JUnitEvidence:
    """Validate that a pytest JUnit report accounts for every collected case."""
    if source is None:
        return _JUnitEvidence(error="JUnit report is missing or is not a regular file")
    if isinstance(source, Path):
        if not source.is_file() or source.is_symlink():
            return _JUnitEvidence(
                error="JUnit report is missing or is not a regular file"
            )
        xml_source: Path | BytesIO = source
    else:
        xml_source = BytesIO(source)
    try:
        root = ET.parse(xml_source).getroot()
        suites = [
            suite
            for suite in root.iter("testsuite")
            if not any(child.tag == "testsuite" for child in suite)
        ]
        if root.tag == "testsuite" and not suites:
            suites = [root]
        if not suites:
            raise ValueError("JUnit report contains no testsuite")

        passing: set[str] = set()
        collected = 0
        observed_failures = 0
        observed_errors = 0
        observed_skipped = 0
        declared_failures = 0
        declared_errors = 0
        declared_skipped = 0
        for suite in suites:
            declared_tests = _junit_count(suite, "tests")
            declared_failures += _junit_count(suite, "failures")
            declared_errors += _junit_count(suite, "errors")
            declared_skipped += _junit_count(suite, "skipped")
            cases = [child for child in suite if child.tag == "testcase"]
            if declared_tests != len(cases):
                raise ValueError(
                    "JUnit testsuite declared "
                    f"{declared_tests} tests but contains {len(cases)} cases"
                )
            collected += len(cases)
            for case in cases:
                name = case.get("name")
                if not name:
                    raise ValueError("JUnit testcase is missing its name")
                outcomes = {
                    child.tag
                    for child in case
                    if child.tag in {"failure", "error", "skipped"}
                }
                if len(outcomes) > 1:
                    raise ValueError(
                        f"JUnit testcase has contradictory outcomes: {name}"
                    )
                if "failure" in outcomes:
                    observed_failures += 1
                elif "error" in outcomes:
                    observed_errors += 1
                elif "skipped" in outcomes:
                    observed_skipped += 1
                else:
                    passing.add(f"{case.get('classname', '')}::{name}")

        observed = (observed_failures, observed_errors, observed_skipped)
        declared = (declared_failures, declared_errors, declared_skipped)
        if observed != declared:
            raise ValueError(
                "JUnit outcome counts do not match testcase evidence: "
                f"declared={declared}, observed={observed}"
            )
        return _JUnitEvidence(
            passing=frozenset(passing),
            collected=collected,
            complete=True,
            failures=observed_failures,
            errors=observed_errors,
            skipped=observed_skipped,
        )
    except (ET.ParseError, OSError, TypeError, ValueError) as exc:
        return _JUnitEvidence(error=f"invalid or incomplete JUnit evidence: {exc}")


def _run_junit(
    sandbox: DockerSandbox,
    repo: Path,
    command: str,
    timeout: int,
    *,
    report_label: str,
    require_nonempty: bool,
) -> tuple[bool, set[str], str]:
    """Run one suite and require internally consistent evaluator evidence."""
    del repo  # Evidence stays in container-local tmpfs, outside the checkout.
    report_path = f"/tmp/repo-eval-{report_label}-{uuid.uuid4().hex}.xml"
    invocation_marker = (
        f"/tmp/repo-eval-{report_label}-{uuid.uuid4().hex}.invocation"
    )
    launched = trusted_evaluator_test_command(
        command,
        trusted_python=sandbox.resolve_trusted_python(),
        junit_path=report_path,
        invocation_marker=invocation_marker,
    )
    run = sandbox.run_shell(launched, timeout)
    invocation_evidence = sandbox.pop_isolated_file(invocation_marker)
    evidence = _parse_junit_evidence(sandbox.pop_isolated_file(report_path))

    evidence_error = evidence.error
    expected_minimum = (
        pytest_selection_lower_bound(command) if require_nonempty else None
    )
    if invocation_evidence != b"single\n":
        evidence_error = (
            "pytest invocation evidence is missing or the benchmark wrapper "
            "invoked pytest more than once"
        )
    elif require_nonempty and evidence.complete and evidence.collected == 0:
        evidence_error = "JUnit evidence contains zero collected tests"
    elif evidence.complete and (evidence.failures or evidence.errors):
        evidence_error = (
            "JUnit evidence contains failing outcomes "
            f"(failures={evidence.failures}, errors={evidence.errors})"
        )
    elif require_nonempty and evidence.complete and evidence.skipped:
        evidence_error = (
            "required hidden JUnit evidence contains skipped outcomes "
            f"(skipped={evidence.skipped})"
        )
    elif (
        evidence.complete
        and expected_minimum is not None
        and evidence.collected < expected_minimum
    ):
        evidence_error = (
            "JUnit evidence contains fewer executed cases than the configured "
            f"selectors ({evidence.collected} < {expected_minimum})"
        )
    elif require_nonempty and evidence.complete and not evidence.passing:
        evidence_error = "JUnit evidence contains no executed passing tests"
    output = run.output
    if evidence_error is not None:
        output = f"{output}\nEvaluator rejected test evidence: {evidence_error}".strip()
    passed = run.success and evidence.complete and evidence_error is None
    return passed, set(evidence.passing), output


def _run_visible_junit(
    sandbox: DockerSandbox, repo: Path, command: str, timeout: int
) -> tuple[bool, set[str], str]:
    """Run the visible suite with a JUnit report and return the passing ids.
    The report is removed afterwards so it never pollutes the repo/diff."""
    return _run_junit(
        sandbox,
        repo,
        command,
        timeout,
        report_label=_JUNIT_LABEL,
        require_nonempty=False,
    )


def _run_hidden_junit(
    sandbox: DockerSandbox, repo: Path, command: str, timeout: int
) -> tuple[bool, str]:
    """Run a hidden suite and reject absent, empty, or incomplete evidence."""
    passed, _passing, output = _run_junit(
        sandbox,
        repo,
        command,
        timeout,
        report_label=_HIDDEN_JUNIT_LABEL,
        require_nonempty=True,
    )
    return passed, output


def _repo_relative_hidden_command(command: str) -> str:
    # Hidden tests are copied into the repo before execution, so a command like
    # ``pytest ../hidden_tests/tests/x.py`` becomes ``pytest tests/x.py``.
    return command.replace("../hidden_tests/", "")


def _hidden_suite_label(name: str) -> str:
    return {
        "hidden": "Hidden tests",
        "hidden_semantic": "Hidden semantic tests",
        "hidden_compat": "Hidden compat tests",
        "hidden_pr_parity": "Hidden PR-parity tests",
    }.get(name, name.replace("_", " ").title())


def _set_hidden_result(result: EvalResult, name: str, passed: bool) -> None:
    result.hidden_suite_results[name] = passed
    if name == "hidden_semantic":
        result.hidden_semantic_passed = passed
    elif name == "hidden_compat":
        result.hidden_compat_passed = passed
    elif name == "hidden_pr_parity":
        result.hidden_pr_parity_passed = passed


def _record_runtime_pytest_config_tampering(
    result: EvalResult,
    repo: Path,
    pristine: Path,
    *,
    phase: str,
) -> None:
    """Restore config changed by code executed inside the previous suite."""
    tampered = sorted(_restore_pytest_config_controls(repo, pristine))
    if not tampered:
        return
    result.test_oracle_tampered = True
    print(f"\n=== {phase} pytest-config tampering detected ===")
    for path in tampered[:20]:
        print(f"  - {path}")


def collect_visible_passing(
    repo: Path,
    command: str,
    *,
    docker_image: str,
    network_disabled: bool,
    test_timeout: int,
    dependency_environment: Path,
    setup_artifact_environment: Path,
    pristine_repo: Path,
    test_commands: list[str],
) -> set[str]:
    """Baseline for the regression metric: visible tests passing on the
    pristine repo, before the agent touches it.

    Dependency setup has already completed in a separate sandbox. Restore the
    frozen setup delta and run this baseline with the same read-only evaluator
    inputs used by final scoring, so the before/after comparison is symmetric.
    """
    baseline_conflicts: list[str] = []
    restored_setup_artifacts = restore_setup_artifacts(
        repo,
        setup_artifact_environment,
        conflicts=baseline_conflicts,
    )
    if baseline_conflicts:
        raise RuntimeError(
            "pristine baseline was not cleaned before setup-artifact restore: "
            + ", ".join(sorted(set(baseline_conflicts))[:20])
        )
    tampered = restore_test_oracle(
        repo,
        pristine_repo=pristine_repo,
        test_commands=test_commands,
    )
    if tampered:
        raise RuntimeError(
            "pristine setup changed the frozen test oracle: "
            + ", ".join(tampered[:20])
        )
    protected_files, protected_directories = scoring_oracle_manifest(
        repo,
        pristine_repo=pristine_repo,
        test_commands=test_commands,
    )
    protected_files = sorted(set(protected_files) | set(restored_setup_artifacts))
    sandbox = _sandbox(
        repo,
        docker_image,
        network_disabled,
        test_timeout,
        dependency_environment=dependency_environment,
        dependency_environment_readonly=True,
    )
    sandbox.configure_protected_test_paths(
        protected_files,
        readonly_directories=protected_directories,
    )
    with tracing_context(enabled=False):
        try:
            sandbox.start()
            passed, passing, output = _run_visible_junit(
                sandbox, repo, command, test_timeout
            )
            if not passed:
                raise RuntimeError(
                    "Pristine visible benchmark tests failed before the agent:\n"
                    + output[-2000:]
                )
            return passing
        finally:
            sandbox.stop()


def prepare_scoring_dependencies(
    repo: Path,
    *,
    docker_image: str,
    network_disabled: bool,
    setup_commands: list[str],
    shell_timeout: int,
    test_timeout: int,
    dependency_environment: Path,
    setup_artifact_environment: Path | None = None,
) -> None:
    """Build a reusable evaluator environment from the pristine checkout."""
    before_setup = (
        setup_artifact_manifest(repo)
        if setup_artifact_environment is not None
        else None
    )
    before_pytest_configs = _pytest_config_candidates(repo)
    sandbox = _sandbox(
        repo,
        docker_image,
        network_disabled and not setup_commands,
        test_timeout,
        dependency_environment=dependency_environment,
    )
    with tracing_context(enabled=False):
        setup_succeeded = False
        try:
            sandbox.start()
            _setup_and_isolate(
                sandbox,
                setup_commands,
                shell_timeout,
                isolate_after_setup=network_disabled,
            )
            setup_succeeded = True
        finally:
            sandbox.stop()
        after_pytest_configs = _pytest_config_candidates(repo)
        if after_pytest_configs != before_pytest_configs:
            changed = sorted(after_pytest_configs ^ before_pytest_configs)
            raise RuntimeError(
                "pristine setup added or removed pytest configuration: "
                + ", ".join(changed[:20])
            )
        if setup_succeeded and setup_artifact_environment is not None:
            # Snapshot only after container teardown. A setup command may have
            # spawned background children; stopping the sandbox makes the host
            # checkout and dependency bind quiescent before hashing/copying.
            freeze_setup_artifacts(
                repo,
                setup_artifact_environment,
                before_setup=before_setup,
            )


def evaluate_solution(
    task: TaskSpec,
    task_ws: Path,
    visible_command: str,
    *,
    baseline_passing: set[str] | None,
    do_hidden: bool,
    docker_image: str,
    network_disabled: bool,
    setup_commands: list[str],
    shell_timeout: int,
    test_timeout: int,
    pristine_repo: Path | None = None,
    dependency_environment: Path | None = None,
    setup_artifact_environment: Path | None = None,
    setup_artifact_conflicts: list[str] | None = None,
) -> EvalResult:
    """Post-run evaluation in one sandbox: the after-state visible run (for
    regressions, before any overlay) and then the withheld hidden tests.

    Hidden tests are overlaid onto the repo package so their in-package
    relative imports resolve; the overlay happens only now, after the agent
    finished, so the tests never leak into its run.
    """
    repo = task_ws / "repo"
    hidden = task_ws / "hidden_tests"
    pristine = (
        pristine_repo.resolve()
        if pristine_repo is not None
        else task.repo_path.resolve()
    )
    result = EvalResult()

    artifact_conflicts = list(setup_artifact_conflicts or [])
    restored_setup_artifacts = []
    if setup_artifact_environment is not None:
        restored_setup_artifacts = restore_setup_artifacts(
            repo,
            setup_artifact_environment,
            conflicts=artifact_conflicts,
        )
    result.setup_artifact_conflicts = sorted(set(artifact_conflicts))
    result.setup_artifact_tampered = bool(result.setup_artifact_conflicts)
    if result.setup_artifact_tampered:
        print("\n=== Frozen setup-artifact conflict detected ===")
        for path in result.setup_artifact_conflicts[:20]:
            print(f"  - {path}")

    hidden_suites = task.hidden_suites() if do_hidden else []
    scoring_commands = [visible_command, *(suite.command for suite in hidden_suites)]
    excluded_agent_test_files: list[str] = []
    tampered = restore_test_oracle(
        repo,
        pristine_repo=pristine,
        test_commands=scoring_commands,
        excluded_agent_test_files=excluded_agent_test_files,
    )
    result.test_oracle_tampered = bool(tampered)
    result.excluded_agent_test_files = excluded_agent_test_files
    if tampered:
        print("\n=== Test oracle tampering detected ===")
        for path in tampered[:20]:
            print(f"  - {path}")
    if excluded_agent_test_files:
        print("\n=== Agent-added tests excluded from final scoring ===")
        for path in excluded_agent_test_files[:20]:
            print(f"  - {path}")

    hidden_fixture_files, hidden_directories, hidden_path_tampered = (
        prepare_hidden_oracle_directories(
            repo,
            hidden,
            pristine_repo=pristine,
        )
        if hidden_suites
        else ([], [], [])
    )
    if hidden_path_tampered:
        result.test_oracle_tampered = True
        print("\n=== Hidden oracle destination tampering detected ===")
        for path in hidden_path_tampered[:20]:
            print(f"  - {path}")

    sandbox = _sandbox(
        repo,
        docker_image,
        network_disabled
        if dependency_environment is not None
        else (network_disabled and not setup_commands),
        test_timeout,
        dependency_environment=dependency_environment,
        dependency_environment_readonly=dependency_environment is not None,
    )
    protected_files, protected_directories = scoring_oracle_manifest(
        repo,
        pristine_repo=pristine,
        test_commands=scoring_commands,
    )
    protected_files = sorted(set(protected_files) | set(restored_setup_artifacts))
    # An exact immutable file mount would keep showing the visible version
    # after the host overlays a hidden augmentation at the same path. Every
    # hidden fixture lives below a read-only directory mount, which already
    # protects its visible bytes from in-container mutation while allowing the
    # evaluator host to swap in the withheld version between suites.
    protected_files = _exact_visible_oracle_mounts(
        protected_files, hidden_fixture_files
    )
    sandbox.configure_protected_test_paths(
        protected_files,
        readonly_directories=sorted(
            set(protected_directories) | set(hidden_directories)
        ),
    )
    with tracing_context(enabled=False):
        try:
            sandbox.start()
            if dependency_environment is None:
                _setup_and_isolate(
                    sandbox,
                    setup_commands,
                    shell_timeout,
                    isolate_after_setup=network_disabled,
                )

            # Setup executes the submitted package metadata and therefore is
            # part of the adversarial surface. Exact baseline files and test
            # directories are already immutable in-container; this second
            # pass removes any newly introduced root-level controls and records
            # every attempted mutation before pytest collection begins.
            post_setup_tampered = restore_test_oracle(
                repo,
                pristine_repo=pristine,
                test_commands=scoring_commands,
                excluded_agent_test_files=excluded_agent_test_files,
            )
            if post_setup_tampered:
                result.test_oracle_tampered = True
                print("\n=== Setup-time test oracle tampering detected ===")
                for path in post_setup_tampered[:20]:
                    print(f"  - {path}")

            visible_passed, after, visible_output = _run_visible_junit(
                sandbox, repo, visible_command, test_timeout
            )
            _record_runtime_pytest_config_tampering(
                result,
                repo,
                pristine,
                phase="Visible-suite",
            )
            result.visible_passed = visible_passed
            verdict = "PASSED" if visible_passed else "FAILED"
            print(f"\n=== Visible tests: {verdict} ===")
            if not visible_passed:
                print(visible_output[-2000:])

            if baseline_passing is not None:
                regressed = sorted(baseline_passing - after)
                result.regressions = len(regressed)
                print(f"\n=== Regressions: {len(regressed)} ===")
                for test_id in regressed[:20]:
                    print(f"  - {test_id}")

            if hidden_suites:
                if hidden.exists():
                    copy_hidden_oracle(hidden, repo, hidden_fixture_files)

                required: list[bool] = []
                for suite in hidden_suites:
                    if suite.reuse_visible_result:
                        passed = bool(result.visible_passed)
                        output = (
                            "Compatibility result reuses the post-run visible "
                            "suite; this local PR task has no separately "
                            "reviewed hidden compatibility fixture."
                        )
                    else:
                        command = _repo_relative_hidden_command(suite.command)
                        passed, output = _run_hidden_junit(
                            sandbox,
                            repo,
                            command,
                            test_timeout,
                        )
                        _record_runtime_pytest_config_tampering(
                            result,
                            repo,
                            pristine,
                            phase=f"{_hidden_suite_label(suite.name)}",
                        )
                    _set_hidden_result(result, suite.name, passed)
                    if suite.required:
                        required.append(passed)
                    verdict = "PASSED" if passed else "FAILED"
                    label = _hidden_suite_label(suite.name)
                    print(f"\n=== {label}: {verdict} ===")
                    print(output[-2000:])

                if required:
                    result.hidden_passed = all(required)
                    verdict = "PASSED" if result.hidden_passed else "FAILED"
                    print(f"\n=== Required hidden tests: {verdict} ===")

            if result.hidden_passed is not None:
                result.task_success = task_success(
                    bool(result.visible_passed),
                    result.hidden_passed,
                    result.regressions,
                    result.test_oracle_tampered,
                    result.setup_artifact_tampered,
                )
        finally:
            sandbox.stop()
    return result
