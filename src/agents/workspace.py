from __future__ import annotations

import hashlib
import posixpath
import re
import stat
import subprocess
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, PrivateAttr

from ..git_safety import safe_git_command, safe_git_env, validate_local_git_config
from .sandbox import combine_output
from .tracing import trace_workspace_op

IGNORED_DIR_NAMES = frozenset(
    {
        ".git",
        ".hg",
        ".mypy_cache",
        ".next",
        ".pytest_cache",
        ".ruff_cache",
        ".svn",
        ".tox",
        ".venv",
        ".eggs",
        "__pycache__",
        "build",
        "coverage",
        "dist",
        "node_modules",
        "target",
        "vendor",
        "venv",
    }
)

TEST_ORACLE_DIR_NAMES = frozenset(
    {"__tests__", "spec", "specs", "test", "tests", "testing"}
)
TEST_ORACLE_CONTROL_NAMES = frozenset(
    {
        "conftest.py",
        "noxfile.py",
        "run_tests.py",
        "run_tests.sh",
        "run_tests_checked.sh",
    }
)
VCS_METADATA_DIR_NAMES = frozenset({".git", ".hg", ".svn"})

SearchMode = Literal["auto", "regex", "literal"]
_SNAPSHOT_CHUNK_BYTES = 1024 * 1024


@dataclass(frozen=True)
class WorktreeSnapshot:
    """Bounded content snapshot used around shell actions.

    Git status alone cannot notice a second edit to a file which was already
    dirty. Content fingerprints make every persistent shell mutation visible
    to the shared executor without storing file contents.
    """

    files: dict[str, str]
    head: str

    def changed_paths(self, other: WorktreeSnapshot) -> list[str]:
        return sorted(
            path
            for path in self.files.keys() | other.files.keys()
            if self.files.get(path) != other.files.get(path)
        )


def _snapshot_fingerprint(path: Path, remaining_bytes: int) -> tuple[str, int]:
    """Hash one path without loading an arbitrarily large file into memory."""
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        metadata = None
    except OSError as exc:
        raise ValueError(f"Failed to inspect {path}: {exc}") from exc

    mode = metadata.st_mode if metadata is not None else 0
    digest = hashlib.sha256()
    if metadata is None:
        digest.update(b"missing\0")
        digest.update(b"0\0")
        return digest.hexdigest(), 0
    if stat.S_ISLNK(mode):
        try:
            payload = str(path.readlink()).encode(
                "utf-8", errors="surrogateescape"
            )
        except OSError as exc:
            raise ValueError(f"Failed to read symlink {path}: {exc}") from exc
        if len(payload) > remaining_bytes:
            raise ValueError("Workspace snapshot exceeds the byte safety limit.")
        digest.update(b"symlink\0")
        digest.update(str(mode).encode("ascii") + b"\0")
        digest.update(payload)
        return digest.hexdigest(), len(payload)
    if not stat.S_ISREG(mode):
        digest.update(b"missing\0")
        digest.update(str(mode).encode("ascii") + b"\0")
        return digest.hexdigest(), 0
    if metadata.st_size > remaining_bytes:
        raise ValueError("Workspace snapshot exceeds the byte safety limit.")

    digest.update(b"file\0")
    digest.update(str(mode).encode("ascii") + b"\0")
    consumed = 0
    try:
        with path.open("rb") as handle:
            while chunk := handle.read(_SNAPSHOT_CHUNK_BYTES):
                consumed += len(chunk)
                if consumed > remaining_bytes:
                    raise ValueError(
                        "Workspace snapshot exceeds the byte safety limit."
                    )
                digest.update(chunk)
    except ValueError:
        raise
    except OSError as exc:
        raise ValueError(f"Failed to snapshot {path}: {exc}") from exc
    return digest.hexdigest(), consumed


def _is_ignored(path: Path) -> bool:
    return any(
        part in IGNORED_DIR_NAMES or part.endswith(".egg-info")
        for part in path.parts
    )


def is_test_oracle_path(path: str) -> bool:
    """Whether a tracked path belongs to the visible benchmark-test oracle."""
    candidate = Path(path)
    directories = {part.lower() for part in candidate.parts[:-1]}
    name = candidate.name.lower()
    return (
        bool(directories & TEST_ORACLE_DIR_NAMES)
        or name.startswith("test_")
        or name in TEST_ORACLE_CONTROL_NAMES
        or name.endswith("_test.go")
        or name.endswith("_test.py")
        or ".test." in name
        or "_spec." in name
        or name.endswith("test.java")
        or name.endswith("tests.java")
        or name.endswith("test.kt")
        or name.endswith("tests.cs")
    )


def _literal_match_starts(content: str, literal: str) -> Iterator[int]:
    """Yield literal match offsets lazily so fallback search stays bounded."""
    if not literal:
        return
    start = 0
    while True:
        found = content.find(literal, start)
        if found < 0:
            return
        yield found
        start = found + len(literal)


def _reject_if_broken_python(rel_path: str, content: str) -> None:
    """SWE-agent-style guard: reject a write/edit that would leave a Python file
    syntactically invalid, so a bad edit is surfaced instead of silently
    breaking the file (the file on disk is left unchanged)."""
    if not rel_path.endswith(".py"):
        return
    try:
        compile(content, rel_path, "exec")
    except SyntaxError as exc:
        where = f" (line {exc.lineno})" if exc.lineno else ""
        raise ValueError(
            f"Rejected: this change would introduce a Python syntax error in "
            f"{rel_path}: {exc.msg}{where}. The file was left unchanged — fix "
            f"the change and try again."
        ) from exc


class Workspace(BaseModel):
    """File and search operations confined to one repository checkout.

    Every path is validated to stay inside the workspace root, so an agent can
    only read and modify the repository it was given.
    """

    root: Path
    _baseline_commit: str | None = PrivateAttr(default=None)
    _git_config_validated: bool = PrivateAttr(default=False)

    def _validate_git_boundary(self) -> None:
        if not self._git_config_validated:
            validate_local_git_config(self.root.resolve())
            self._git_config_validated = True

    @staticmethod
    def search_key(
        query: str,
        path: str = ".",
        max_results: int = 80,
        mode: SearchMode = "auto",
    ) -> str:
        """Stable cache key for searches whose result survives compaction.

        Query bytes are part of search semantics. In particular, whitespace
        and ``\\|`` mean different things in literal/regex modes. Auto-mode
        retries are an execution detail and must not alias two requests in the
        durable cache.
        """
        normalized_path = posixpath.normpath(Path(path).as_posix())
        return f"{mode}\n{max_results}\n{normalized_path}\n{query}"

    def resolve(self, path: str) -> Path:
        candidate = Path(path)
        if candidate.is_absolute():
            raise ValueError(f"Absolute paths are not allowed: {path}")
        if any(part.lower() in VCS_METADATA_DIR_NAMES for part in candidate.parts):
            raise ValueError(f"VCS metadata is outside the agent workspace: {path}")

        root = self.root.resolve()
        resolved = (root / path).resolve()
        if not resolved.is_relative_to(root):
            raise ValueError(f"Path escapes workspace: {path}")
        relative = resolved.relative_to(root)
        if any(part.lower() in VCS_METADATA_DIR_NAMES for part in relative.parts):
            raise ValueError(f"VCS metadata is outside the agent workspace: {path}")
        return resolved

    def to_relative(self, path: Path) -> str:
        return path.resolve().relative_to(self.root.resolve()).as_posix()

    def _initial_commit(self) -> str:
        if self._baseline_commit is None:
            self._validate_git_boundary()
            result = subprocess.run(
                safe_git_command(self.root, "rev-parse", "HEAD"),
                capture_output=True,
                check=False,
                text=True,
                env=safe_git_env(),
                timeout=30,
            )
            if result.returncode != 0:
                raise ValueError(f"Workspace is not a Git checkout: {self.root}")
            self._baseline_commit = result.stdout.strip()
        return self._baseline_commit

    def is_git_checkout(self) -> bool:
        try:
            self._validate_git_boundary()
        except RuntimeError:
            return False
        result = subprocess.run(
            safe_git_command(self.root, "rev-parse", "--is-inside-work-tree"),
            capture_output=True,
            check=False,
            text=True,
            env=safe_git_env(),
            timeout=30,
        )
        return result.returncode == 0 and result.stdout.strip() == "true"

    def is_tracked(self, path: str) -> bool:
        """Whether ``path`` belongs to the pristine Git checkout (HEAD/index)."""
        file_path = self.resolve(path)
        rel_path = self.to_relative(file_path)
        if not self.is_git_checkout():
            return False
        try:
            baseline = self._initial_commit()
        except ValueError:
            # A newly initialized repository has no pristine tree yet.
            return False
        result = subprocess.run(
            safe_git_command(self.root, "cat-file", "-e", f"{baseline}:{rel_path}"),
            capture_output=True,
            check=False,
            text=True,
            env=safe_git_env(),
            timeout=30,
        )
        return result.returncode == 0

    def baseline_tracked_files(self) -> list[str]:
        """Files in the immutable commit captured at the start of this run."""
        if not self.is_git_checkout():
            return []
        try:
            baseline = self._initial_commit()
        except ValueError:
            return []
        result = subprocess.run(
            safe_git_command(
                self.root,
                "ls-tree",
                "-r",
                "-z",
                "--name-only",
                baseline,
            ),
            capture_output=True,
            check=False,
            env=safe_git_env(),
            timeout=60,
        )
        if result.returncode != 0:
            raise ValueError("Failed to enumerate files in the baseline Git tree.")
        return [
            name.decode("utf-8", errors="surrogateescape")
            for name in result.stdout.split(b"\0")
            if name
        ]

    def _reject_aliased_write_target(self, path: str) -> None:
        """Reject writes through symlinks or pre-existing hardlink aliases.

        File actions execute on the host, outside Docker's read-only oracle
        mounts. Without this check, an innocently named alias could target a
        protected test while bypassing the lexical policy in the executor.
        """
        candidate = Path(path)
        if candidate.is_absolute() or ".." in candidate.parts:
            raise ValueError(f"Unsafe write path: {path}")

        current = self.root.resolve()
        for part in candidate.parts:
            if part in ("", "."):
                continue
            current = current / part
            try:
                metadata = current.lstat()
            except FileNotFoundError:
                continue
            except OSError as exc:
                raise ValueError(f"Failed to inspect write path {path}: {exc}") from exc
            if stat.S_ISLNK(metadata.st_mode):
                raise ValueError(f"Writes through symlinks are not allowed: {path}")

        try:
            metadata = (self.root.resolve() / candidate).lstat()
        except FileNotFoundError:
            return
        except OSError as exc:
            raise ValueError(f"Failed to inspect write target {path}: {exc}") from exc
        if stat.S_ISREG(metadata.st_mode) and metadata.st_nlink > 1:
            raise ValueError(f"Writes through hardlinked files are not allowed: {path}")

    def restore_tracked_files(self, paths: list[str]) -> None:
        """Restore protected tracked files from the run's original commit."""
        if not paths:
            return
        for start in range(0, len(paths), 200):
            chunk = paths[start : start + 200]
            result = subprocess.run(
                safe_git_command(
                    self.root,
                    "checkout",
                    self._initial_commit(),
                    "--",
                    *chunk,
                ),
                capture_output=True,
                check=False,
                text=True,
                env=safe_git_env(),
                timeout=60,
            )
            if result.returncode != 0:
                raise ValueError(
                    "Failed to restore protected tests: " + result.stderr.strip()
                )

    def restore_git_head(self) -> None:
        """Undo a shell action's commit/checkout while preserving file edits."""
        result = subprocess.run(
            safe_git_command(
                self.root,
                "reset",
                "--mixed",
                self._initial_commit(),
            ),
            capture_output=True,
            check=False,
            text=True,
            env=safe_git_env(),
            timeout=60,
        )
        if result.returncode != 0:
            raise ValueError("Failed to restore Git HEAD: " + result.stderr.strip())

    def worktree_snapshot(
        self,
        *,
        max_files: int = 25_000,
        max_bytes: int = 512 * 1024 * 1024,
    ) -> WorktreeSnapshot:
        """Fingerprint tracked and untracked repository files, with hard caps."""
        root = self.root.resolve()
        # Capture the immutable oracle/base before an untrusted shell command
        # has any opportunity to move HEAD.
        self._initial_commit()
        tracked_result = subprocess.run(
            safe_git_command(root, "ls-files", "-z", "--cached"),
            capture_output=True,
            check=False,
            env=safe_git_env(),
            timeout=60,
        )
        untracked_result = subprocess.run(
            safe_git_command(
                root,
                "ls-files",
                "-z",
                "--others",
                "--exclude-standard",
            ),
            capture_output=True,
            check=False,
            env=safe_git_env(),
            timeout=60,
        )
        if tracked_result.returncode != 0 or untracked_result.returncode != 0:
            raise ValueError("Failed to snapshot workspace with git ls-files.")

        tracked_names = [
            name.decode("utf-8", errors="surrogateescape")
            for name in tracked_result.stdout.split(b"\0")
            if name
        ]
        untracked_names = [
            name.decode("utf-8", errors="surrogateescape")
            for name in untracked_result.stdout.split(b"\0")
            if name
        ]
        # Ignore generated/dependency trees only when they are untracked.
        # Every tracked file remains part of the mutation and oracle boundary,
        # even in repositories that intentionally version vendor/build trees.
        names = sorted(
            set(tracked_names)
            | {name for name in untracked_names if not _is_ignored(Path(name))}
        )
        if len(names) > max_files:
            raise ValueError(
                f"Workspace snapshot exceeds the {max_files}-file safety limit."
            )

        total_bytes = 0
        fingerprints: dict[str, str] = {}
        for name in names:
            relative = Path(name)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"Unsafe path returned by git ls-files: {name}")
            # Do not call resolve(): tracked symlinks may intentionally point
            # outside the checkout, and snapshotting must hash the link itself
            # without dereferencing it.
            path = root / relative
            fingerprint, consumed = _snapshot_fingerprint(
                path, max_bytes - total_bytes
            )
            total_bytes += consumed
            fingerprints[name] = fingerprint

        head_result = subprocess.run(
            safe_git_command(root, "rev-parse", "HEAD"),
            capture_output=True,
            check=False,
            text=True,
            env=safe_git_env(),
            timeout=30,
        )
        head = head_result.stdout.strip() if head_result.returncode == 0 else ""
        return WorktreeSnapshot(files=fingerprints, head=head)

    @trace_workspace_op("Workspace.read_file")
    def read_file(self, path: str) -> str:
        file_path = self.resolve(path)
        if not file_path.is_file():
            raise ValueError(f"File does not exist: {path}")
        return file_path.read_text(encoding="utf-8", errors="replace")

    @trace_workspace_op("Workspace.write_file")
    def write_file(self, path: str, content: str) -> str:
        self._reject_aliased_write_target(path)
        file_path = self.resolve(path)
        rel_path = self.to_relative(file_path)
        _reject_if_broken_python(rel_path, content)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding="utf-8")
        return rel_path

    @trace_workspace_op("Workspace.edit_file")
    def edit_file(
        self,
        path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> tuple[str, int]:
        if old_string == new_string:
            raise ValueError("old_string and new_string must differ.")
        if not old_string:
            raise ValueError("old_string must not be empty.")

        self._reject_aliased_write_target(path)
        file_path = self.resolve(path)
        if not file_path.is_file():
            raise ValueError(f"File does not exist: {path}")

        content = file_path.read_text(encoding="utf-8", errors="replace")
        count = content.count(old_string)
        if count == 0:
            raise ValueError(f"old_string not found in {path}.")
        if not replace_all and count > 1:
            raise ValueError(
                f"old_string is not unique in {path} ({count} occurrences). "
                "Include more surrounding context or set replace_all=true."
            )

        new_count = -1 if replace_all else 1
        new_content = content.replace(old_string, new_string, new_count)
        rel_path = self.to_relative(file_path)
        _reject_if_broken_python(rel_path, new_content)
        file_path.write_text(new_content, encoding="utf-8")
        return rel_path, count

    @trace_workspace_op("Workspace.list_files")
    def list_files(self, path: str = ".") -> str:
        directory = self.resolve(path)
        if not directory.exists():
            raise ValueError(f"Path does not exist: {path}")

        base = self.root.resolve()
        paths = sorted(
            p.relative_to(base).as_posix()
            for p in directory.rglob("*")
            if p.is_file() and not _is_ignored(p)
        )
        if not paths:
            return "No files found."
        return "\n".join(paths)

    @trace_workspace_op("Workspace.search")
    def search(
        self,
        query: str,
        path: str = ".",
        max_results: int = 80,
        mode: SearchMode = "auto",
    ) -> str:
        search_root = self.resolve(path)
        if not search_root.exists():
            raise ValueError(f"Search path does not exist: {path}")

        multiline = "\n" in query or "\r" in query

        relative_root = self.to_relative(search_root) or "."

        def run_rg(*, fixed_strings: bool = False) -> subprocess.CompletedProcess[str]:
            command = ["rg", "-n", "--no-heading", "--with-filename"]
            if multiline:
                # ripgrep rejects a literal newline unless multiline mode is
                # enabled. Agents often paste a short code block as a query.
                command.append("--multiline")
            if (
                fixed_strings
                or mode == "literal"
                or (multiline and mode == "auto")
            ):
                # A pasted multi-line code fragment is almost always a
                # literal lookup. Treat it as such: otherwise regex escapes
                # inside the fragment can change its meaning or yield no
                # match even though multiline mode accepted the pattern.
                command.append("--fixed-strings")
            return subprocess.run(
                [*command, query, relative_root],
                capture_output=True,
                check=False,
                text=True,
                timeout=30,
                cwd=self.root.resolve(),
            )

        try:
            result = run_rg()
            if (
                mode == "auto"
                and result.returncode not in (0, 1)
                and "regex parse error" in result.stderr
            ):
                # Models routinely search for literal code (`def parse(self,`),
                # which is rarely valid regex. Retry literally instead of
                # burning the agent's step on a pattern syntax error.
                result = run_rg(fixed_strings=True)
            # Basic grep uses ``\|`` for alternation; ripgrep/Rust regex uses
            # bare ``|`` and treats the escaped form as a literal pipe. Retry
            # the model's likely intent only after an empty valid result, so a
            # real literal-pipe match still wins.
            if mode == "auto" and result.returncode == 1 and r"\|" in query:
                original_query = query
                query = query.replace(r"\|", "|")
                result = run_rg()
                if (
                    result.returncode not in (0, 1)
                    and "regex parse error" in result.stderr
                ):
                    result = run_rg(fixed_strings=True)
                query = original_query
        except FileNotFoundError:
            return self._python_search(search_root, query, max_results, mode)

        if result.returncode not in (0, 1):
            output = combine_output(result.stdout, result.stderr)
            raise ValueError(f"Search failed:\n{output}")

        output = result.stdout.strip()
        if not output:
            return "No matches."

        lines = output.splitlines()
        suffix = "" if len(lines) <= max_results else "\n... truncated ..."
        return "\n".join(lines[:max_results]) + suffix

    def _python_search(
        self,
        root: Path,
        query: str,
        max_results: int,
        mode: SearchMode = "auto",
    ) -> str:
        multiline = "\n" in query or "\r" in query
        pattern: re.Pattern[str] | None = None
        if mode == "regex" or (mode == "auto" and not multiline):
            try:
                pattern = re.compile(query)
            except re.error as exc:
                if mode == "regex":
                    raise ValueError(f"Search failed: invalid regex: {exc}") from exc
                # Auto mode mirrors ripgrep: malformed model-generated regex
                # is retried as literal text rather than burning a step.
                pattern = None

        def collect(compiled: re.Pattern[str] | None, literal: str) -> list[str]:
            matches: list[str] = []
            for path in root.rglob("*"):
                if len(matches) >= max_results:
                    break
                # Match ripgrep's default confinement: do not follow a file
                # symlink that can point outside the repository.
                if path.is_symlink() or not path.is_file() or _is_ignored(path):
                    continue

                try:
                    content = path.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    continue

                lines = content.splitlines()
                if multiline:
                    starts: Iterator[int]
                    if compiled is not None:
                        starts = (match.start() for match in compiled.finditer(content))
                    else:
                        starts = _literal_match_starts(content, literal)
                    for match_start in starts:
                        line_number = content.count("\n", 0, match_start) + 1
                        line = (
                            lines[min(line_number - 1, len(lines) - 1)]
                            if lines
                            else ""
                        )
                        rel_path = self.to_relative(path)
                        matches.append(f"{rel_path}:{line_number}:{line}")
                        if len(matches) >= max_results:
                            break
                    continue

                for line_number, line in enumerate(lines, start=1):
                    matched = (
                        compiled.search(line) is not None
                        if compiled is not None
                        else literal in line
                    )
                    if matched:
                        rel_path = self.to_relative(path)
                        matches.append(f"{rel_path}:{line_number}:{line}")
                        if len(matches) >= max_results:
                            break
            return matches

        matches = collect(pattern, query)
        if mode == "auto" and not matches and r"\|" in query:
            normalized = query.replace(r"\|", "|")
            try:
                matches = collect(re.compile(normalized), normalized)
            except re.error:
                matches = collect(None, query)

        if not matches:
            return "No matches."
        return "\n".join(matches)
