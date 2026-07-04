from __future__ import annotations

import subprocess
from pathlib import Path

from pydantic import BaseModel

from .sandbox import combine_output
from .tracing import trace_workspace_op

IGNORED_DIR_NAMES = frozenset({".git", "__pycache__", ".pytest_cache"})


def _is_ignored(path: Path) -> bool:
    return any(part in IGNORED_DIR_NAMES for part in path.parts)


class Workspace(BaseModel):
    """File and search operations confined to one repository checkout.

    Every path is validated to stay inside the workspace root, so an agent can
    only read and modify the repository it was given.
    """

    root: Path

    def resolve(self, path: str) -> Path:
        if Path(path).is_absolute():
            raise ValueError(f"Absolute paths are not allowed: {path}")

        root = self.root.resolve()
        resolved = (root / path).resolve()
        if not resolved.is_relative_to(root):
            raise ValueError(f"Path escapes workspace: {path}")
        return resolved

    def to_relative(self, path: Path) -> str:
        return str(path.relative_to(self.root.resolve()))

    @trace_workspace_op("Workspace.read_file")
    def read_file(self, path: str) -> str:
        file_path = self.resolve(path)
        if not file_path.is_file():
            raise ValueError(f"File does not exist: {path}")
        return file_path.read_text(encoding="utf-8", errors="replace")

    @trace_workspace_op("Workspace.write_file")
    def write_file(self, path: str, content: str) -> str:
        file_path = self.resolve(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding="utf-8")
        return self.to_relative(file_path)

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
        file_path.write_text(new_content, encoding="utf-8")
        return self.to_relative(file_path), count

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
    def search(self, query: str, path: str = ".", max_results: int = 80) -> str:
        search_root = self.resolve(path)
        if not search_root.exists():
            raise ValueError(f"Search path does not exist: {path}")

        try:
            result = subprocess.run(
                ["rg", "-n", "--no-heading", query, str(search_root)],
                capture_output=True,
                check=False,
                text=True,
                timeout=30,
            )
        except FileNotFoundError:
            return self._python_search(search_root, query, max_results)

        if result.returncode not in (0, 1):
            output = combine_output(result.stdout, result.stderr)
            raise ValueError(f"Search failed:\n{output}")

        output = result.stdout.strip()
        if not output:
            return "No matches."

        lines = output.splitlines()
        suffix = "" if len(lines) <= max_results else "\n... truncated ..."
        return "\n".join(lines[:max_results]) + suffix

    def _python_search(self, root: Path, query: str, max_results: int) -> str:
        matches: list[str] = []

        for path in root.rglob("*"):
            if len(matches) >= max_results:
                break
            if not path.is_file() or _is_ignored(path):
                continue

            try:
                lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
            except OSError:
                continue

            for line_number, line in enumerate(lines, start=1):
                if query in line:
                    matches.append(f"{path}:{line_number}:{line}")
                    if len(matches) >= max_results:
                        break

        if not matches:
            return "No matches."
        return "\n".join(matches)
