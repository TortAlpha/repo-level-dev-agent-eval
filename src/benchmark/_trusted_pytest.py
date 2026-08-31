"""Bootstrap pytest before repository-controlled import paths are exposed.

This file is bind-mounted read-only into evaluator containers and executed
with ``python -I -S``.  It deliberately avoids normal ``site`` startup so a
frozen editable-install ``.pth`` file cannot execute code from the submitted
checkout before the evaluator has imported its own pytest runtime.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

_WORKSPACE = Path("/workspace")
_CRITICAL_IMPORT_ROOTS = frozenset(
    {"pytest", "_pytest", "pluggy", "iniconfig", "packaging"}
)


def _trusted_site_directories() -> list[Path]:
    version = f"python{sys.version_info.major}.{sys.version_info.minor}"
    patterns = [
        f"/home/agent/.local/lib/{version}/site-packages",
        f"{sys.prefix}/lib/{version}/site-packages",
        f"{sys.base_prefix}/lib/{version}/site-packages",
        f"/usr/local/lib/{version}/site-packages",
        f"/usr/lib/{version}/site-packages",
        f"/usr/lib/{version}/dist-packages",
        "/opt/*/lib/python*/site-packages",
    ]
    directories: list[Path] = []
    for pattern in patterns:
        for raw in glob.glob(pattern):
            candidate = Path(raw).resolve()
            if candidate.is_dir() and candidate not in directories:
                directories.append(candidate)
    return directories


def _is_within(path: Path, roots: list[Path]) -> bool:
    return any(path == root or root in path.parents for root in roots)


def _attest_loaded_runtime(trusted_roots: list[Path]) -> None:
    for module_name, module in tuple(sys.modules.items()):
        if module_name.partition(".")[0] not in _CRITICAL_IMPORT_ROOTS:
            continue
        raw_origin = getattr(module, "__file__", None)
        if not raw_origin:
            continue
        origin = Path(raw_origin).resolve()
        if not _is_within(origin, trusted_roots) or _is_within(origin, [_WORKSPACE]):
            raise RuntimeError(
                "untrusted pytest runtime module origin: "
                f"{module_name}={origin}"
            )


def _project_paths(encoded: str) -> list[str]:
    values = [str(_WORKSPACE)]
    conventional_source_root = (_WORKSPACE / "src").resolve()
    if conventional_source_root.is_dir():
        values.append(str(conventional_source_root))
    for raw in encoded.split(os.pathsep):
        if not raw:
            continue
        candidate = Path(raw)
        if not candidate.is_absolute():
            candidate = _WORKSPACE / candidate
        resolved = candidate.resolve()
        if resolved != _WORKSPACE and _WORKSPACE not in resolved.parents:
            raise RuntimeError(
                "configured pytest PYTHONPATH escapes the benchmark workspace: "
                f"{raw}"
            )
        value = str(resolved)
        if value not in values:
            values.append(value)
    return values


def main() -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--repo-eval-pythonpath", default="")
    parser.add_argument("pytest_args", nargs=argparse.REMAINDER)
    parsed = parser.parse_args()

    trusted_roots = _trusted_site_directories()
    if not trusted_roots:
        raise RuntimeError("no trusted Python package directory is available")
    # ``-S`` leaves only the interpreter/stdlib paths.  Appending immutable
    # package roots imports pytest without executing any .pth startup hooks.
    sys.path.extend(str(path) for path in trusted_roots)
    import pytest  # noqa: PLC0415 - ordering is the security boundary

    _attest_loaded_runtime(trusted_roots)
    # Project imports are necessary for the tests, but they become visible
    # only after the runner and its critical dependencies are resident.
    sys.path[0:0] = _project_paths(parsed.repo_eval_pythonpath)
    arguments = parsed.pytest_args
    if arguments[:1] == ["--"]:
        arguments = arguments[1:]
    return int(pytest.main(arguments))


if __name__ == "__main__":
    raise SystemExit(main())
