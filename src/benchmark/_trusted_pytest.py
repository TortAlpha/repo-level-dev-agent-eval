"""Bootstrap pytest before repository-controlled import paths are exposed.

This file is bind-mounted read-only into evaluator containers and executed
with ``python -I -S``. It deliberately avoids normal ``site`` startup so a
frozen editable-install ``.pth`` file cannot execute submitted code before the
evaluator has imported and attested its pytest runtime.

The pluggy repository is a self-hosting edge case: pytest needs pluggy while
the tests must observe the candidate pluggy. For that one distribution,
frozen editable metadata selects an exact source root. Pytest first imports
against a separate pristine package snapshot; the pristine canonical pluggy
modules are then removed before test collection exposes the candidate root.
"""

from __future__ import annotations

import argparse
import base64
import csv
import glob
import hashlib
import importlib.util
import json
import os
import re
import stat
import sys
from pathlib import Path, PurePosixPath
from types import ModuleType
from urllib.parse import unquote, urlsplit

_WORKSPACE = Path("/workspace")
_BOOTSTRAP = Path("/tmp/repo-eval-bootstrap")
_CRITICAL_IMPORT_ROOTS = frozenset(
    {"pytest", "_pytest", "pluggy", "iniconfig", "packaging"}
)
_EDITABLE_BOOTSTRAP_ROOTS = frozenset({"pluggy"})


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


def _regular_file(path: Path) -> bool:
    try:
        metadata = path.lstat()
    except OSError:
        return False
    return stat.S_ISREG(metadata.st_mode) and metadata.st_nlink == 1


def _module_origin(module_name: str, module: object) -> Path:
    raw_file = getattr(module, "__file__", None)
    spec = getattr(module, "__spec__", None)
    raw_origin = getattr(spec, "origin", None)
    loader = getattr(spec, "loader", None)
    if not raw_file or not raw_origin or loader is None:
        raise RuntimeError(
            f"critical pytest runtime module lacks concrete provenance: {module_name}"
        )
    file_origin = Path(str(raw_file)).resolve()
    spec_origin = Path(str(raw_origin)).resolve()
    if file_origin != spec_origin:
        raise RuntimeError(
            "critical pytest runtime module has inconsistent provenance: "
            f"{module_name}={file_origin} != {spec_origin}"
        )
    return file_origin


def _attest_loaded_runtime(
    trusted_roots: list[Path],
    *,
    bootstrap_roots: frozenset[str] = frozenset(),
) -> None:
    trusted = [path.resolve() for path in trusted_roots]
    bootstrap = _BOOTSTRAP.resolve()
    workspace = _WORKSPACE.resolve()
    for module_name, module in tuple(sys.modules.items()):
        root_name = module_name.partition(".")[0]
        if root_name not in _CRITICAL_IMPORT_ROOTS:
            continue
        origin = _module_origin(module_name, module)
        trusted_origin = _is_within(origin, trusted) and not _is_within(
            origin, [workspace]
        )
        bootstrap_origin = root_name in bootstrap_roots and _is_within(
            origin, [bootstrap]
        )
        if not trusted_origin and not bootstrap_origin:
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


def _normalized_distribution(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).lower()


def _metadata_name(path: Path) -> str | None:
    if not _regular_file(path) or path.stat().st_size > 1024 * 1024:
        return None
    try:
        names = [
            line.partition(":")[2].strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.lower().startswith("name:")
        ]
    except (OSError, UnicodeError):
        return None
    return names[0] if len(names) == 1 and names[0] else None


def _recorded_simple_path(
    metadata_dir: Path,
    trusted_root: Path,
    *,
    workspace: Path,
) -> Path:
    record = metadata_dir / "RECORD"
    if not _regular_file(record) or record.stat().st_size > 4 * 1024 * 1024:
        raise RuntimeError(
            f"editable pytest bootstrap metadata lacks a bounded RECORD: {metadata_dir}"
        )
    try:
        rows = list(csv.reader(record.read_text(encoding="utf-8").splitlines()))
    except (csv.Error, OSError, UnicodeError) as exc:
        raise RuntimeError(
            f"editable pytest bootstrap RECORD is invalid: {metadata_dir}"
        ) from exc
    candidates: list[Path] = []
    for row in rows:
        if not row or not row[0].endswith(".pth"):
            continue
        if len(row) != 3:
            raise RuntimeError("editable pytest bootstrap .pth RECORD row is invalid")
        relative = PurePosixPath(row[0])
        if relative.is_absolute() or ".." in relative.parts or len(relative.parts) != 1:
            raise RuntimeError("editable pytest bootstrap .pth path is unsafe")
        source = trusted_root / relative.as_posix()
        if not _regular_file(source) or source.stat().st_size > 4096:
            raise RuntimeError("editable pytest bootstrap .pth is not a regular file")
        data = source.read_bytes()
        expected_hash = "sha256=" + base64.urlsafe_b64encode(
            hashlib.sha256(data).digest()
        ).rstrip(b"=").decode("ascii")
        try:
            expected_size = int(row[2])
        except ValueError as exc:
            raise RuntimeError(
                "editable pytest bootstrap .pth size is invalid"
            ) from exc
        if row[1] != expected_hash or expected_size != len(data):
            raise RuntimeError("editable pytest bootstrap .pth RECORD mismatch")
        try:
            lines = source.read_text(encoding="utf-8").splitlines()
        except UnicodeError as exc:
            raise RuntimeError("editable pytest bootstrap .pth is not UTF-8") from exc
        if (
            len(lines) != 1
            or not lines[0]
            or lines[0] != lines[0].strip()
            or lines[0].startswith("import")
        ):
            raise RuntimeError(
                "editable pytest bootstrap requires one inert absolute .pth path"
            )
        raw_path = Path(lines[0])
        if not raw_path.is_absolute():
            raise RuntimeError("editable pytest bootstrap .pth path is not absolute")
        if ".." in raw_path.parts:
            raise RuntimeError("editable pytest bootstrap .pth path is not exact")
        source_root = raw_path
        workspace = workspace.resolve()
        if source_root != workspace and workspace not in source_root.parents:
            raise RuntimeError("editable pytest bootstrap .pth escapes the workspace")
        candidates.append(source_root)
    if len(candidates) != 1:
        raise RuntimeError(
            "editable pytest bootstrap requires exactly one recorded simple-path .pth"
        )
    return candidates[0]


def _editable_workspace_roots(
    trusted_roots: list[Path],
    *,
    workspace: Path = _WORKSPACE,
) -> dict[str, Path]:
    """Map an approved editable distribution to its exact frozen .pth root."""
    workspace = workspace.resolve()
    declared: dict[str, Path] = {}
    for trusted_root_raw in trusted_roots:
        trusted_root = trusted_root_raw.resolve()
        for metadata_dir in sorted(trusted_root.glob("*.dist-info")):
            stem = metadata_dir.name.removesuffix(".dist-info")
            distribution, separator, _version = stem.rpartition("-")
            if not separator:
                continue
            normalized_distribution = _normalized_distribution(distribution)
            if normalized_distribution not in _EDITABLE_BOOTSTRAP_ROOTS:
                continue
            try:
                metadata = metadata_dir.lstat()
            except OSError as exc:
                raise RuntimeError(
                    "approved pytest bootstrap metadata cannot be inspected: "
                    f"{metadata_dir}"
                ) from exc
            if not stat.S_ISDIR(metadata.st_mode):
                raise RuntimeError(
                    "approved pytest bootstrap metadata is an alias: "
                    f"{metadata_dir}"
                )

            direct_url = metadata_dir / "direct_url.json"
            top_level = metadata_dir / "top_level.txt"
            metadata_file = metadata_dir / "METADATA"
            if not direct_url.exists() and not direct_url.is_symlink():
                # A normal physical pytest dependency is not a self-hosted
                # editable declaration and needs no special bootstrap path.
                continue
            required = (direct_url, top_level, metadata_file)
            if not all(_regular_file(path) for path in required):
                raise RuntimeError(
                    f"editable pytest bootstrap metadata is incomplete: {metadata_dir}"
                )
            if direct_url.stat().st_size > 64 * 1024 or top_level.stat().st_size > 4096:
                raise RuntimeError("editable pytest bootstrap metadata is oversized")
            try:
                payload = json.loads(direct_url.read_text(encoding="utf-8"))
                if not isinstance(payload, dict):
                    raise ValueError("direct_url.json is not an object")
                parsed = urlsplit(str(payload.get("url", "")))
                directory_info = payload.get("dir_info")
                editable = (
                    isinstance(directory_info, dict)
                    and directory_info.get("editable") is True
                )
                decoded_path = unquote(parsed.path)
                target_path = Path(decoded_path)
                target = (
                    target_path
                    if decoded_path
                    and target_path.is_absolute()
                    and ".." not in target_path.parts
                    else None
                )
                top_levels = top_level.read_text(encoding="utf-8").splitlines()
            except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
                raise RuntimeError(
                    f"editable pytest bootstrap metadata is invalid: {metadata_dir}"
                ) from exc
            root_name = top_levels[0] if len(top_levels) == 1 else ""
            declared_name = _metadata_name(metadata_file)
            if (
                not editable
                or parsed.scheme != "file"
                or parsed.netloc not in {"", "localhost"}
                or parsed.query
                or parsed.fragment
                or target is None
                or target != workspace
                or not root_name.isidentifier()
                or root_name not in _EDITABLE_BOOTSTRAP_ROOTS
                or _normalized_distribution(root_name) != normalized_distribution
                or declared_name is None
                or _normalized_distribution(declared_name) != normalized_distribution
            ):
                raise RuntimeError(
                    "editable pytest bootstrap declaration is inconsistent: "
                    f"{metadata_dir}"
                )
            if root_name in declared:
                raise RuntimeError(
                    f"duplicate editable pytest bootstrap declaration: {root_name}"
                )
            declared[root_name] = _recorded_simple_path(
                metadata_dir,
                trusted_root,
                workspace=workspace,
            )
    return declared


def _plain_package_entry(source_root: Path, root_name: str, *, root: Path) -> Path:
    root = root.resolve()
    if not source_root.is_absolute() or ".." in source_root.parts:
        raise RuntimeError(
            f"pytest bootstrap source root is not an exact absolute path: {source_root}"
        )
    if source_root != root and root not in source_root.parents:
        raise RuntimeError(
            f"pytest bootstrap source root escapes its tree: {source_root}"
        )
    current = root
    relative_root = source_root.relative_to(root)
    for part in relative_root.parts:
        current /= part
        try:
            metadata = current.lstat()
        except OSError as exc:
            raise RuntimeError(
                f"pytest bootstrap source root is missing: {source_root}"
            ) from exc
        if not stat.S_ISDIR(metadata.st_mode):
            raise RuntimeError("pytest bootstrap source root traverses an alias")
    package = source_root / root_name
    try:
        package_metadata = package.lstat()
    except OSError as exc:
        raise RuntimeError(
            f"pytest bootstrap package is not a directory: {root_name}"
        ) from exc
    if not stat.S_ISDIR(package_metadata.st_mode):
        raise RuntimeError(f"pytest bootstrap package is not a directory: {root_name}")
    entry = package / "__init__.py"
    if not _regular_file(entry):
        raise RuntimeError(f"pytest bootstrap package entry is unsafe: {root_name}")
    return entry.resolve()


def _load_pristine_bootstrap_root(
    root_name: str,
    candidate_source_root: Path,
) -> None:
    workspace = _WORKSPACE.resolve()
    bootstrap = _BOOTSTRAP.resolve()
    relative_source_root = candidate_source_root.relative_to(workspace)
    bootstrap_source_root = bootstrap / relative_source_root
    entry = _plain_package_entry(
        bootstrap_source_root,
        root_name,
        root=bootstrap,
    )
    spec = importlib.util.spec_from_file_location(
        root_name,
        entry,
        submodule_search_locations=[str(entry.parent)],
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"cannot construct pristine pytest bootstrap loader: {root_name}"
        )
    module = importlib.util.module_from_spec(spec)
    sys.modules[root_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(root_name, None)
        raise


def _frozen_bootstrap_source_roots() -> dict[str, Path]:
    bootstrap = _BOOTSTRAP.resolve()
    discovered: dict[str, Path] = {}
    for prefix in (Path("src"), Path(".")):
        source_root = bootstrap / prefix
        for root_name in sorted(_EDITABLE_BOOTSTRAP_ROOTS):
            package = source_root / root_name
            if not package.exists() and not package.is_symlink():
                continue
            _plain_package_entry(source_root, root_name, root=bootstrap)
            if root_name in discovered:
                raise RuntimeError(
                    f"duplicate frozen pytest bootstrap package: {root_name}"
                )
            discovered[root_name] = source_root
    return discovered


def _validate_frozen_bootstrap_routes(editable_roots: dict[str, Path]) -> None:
    frozen_roots = _frozen_bootstrap_source_roots()
    if set(frozen_roots) != set(editable_roots):
        raise RuntimeError(
            "frozen pytest bootstrap packages do not match editable routes: "
            f"frozen={sorted(frozen_roots)}, editable={sorted(editable_roots)}"
        )
    workspace = _WORKSPACE.resolve()
    bootstrap = _BOOTSTRAP.resolve()
    for root_name, candidate_source_root in editable_roots.items():
        expected = bootstrap / candidate_source_root.relative_to(workspace)
        if frozen_roots[root_name] != expected:
            raise RuntimeError(
                "frozen pytest bootstrap source root does not match editable route: "
                f"{root_name}"
            )


def _load_candidate_root(root_name: str, source_root: Path) -> tuple[ModuleType, Path]:
    entry = _plain_package_entry(
        source_root,
        root_name,
        root=_WORKSPACE,
    )
    for current_raw, directory_names, file_names in os.walk(
        entry.parent, followlinks=False
    ):
        current = Path(current_raw)
        for name in directory_names:
            metadata = (current / name).lstat()
            if not stat.S_ISDIR(metadata.st_mode):
                raise RuntimeError(
                    "candidate pytest bootstrap package contains a directory alias"
                )
        for name in file_names:
            metadata = (current / name).lstat()
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                raise RuntimeError(
                    "candidate pytest bootstrap package contains an aliased or "
                    "special file"
                )
    spec = importlib.util.spec_from_file_location(
        root_name,
        entry,
        submodule_search_locations=[str(entry.parent)],
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot construct exact candidate loader: {root_name}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[root_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(root_name, None)
        raise
    return module, entry


def _critical_runtime_snapshot() -> dict[str, object]:
    return {
        name: module
        for name, module in sys.modules.items()
        if name.partition(".")[0] in _CRITICAL_IMPORT_ROOTS
        and name.partition(".")[0] not in _EDITABLE_BOOTSTRAP_ROOTS
    }


def _attest_post_run_runtime(
    trusted_roots: list[Path],
    snapshot: dict[str, object],
) -> None:
    for module_name, expected in snapshot.items():
        if sys.modules.get(module_name) is not expected:
            raise RuntimeError(
                "pytest critical runtime module identity changed during tests: "
                f"{module_name}"
            )
    trusted = [path.resolve() for path in trusted_roots]
    workspace = _WORKSPACE.resolve()
    for module_name, module in tuple(sys.modules.items()):
        root_name = module_name.partition(".")[0]
        if (
            root_name not in _CRITICAL_IMPORT_ROOTS
            or root_name in _EDITABLE_BOOTSTRAP_ROOTS
        ):
            continue
        origin = _module_origin(module_name, module)
        if not _is_within(origin, trusted) or _is_within(origin, [workspace]):
            raise RuntimeError(
                "untrusted post-run pytest runtime module origin: "
                f"{module_name}={origin}"
            )


def _attest_candidate_root(
    root_name: str,
    expected_module: ModuleType,
    expected_entry: Path,
) -> None:
    current = sys.modules.get(root_name)
    if current is not expected_module:
        raise RuntimeError(
            f"candidate pytest bootstrap module identity changed: {root_name}"
        )
    origin = _module_origin(root_name, current)
    if origin != expected_entry:
        raise RuntimeError(
            "candidate pytest bootstrap module origin changed: "
            f"{root_name}={origin}"
        )
    package_root = expected_entry.parent
    for module_name, module in tuple(sys.modules.items()):
        if module_name != root_name and not module_name.startswith(f"{root_name}."):
            continue
        submodule_origin = _module_origin(module_name, module)
        if not _is_within(submodule_origin, [package_root]):
            raise RuntimeError(
                "candidate pytest bootstrap submodule escaped its exact package: "
                f"{module_name}={submodule_origin}"
            )


def _pristine_pluggy_bindings() -> tuple[set[int], set[int]]:
    modules = [
        module
        for name, module in sys.modules.items()
        if name == "pluggy" or name.startswith("pluggy.")
    ]
    module_ids = {id(module) for module in modules}
    object_ids = {
        id(value)
        for module in modules
        for value in getattr(module, "__dict__", {}).values()
    }
    for module_name, module in tuple(sys.modules.items()):
        if module_name != "_pytest" and not module_name.startswith("_pytest."):
            continue
        for value in getattr(module, "__dict__", {}).values():
            if isinstance(value, ModuleType):
                if value.__name__.partition(".")[0] == "pluggy":
                    module_ids.add(id(value))
                continue
            value_module = str(getattr(value, "__module__", "")).partition(".")[0]
            if value_module == "pluggy":
                object_ids.add(id(value))
    return module_ids, object_ids


def _attest_pytest_pluggy_bindings(
    pristine_module_ids: set[int],
    pristine_object_ids: set[int],
) -> None:
    for module_name, module in tuple(sys.modules.items()):
        if module_name != "_pytest" and not module_name.startswith("_pytest."):
            continue
        for name, value in getattr(module, "__dict__", {}).items():
            if isinstance(value, ModuleType):
                value_module = value.__name__.partition(".")[0]
                if value_module == "pluggy" and id(value) not in pristine_module_ids:
                    raise RuntimeError(
                        "pytest loaded candidate pluggy into its runtime: "
                        f"{module_name}.{name}"
                    )
                continue
            value_module = str(getattr(value, "__module__", "")).partition(".")[0]
            if value_module == "pluggy" and id(value) not in pristine_object_ids:
                raise RuntimeError(
                    "pytest bound a candidate pluggy object into its runtime: "
                    f"{module_name}.{name}"
                )


def _reject_explicit_plugins(arguments: list[str]) -> None:
    for index, argument in enumerate(arguments):
        if argument == "-p" or argument.startswith("-p=") or (
            argument.startswith("-p")
            and not argument.startswith("--")
            and len(argument) > 2
        ):
            raise RuntimeError(
                "self-hosted pluggy scoring does not allow explicit pytest plugins: "
                f"argument {index + 1}"
            )


def main() -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--repo-eval-pythonpath", default="")
    parser.add_argument("pytest_args", nargs=argparse.REMAINDER)
    parsed = parser.parse_args()

    trusted_roots = _trusted_site_directories()
    if not trusted_roots:
        raise RuntimeError("no trusted Python package directory is available")
    project_paths = _project_paths(parsed.repo_eval_pythonpath)
    editable_roots = _editable_workspace_roots(trusted_roots)
    _validate_frozen_bootstrap_routes(editable_roots)

    # ``-S`` leaves only the interpreter/stdlib paths. Appending immutable
    # package roots imports pytest without executing any .pth startup hooks.
    sys.path.extend(str(path) for path in trusted_roots)
    for root_name, source_root in sorted(editable_roots.items()):
        _load_pristine_bootstrap_root(root_name, source_root)
    import pytest  # noqa: PLC0415 - ordering is the security boundary

    _attest_loaded_runtime(
        trusted_roots,
        bootstrap_roots=frozenset(editable_roots),
    )
    critical_snapshot = _critical_runtime_snapshot()
    pytest_main = pytest.main
    arguments = parsed.pytest_args
    if arguments[:1] == ["--"]:
        arguments = arguments[1:]

    pristine_module_ids: set[int] = set()
    pristine_object_ids: set[int] = set()
    candidate_modules: dict[str, tuple[ModuleType, Path]] = {}
    target_paths = [str(path) for path in editable_roots.values()]
    if editable_roots:
        _reject_explicit_plugins(arguments)
        os.environ["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
        pristine_module_ids, pristine_object_ids = _pristine_pluggy_bindings()
        for name in tuple(sys.modules):
            if name == "pluggy" or name.startswith("pluggy."):
                sys.modules.pop(name, None)

    # The exact frozen .pth root comes first for a self-hosted package; a model
    # cannot redirect collection by adding a same-named root package.
    ordered_paths = [*target_paths, *project_paths]
    sys.path[0:0] = list(dict.fromkeys(ordered_paths))
    for root_name, source_root in sorted(editable_roots.items()):
        candidate_modules[root_name] = _load_candidate_root(root_name, source_root)
    result = int(pytest_main(arguments))
    if editable_roots:
        for root_name, (module, entry) in candidate_modules.items():
            _attest_candidate_root(root_name, module, entry)
        _attest_pytest_pluggy_bindings(
            pristine_module_ids,
            pristine_object_ids,
        )
        _attest_post_run_runtime(trusted_roots, critical_snapshot)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
