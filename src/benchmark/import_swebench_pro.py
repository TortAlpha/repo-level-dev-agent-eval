"""CLI: import SWE-bench Pro instances as benchmark tasks.

Materializes instances from the public HuggingFace split
(``ScaleAI/SWE-bench_Pro``) into this project's task layout:

- repo @ ``base_commit``      -> ``repositories/tasks/<task_id>/repo``
- problem + requirements      -> ``task.md`` (the augmented spec is the whole
  point: it removes the underspecification our PR-derived feature tasks have)
- ``test_patch``-touched files-> ``hidden_tests/`` (applied on a scratch copy)
- ``fail_to_pass``            -> ``hidden_semantic_test_command``
- ``pass_to_pass``            -> ``hidden_compat_test_command`` (capped)
- gold ``patch``              -> ``<task_id>/reference.patch`` (validate.py
  uses it as the reference state when ``merge_commit`` is empty)
- ``dockerhub_tag``           -> ``docker_image`` column (prebuilt env,
  ``jefzda/sweap-images:<tag>``)

Imported rows land as ``pr_task_unverified``; run ``src.benchmark.validate``
afterwards. Instance repos are GPL-family — they stay local (the tasks dir is
gitignored) and are used for evaluation only.
"""

from __future__ import annotations

import argparse
import ast
import csv
import re
import shutil
import subprocess
import tempfile
import urllib.request
from pathlib import Path

from .collection import DEFAULT_COLLECTION

PARQUET_URL = (
    "https://huggingface.co/api/datasets/ScaleAI/SWE-bench_Pro/parquet/default/test/0.parquet"
)
RUN_SCRIPT_URL = (
    "https://raw.githubusercontent.com/scaleapi/SWE-bench_Pro-os/main/run_scripts/{instance_id}/run_script.sh"
)
CONTAINER_WORKDIR = "/workspace"  # must match DockerSandbox.container_workdir
RUN_SCRIPT_NAME = "run_tests.sh"
DEFAULT_PARQUET_CACHE = Path("repositories/repos/_swebench_pro.parquet")
DEFAULT_CLONE_CACHE = Path("repositories/repos/_swepro_cache")
DEFAULT_IMAGE_PREFIX = "jefzda/sweap-images"
MAX_SEMANTIC_IDS = 30
MAX_COMPAT_IDS = 15
MAX_VISIBLE_FILES = 5


def _run(cmd: list[str], cwd: Path | None = None, timeout: int = 1800) -> subprocess.CompletedProcess:
    proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd)} failed:\n{proc.stderr[-1500:]}")
    return proc


def load_instances(parquet: Path):
    import pandas as pd

    if not parquet.exists():
        parquet.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading dataset to {parquet} ...")
        urllib.request.urlretrieve(PARQUET_URL, parquet)
    return pd.read_parquet(parquet)


def short_task_id(row) -> str:
    repo_name = row["repo"].split("/")[-1].replace("-", "_")
    match = re.search(r"-([0-9a-f]{8})[0-9a-f]*-", row["instance_id"])
    suffix = match.group(1) if match else row["base_commit"][:8]
    return f"swepro_{repo_name}_{suffix}"


def ensure_repo_cache(repo: str, cache_dir: Path) -> Path:
    """One bare clone per upstream repo, shared by all its instances."""
    bare = cache_dir / f"{repo.split('/')[-1]}.git"
    if not bare.exists():
        cache_dir.mkdir(parents=True, exist_ok=True)
        print(f"Cloning {repo} (bare, one-time) ...")
        _run(["git", "clone", "--bare", f"https://github.com/{repo}", str(bare)], timeout=3600)
    return bare


def materialize_repo(bare: Path, base_commit: str, dest: Path) -> None:
    _run(["git", "clone", str(bare), str(dest)])
    _run(["git", "checkout", "--detach", "-q", base_commit], cwd=dest)


def parse_id_list(raw) -> list[str]:
    if isinstance(raw, (list, tuple)):
        return [str(item) for item in raw]
    raw = str(raw or "").strip()
    if not raw:
        return []
    try:
        return [str(item) for item in ast.literal_eval(raw)]
    except (ValueError, SyntaxError):
        return [part.strip() for part in raw.splitlines() if part.strip()]


def extract_hidden_tests(bare: Path, base_commit: str, test_patch: str, dest: Path) -> list[str]:
    """Apply the solution PR's test patch on a scratch checkout and copy the
    touched files into hidden_tests/, preserving repo-relative paths."""
    touched = re.findall(r"^\+\+\+ b/(.+)$", test_patch, flags=re.MULTILINE)
    touched = [p for p in touched if p != "/dev/null"]
    with tempfile.TemporaryDirectory() as tmp:
        scratch = Path(tmp) / "repo"
        materialize_repo(bare, base_commit, scratch)
        patch_file = Path(tmp) / "tests.patch"
        patch_file.write_text(test_patch, encoding="utf-8")
        _run(["git", "apply", "--whitespace=nowarn", str(patch_file)], cwd=scratch)
        for rel in touched:
            src = scratch / rel
            if not src.is_file():
                continue
            target = dest / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, target)
    return touched


def install_run_script(instance_id: str, repo_dir: Path) -> bool:
    """Drop the instance's upstream test runner into the checkout.

    SWE-bench Pro invokes tests through a per-instance ``run_script.sh``
    (family-specific env vars, native runners like ansible-test); raw pytest
    fails whole families. The script assumes the repo at /app — rewritten to
    the sandbox workdir — and is excluded from git locally so it never shows
    up in agent patches.
    """
    url = RUN_SCRIPT_URL.format(instance_id=instance_id)
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            script = response.read().decode("utf-8")
    except OSError:
        return False
    script = script.replace("/app", CONTAINER_WORKDIR)
    # Keep the image's baked-in checkout (/app) on the import path as a
    # fallback: some repos vendor deps as submodules that a plain git clone
    # of the task repo lacks (e.g. openlibrary's infogami).
    script = re.sub(
        rf"^(\s*export PYTHONPATH={re.escape(CONTAINER_WORKDIR)})(?=[:\s]|$)",
        r"\1:/app",
        script,
        flags=re.MULTILINE,
    )
    target = repo_dir / RUN_SCRIPT_NAME
    target.write_text(script, encoding="utf-8")
    target.chmod(0o755)
    exclude = repo_dir / ".git" / "info" / "exclude"
    exclude.parent.mkdir(parents=True, exist_ok=True)
    with exclude.open("a", encoding="utf-8") as handle:
        handle.write(f"{RUN_SCRIPT_NAME}\n")
    return True


def build_task_md(row) -> str:
    parts = [str(row["problem_statement"]).strip()]
    requirements = str(row.get("requirements") or "").strip()
    if requirements:
        parts.append("## Requirements\n\n" + requirements)
    interface = str(row.get("interface") or "").strip()
    if interface:
        parts.append("## Interface\n\n" + interface)
    return "\n\n".join(parts) + "\n"


def infer_task_type(row) -> str:
    categories = " ".join(parse_id_list(row.get("issue_categories"))).lower()
    return "bugfix" if "bug" in categories else "feature"


def import_instance(row, args, fieldnames: list[str]) -> dict:
    task_id = short_task_id(row)
    task_dir = Path("repositories/tasks") / task_id
    if task_dir.exists():
        if not args.force:
            raise RuntimeError(f"{task_dir} already exists (use --force to overwrite)")
        shutil.rmtree(task_dir)
    task_dir.mkdir(parents=True)

    bare = ensure_repo_cache(row["repo"], args.clone_cache)
    materialize_repo(bare, row["base_commit"], task_dir / "repo")
    (task_dir / "task.md").write_text(build_task_md(row), encoding="utf-8")
    (task_dir / "reference.patch").write_text(str(row["patch"]), encoding="utf-8")
    extract_hidden_tests(bare, row["base_commit"], str(row["test_patch"]), task_dir / "hidden_tests")

    fail_to_pass = parse_id_list(row["fail_to_pass"])[:MAX_SEMANTIC_IDS]
    pass_to_pass = parse_id_list(row["pass_to_pass"])
    has_script = install_run_script(str(row["instance_id"]), task_dir / "repo")
    runner = f"bash ./{RUN_SCRIPT_NAME}" if has_script else "python -m pytest"
    semantic = f"{runner} " + " ".join(
        f"'../hidden_tests/{test_id}'" for test_id in fail_to_pass
    )
    compat = ""
    if pass_to_pass:
        compat = f"{runner} " + " ".join(
            f"'{test_id}'" for test_id in pass_to_pass[:MAX_COMPAT_IDS]
        )
    # Visible command must be green on the *base* checkout. P2P ids are
    # evaluated with the test patch applied upstream, so their files can be
    # patch-created — keep only files that exist at base.
    visible_files: list[str] = []
    for test_id in pass_to_pass:
        path = test_id.split("::")[0]
        if path in visible_files or not (task_dir / "repo" / path).is_file():
            continue
        visible_files.append(path)
        if len(visible_files) >= MAX_VISIBLE_FILES:
            break
    if not visible_files:
        # No P2P files at base (or no P2P at all): fall back to the F2P
        # files' base versions instead of a bare runner — a bare command
        # runs the whole suite, which drags in slow/flaky families (e.g.
        # qutebrowser's end2end tests) and blows the visible-on-base check.
        for test_id in fail_to_pass:
            path = test_id.split("::")[0]
            if path in visible_files or not (task_dir / "repo" / path).is_file():
                continue
            visible_files.append(path)
            if len(visible_files) >= MAX_VISIBLE_FILES:
                break
    visible = f"{runner} " + " ".join(visible_files) if visible_files else runner

    record = {name: "" for name in fieldnames}
    record.update(
        {
            "task_id": task_id,
            "repo_path": f"repositories/tasks/{task_id}/repo",
            "task_file_path": f"repositories/tasks/{task_id}/task.md",
            "size": "large",
            "task_type": infer_task_type(row),
            "repo_url": f"https://github.com/{row['repo']}",
            "base_commit": row["base_commit"],
            "hidden_tests_path": f"repositories/tasks/{task_id}/hidden_tests",
            "visible_test_command": visible,
            "hidden_test_command": semantic,
            "hidden_semantic_test_command": semantic,
            "hidden_compat_test_command": compat,
            "task_status": "pr_task_unverified",
            # Their images ship deps preinstalled; "true" keeps the harness
            # from running the default pip install into a foreign env.
            "setup_commands": "true",
            "docker_image": f"{args.image_prefix}:{row['dockerhub_tag']}",
            "notes": (
                f"Imported from SWE-bench Pro public split; instance_id={row['instance_id']}. "
                f"Augmented spec (requirements/interface) included in task.md. "
                f"F2P ids: {len(fail_to_pass)}, P2P ids: {len(pass_to_pass)} (compat capped at {MAX_COMPAT_IDS})."
            ),
        }
    )
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Import SWE-bench Pro instances as tasks.")
    parser.add_argument("--parquet", type=Path, default=DEFAULT_PARQUET_CACHE)
    parser.add_argument("--language", default="python")
    parser.add_argument("--repo", default=None, help="Filter by upstream repo (org/name).")
    parser.add_argument("--instance-id", action="append", default=None)
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--collection", type=Path, default=DEFAULT_COLLECTION)
    parser.add_argument("--clone-cache", type=Path, default=DEFAULT_CLONE_CACHE)
    parser.add_argument("--image-prefix", default=DEFAULT_IMAGE_PREFIX)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    df = load_instances(args.parquet)
    picked = df[df["repo_language"].str.lower() == args.language.lower()]
    if args.repo:
        picked = picked[picked["repo"] == args.repo]
    if args.instance_id:
        picked = picked[picked["instance_id"].isin(args.instance_id)]
    picked = picked.head(args.limit)
    if picked.empty:
        raise SystemExit("no instances match the filters")

    with args.collection.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    if "docker_image" not in fieldnames:
        fieldnames.append("docker_image")
        for row in rows:
            row.setdefault("docker_image", "")
    existing = {row["task_id"] for row in rows}

    imported = []
    skipped = 0
    for index, (_, row) in enumerate(picked.iterrows(), 1):
        task_id = short_task_id(row)
        if task_id in existing and not args.force:
            skipped += 1
            continue
        try:
            record = import_instance(row, args, fieldnames)
        except (RuntimeError, subprocess.SubprocessError, OSError) as exc:
            print(f"[{index}/{len(picked)}] SKIP {task_id}: {exc}")
            continue
        if record["task_id"] in existing:
            rows = [r for r in rows if r["task_id"] != record["task_id"]]
        rows.append(record)
        existing.add(record["task_id"])
        imported.append(record["task_id"])
        print(f"[{index}/{len(picked)}] imported {record['task_id']}")
    if skipped:
        print(f"skipped {skipped} already-imported task(s)")

    with args.collection.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n{len(imported)} task(s) added to {args.collection} as pr_task_unverified.")
    print("Validate with: python -m src.benchmark.validate " + " ".join(f"--task-id {t}" for t in imported))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
