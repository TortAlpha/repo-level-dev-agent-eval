"""Prepare small benchmark-image derivatives that hide the baked /app checkout.

Use --tasks TASK... to inspect/plan; add --build to build and verify. This
script never pulls, prunes, removes original images, or edits task inputs or
collection.csv. A successful record supplies the image ID for a frozen,
plan-local collection. Known /app exposure is removed; this is not a general
forensic audit of every byte in the upstream image.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[3]
PLAN = Path(__file__).resolve().parent
DOCKERFILE = PLAN / "Sanitized.Dockerfile"
REPORT = PLAN / "sanitized_images.json"
SUPPORTED = {"ansible", "qutebrowser"}


def inspect_image(image: str) -> dict:
    result = subprocess.run(
        ["docker", "image", "inspect", image],
        capture_output=True, text=True, timeout=30, check=False,
    )
    if result.returncode:
        raise RuntimeError(f"Image must already be cached: {image}\n{result.stderr[-1500:]}")
    return json.loads(result.stdout)[0]


def known_manifest(task_id: str, family: str) -> dict:
    payload = json.loads((PLAN / f"image_manifests_{family}.json").read_text())
    return next(item for item in payload["images"] if item["task_id"] == task_id)


def write_report(records: dict[str, dict]) -> None:
    temporary = REPORT.with_suffix(".json.tmp")
    temporary.write_text(json.dumps({
        "schema_version": 1,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Remove the known baked /app checkout; preserve installed dependencies.",
        "images": list(records.values()),
    }, indent=2) + "\n")
    temporary.replace(REPORT)


def verify_image(image: str, task_id: str, row: dict, family: str) -> dict:
    """Verify no /app and normal imports resolve the mounted candidate tree."""
    repository = (ROOT / row["repo_path"]).resolve()
    if not repository.is_dir() or not repository.is_relative_to(ROOT / "repositories/tasks"):
        raise RuntimeError(f"Unexpected task repository path: {repository}")
    program = """
import importlib.util, json, os, pwd, grp, sys
from pathlib import Path
assert not os.path.lexists('/app'), 'baked /app is still accessible'
assert sys.version_info >= (3, 9), 'strict evaluator needs Python 3.9 or newer'
assert pwd.getpwuid(os.getuid()), 'sandbox UID has no passwd entry'
assert grp.getgrgid(os.getgid()), 'sandbox GID has no group entry'
family = sys.argv[1]
spec = importlib.util.find_spec(family)
assert spec is not None and spec.origin, 'target package is unavailable'
origin = Path(spec.origin).resolve()
assert Path('/workspace') in origin.parents, ('target import is not the candidate tree', str(origin))
import pytest
pytest_origin = Path(pytest.__file__).resolve()
assert Path('/workspace') not in pytest_origin.parents
assert Path('/app') not in pytest_origin.parents
for directory in sys.path:
    if directory and (directory == '/app' or directory.startswith('/app/')):
        assert not os.path.exists(directory), ('live baked import root', directory)
print(json.dumps({'python_version': sys.version.split()[0], 'target_origin': str(origin), 'pytest_origin': str(pytest_origin), 'baked_app_absent': True, 'sys_path': sys.path}))
"""
    command = [
        "docker", "run", "--rm", "--network", "none", "--read-only",
        "--cap-drop", "ALL", "--security-opt", "no-new-privileges:true",
        "--pids-limit", "64", "--memory", "512m",
        "--user", f"{os.getuid()}:{os.getgid()}",
        "--tmpfs", "/tmp:rw,nosuid,nodev,size=64m,mode=1777",
        "--env", "HOME=/tmp", "--env", "PYTHONDONTWRITEBYTECODE=1",
        "--mount", f"type=bind,src={repository},dst=/workspace,readonly",
        "--workdir", "/workspace", "--entrypoint", "/bin/sh", image,
        "-c", 'exec python -c "$1" "$2"', "verify-image", program, family,
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=60, check=False)
    if result.returncode:
        raise RuntimeError(f"Derived image verification failed for {task_id}:\n{result.stdout}\n{result.stderr}")
    verification = json.loads(result.stdout.strip().splitlines()[-1])
    verification["stderr"] = result.stderr[-2000:]
    return verification


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="+", required=True)
    parser.add_argument("--build", action="store_true", help="Build and verify; omitted means plan only.")
    args = parser.parse_args()
    rows = {row["task_id"]: row for row in csv.DictReader((ROOT / "repositories/collection.csv").open())}
    build_environment = None
    if args.build:
        endpoint = subprocess.check_output(
            ["docker", "context", "inspect", "--format", "{{.Endpoints.docker.Host}}"],
            text=True, timeout=15,
        ).strip()
        anonymous_config = PLAN / "anonymous_docker"
        if not (anonymous_config / "config.json").is_file():
            raise RuntimeError("Prepare the plan-local anonymous Docker configuration first.")
        build_environment = {**os.environ, "DOCKER_CONFIG": str(anonymous_config), "DOCKER_HOST": endpoint}
    recipe = DOCKERFILE.read_bytes()
    recipe_sha = hashlib.sha256(recipe).hexdigest()
    records = {}
    if REPORT.exists():
        records = {record["task_id"]: record for record in json.loads(REPORT.read_text())["images"]}
    for task_id in dict.fromkeys(args.tasks):
        row = rows[task_id]
        family = task_id.split("_")[1]
        if family not in SUPPORTED:
            raise RuntimeError(f"Unsupported image family: {family}")
        source = inspect_image(row["docker_image"])
        expected = known_manifest(task_id, family)
        source_digest = expected["digest"]
        pinned_source = expected["image"].split(":", 1)[0] + "@" + source_digest
        if pinned_source not in source.get("RepoDigests", []):
            raise RuntimeError(f"Cached source digest differs from recorded manifest: {task_id}")
        for value in source.get("Config", {}).get("Env", []) or []:
            if value.startswith(("VIRTUAL_ENV=/app", "PYTHONHOME=/app")):
                raise RuntimeError(f"Image stores its Python environment under /app: {task_id}")
        key = hashlib.sha256((source["Id"] + recipe_sha + f":{os.getuid()}:{os.getgid()}").encode()).hexdigest()[:12]
        tag = f"repo-eval-defense:sanitized-{task_id}-{key}"
        record = {
            "task_id": task_id, "status": "planned", "original_image": row["docker_image"],
            "original_digest": source_digest, "original_id": source["Id"],
            "original_repo_digests": source.get("RepoDigests", []),
            "original_size_bytes": source.get("Size"), "architecture": source["Architecture"],
            "derived_tag": tag, "derived_id": None, "recipe_sha256": recipe_sha,
            "removed_paths": ["/app"], "registry_commands_changed": False,
            "agent_uid": os.getuid(), "agent_gid": os.getgid(),
            "identity_mapping_policy": "Add a passwd/group entry only when the sandbox UID/GID is absent.",
            "free_bytes_before": shutil.disk_usage(ROOT).free,
        }
        if not args.build:
            print(json.dumps(record), flush=True)
            continue
        records[task_id] = record
        write_report(records)
        try:
            if shutil.disk_usage(ROOT).free < 2 * 1024 ** 3:
                raise RuntimeError("Disk guard: fewer than 2 GiB free before the minimal derivative build.")
            logs = PLAN / "sanitized_image_logs"
            logs.mkdir(exist_ok=True)
            log_path = logs / f"{task_id}.log"
            with tempfile.TemporaryDirectory(prefix="image-context-", dir=PLAN) as temporary:
                context = Path(temporary)
                (context / "Dockerfile").write_bytes(recipe)
                command = [
                    "docker", "build", "--pull=false", "--network=none",
                    "--platform", f"{source['Os']}/{source['Architecture']}",
                    "--build-arg", f"SOURCE_IMAGE={pinned_source}",
                    "--build-arg", f"SOURCE_ID={source['Id']}",
                    "--build-arg", f"SOURCE_DIGEST={source_digest}",
                    "--build-arg", f"RECIPE_SHA256={recipe_sha}",
                    "--build-arg", f"AGENT_UID={os.getuid()}",
                    "--build-arg", f"AGENT_GID={os.getgid()}",
                    "--tag", tag, str(context),
                ]
                with log_path.open("w") as log:
                    result = subprocess.run(command, env=build_environment, stdout=log, stderr=subprocess.STDOUT, timeout=600, check=False)
                if result.returncode:
                    raise RuntimeError(f"Derivative build failed; see {log_path}")
            derived = inspect_image(tag)
            expected_layers = source.get("RootFS", {}).get("Layers", [])
            derived_layers = derived.get("RootFS", {}).get("Layers", [])
            if derived_layers[:len(expected_layers)] != expected_layers:
                raise RuntimeError("Derivative does not preserve the recorded base layers.")
            labels = derived.get("Config", {}).get("Labels", {}) or {}
            if labels.get("repo-eval.source-id") != source["Id"] or labels.get("repo-eval.recipe-sha256") != recipe_sha:
                raise RuntimeError("Derivative provenance labels do not match.")
            record.update({"derived_id": derived["Id"], "derived_size_bytes": derived.get("Size"),
                           "added_layer_count": len(derived_layers) - len(expected_layers), "build_log": str(log_path)})
            record["verification"] = verify_image(derived["Id"], task_id, row, family)
            record["status"] = "verified"
        except Exception as error:
            record.update({"status": "failed", "error": str(error)})
            write_report(records)
            raise
        record["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        write_report(records)
        print(json.dumps({"task_id": task_id, "status": record["status"], "derived_id": record["derived_id"]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
