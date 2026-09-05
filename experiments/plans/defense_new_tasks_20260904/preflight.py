"""Validate the preselected task inputs before any benchmark model calls."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
PLAN = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

PRIMARY = [
    "swepro_ansible_1ee70fc2", "swepro_ansible_3889ddeb",
    "swepro_ansible_1b70260d", "swepro_ansible_1bd7dcf3",
    "swepro_openlibrary_c05ccf2c", "swepro_openlibrary_0dc5b20f",
    "swepro_openlibrary_08ac40d0", "swepro_openlibrary_5c6c22f3",
    "swepro_qutebrowser_6dd402c0", "swepro_qutebrowser_e64622cd",
    "swepro_qutebrowser_ed19d7f5", "swepro_qutebrowser_0fc6d110",
]

def one(task_id: str) -> int:
    from src.benchmark.validate import validate_task, _print_verdict
    rows = {r['task_id']: r for r in csv.DictReader((ROOT / 'repositories/collection.csv').open())}
    args = argparse.Namespace(
        workspaces_dir=ROOT/'experiments/workspaces', docker_image='python:3.11-slim',
        setup_commands=None, shell_timeout=600, test_timeout=600,
        keep_workdirs=False, clean_failures=True,
    )
    verdict = validate_task(rows[task_id], args)
    _print_verdict(verdict)
    payload = {**asdict(verdict), 'verified': verdict.verified, 'finished_at': datetime.now(timezone.utc).isoformat()}
    (PLAN/'preflight'/f'{task_id}.json').write_text(json.dumps(payload, indent=2)+'\n')
    return 0 if verdict.verified else 1

def batch(tasks: list[str]) -> int:
    (PLAN/'preflight').mkdir(parents=True, exist_ok=True)
    def execute(task_id):
        with (PLAN/'preflight'/f'{task_id}.log').open('w') as log:
            result = subprocess.run([sys.executable, '-u', str(Path(__file__).resolve()), '--one', task_id], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=False)
        return task_id, result.returncode
    failed=[]
    with ThreadPoolExecutor(max_workers=2) as pool:
        pending=[pool.submit(execute, task) for task in tasks]
        for future in as_completed(pending):
            task_id, code=future.result()
            print(json.dumps({'task_id':task_id,'exit_code':code,'log':str(PLAN/'preflight'/f'{task_id}.log')}),flush=True)
            if code:failed.append(task_id)
    print(json.dumps({'checked':len(tasks),'failed':failed}),flush=True)
    return bool(failed)

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--one')
    parser.add_argument('--tasks', nargs='*')
    args=parser.parse_args()
    (PLAN/'preflight').mkdir(parents=True, exist_ok=True)
    raise SystemExit(one(args.one) if args.one else batch(args.tasks or PRIMARY))
