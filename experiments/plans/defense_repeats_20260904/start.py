"""Start two bounded, independently logged repetitions of the existing core block."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
PLAN = Path(__file__).resolve().parent
OVERRIDES = {
    'TEMPERATURE':'0', 'MAX_TOKENS':'16384', 'REQUEST_TIMEOUT_SECONDS':'300',
    'MAX_RETRIES':'0', 'COMPACTION_MODE':'checkpoint', 'CONTEXT_BUDGET_TOKENS':'48000',
    'OPENROUTER_CONTEXT_WINDOW_TOKENS':'256000', 'DOCKER_IMAGE':'python:3.11-slim',
    'TEST_TIMEOUT_SECONDS':'600', 'SHELL_TIMEOUT_SECONDS':'300', 'PYTHONUNBUFFERED':'1',
}

def command(repetition):
    session=f'kernel_v2_ds_defense_repeat_20260904_r{repetition}'
    return [str(ROOT/'.venv/bin/python'), '-u', '-m', 'src.benchmark.sweep',
        '--task-set','final-v1','--task-groups','core_small','--agents','single,multi-graph',
        '--session',session,'--campaign',session,'--provider','openrouter',
        '--models','deepseek/deepseek-v4-flash-0731:nitro',
        '--reasoning-effort','low','--action-transport','text_json',
        '--max-steps','50','--max-iterations','5','--max-cost-usd','0.15',
        '--max-total-cost-usd','2','--no-network','--concurrency','1',
        '--single-research-guard','--single-compatibility-guard',
        '--single-research-warning-steps','12','--single-research-hard-limit','20',
        '--single-post-plan-research-warning-steps','5','--single-post-plan-research-hard-limit','8',
        '--order','task-major','--seed',str(20260903+repetition),
        '--infrastructure-retries','1','--archive-patches']

def main(launch):
    os.chdir(ROOT)
    sys.path.insert(0,str(ROOT))
    from src.config import load_config
    os.environ.update(OVERRIDES)
    config=load_config('openrouter')
    assert config.openrouter_api_key, 'Configured OpenRouter credential is required.'
    for name in type(config).model_fields:
        if name.startswith(('role_model_', 'role_reasoning_effort_')) or name in {
            'developer_escalation_model','developer_escalation_reasoning_effort'}:
            assert getattr(config,name) is None, f'Unexpected override: {name}'
    if (PLAN/'launch.json').exists():
        raise RuntimeError('Launch metadata already exists; inspect existing workers before restarting.')
    for repetition in (1,2):
        session=f'kernel_v2_ds_defense_repeat_20260904_r{repetition}'
        if (ROOT/'experiments/results/sessions'/session/'sweep.json').exists():
            raise RuntimeError(f'Session already exists: {session}')
        with (PLAN/f'r{repetition}.plan.log').open('w') as log:
            checked=subprocess.run(command(repetition)+['--plan-only'],cwd=ROOT,
                stdout=log,stderr=subprocess.STDOUT,timeout=180,check=False)
        if checked.returncode:
            raise RuntimeError(f'Repetition {repetition} plan failed; inspect its log.')
    metadata={'created_at':datetime.now(timezone.utc).isoformat(),
        'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'public_environment_overrides':OVERRIDES,'planned_runs':48,
        'experiment':'Two additional repetitions on the existing twelve core_small tasks; not a new held-out set.',
        'cost_limits':'USD 0.15 per run, USD 2 per independent campaign; checked between calls/runs, possible bounded overshoot.',
        'workers':[]}
    if not launch:
        print('Both 24-run plans validated. No model calls started.')
        return
    for repetition in (1,2):
        path=PLAN/f'r{repetition}.log'
        with path.open('w') as log:
            process=subprocess.Popen(command(repetition),cwd=ROOT,stdout=log,
                stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
        metadata['workers'].append({'repetition':repetition,'pid':process.pid,
            'log':str(path),'command':command(repetition)})
        (PLAN/'launch.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps(metadata,indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--launch',action='store_true')
    main(parser.parse_args().launch)
