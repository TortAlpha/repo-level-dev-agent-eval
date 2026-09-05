# SWE-agent as an alternative agent

The benchmark scores a **patch**, not an agent. Any agent that turns
`repo@base + task.md` into edits on the workspace can be evaluated by the same
downstream pipeline (visible/hidden tests, regressions, quality, cost). This is
what `src/agents/swe_agent.py` (`SweAgentAdapter`) does: it shells out to
[SWE-agent](https://swe-agent.com), applies the patch it produces onto the
prepared workspace, and maps its trajectory onto a `State` so the runner and
metrics work unchanged.

Final benchmark runs the same tasks through **single-agent** (ours),
**swe-agent** (this adapter), and **multi-agent** (ours) for comparison.

## Install

SWE-agent 1.x must be installed **editable from source** — the PyPI wheel we
tried had a broken dep and, when non-editable, ships without its `config/`
directory (so `CONFIG_DIR` fails):

```bash
git clone https://github.com/SWE-agent/SWE-agent /tmp/SWE-agent
pip install -e /tmp/SWE-agent
sweagent --version   # expect 1.1.0
```

Docker must be running (SWE-agent spins its own container via SWE-ReX).

## Run

Through the harness (recommended — gives eval + metrics + review):

```bash
set -a; source .env; set +a          # OPENROUTER_API_KEY
python -m src.benchmark --task-id parse_pr_227 \
    --agent swe-agent --model openai/gpt-4o-mini --enable-review
```

Live progress streams to
`experiments/workspaces/<task>-<run>/swe_agent/sweagent.log`
(`tail -f` it). SWE-agent's own trajectory persists in the same dir — inspect
with `sweagent inspect <file>.traj` (terminal) or `sweagent inspector <dir>`
(web). LangSmith only sees this adapter's outer span, not SWE-agent's internal
steps.

Direct run (bypasses the adapter; useful for debugging):

```bash
sweagent run --config /tmp/SWE-agent/config/default.yaml \
  --agent.model.name=openrouter/openai/gpt-4o-mini \
  --agent.model.per_instance_cost_limit=0 --agent.model.total_cost_limit=0 \
  --env.repo.path=/path/to/repo/at/base \
  --problem_statement.path=/path/to/task.md \
  --output_dir=/tmp/sweout
```

## Gotchas (all handled in the adapter; listed so they aren't rediscovered)

- **litellm can't price many OpenRouter models** → SWE-agent's cost guard
  aborts (`ModelConfigurationError: ... set ... cost_limit ... to 0`). The
  adapter passes `per_instance_cost_limit=0 total_cost_limit=0`
  (`disable_cost_limit=True`) and we derive cost ourselves from token counts via
  `pricing.csv`.
- **Don't force `thought_action`.** litellm *warns* that every `openrouter/*`
  model "does not support function calling", but for tool-capable models
  (gpt-4o(-mini), sonnet, deepseek, ...) OpenRouter still serves clean tool
  calls, and SWE-agent's default (`function_calling`) works well. The adapter
  defaults `parse_function=None` (⇒ SWE-agent's default). Forcing
  `thought_action` on a tool-capable model actually *breaks* it: it emits raw
  Python as the command and SWE-ReX's bashlex pre-check rejects it
  (`BashIncorrectSyntaxError`) → dozens of wasted steps. Only set
  `thought_action` for a model that genuinely lacks tool-calling (e.g.
  `z-ai/glm-5.2`) — and expect thrash there, so prefer a tool-capable model.
- **The ACI is not model-neutral.** `z-ai/glm-5.2` has no tool-calling, is forced
  onto `thought_action`, and mangles it (raw-Python-as-command → bashlex reject).
  The *same* glm solves cleanly in our JSON-action harness — scaffold
  performance depends on the **model × ACI** pair, not the model alone.
- **Bound the run with a call limit.** We disable SWE-agent's cost guard (it
  can't price OpenRouter models), so *nothing* stops a weak model that
  degenerates into a loop (seen: gpt-4o-mini did 269 steps re-running the same
  repro script). The adapter always passes
  `--agent.model.per_instance_call_limit` (`call_limit=75`), which raises
  `InstanceCallLimitExceededError` and ends the run. A legit solve here finishes
  in ~15-25 calls.
- **Strip build artifacts from the patch and changed-file list.** `pip install
  -e .` (baseline + the fast path's `post_startup_commands`) drops
  `*.egg-info/`, `__pycache__/`, `*.pyc` into the repo; SWE-agent's submit diff
  includes them, which (a) breaks `git apply` onto the clean workspace and (b)
  makes the quality reviewer ding "unrelated egg-info files". Fixes:
  `_strip_artifact_hunks` drops those diff sections before apply, and the
  changed-file / review-patch git-diffs exclude them with **`:(exclude,glob)`**
  pathspecs (plain `:(exclude)**/…` does *not* match root-level `parse.egg-info/`
  — the `glob` magic is required).
- **SWE-ReX containers leak on kill.** `pkill` of the python process does **not**
  stop the detached Docker container (a *clean* finish does stop it). Clean up
  with `docker ps -a --filter name=python3.11- -q | xargs docker rm -f` (and
  `--filter ancestor=swe-adapter-base:py311`). Killing runs mid-flight can wedge
  SWE-ReX; if a run hangs before "Starting environment", `docker system prune
  -f` and start fresh.
- **Streamed log, live.** The adapter runs SWE-agent with `PYTHONUNBUFFERED=1`
  and streams stdout to `<workspace>/swe_agent/sweagent.log`; without the
  unbuffering the file updates in chunks and a healthy run *looks* hung.

## Fast path (default): skip the slow standalone-python build

By default the adapter runs on a prebuilt base image and skips SWE-ReX's
multi-minute standalone-python build, cutting startup from minutes to seconds —
using SWE-agent's *own* config, nothing bespoke:

- `_ensure_base_image()` builds `swe-adapter-base:py311` once
  (`python:3.11-slim` + `git` + `swe-rex`, pinned to the installed version). No
  repo/task contents ⇒ no answer leak.
- An overlay config (merged after `default.yaml`) sets
  `env.deployment.image`, `env.deployment.python_standalone_dir: ""` (skip the
  build — needs swe-rex already in the image, hence the base), and
  `env.post_startup_commands` = the **same** setup commands single-agent uses
  (`DEFAULT_SETUP_COMMANDS`) so deps are installed identically, at run time, with
  nothing baked in.

This keeps the comparison fair: the fast path is only infrastructure. Compare on
`resolved@1`, overfitting, tool-use, regression, **token** cost, quality — *not*
wall-clock (which the prebuilt image changes by design). `--swe-no-fast` reverts
to SWE-agent's slow default startup.

macOS is workable (the fast path removes most of the pain); a Linux/CI host or
`--env.deployment.type=modal` is still smoother, and our `single`/`multi` agents
have no SWE-ReX dependency at all.

## Validated result (parse_pr_227, on macOS + Docker)

| agent | model | outcome |
|---|---|---|
| single (ours) | z-ai/glm-5.2 | **solved** — hidden pass, 25 clean steps |
| swe-agent | z-ai/glm-5.2 | ACI mismatch — forced onto thought_action, bashlex thrash, no usable patch |
| swe-agent | openai/gpt-4o-mini | clean bounded run (fast path, function_calling, ≤17 steps, no artifacts) but the **fix is wrong** → hidden **fail** |

The end-to-end `--agent swe-agent` pipeline is **confirmed working on macOS**:
fast start, clean tool-calls, bounded steps, artifact-free patch/`changed_files`,
patch applied + evaluated + reviewed + recorded, container cleaned up.

gpt-4o-mini is **unreliable** on this subtle bug — across runs it variously
patched the wrong regex branch, only wrote a throwaway repro script, or
degenerated into a 269-step loop. That's a model-quality signal the eval catches
(hidden fail / low quality), not a pipeline defect. For a meaningful
single-vs-swe comparison use a stronger tool-capable model
(`deepseek/deepseek-chat`, `anthropic/claude-sonnet-4.5`).
