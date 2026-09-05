# Kernel v2 campaign preflight review

The original `preflight.py` wraps `src.benchmark.validate.validate_task`. It is useful as an older collection check, but is not an adequate gate for this campaign. `validate.py` starts network-enabled containers, accepts shell exit codes without trusted JUnit evidence, accepts arbitrary hidden-base failures, and does not require visible tests to pass on the reference patch.

Use the plan-local `strict_preflight.py` with a frozen campaign collection. It imports the current evaluator and does not modify harness or task inputs. The script has been authored but has not been executed by its author; the root agent must inspect and run it.

## Implemented gates

- Require locally available Docker images and resolve them to immutable image IDs before creating containers.
- Use `prepare_workspace` and `clean_workspace_repo`; require copied HEAD to equal `base_commit`.
- Build the reference in a separate workspace by applying the local reference patch.
- Freeze test controls, pytest bootstrap packages, setup dependencies, and setup artifacts with the v2 APIs.
- Force dependency setup and all evaluation containers offline. An unavailable dependency is an environment failure, not a task failure or automatic reason to enable the internet.
- Evaluate base and reference with `evaluate_solution`, including reference visible tests and regression comparison.
- Capture raw JUnit XML before the evaluator removes its isolated temporary reports. Preserve XML, command output, parsed evidence, and immutable timestamped verdicts.
- Require a nonempty visible passing set. Visible suites may contain recorded platform skips, but skip-only success does not qualify.
- Require real failing cases for the base semantic suite: complete JUnit, enough selected cases, at least one failure, no errors, and no skips. A missing newly introduced API causing collection failure is deliberately rejected by this campaign's strong acceptance rule; it must not silently count as demonstrated behavioral failure.
- Require all reference hidden suites to pass the actual strict v2 evaluator, including no skipped hidden outcomes. Require reference visible success and zero regressions.
- Preserve previous artifacts; the stable `<task_id>.json` points to the latest result while each run retains a timestamped verdict directory.
- Recompute task input hashes at the end to detect concurrent source changes.

## Wrapper and image checks

Only invoked `.sh` wrappers are inspected. Unused legacy wrappers do not cause rejection. Original selected Ansible registry commands invoke pytest directly; their old `run_tests.sh` is irrelevant unless the frozen collection selects it.

Selected OpenLibrary wrappers contain `PYTHONPATH=/workspace:/app`. The v2 trusted pytest launcher explicitly rejects `/app`. Merely obtaining a successful result from the old network-enabled validator would not make these tasks executable under v2. The wrapper cannot be rescued by weakening the trusted evaluator; any approved dependency/runner adaptation must be frozen as campaign inputs before model calls.

The sandbox mounts `/workspace` but does not remove the image's baked `/app` checkout. Read-only rootfs does not prevent reading that second copy. The strict script compares baked files at changed production paths against reference bytes, and withheld paths against hidden bytes, excluding bytes already present in the task base. Exact reference/withheld matches block validation. If `/app/.git` exists, its HEAD must match the declared base. This is a targeted audit, not a proof that arbitrary image paths/history contain no extra information. A derivative image with the baked checkout removed is preferable; independently required vendored dependencies must be preserved as dependencies rather than exposing the project source tree.

## Launcher contract

Import `compute_input_hashes(row)` without calling `main`. It returns `registry_row_sha256`, `task_sha256`, `reference_patch_sha256`, `repository_tree_sha256`, and `hidden_tree_sha256`. It performs local read-only hashing and `git ls-files`, with no Docker or network calls.

Latest verdict: `strict_preflight/<task_id>.json`.

Required summary fields: `task_id`, `verified`, `base_visible_passed`, `base_hidden_semantic_failed`, `reference_visible_passed`, `reference_hidden_semantic_passed`, `reference_hidden_compat_passed`, `input_hashes`, `image_identity`, and `evidence`. Compatibility fields are `null` when no separate compatibility command exists. In that case the launcher should also require `reference_evaluation.task_success == true` and the visible/regression gates rather than interpreting null as failure.

Each evidence entry includes `path` and `sha256` for saved XML when available, plus `phase`, `report_label`, `command`, `junit`, `passing_ids`, `passed`, `output_path`, `output_sha256`, `invocation_valid`, and `selection_lower_bound`. An entry with missing XML has no `path` and cannot qualify for a verified task. Launcher checks should require existing evidence files matching their hashes.

The script writes only below this plan directory and does not append benchmark runs, sweep attempts, infrastructure failures, or collection status. It creates no model clients and makes no LLM calls.
