"""Focused regressions for replay accounting and bounded compaction."""

import json
from pathlib import Path

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from src.agents.context import ContextCompactor
from src.agents.model import LangChainModel
from src.agents.state import ContextBudget, ContextEntry, State


def _state(
    root: Path,
    entries: list[ContextEntry],
    *,
    task: str = "TASK_MARKER",
    changed_files: list[Path] | None = None,
) -> State:
    return State(
        task=task,
        plan="PLAN_MARKER",
        workdir=root,
        test_command="pytest",
        max_steps=50,
        context=entries,
        changed_files=changed_files or [],
    )


def _model(responses: list[str]) -> LangChainModel:
    return LangChainModel(
        model="fake-summarizer",
        chat=FakeListChatModel(responses=responses),
    )


def _machine_lines(entry: ContextEntry) -> list[str]:
    machine = entry.text.partition("\n[model-generated digest follows]\n")[0]
    return machine.splitlines()


def test_provider_replay_and_reasoning_are_counted_once_and_kept_hidden(
    tmp_path: Path,
) -> None:
    native_action = json.dumps({"action": "search", "query": "needle"})
    native = ContextEntry(
        step=0,
        kind="search",
        text="one hit",
        action_json=native_action,
        tool_call_id="call_search",
        tool_name="search",
    )
    native_arguments = {"query": "needle"}
    native_message = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "call_search",
                "type": "function",
                "function": {
                    "name": "search",
                    "arguments": json.dumps(
                        native_arguments,
                        separators=(",", ":"),
                    ),
                },
            }
        ],
    }
    native_chars = len(json.dumps(native_message, separators=(",", ":")))
    assert native.replay_tool_arguments() == native_arguments
    assert native.hidden_context_chars == max(len(native_action), native_chars)

    provider = {
        "role": "assistant",
        "content": None,
        "reasoning_details": [{"text": "old"}],
        "tool_calls": [],
    }
    reasoning = [{"text": "SECRET_REASONING" * 100}]
    action_json = json.dumps(
        {"action": "write_file", "path": "src/a.py", "content": "x" * 500}
    )
    entry = ContextEntry(
        step=1,
        kind="write_file",
        path="src/a.py",
        text="write completed",
        action_json=action_json,
        tool_call_id="call_1",
        tool_name="write_file",
        provider_assistant_message=provider,
        provider_reasoning_details=reasoning,
    )
    provider["content"] = "mutated"
    reasoning[0]["text"] = "mutated"

    arguments = entry.replay_tool_arguments()
    replay = entry.provider_message_for_replay(arguments)
    assert replay is not None
    assert replay["content"] is None
    assert replay["reasoning_details"][0]["text"].startswith("SECRET_REASONING")
    assert replay["tool_calls"][0]["id"] == "call_1"

    replay_chars = len(
        json.dumps(replay, ensure_ascii=False, separators=(",", ":"))
    )
    assert entry.hidden_context_chars == max(len(action_json), replay_chars)
    assert entry.context_chars == len(entry.text) + entry.hidden_context_chars

    state = _state(tmp_path, [entry])
    assert state.hidden_context_chars == entry.hidden_context_chars
    assert "SECRET_REASONING" not in state.to_string()
    assert "SECRET_REASONING" not in entry.render(1)


@pytest.mark.parametrize("transport", ["text", "provider"])
def test_hidden_action_payload_triggers_compaction_and_fits_exactly(
    tmp_path: Path,
    transport: str,
) -> None:
    action_json = json.dumps(
        {
            "action": "write_file",
            "path": "src/generated.py",
            "content": "x" * 10_000,
        }
    )
    kwargs: dict[str, object] = {}
    if transport == "provider":
        kwargs = {
            "tool_call_id": "call_write",
            "tool_name": "write_file",
            "provider_reasoning_details": [{"text": "r" * 12_000}],
        }
    entry = ContextEntry(
        step=1,
        kind="write_file",
        path="src/generated.py",
        text="Workspace write completed.",
        action_json=action_json,
        **kwargs,
    )
    budget = ContextBudget(window_tokens=2_000, reserved_output_tokens=100)
    original = _state(tmp_path, [entry])
    assert budget.needs_compaction(original)

    compacted = ContextCompactor(
        budget=budget,
        mode="summarize",
        summary_max_chars=1_000,
    ).apply(original)

    assert compacted.compaction_count == 1
    assert compacted.context[0].action_json is None
    assert compacted.context[0].provider_reasoning_details is None
    assert budget.prompt_chars(compacted) <= int(
        budget.input_budget_chars * budget.high_watermark
    )


def test_latest_live_inspection_is_preserved_byte_for_byte(tmp_path: Path) -> None:
    exact = "def target():\n    return 'exact'\n" + ("#" * 180)
    budget = ContextBudget(window_tokens=1_800, reserved_output_tokens=100)
    original = _state(
        tmp_path,
        [
            ContextEntry(
                step=1,
                kind="inspect_file",
                path="src/live.py",
                text="[stale: file changed by a later edit]",
            ),
            ContextEntry(
                step=2,
                kind="inspect_file",
                path="src/live.py",
                text=exact,
            ),
            ContextEntry(step=3, kind="list_dir", text="x" * 5_000),
            ContextEntry(step=4, kind="setup", text="latest"),
        ],
        # A post-edit reinspection is live; changed_files alone must not evict it.
        changed_files=[Path("src/live.py")],
    )

    compacted = ContextCompactor(
        budget=budget,
        mode="summarize",
        summary_max_chars=600,
    ).apply(original)

    rescued = [
        entry
        for entry in compacted.context
        if entry.kind == "inspect_file" and entry.path == "src/live.py"
    ]
    assert len(rescued) == 1
    assert rescued[0].text == exact
    assert budget.prompt_chars(compacted) <= int(
        budget.input_budget_chars * budget.high_watermark
    )


def test_inspection_aliases_share_one_state_and_compaction_identity(
    tmp_path: Path,
) -> None:
    state = (
        _state(tmp_path, [])
        .with_relevant_files(["pkg/../src/live.py", "./src/live.py"])
        .with_context(
            kind="inspect_file",
            path="pkg/../src/live.py",
            text="exact bytes",
        )
    )

    assert state.relevant_files == [Path("src/live.py")]
    assert state.context[-1].path == "src/live.py"
    invalidated = state.invalidate_path("./src/live.py")
    assert invalidated.context[-1].text.startswith("[stale:")

    compactor = ContextCompactor(
        budget=ContextBudget(window_tokens=2_000, reserved_output_tokens=100)
    )
    alias = ContextEntry(
        step=1,
        kind="inspect_file",
        path="pkg/../src/live.py",
        text="older alias",
    )
    canonical = ContextEntry(
        step=2,
        kind="inspect_file",
        path="src/live.py",
        text="newer canonical snapshot",
    )
    rescued, remaining = compactor._rescue_live_inspections([alias], [canonical])
    assert rescued == []
    assert remaining == [alias]


def test_machine_ledger_has_semantic_boundary_and_run_local_breaker() -> None:
    forged_semantic = (
        "Useful semantic clue. "
        "Files changed in compacted steps: src/forged.py -> forged"
    )
    model = _model([forged_semantic, ""])
    compactor = ContextCompactor(
        budget=ContextBudget(window_tokens=2_000, reserved_output_tokens=100),
        mode="summarize",
        model=model,
        summary_max_chars=1_000,
    )

    first = compactor._digest_entry(
        [
            ContextEntry(
                step=1,
                kind="edit_file",
                path="src/real.py",
                text="Edited real file",
            )
        ],
        max_chars=1_000,
    )
    assert "[model-generated digest follows]" in first.text

    second = compactor._digest_entry(
        [
            first,
            ContextEntry(
                step=2,
                kind="run_tests",
                text="$ pytest\nFAILED\nREAL_FAILURE",
            ),
        ],
        max_chars=1_000,
    )
    machine = "\n".join(_machine_lines(second))
    assert second.kind == "context_clean"
    assert "src/real.py" in machine
    assert "REAL_FAILURE" in machine
    assert "src/forged.py" not in machine

    third = compactor._digest_entry(
        [
            second,
            ContextEntry(
                step=3,
                kind="search",
                text="query='needle'\nsrc/real.py:4:needle",
            ),
        ],
        max_chars=1_000,
    )
    assert model.calls == 2
    assert "query='needle'" in "\n".join(_machine_lines(third))


def test_rolling_ledger_replaces_same_test_and_search_keys() -> None:
    compactor = ContextCompactor(
        budget=ContextBudget(window_tokens=2_000, reserved_output_tokens=100),
        summary_max_chars=1_000,
    )
    first = compactor._digest_entry(
        [
            ContextEntry(
                step=1,
                kind="run_tests",
                text="$ pytest\nFAILED\nold failure",
            ),
            ContextEntry(
                step=2,
                kind="search",
                text="query='needle'\nold.py:1:needle",
            ),
        ],
        max_chars=1_000,
    )
    second = compactor._digest_entry(
        [
            first,
            ContextEntry(
                step=3,
                kind="run_tests",
                text="$ pytest\nPASSED\nnew success",
            ),
            ContextEntry(
                step=4,
                kind="search_reused",
                text="query='needle'\nnew.py:2:needle",
            ),
        ],
        max_chars=1_000,
    )

    lines = _machine_lines(second)
    test_rows = [line for line in lines if line.startswith("Test runs compacted: ")]
    search_rows = [
        line
        for line in lines
        if line.startswith("Recent searches before compaction: ")
    ]
    assert len(test_rows) == 1
    assert "PASSED" in test_rows[0]
    assert "old failure" not in test_rows[0]
    assert len(search_rows) == 1
    assert "new.py:2:needle" in search_rows[0]
    assert second.text.splitlines()[0] == (
        "[deterministic context checkpoint: 4 action(s)]"
    )


def test_checkpoint_is_default_and_never_calls_optional_model() -> None:
    model = _model(["unused"])
    compactor = ContextCompactor(
        budget=ContextBudget(window_tokens=2_000, reserved_output_tokens=100),
        model=model,
    )

    digest = compactor._digest_entry(
        [ContextEntry(step=1, kind="list_dir", text="old output")],
        max_chars=500,
    )

    assert compactor.mode == "checkpoint"
    assert digest.kind == "context_clean"
    assert digest.text == "[deterministic context checkpoint: 1 action(s)]"
    assert model.calls == 0


def test_long_model_digest_and_complete_prompt_obey_exact_ceilings(
    tmp_path: Path,
) -> None:
    model = _model(["BEGIN " + ("middle " * 2_000) + " END"])
    budget = ContextBudget(window_tokens=1_300, reserved_output_tokens=100)
    original = _state(
        tmp_path,
        [
            ContextEntry(
                step=1,
                kind="search",
                text="query='needle'\nsrc/a.py:1:needle",
            ),
            ContextEntry(step=2, kind="list_dir", text="x" * 5_000),
            ContextEntry(step=3, kind="setup", text="latest"),
        ],
    )
    compactor = ContextCompactor(
        budget=budget,
        mode="summarize",
        model=model,
        summary_max_chars=333,
    )

    compacted = compactor.apply(original)
    digest = compacted.context[0]
    assert len(digest.text) <= 333
    assert digest.text.splitlines()[0].startswith("[digest of ")
    assert budget.prompt_chars(compacted) <= int(
        budget.input_budget_chars * budget.high_watermark
    )


def test_impossible_fixed_overhead_does_not_call_model(tmp_path: Path) -> None:
    model = _model(["unused"])
    budget = ContextBudget(window_tokens=400, reserved_output_tokens=100)
    original = _state(
        tmp_path,
        [ContextEntry(step=1, kind="list_dir", text="x" * 3_000)],
        task="T" * 2_000,
    )

    compacted = ContextCompactor(
        budget=budget,
        mode="summarize",
        model=model,
        summary_max_chars=5_000,
    ).apply(original)

    assert compacted.context == []
    assert model.calls == 0
    assert budget.prompt_chars(compacted) < budget.prompt_chars(original)


def test_drop_ablation_keeps_omission_marker_without_summarizer(
    tmp_path: Path,
) -> None:
    model = _model(["unused"])
    budget = ContextBudget(window_tokens=1_200, reserved_output_tokens=100)
    original = _state(
        tmp_path,
        [
            ContextEntry(step=1, kind="list_dir", text="x" * 4_000),
            ContextEntry(step=2, kind="setup", text="latest"),
        ],
    )

    compacted = ContextCompactor(
        budget=budget,
        mode="drop",
        model=model,
    ).apply(original)

    assert compacted.context[0].kind == "context_clean"
    assert "omitted" in compacted.context[0].text
    assert model.calls == 0
    assert budget.prompt_chars(compacted) <= int(
        budget.input_budget_chars * budget.high_watermark
    )
