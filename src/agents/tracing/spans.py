"""The concrete ``trace_*`` decorators, wiring payloads and describers.

Each pairs a static fallback name with its input/output shaping and its
dynamic ``on_enter`` / ``on_exit`` describers. These are what the agent
modules import and apply to their methods.
"""

from __future__ import annotations

from . import describers, payloads
from .decorator import make_traced


def trace_workspace_op(name: str):
    """Trace one workspace file operation as a LangSmith tool run.

    Failed calls (e.g. inspecting a hallucinated path) become error runs,
    so tool-use validity can be read straight from the trace.
    """
    return make_traced(
        name=name,
        run_type="tool",
        process_inputs=payloads.tool_inputs,
        process_outputs=payloads.tool_outputs,
        on_enter=describers.describe_workspace(name.split(".")[-1]),
    )


trace_model_call = make_traced(
    name="LangChainModel.generate",
    run_type="chain",
    process_inputs=payloads.generate_inputs,
    process_outputs=payloads.generate_outputs,
    on_enter=describers.enter_generate,
)

trace_summarize_call = make_traced(
    name="LangChainModel.summarize",
    run_type="chain",
    process_inputs=payloads.summarize_inputs,
    process_outputs=payloads.generate_outputs,
)

trace_agent_run = make_traced(
    name="SingleAgent.run",
    run_type="chain",
    process_inputs=payloads.run_inputs,
    process_outputs=payloads.run_outputs,
)

trace_agent_step = make_traced(
    name="SingleAgent.step",
    run_type="chain",
    process_inputs=payloads.step_inputs,
    process_outputs=payloads.step_outputs,
    on_exit=describers.exit_step,
)

trace_action_execution = make_traced(
    name="ActionExecutor.execute",
    run_type="tool",
    process_inputs=payloads.action_inputs,
    process_outputs=payloads.action_outputs,
    on_enter=describers.enter_execute,
)

trace_shell_command = make_traced(
    name="DockerSandbox.run_shell",
    run_type="tool",
    process_inputs=payloads.shell_inputs,
    process_outputs=payloads.shell_outputs,
    on_enter=describers.enter_shell,
    on_exit=describers.exit_shell,
)

trace_compaction = make_traced(
    name="ContextCompactor.compact",
    run_type="chain",
    process_inputs=payloads.compaction_inputs,
    process_outputs=payloads.compaction_outputs,
    on_enter=describers.enter_compaction,
)
