import {
  Activity,
  AlertTriangle,
  BarChart3,
  CheckCircle2,
  Clock3,
  Database,
  Gauge,
  Layers,
  Play,
  RefreshCw,
  Search,
  Square,
  TableProperties,
  Terminal,
  XCircle
} from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import type { ReactNode } from "react";
import { fetchJobs, fetchMeta, fetchOverview, fetchRunDetail, startJob, startSweep, stopJob } from "./api";
import type {
  ActionTransport,
  JobRecord,
  LaunchPayload,
  Meta,
  MetricGroups,
  MetricSet,
  Overview,
  RunDetail,
  RunRecord,
  SessionOverview,
  SweepPayload,
  TaskDifficulty,
  TaskRecord
} from "./types";

type Tab = "overview" | "runs" | "tasks" | "launcher";
type MetricKey = keyof MetricSet;

const statusOrder = ["all", "solved", "failed", "handoff", "running"];

function splitCsv(value: string): string[] {
  return value.split(",").map((item) => item.trim()).filter(Boolean);
}

function normalizeActionTransport(value: string | null | undefined): ActionTransport | undefined {
  return value === "text_json" || value === "tools" || value === "auto" ? value : undefined;
}

function normalizeProvider(value: string | null | undefined): "openrouter" | "local" | undefined {
  return value === "openrouter" || value === "local" ? value : undefined;
}

function inferProvider(model: string | null | undefined): "openrouter" | "local" {
  return model && model.includes("/") ? "openrouter" : "local";
}

function launchPayloadFromRun(run: RunRecord): LaunchPayload {
  const agent = run.agent_mode || "single";
  return {
    task_id: run.task_id,
    provider: normalizeProvider(run.provider) ?? inferProvider(run.model),
    model: run.model || undefined,
    agent,
    action_transport: agent === "swe-agent"
      ? undefined
      : normalizeActionTransport(run.action_transport) ?? "text_json",
    session: run.session_id || undefined
  };
}

function commandOption(command: string[], flag: string): string | undefined {
  const index = command.indexOf(flag);
  if (index === -1 || index + 1 >= command.length) {
    return undefined;
  }
  return command[index + 1];
}

function commandNumberOption(command: string[], flag: string): number | undefined {
  const value = commandOption(command, flag);
  if (!value) {
    return undefined;
  }
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : undefined;
}

function commandHasFlag(command: string[], flag: string): boolean {
  return command.includes(flag);
}

function nextSessionName(value: string | null | undefined): string {
  const base = (value || "default").trim() || "default";
  const stamp = new Date().toISOString().slice(0, 16).replace(/[-:T]/g, "");
  return `${base}_rerun_${stamp}`;
}

function sweepPayloadFromJob(job: JobRecord): SweepPayload {
  const command = job.command || [];
  return {
    tasks: commandOption(command, "--tasks") || job.task_id || "all",
    models: commandOption(command, "--models") || job.model || "",
    agents: commandOption(command, "--agents") || job.agent || "single",
    action_transport: normalizeActionTransport(
      commandOption(command, "--action-transport") || job.action_transport
    ) ?? "text_json",
    reasoning_effort:
      commandOption(command, "--reasoning-effort") || job.reasoning_effort || undefined,
    session: nextSessionName(commandOption(command, "--session") || job.session),
    provider: normalizeProvider(commandOption(command, "--provider") || job.provider) ?? "openrouter",
    concurrency: commandNumberOption(command, "--concurrency"),
    max_steps: commandNumberOption(command, "--max-steps"),
    max_iterations: commandNumberOption(command, "--max-iterations"),
    no_score: commandHasFlag(command, "--no-score"),
    no_regression: commandHasFlag(command, "--no-regression"),
    enable_review: commandHasFlag(command, "--enable-review"),
    no_network: commandHasFlag(command, "--no-network"),
    no_setup: commandHasFlag(command, "--no-setup")
  };
}

function readCookie(name: string): string | null {
  const escaped = name.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  const match = document.cookie.match(new RegExp(`(?:^|; )${escaped}=([^;]*)`));
  return match ? decodeURIComponent(match[1]) : null;
}

function writeCookie(name: string, value: string, days = 180): void {
  const expires = new Date(Date.now() + days * 864e5).toUTCString();
  document.cookie = `${name}=${encodeURIComponent(value)}; expires=${expires}; path=/; SameSite=Lax`;
}

// useState that persists its (string) value to a cookie — remembers the last
// selected session across reloads.
function useCookieState(key: string, fallback: string): [string, (next: string) => void] {
  const [value, setValue] = useState<string>(() => readCookie(key) ?? fallback);
  const set = (next: string) => {
    setValue(next);
    writeCookie(key, next);
  };
  return [value, set];
}

export function App() {
  const [overview, setOverview] = useState<Overview | null>(null);
  const [jobs, setJobs] = useState<JobRecord[]>([]);
  const [activeTab, setActiveTab] = useState<Tab>("overview");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  async function load() {
    setLoading(true);
    setError(null);
    try {
      const [nextOverview, nextJobs] = await Promise.all([fetchOverview(), fetchJobs()]);
      setOverview(nextOverview);
      setJobs(nextJobs);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    void load();
  }, []);

  useEffect(() => {
    const timer = window.setInterval(() => {
      void fetchJobs().then(setJobs).catch(() => undefined);
    }, 4000);
    return () => window.clearInterval(timer);
  }, []);

  const runningJobs = jobs.filter((job) => job.status === "running").length;

  return (
    <main className="app">
      <header className="topbar">
        <div>
          <div className="eyebrow">Repository-level dev agent evaluation</div>
          <h1>Benchmark Control Console</h1>
        </div>
        <div className="topbarActions">
          <span className={runningJobs ? "liveBadge active" : "liveBadge"}>
            {runningJobs ? `${runningJobs} running` : "idle"}
          </span>
          <button className="iconButton" onClick={() => void load()} disabled={loading} title="Refresh">
            <RefreshCw size={17} />
          </button>
        </div>
      </header>

      {error ? <div className="errorBand">{error}</div> : null}

      <nav className="tabs" aria-label="Console sections">
        <TabButton icon={<BarChart3 size={16} />} label="Overview" tab="overview" activeTab={activeTab} onClick={setActiveTab} />
        <TabButton icon={<TableProperties size={16} />} label="Runs" tab="runs" activeTab={activeTab} onClick={setActiveTab} />
        <TabButton icon={<Database size={16} />} label="Tasks" tab="tasks" activeTab={activeTab} onClick={setActiveTab} />
        <TabButton icon={<Terminal size={16} />} label="Launcher" tab="launcher" activeTab={activeTab} onClick={setActiveTab} />
      </nav>

      {!overview && loading ? (
        <div className="loading">Loading console data...</div>
      ) : null}

      {overview && activeTab === "overview" ? <OverviewTab overview={overview} /> : null}
      {overview && activeTab === "runs" ? (
        <RunsTab overview={overview} onJobsChange={setJobs} />
      ) : null}
      {overview && activeTab === "tasks" ? <TasksTab tasks={overview.tasks} /> : null}
      {overview && activeTab === "launcher" ? (
        <LauncherTab overview={overview} jobs={jobs} onJobsChange={setJobs} />
      ) : null}
    </main>
  );
}

function TabButton({
  icon,
  label,
  tab,
  activeTab,
  onClick
}: {
  icon: ReactNode;
  label: string;
  tab: Tab;
  activeTab: Tab;
  onClick: (tab: Tab) => void;
}) {
  return (
    <button className={activeTab === tab ? "tab active" : "tab"} onClick={() => onClick(tab)}>
      {icon}
      {label}
    </button>
  );
}

function OverviewTab({ overview }: { overview: Overview }) {
  const [sessionId, setSessionId] = useCookieState("console.session", "all");
  const activeSession = useMemo(
    () => overview.sessions.find((item) => item.session_id === sessionId) ?? null,
    [overview.sessions, sessionId]
  );
  useEffect(() => {
    if (sessionId !== "all" && !overview.sessions.some((item) => item.session_id === sessionId)) {
      setSessionId("all");
    }
  }, [overview.sessions, sessionId]);

  const sessionRuns = useMemo(
    () =>
      sessionId === "all"
        ? overview.runs
        : overview.runs.filter((run) => (run.session_id || "default") === sessionId),
    [overview.runs, sessionId]
  );
  const groups: MetricGroups = activeSession?.groups ?? overview.metrics;
  const metrics = activeSession?.metrics ?? overview.metrics.overall;
  const calls = activeSession?.calls ?? overview.calls;
  const actions = activeSession?.actions ?? overview.actions;
  const taskDifficultyRows = activeSession?.task_difficulty ?? overview.task_difficulty;
  const recentRuns = useMemo(() => newestRuns(sessionRuns).slice(0, 12), [sessionRuns]);
  const agentModes = Object.keys(groups.by_agent_mode);
  const scopedRunCount = activeSession?.runs ?? overview.summary.runs;
  const scopedTaskCount = activeSession?.tasks ?? overview.summary.run_tasks;
  const scopedModelCount = activeSession?.models.length ?? Object.keys(groups.by_model).length;
  const sessionLabel = sessionId === "all" ? "all sessions" : sessionId;

  return (
    <section className="page">
      <div className="kpis">
        <Kpi icon={<Gauge size={17} />} label="Task success" value={formatRate(metrics?.task_success_rate)} />
        <Kpi icon={<CheckCircle2 size={17} />} label="Visible pass" value={formatRate(metrics?.visible_test_pass_rate)} />
        <Kpi icon={<CheckCircle2 size={17} />} label="Semantic hidden" value={formatRate(metrics?.hidden_semantic_pass_rate)} />
        <Kpi icon={<Activity size={17} />} label="Compat hidden" value={formatRate(metrics?.hidden_compat_pass_rate)} />
        <Kpi icon={<Layers size={17} />} label="PR parity" value={formatRate(metrics?.hidden_pr_parity_pass_rate)} />
        <Kpi icon={<Activity size={17} />} label="Runs / tasks" value={`${formatNumber(scopedRunCount)}/${formatNumber(scopedTaskCount)}`} />
        <Kpi icon={<TableProperties size={17} />} label="Agent modes" value={String(agentModes.length)} />
        <Kpi icon={<Clock3 size={17} />} label="Mean duration" value={formatSeconds(metrics?.mean_duration_s)} />
        <Kpi icon={<Terminal size={17} />} label="LLM calls" value={formatNumber(calls.llm_calls)} />
      </div>

      <section className="panel sessionPanel">
        <PanelHeader title="Session Overview" meta={sessionLabel} />
        <div className="sessionControls">
          <label className="sessionSelect">
            Session
            <select value={sessionId} onChange={(event) => setSessionId(event.target.value)}>
              <option value="all">all sessions</option>
              {overview.sessions.map((session) => (
                <option key={session.session_id} value={session.session_id}>
                  {session.session_id}
                </option>
              ))}
            </select>
          </label>
          <div className="sessionFacts">
            <SessionFact label="Runs" value={formatNumber(scopedRunCount)} />
            <SessionFact label="Tasks" value={formatNumber(scopedTaskCount)} />
            <SessionFact label="Agents" value={formatNumber(agentModes.length)} />
            <SessionFact label="Models" value={formatNumber(scopedModelCount)} />
            <SessionFact label="Cost" value={formatCost(calls.total_cost_usd)} />
            <SessionFact label="Last run" value={formatDate(activeSession?.last_finished_at ?? overview.summary.last_finished_at)} />
          </div>
        </div>
        <SessionComparison
          sessions={overview.sessions}
          selectedSession={sessionId}
          onSelect={setSessionId}
        />
      </section>

      <div className="grid two">
        <section className="panel">
          <PanelHeader title="Outcome By Agent Mode" meta={sessionLabel} />
          <GroupedBars groups={groups.by_agent_mode} primary="task_success_rate" secondary="visible_test_pass_rate" />
        </section>
        <section className="panel">
          <PanelHeader title="Outcome By Task Type" meta={sessionLabel} />
          <GroupedBars groups={groups.by_task_type} primary="task_success_rate" secondary="mean_quality_score" secondaryMax={5} />
        </section>
      </div>

      <section className="panel widePanel">
        <PanelHeader title="Agent Comparison" meta={sessionLabel} />
        <AgentComparison groups={groups.by_agent_mode} />
      </section>

      <div className="grid two">
        <section className="panel">
          <PanelHeader title="Outcome By Difficulty" meta={sessionLabel} />
          <GroupedBars groups={groups.by_difficulty} primary="task_success_rate" secondary="visible_test_pass_rate" />
        </section>
        <section className="panel">
          <PanelHeader title="Measured Task Difficulty" meta={sessionLabel} />
          <DifficultyTable rows={taskDifficultyRows} />
        </section>
      </div>

      <div className="grid two">
        <section className="panel">
          <PanelHeader title="Static Difficulty Estimate" meta={sessionLabel} />
          <GroupedBars groups={groups.by_difficulty_estimate} primary="task_success_rate" secondary="mean_quality_score" secondaryMax={5} />
        </section>
        <section className="panel">
          <PanelHeader title="Outcome By Action Transport" meta={sessionLabel} />
          <GroupedBars groups={groups.by_action_transport} primary="task_success_rate" secondary="tool_use_validity_rate" />
        </section>
      </div>

      <div className="grid two">
        <section className="panel">
          <PanelHeader title="Token Use By Model" meta={sessionLabel} />
          <TokenChart runs={sessionRuns} />
        </section>
        <section className="panel">
          <PanelHeader title="Recent Run Timeline" meta={sessionLabel} />
          <TimelineChart runs={recentRuns} />
        </section>
      </div>

      <div className="grid two">
        <section className="panel">
          <PanelHeader title="Action Mix" meta={sessionLabel} />
          <ActionBars actions={actions} />
        </section>
      </div>

      <div className="grid two">
        <section className="panel">
          <PanelHeader title="Collection State" meta="benchmark inventory" />
          <div className="inventory">
            <InventoryItem label="Collection tasks" value={overview.summary.collection_tasks} />
            <InventoryItem label="Verified tasks" value={overview.summary.verified_tasks} />
            <InventoryItem label="Unverified tasks" value={overview.summary.unverified_tasks} />
            <InventoryItem label="Unknown run tasks" value={overview.summary.unknown_run_tasks.length} warning={overview.summary.unknown_run_tasks.length > 0} />
          </div>
          {overview.summary.unknown_run_tasks.length ? (
            <div className="warningLine">
              <AlertTriangle size={15} />
              {overview.summary.unknown_run_tasks.join(", ")}
            </div>
          ) : null}
        </section>
      </div>
    </section>
  );
}

function RunsTab({
  overview,
  onJobsChange
}: {
  overview: Overview;
  onJobsChange: (jobs: JobRecord[]) => void;
}) {
  const [query, setQuery] = useState("");
  const [status, setStatus] = useState("all");
  const [agentMode, setAgentMode] = useState("all");
  const [transport, setTransport] = useState("all");
  const [session, setSession] = useCookieState("console.session", "all");
  const [selectedRunId, setSelectedRunId] = useState<string | null>(overview.runs[0]?.run_id ?? null);
  const [detail, setDetail] = useState<RunDetail | null>(null);
  const [detailError, setDetailError] = useState<string | null>(null);
  const [rerunRunId, setRerunRunId] = useState<string | null>(null);

  const runs = useMemo(() => newestRuns(overview.runs), [overview.runs]);
  const agentModes = useMemo(
    () => Array.from(new Set(runs.map((run) => run.agent_mode))).sort(),
    [runs]
  );
  const sessions = useMemo(
    () => Array.from(new Set(runs.map((run) => run.session_id || "default"))).sort(),
    [runs]
  );
  const transports = useMemo(
    () => Array.from(new Set(runs.map((run) => run.action_transport || "text_json"))).sort(),
    [runs]
  );
  const filtered = useMemo(() => {
    const needle = query.trim().toLowerCase();
    return runs.filter((run) => {
      if (status !== "all" && run.status !== status) {
        return false;
      }
      if (agentMode !== "all" && run.agent_mode !== agentMode) {
        return false;
      }
      if (session !== "all" && (run.session_id || "default") !== session) {
        return false;
      }
      if (transport !== "all" && (run.action_transport || "text_json") !== transport) {
        return false;
      }
      if (!needle) {
        return true;
      }
      return [
        run.task_id,
        run.run_id,
        run.model,
        run.agent_mode,
        run.provider,
        run.action_transport,
        run.session_id,
        run.status
      ]
        .filter(Boolean)
        .some((value) => String(value).toLowerCase().includes(needle));
    });
  }, [agentMode, query, runs, session, status, transport]);

  useEffect(() => {
    if (!selectedRunId) {
      setDetail(null);
      return;
    }
    setDetailError(null);
    void fetchRunDetail(selectedRunId)
      .then(setDetail)
      .catch((err) => setDetailError(err instanceof Error ? err.message : String(err)));
  }, [selectedRunId]);

  async function rerun(run: RunRecord) {
    setRerunRunId(run.run_id);
    setDetailError(null);
    try {
      await startJob(launchPayloadFromRun(run));
      onJobsChange(await fetchJobs());
    } catch (err) {
      setDetailError(err instanceof Error ? err.message : String(err));
    } finally {
      setRerunRunId(null);
    }
  }

  return (
    <section className="page runsPage">
      <div className="toolbar">
        <label className="searchBox">
          <Search size={16} />
          <input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search task, run, model" />
        </label>
        <select value={status} onChange={(event) => setStatus(event.target.value)}>
          {statusOrder.map((item) => (
            <option key={item} value={item}>
              {item}
            </option>
          ))}
        </select>
        <select value={agentMode} onChange={(event) => setAgentMode(event.target.value)}>
          <option value="all">all agents</option>
          {agentModes.map((item) => (
            <option key={item} value={item}>
              {item}
            </option>
          ))}
        </select>
        <select value={session} onChange={(event) => setSession(event.target.value)}>
          <option value="all">all sessions</option>
          {sessions.map((item) => (
            <option key={item} value={item}>
              {item}
            </option>
          ))}
        </select>
        <select value={transport} onChange={(event) => setTransport(event.target.value)}>
          <option value="all">all transports</option>
          {transports.map((item) => (
            <option key={item} value={item}>
              {item}
            </option>
          ))}
        </select>
      </div>

      <div className="split">
        <section className="panel tablePanel">
          <table className="dataTable runsTable">
            <thead>
              <tr>
                <th>Task</th>
                <th>Agent</th>
                <th>Transport</th>
                <th>Session</th>
                <th>Status</th>
                <th>Success</th>
                <th>Semantic Hidden</th>
                <th>Compat Hidden</th>
                <th>PR Parity</th>
                <th>Difficulty</th>
                <th>Model</th>
                <th>Calls</th>
                <th>Tokens</th>
                <th>Cost</th>
                <th>Finished</th>
                <th></th>
              </tr>
            </thead>
            <tbody>
              {filtered.map((run) => (
                <tr
                  key={run.run_id}
                  className={selectedRunId === run.run_id ? "selected" : ""}
                  onClick={() => setSelectedRunId(run.run_id)}
                >
                  <td>
                    <div className="strong">{run.task_id}</div>
                    <div className="muted">{shortId(run.run_id)}</div>
                  </td>
                  <td><AgentPill value={run.agent_mode} /></td>
                  <td className="mono smallCell">{run.action_transport || "text_json"}</td>
                  <td className="mono smallCell">{run.session_id || "default"}</td>
                  <td><StatusPill status={run.status} /></td>
                  <td><HiddenPill value={taskSuccessValue(run)} /></td>
                  <td><HiddenPill value={run.hidden_semantic_tests_passed} /></td>
                  <td><HiddenPill value={run.hidden_compat_tests_passed} /></td>
                  <td><HiddenPill value={run.hidden_pr_parity_tests_passed} /></td>
                  <td><DifficultyPill value={run.difficulty} fallback={run.difficulty_estimate} /></td>
                  <td className="mono smallCell">{run.model}</td>
                  <td>{formatNumber(run.llm_calls)}</td>
                  <td>{formatNumber(run.total_tokens)}</td>
                  <td>{formatCost(run.cost_usd)}</td>
                  <td>{formatDate(run.finished_at)}</td>
                  <td>
                    <button
                      className="iconButton small"
                      disabled={rerunRunId === run.run_id}
                      onClick={(event) => {
                        event.stopPropagation();
                        void rerun(run);
                      }}
                      title="Rerun in same session"
                    >
                      <Play size={14} />
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>

        <aside className="panel detailPanel">
          {detailError ? <div className="errorBand compact">{detailError}</div> : null}
          {detail ? (
            <RunDetailView
              detail={detail}
              onRerun={rerun}
              rerunning={rerunRunId === detail.run.run_id}
            />
          ) : <div className="muted">Select a run.</div>}
        </aside>
      </div>
    </section>
  );
}

function RunDetailView({
  detail,
  onRerun,
  rerunning
}: {
  detail: RunDetail;
  onRerun: (run: RunRecord) => void;
  rerunning: boolean;
}) {
  const run = detail.run;
  return (
    <div className="detailStack">
      <div className="detailHeader">
        <div>
          <div className="eyebrow">Run detail</div>
          <h2>{run.task_id}</h2>
          <div className="muted mono">{run.run_id}</div>
        </div>
        <button
          className="miniButton"
          disabled={rerunning}
          onClick={() => onRerun(run)}
        >
          <Play size={14} />
          Run
        </button>
      </div>
      <div className="detailGrid">
        <DetailItem label="Status" value={<StatusPill status={run.status} />} />
        <DetailItem label="Agent" value={<AgentPill value={run.agent_mode} />} />
        <DetailItem label="Transport" value={run.action_transport || "text_json"} />
        <DetailItem label="Task Success" value={<HiddenPill value={taskSuccessValue(run)} />} />
        <DetailItem label="Required Hidden" value={<HiddenPill value={run.hidden_required_tests_passed ?? run.hidden_tests_passed} />} />
        <DetailItem label="Semantic Hidden" value={<HiddenPill value={run.hidden_semantic_tests_passed} />} />
        <DetailItem label="Compat Hidden" value={<HiddenPill value={run.hidden_compat_tests_passed} />} />
        <DetailItem label="PR Parity" value={<HiddenPill value={run.hidden_pr_parity_tests_passed} />} />
        <DetailItem label="Quality" value={run.quality_score == null ? "n/a" : `${run.quality_score}/5`} />
        <DetailItem label="Difficulty" value={<DifficultyPill value={run.difficulty} fallback={run.difficulty_estimate} />} />
        <DetailItem label="Duration" value={formatSeconds(run.duration_s)} />
        <DetailItem label="LLM calls" value={formatNumber(run.llm_calls)} />
        <DetailItem label="Tokens" value={formatNumber(run.total_tokens)} />
        <DetailItem label="Cost" value={formatCost(run.cost_usd)} />
        <DetailItem label="Regressions" value={run.regressions == null ? "n/a" : String(run.regressions)} />
      </div>
      <section>
        <h3>Changed Files</h3>
        <div className="chips">
          {run.changed_files.length ? run.changed_files.map((file) => <span key={file}>{file}</span>) : <span>none</span>}
        </div>
      </section>
      <section>
        <h3>Action Counts</h3>
        <ActionBars actions={run.action_counts} compact />
      </section>
      {run.quality_rationale ? (
        <section>
          <h3>Quality Rationale</h3>
          <p className="rationale">{run.quality_rationale}</p>
        </section>
      ) : null}
      <section>
        <h3>Agent Patch</h3>
        <pre className="patch">{detail.patch || "No agent.patch found for this run."}</pre>
      </section>
    </div>
  );
}

function TasksTab({ tasks }: { tasks: TaskRecord[] }) {
  const [query, setQuery] = useState("");
  const filtered = useMemo(() => {
    const needle = query.trim().toLowerCase();
    if (!needle) {
      return tasks;
    }
    return tasks.filter((task) =>
      [task.task_id, task.repo_url, task.task_type, task.size, task.task_status]
        .filter(Boolean)
        .some((value) => String(value).toLowerCase().includes(needle))
    );
  }, [query, tasks]);

  return (
    <section className="page">
      <div className="toolbar">
        <label className="searchBox">
          <Search size={16} />
          <input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search tasks" />
        </label>
      </div>
      <section className="panel tablePanel">
        <table className="dataTable">
          <thead>
            <tr>
              <th>Task</th>
              <th>Size</th>
              <th>Difficulty</th>
              <th>Type</th>
              <th>Status</th>
              <th>Source LOC</th>
              <th>Patch LOC</th>
              <th>Patch Files</th>
              <th>Files</th>
              <th>Visible Command</th>
              <th>Hidden Command</th>
              <th>Semantic Hidden</th>
              <th>Compat Hidden</th>
              <th>PR Parity Hidden</th>
            </tr>
          </thead>
          <tbody>
            {filtered.map((task) => (
              <tr key={task.task_id}>
                <td>
                  <div className="strong">{task.task_id}</div>
                  <div className="muted">{task.pr_number}</div>
                </td>
                <td>
                  <span className={task.large_by_spec ? "pill warn" : "pill neutral"}>{task.size}</span>
                </td>
                <td><DifficultyPill value={task.difficulty_estimate} /></td>
                <td>{task.task_type}</td>
                <td>{task.task_status}</td>
                <td>{formatNumber(task.source_loc)}</td>
                <td>{formatNumber(task.patch_loc)}</td>
                <td>{formatNumber(task.patch_files)}</td>
                <td>{formatNumber(task.source_files)}</td>
                <td className="mono smallCell">{task.visible_test_command}</td>
                <td className="mono smallCell">{task.hidden_test_command}</td>
                <td className="mono smallCell">{task.hidden_semantic_test_command}</td>
                <td className="mono smallCell">{task.hidden_compat_test_command}</td>
                <td className="mono smallCell">{task.hidden_pr_parity_test_command}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>
    </section>
  );
}

function LauncherTab({
  overview,
  jobs,
  onJobsChange
}: {
  overview: Overview;
  jobs: JobRecord[];
  onJobsChange: (jobs: JobRecord[]) => void;
}) {
  // Only verified tasks are offered; rejected/candidate rows and fixtures can
  // still be launched from the CLI by naming them explicitly.
  const pickableTasks = overview.tasks.filter((task) => task.task_status.endsWith("_verified"));
  const firstTask = pickableTasks[0]?.task_id ?? "";
  const seenModels = Array.from(new Set(overview.runs.map((run) => run.model))).sort();
  const [meta, setMeta] = useState<Meta | null>(null);
  useEffect(() => {
    void fetchMeta().then(setMeta).catch(() => setMeta(null));
  }, []);
  const modelOptions = Array.from(
    new Set([...(meta?.model_presets ?? []), ...seenModels])
  ).filter(Boolean);
  const agentOptions = meta?.agents ?? ["single", "swe-agent", "multi"];
  const actionTransportOptions = meta?.action_transports ?? ["text_json", "tools", "auto"];
  const reasoningEffortOptions = meta?.reasoning_efforts ?? ["low", "medium", "high"];
  const reasoningModels = meta?.reasoning_models ?? null;
  const knownSessions = meta?.sessions ?? [];

  const [mode, setMode] = useState<"single" | "sweep">("single");
  const [session, setSession] = useCookieState("console.launch_session", "");
  const [taskId, setTaskId] = useState(firstTask);
  const [agent, setAgent] = useState("single");
  const [provider, setProvider] = useState<"openrouter" | "local">("openrouter");
  const [actionTransport, setActionTransport] = useState<ActionTransport>("text_json");
  const [reasoningEffort, setReasoningEffort] = useState("");
  const [model, setModel] = useState("");
  const [allTasks, setAllTasks] = useState(true);
  const [selectedTasks, setSelectedTasks] = useState<string[]>([]);
  const [taskSearch, setTaskSearch] = useState("");
  const [modelsCsv, setModelsCsv] = useState("");
  const [sweepAgents, setSweepAgents] = useState<string[]>(["single"]);
  const [concurrency, setConcurrency] = useState(1);
  const [maxSteps, setMaxSteps] = useState(50);
  const [maxIterations, setMaxIterations] = useState(5);
  const [score, setScore] = useState(true);
  const [regression, setRegression] = useState(true);
  const [review, setReview] = useState(false);
  const [network, setNetwork] = useState(true);
  const [setup, setSetup] = useState(true);
  const [dryRun, setDryRun] = useState(false);
  const [busy, setBusy] = useState(false);
  const [jobBusyId, setJobBusyId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  function toggleSweepAgent(value: string) {
    setSweepAgents((prev) =>
      prev.includes(value) ? prev.filter((item) => item !== value) : [...prev, value]
    );
  }
  function toggleTask(id: string) {
    setSelectedTasks((prev) =>
      prev.includes(id) ? prev.filter((item) => item !== id) : [...prev, id]
    );
  }

  // One facet group per collection dimension; a chip toggles its whole group.
  const facetGroups = [
    facetGroup("Size", pickableTasks, (task) => task.size, ["small", "medium", "large"]),
    facetGroup("Type", pickableTasks, (task) => task.task_type, ["bugfix", "feature"]),
    facetGroup("Difficulty", pickableTasks, (task) => task.difficulty_estimate, ["easy", "medium", "hard"])
  ].filter((group) => group.values.length > 0);

  function groupState(ids: string[]): "on" | "partial" | "off" {
    const picked = ids.filter((id) => selectedTasks.includes(id)).length;
    return picked === ids.length ? "on" : picked > 0 ? "partial" : "off";
  }
  function toggleGroup(ids: string[]) {
    setSelectedTasks((prev) =>
      ids.every((id) => prev.includes(id))
        ? prev.filter((id) => !ids.includes(id))
        : Array.from(new Set([...prev, ...ids]))
    );
  }

  const visibleTasks = pickableTasks.filter(
    (task) => !taskSearch.trim() || task.task_id.toLowerCase().includes(taskSearch.trim().toLowerCase())
  );
  const taskCount = allTasks ? pickableTasks.length : selectedTasks.length;

  // Reasoning effort only means something for reasoning models on OpenRouter.
  // OpenRouter ignores the field for unsupported models, so this is guidance,
  // not a hard gate; with the catalog unavailable, support stays unknown and
  // the select remains enabled.
  const enteredModels = (mode === "sweep" ? splitCsv(modelsCsv) : splitCsv(model)).filter(Boolean);
  const effortSupport = (() => {
    if (provider === "local") {
      return { enabled: false, note: "not sent to the local provider" };
    }
    if (!reasoningModels || enteredModels.length === 0) {
      return { enabled: true, note: null as string | null };
    }
    const supporting = enteredModels.filter((id) => reasoningModels.includes(id));
    if (supporting.length === 0) {
      return { enabled: false, note: "entered model(s) have no reasoning — effort would be ignored" };
    }
    if (supporting.length < enteredModels.length) {
      return { enabled: true, note: `applies only to: ${supporting.join(", ")}` };
    }
    return { enabled: true, note: null };
  })();
  const effortToSend = effortSupport.enabled ? reasoningEffort || undefined : undefined;
  const sweepCount =
    taskCount * Math.max(1, splitCsv(modelsCsv).length) * Math.max(1, sweepAgents.length);

  async function submit() {
    setBusy(true);
    setError(null);
    try {
      if (mode === "sweep") {
        const sweep: SweepPayload = {
          tasks: allTasks ? "all" : selectedTasks.join(","),
          models: modelsCsv.trim(),
          agents: (sweepAgents.length ? sweepAgents : ["single"]).join(","),
          action_transport: actionTransport,
          reasoning_effort: effortToSend,
          session: session.trim() || undefined,
          provider,
          concurrency,
          max_steps: maxSteps,
          max_iterations: maxIterations,
          no_score: !score,
          no_regression: !regression,
          enable_review: review,
          no_network: !network,
          no_setup: !setup
        };
        await startSweep(sweep);
      } else {
        const payload: LaunchPayload = {
          task_id: taskId,
          provider,
          model: model.trim() || undefined,
          agent,
          action_transport: actionTransport,
          reasoning_effort: effortToSend,
          session: session.trim() || undefined,
          max_steps: maxSteps,
          max_iterations: maxIterations,
          dry_run: dryRun,
          no_score: !score,
          no_regression: !regression,
          enable_review: review,
          no_network: !network,
          no_setup: !setup
        };
        await startJob(payload);
      }
      onJobsChange(await fetchJobs());
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  }

  async function stop(jobId: string) {
    await stopJob(jobId);
    onJobsChange(await fetchJobs());
  }

  async function rerunSweep(job: JobRecord) {
    const payload = sweepPayloadFromJob(job);
    setJobBusyId(job.job_id);
    setError(null);
    try {
      setSession(payload.session || "");
      await startSweep(payload);
      onJobsChange(await fetchJobs());
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setJobBusyId(null);
    }
  }

  return (
    <section className="page">
      <div className="grid two">
        <section className="panel launcher">
          <PanelHeader title="Start Benchmark Run" meta={mode === "sweep" ? "sweep / full benchmark" : "single-run launcher"} />
          {error ? <div className="errorBand compact">{error}</div> : null}

          <div className="segmented">
            <button className={mode === "single" ? "active" : ""} onClick={() => setMode("single")}>Single run</button>
            <button className={mode === "sweep" ? "active" : ""} onClick={() => setMode("sweep")}>
              <Layers size={14} /> Sweep
            </button>
          </div>

          <label className="fullWidth">
            Session
            <input list="sessionList" value={session} onChange={(event) => setSession(event.target.value)} placeholder="default" />
            <datalist id="sessionList">
              {knownSessions.map((item) => (
                <option key={item.session_id} value={item.session_id}>{`${item.runs} runs`}</option>
              ))}
            </datalist>
          </label>

          {mode === "single" ? (
            <div className="formGrid">
              <label>
                Task
                <select value={taskId} onChange={(event) => setTaskId(event.target.value)}>
                  {pickableTasks.map((task) => (
                    <option key={task.task_id} value={task.task_id}>{task.task_id}</option>
                  ))}
                </select>
              </label>
              <label>
                Agent
                <select value={agent} onChange={(event) => setAgent(event.target.value)}>
                  {agentOptions.map((item) => <option key={item} value={item}>{item}</option>)}
                </select>
              </label>
              <label>
                Model
                <input list="modelList" value={model} onChange={(event) => setModel(event.target.value)} placeholder="provider default" />
              </label>
            </div>
          ) : (
            <div className="formGrid">
              <label className="fullWidth">
                <span className="rowBetween">Tasks<Toggle label="all" checked={allTasks} onChange={setAllTasks} /></span>
                {!allTasks ? (
                  <div className="taskPicker">
                    {facetGroups.map((group) => (
                      <div key={group.label} className="facetRow">
                        <span className="facetLabel">{group.label}</span>
                        {group.values.map(({ value, ids }) => (
                          <button
                            key={value}
                            type="button"
                            className={`chip ${groupState(ids)}`}
                            title={`Select/deselect all ${value} tasks`}
                            onClick={() => toggleGroup(ids)}
                          >
                            {value} · {ids.length}
                          </button>
                        ))}
                      </div>
                    ))}
                    <div className="taskPickerBar">
                      <button type="button" className={`chip ${groupState(pickableTasks.map((task) => task.task_id))}`} onClick={() => toggleGroup(pickableTasks.map((task) => task.task_id))}>
                        All · {pickableTasks.length}
                      </button>
                      <button type="button" className="chip" onClick={() => setSelectedTasks([])}>Clear</button>
                      <input
                        className="taskSearch"
                        value={taskSearch}
                        onChange={(event) => setTaskSearch(event.target.value)}
                        placeholder="filter by id…"
                      />
                      <span className="muted">{selectedTasks.length} selected</span>
                    </div>
                    <div className="taskList">
                      {visibleTasks.map((task) => (
                        <label key={task.task_id} className="taskItem">
                          <input type="checkbox" checked={selectedTasks.includes(task.task_id)} onChange={() => toggleTask(task.task_id)} />
                          <span className="taskItemId">{task.task_id}</span>
                          <SizePill value={task.size} />
                          <TypePill value={task.task_type} />
                          <DifficultyPill value={task.difficulty_estimate} />
                        </label>
                      ))}
                      {visibleTasks.length === 0 ? <span className="muted taskListEmpty">no tasks match “{taskSearch}”</span> : null}
                    </div>
                  </div>
                ) : null}
              </label>
              <label className="fullWidth">
                Models (comma-separated; blank = provider default)
                <input list="modelList" value={modelsCsv} onChange={(event) => setModelsCsv(event.target.value)} placeholder="openai/gpt-4o-mini, deepseek/deepseek-chat" />
              </label>
              <label className="fullWidth">
                Agents
                <div className="checkRow">
                  {agentOptions.map((item) => (
                    <label key={item} className="checkPill">
                      <input type="checkbox" checked={sweepAgents.includes(item)} onChange={() => toggleSweepAgent(item)} />
                      {item}
                    </label>
                  ))}
                </div>
              </label>
              <label className="fullWidth">
                <span className="rowBetween">
                  Concurrency (parallel runs)
                  <span className="muted">{concurrency === 1 ? "sequential" : `${concurrency}× parallel`}</span>
                </span>
                <input type="number" min={1} max={8} value={concurrency} onChange={(event) => setConcurrency(Math.max(1, Math.min(8, Number(event.target.value) || 1)))} />
                <span className="muted">Runs are model/Docker-bound, so 2–4 speeds a sweep up a lot. swe-agent spins a container per run — keep it modest.</span>
              </label>
              <div className="muted">
                {sweepCount} run(s): {taskCount} task(s) × {Math.max(1, splitCsv(modelsCsv).length)} model(s) × {Math.max(1, sweepAgents.length)} agent(s){concurrency > 1 ? ` · ${concurrency} at a time` : ""}
              </div>
            </div>
          )}

          <datalist id="modelList">
            {modelOptions.map((item) => <option key={item} value={item} />)}
          </datalist>

          <div className="formGrid">
            <label>
              Max steps
              <input type="number" min={1} max={200} value={maxSteps} onChange={(event) => setMaxSteps(Number(event.target.value))} />
            </label>
            <label>
              Max iterations
              <input type="number" min={1} max={50} value={maxIterations} onChange={(event) => setMaxIterations(Number(event.target.value))} />
            </label>
          </div>

          <div className="segmented">
            <button className={provider === "openrouter" ? "active" : ""} onClick={() => setProvider("openrouter")}>OpenRouter</button>
            <button className={provider === "local" ? "active" : ""} onClick={() => setProvider("local")}>Local</button>
          </div>

          <div className="formGrid">
            <label>
              Action transport
              <select
                value={actionTransport}
                onChange={(event) => setActionTransport(event.target.value as ActionTransport)}
              >
                {actionTransportOptions.map((item) => (
                  <option key={item} value={item}>{item}</option>
                ))}
              </select>
            </label>
            <label>
              Reasoning effort
              <select
                value={effortSupport.enabled ? reasoningEffort : ""}
                disabled={!effortSupport.enabled}
                onChange={(event) => setReasoningEffort(event.target.value)}
                title="Reasoning budget for reasoning models (OpenRouter). Hidden reasoning shares max_tokens with the answer; on long contexts an uncapped model can burn the whole budget and return empty actions."
              >
                <option value="">{effortSupport.enabled ? "provider default" : "n/a"}</option>
                {reasoningEffortOptions.map((item) => (
                  <option key={item} value={item}>{item}</option>
                ))}
              </select>
              {effortSupport.note ? <span className="muted">{effortSupport.note}</span> : null}
            </label>
          </div>

          <div className="toggleGrid">
            <Toggle label="Hidden score" checked={score} onChange={setScore} />
            <Toggle label="Regression" checked={regression} onChange={setRegression} />
            <Toggle label="Review" checked={review} onChange={setReview} />
            <Toggle label="Network" checked={network} onChange={setNetwork} />
            <Toggle label="Setup" checked={setup} onChange={setSetup} />
            {mode === "single" ? <Toggle label="Dry run" checked={dryRun} onChange={setDryRun} /> : null}
          </div>

          <button
            className="primaryButton"
            onClick={() => void submit()}
            disabled={busy || (mode === "single" ? !taskId : (!allTasks && !selectedTasks.length) || !sweepAgents.length)}
          >
            {mode === "sweep" ? <Layers size={16} /> : <Play size={16} />}
            {busy ? "Starting..." : mode === "sweep" ? `Start sweep (${sweepCount})` : "Start job"}
          </button>
        </section>

        <section className="panel">
          <PanelHeader title="Jobs" meta={`${jobs.length} recorded`} />
          <div className="jobs">
            {jobs.map((job) => (
              <div key={job.job_id} className="jobRow">
                <div>
                  <div className="strong">{job.kind === "sweep" ? `sweep · ${job.task_id}` : job.task_id}</div>
                  <div className="muted mono">
                    {job.session || "default"} · {job.agent || "single"} · {job.action_transport || "text_json"}{job.reasoning_effort ? ` · effort ${job.reasoning_effort}` : ""} · {job.model || "default"}
                  </div>
                  <div className="muted mono commandLine">{job.command.join(" ")}</div>
                </div>
                <div className="jobActions">
                  <StatusPill status={job.status} />
                  {job.kind === "sweep" && job.status !== "running" ? (
                    <button
                      className="miniButton"
                      disabled={jobBusyId === job.job_id}
                      onClick={() => void rerunSweep(job)}
                      title="Run sweep in new session"
                    >
                      <Play size={14} />
                      Run
                    </button>
                  ) : null}
                  {job.status === "running" ? (
                    <button className="iconButton danger" onClick={() => void stop(job.job_id)} title="Stop job">
                      <Square size={14} />
                    </button>
                  ) : null}
                </div>
              </div>
            ))}
            {!jobs.length ? <div className="muted">No jobs recorded.</div> : null}
          </div>
        </section>
      </div>
    </section>
  );
}

function Kpi({ icon, label, value }: { icon: ReactNode; label: string; value: string }) {
  return (
    <div className="kpi">
      <div className="kpiIcon">{icon}</div>
      <div>
        <div className="kpiLabel">{label}</div>
        <div className="kpiValue">{value}</div>
      </div>
    </div>
  );
}

function PanelHeader({ title, meta }: { title: string; meta?: string }) {
  return (
    <div className="panelHeader">
      <h2>{title}</h2>
      {meta ? <span>{meta}</span> : null}
    </div>
  );
}

function GroupedBars({
  groups,
  primary,
  secondary,
  secondaryMax = 1
}: {
  groups: Record<string, MetricSet>;
  primary: MetricKey;
  secondary: MetricKey;
  secondaryMax?: number;
}) {
  const rows = Object.entries(groups).map(([name, metrics]) => ({
    name,
    primary: numberMetric(metrics, primary),
    secondary: numberMetric(metrics, secondary)
  }));

  return (
    <div className="barRows">
      {rows.map((row) => (
        <div key={row.name} className="barRow">
          <div className="barName">{row.name}</div>
          <div className="bars">
            <div className="barTrack">
              <div className="barFill good" style={{ width: `${clampPct(row.primary)}%` }} />
            </div>
            <div className="barTrack">
              <div className="barFill info" style={{ width: `${clampPct((row.secondary ?? 0) / secondaryMax)}%` }} />
            </div>
          </div>
          <div className="barValue">
            {formatMetricValue(row.primary, primary)} / {formatMetricValue(row.secondary, secondary)}
          </div>
        </div>
      ))}
      {!rows.length ? <div className="muted">No grouped metrics.</div> : null}
    </div>
  );
}

function AgentComparison({ groups }: { groups: Record<string, MetricSet> }) {
  const agents = Object.keys(groups);
  const rows: Array<[MetricKey, string]> = [
    ["n_runs", "Runs"],
    ["n_tasks", "Tasks"],
    ["task_success_rate", "Task success"],
    ["visible_test_pass_rate", "Visible pass"],
    ["hidden_semantic_pass_rate", "Semantic hidden"],
    ["hidden_compat_pass_rate", "Compat hidden"],
    ["hidden_pr_parity_pass_rate", "PR parity"],
    ["test_overfitting_rate", "Overfit"],
    ["mean_quality_score", "Quality"],
    ["mean_steps", "Steps"],
    ["mean_duration_s", "Duration"],
    ["mean_total_tokens", "Tokens"],
    ["mean_cost_usd", "Mean cost"]
  ];

  return (
    <div className="agentMatrixWrap">
      <table className="agentMatrix">
        <thead>
          <tr>
            <th>Metric</th>
            {agents.map((agent) => (
              <th key={agent}>
                <AgentPill value={agent} />
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map(([key, label]) => (
            <tr key={key}>
              <td>{label}</td>
              {agents.map((agent) => (
                <td key={`${agent}-${key}`}>
                  {formatAgentMetric(groups[agent], key)}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
      {!agents.length ? <div className="muted">No agent modes recorded.</div> : null}
    </div>
  );
}

function SessionFact({ label, value }: { label: string; value: string }) {
  return (
    <div className="sessionFact">
      <div>{label}</div>
      <strong>{value}</strong>
    </div>
  );
}

function SessionComparison({
  sessions,
  selectedSession,
  onSelect
}: {
  sessions: SessionOverview[];
  selectedSession: string;
  onSelect: (sessionId: string) => void;
}) {
  return (
    <div className="sessionTableWrap">
      <table className="sessionTable">
        <thead>
          <tr>
            <th>Session</th>
            <th>Runs</th>
            <th>Tasks</th>
            <th>Agents</th>
            <th>Success</th>
            <th>Visible</th>
            <th>Semantic</th>
            <th>Compat</th>
            <th>PR parity</th>
            <th>Cost</th>
            <th>Last run</th>
          </tr>
        </thead>
        <tbody>
          {sessions.map((session) => (
            <tr
              key={session.session_id}
              className={selectedSession === session.session_id ? "selected" : ""}
              onClick={() => onSelect(session.session_id)}
            >
              <td>
                <div className="strong">{session.session_id}</div>
                <div className="muted">{session.models.length} model(s)</div>
              </td>
              <td>{formatNumber(session.runs)}</td>
              <td>{formatNumber(session.tasks)}</td>
              <td>{session.agents.map((agent) => <AgentPill key={agent} value={agent} />)}</td>
              <td>{formatRate(session.metrics.task_success_rate)}</td>
              <td>{formatRate(session.metrics.visible_test_pass_rate)}</td>
              <td>{formatRate(session.metrics.hidden_semantic_pass_rate)}</td>
              <td>{formatRate(session.metrics.hidden_compat_pass_rate)}</td>
              <td>{formatRate(session.metrics.hidden_pr_parity_pass_rate)}</td>
              <td>{formatCost(session.calls.total_cost_usd)}</td>
              <td>{formatDate(session.last_finished_at)}</td>
            </tr>
          ))}
        </tbody>
      </table>
      {!sessions.length ? <div className="muted">No sessions recorded.</div> : null}
    </div>
  );
}

function TokenChart({ runs }: { runs: RunRecord[] }) {
  const rows = Array.from(
    runs.reduce((map, run) => {
      const item = map.get(run.model) ?? { model: run.model, input: 0, output: 0, runs: 0 };
      item.input += run.input_tokens ?? 0;
      item.output += run.output_tokens ?? 0;
      item.runs += 1;
      map.set(run.model, item);
      return map;
    }, new Map<string, { model: string; input: number; output: number; runs: number }>())
  )
    .map(([, value]) => value)
    .sort((a, b) => b.input + b.output - (a.input + a.output));
  const max = Math.max(1, ...rows.map((row) => row.input + row.output));

  return (
    <div className="tokenRows">
      {rows.map((row) => {
        const inputPct = (row.input / max) * 100;
        const outputPct = (row.output / max) * 100;
        return (
          <div key={row.model} className="tokenRow">
            <div className="tokenName">{row.model}</div>
            <div className="tokenTrack">
              <div className="tokenInput" style={{ width: `${inputPct}%` }} />
              <div className="tokenOutput" style={{ width: `${outputPct}%` }} />
            </div>
            <div className="tokenValue">{formatNumber(row.input + row.output)}</div>
          </div>
        );
      })}
    </div>
  );
}

function TimelineChart({ runs }: { runs: RunRecord[] }) {
  const ordered = [...runs].reverse();
  const maxDuration = Math.max(1, ...ordered.map((run) => run.duration_s ?? 0));
  const width = 640;
  const height = 210;
  const padding = 28;
  const points = ordered.map((run, index) => {
    const x = padding + (index / Math.max(1, ordered.length - 1)) * (width - padding * 2);
    const y = height - padding - ((run.duration_s ?? 0) / maxDuration) * (height - padding * 2);
    return { x, y, run };
  });

  return (
    <svg className="timeline" viewBox={`0 0 ${width} ${height}`} role="img">
      <line x1={padding} y1={height - padding} x2={width - padding} y2={height - padding} />
      <line x1={padding} y1={padding} x2={padding} y2={height - padding} />
      <polyline points={points.map((point) => `${point.x},${point.y}`).join(" ")} />
      {points.map((point) => (
        <circle key={point.run.run_id} cx={point.x} cy={point.y} r="5" className={`dot ${statusClass(point.run.status)}`}>
          <title>{`${point.run.task_id}: ${formatSeconds(point.run.duration_s)}`}</title>
        </circle>
      ))}
    </svg>
  );
}

function ActionBars({ actions, compact = false }: { actions: Record<string, number>; compact?: boolean }) {
  const rows = Object.entries(actions).sort((a, b) => b[1] - a[1]);
  const max = Math.max(1, ...rows.map(([, value]) => value));
  return (
    <div className={compact ? "actionBars compact" : "actionBars"}>
      {rows.map(([name, value]) => (
        <div key={name} className="actionRow">
          <div className="actionName">{name}</div>
          <div className="actionTrack">
            <div style={{ width: `${(value / max) * 100}%` }} />
          </div>
          <div className="actionValue">{value}</div>
        </div>
      ))}
      {!rows.length ? <div className="muted">No actions recorded.</div> : null}
    </div>
  );
}

function DifficultyTable({ rows }: { rows: TaskDifficulty[] }) {
  return (
    <div className="difficultyRows">
      <div className="difficultyHeader">
        <span>Task</span>
        <span>Runs</span>
        <span>Models</span>
        <span>Solve</span>
        <span>Diff</span>
        <span>Discrim</span>
      </div>
      {rows.map((row) => (
        <div key={row.task_id} className="difficultyRow">
          <span className="strong">{row.task_id}</span>
          <span>{row.n_runs}</span>
          <span>{row.n_models}</span>
          <span>{formatRate(row.solve_rate)}</span>
          <span>{row.difficulty.toFixed(2)}</span>
          <span>{row.discrimination == null ? "n/a" : row.discrimination.toFixed(2)}</span>
        </div>
      ))}
      {!rows.length ? <div className="muted">No measured difficulty yet.</div> : null}
    </div>
  );
}

function InventoryItem({ label, value, warning }: { label: string; value: number; warning?: boolean }) {
  return (
    <div className={warning ? "inventoryItem warning" : "inventoryItem"}>
      <div>{label}</div>
      <strong>{formatNumber(value)}</strong>
    </div>
  );
}

function DetailItem({ label, value }: { label: string; value: ReactNode }) {
  return (
    <div className="detailItem">
      <div>{label}</div>
      <strong>{value}</strong>
    </div>
  );
}

function Toggle({ label, checked, onChange }: { label: string; checked: boolean; onChange: (checked: boolean) => void }) {
  return (
    <label className="toggle">
      <input type="checkbox" checked={checked} onChange={(event) => onChange(event.target.checked)} />
      <span>{label}</span>
    </label>
  );
}

function StatusPill({ status }: { status: string }) {
  return <span className={`pill ${statusClass(status)}`}>{status}</span>;
}

function AgentPill({ value }: { value: string }) {
  return <span className={`pill ${agentClass(value)}`}>{value}</span>;
}

function TypePill({ value }: { value: string }) {
  const variant = value === "feature" ? "feature" : value === "bugfix" ? "bugfix" : "";
  return <span className={`pill ${variant}`}>{value}</span>;
}

function SizePill({ value }: { value: string | null }) {
  if (!value) return null;
  return <span className={`pill size-${value}`}>{value}</span>;
}

function facetGroup(
  label: string,
  tasks: TaskRecord[],
  key: (task: TaskRecord) => string | null,
  order: string[]
): { label: string; values: { value: string; ids: string[] }[] } {
  const buckets = new Map<string, string[]>();
  for (const task of tasks) {
    const value = key(task);
    if (!value) continue;
    buckets.set(value, [...(buckets.get(value) ?? []), task.task_id]);
  }
  const ordered = [
    ...order.filter((value) => buckets.has(value)),
    ...[...buckets.keys()].filter((value) => !order.includes(value)).sort()
  ];
  return { label, values: ordered.map((value) => ({ value, ids: buckets.get(value)! })) };
}

function HiddenPill({ value }: { value: boolean | null }) {
  if (value === true) {
    return <span className="pill solved"><CheckCircle2 size={13} />pass</span>;
  }
  if (value === false) {
    return <span className="pill failed"><XCircle size={13} />fail</span>;
  }
  return <span className="pill neutral">n/a</span>;
}

function taskSuccessValue(run: RunRecord): boolean | null {
  return run.task_success ?? run.hidden_tests_passed;
}

function DifficultyPill({ value, fallback }: { value: string | null; fallback?: string | null }) {
  const label = value ?? fallback ?? "n/a";
  return <span className={`pill ${difficultyClass(label)}`}>{label}</span>;
}

function newestRuns(runs: RunRecord[]) {
  return [...runs].sort((a, b) => String(b.finished_at ?? "").localeCompare(String(a.finished_at ?? "")));
}

function numberMetric(metrics: MetricSet, key: MetricKey): number | null {
  const value = metrics[key];
  return typeof value === "number" ? value : null;
}

function formatMetricValue(value: number | null, key: MetricKey): string {
  if (value == null) {
    return "n/a";
  }
  if (key.endsWith("_rate") || key === "resolved_at_1") {
    return formatRate(value);
  }
  if (key.includes("cost")) {
    return formatCost(value);
  }
  return value.toFixed(2);
}

function formatAgentMetric(metrics: MetricSet, key: MetricKey): string {
  const value = numberMetric(metrics, key);
  if (value == null) {
    return "n/a";
  }
  if (key === "n_runs" || key === "n_tasks") {
    return formatNumber(value);
  }
  if (key === "mean_duration_s") {
    return formatSeconds(value);
  }
  if (key === "mean_total_tokens") {
    return formatNumber(Math.round(value));
  }
  if (key.includes("cost")) {
    return formatCost(value);
  }
  return formatMetricValue(value, key);
}

function clampPct(value: number | null): number {
  if (value == null || Number.isNaN(value)) {
    return 0;
  }
  return Math.max(0, Math.min(100, value * 100));
}

function statusClass(status: string): string {
  if (status === "solved" || status === "completed") {
    return "solved";
  }
  if (status === "failed" || status === "stopped") {
    return "failed";
  }
  if (status === "handoff") {
    return "warn";
  }
  if (status === "running") {
    return "running";
  }
  return "neutral";
}

function agentClass(value: string): string {
  if (value === "multi") {
    return "running";
  }
  if (value === "single") {
    return "neutral";
  }
  return "warn";
}

function difficultyClass(value: string): string {
  if (value === "easy") {
    return "solved";
  }
  if (value === "medium") {
    return "warn";
  }
  if (value === "hard") {
    return "failed";
  }
  return "neutral";
}

function formatRate(value: number | null | undefined): string {
  return value == null ? "n/a" : `${(value * 100).toFixed(1)}%`;
}

function formatSeconds(value: number | null | undefined): string {
  if (value == null) {
    return "n/a";
  }
  if (value < 60) {
    return `${value.toFixed(1)}s`;
  }
  return `${(value / 60).toFixed(1)}m`;
}

function formatNumber(value: number | null | undefined): string {
  return value == null ? "n/a" : new Intl.NumberFormat("en-US").format(value);
}

function formatCost(value: number | null | undefined): string {
  return value == null ? "n/a" : `$${value.toFixed(4)}`;
}

function formatDate(value: string | null): string {
  if (!value) {
    return "n/a";
  }
  return new Intl.DateTimeFormat("en-US", {
    month: "short",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit"
  }).format(new Date(value));
}

function shortId(value: string): string {
  return value.slice(0, 8);
}
