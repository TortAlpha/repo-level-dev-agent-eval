export type AgentStatus = "running" | "planning" | "editing" | "needs_revision" | "solved" | "failed" | "handoff" | string;
export type ActionTransport = "text_json" | "tools" | "auto";
export type ReasoningEffort = "none" | "minimal" | "low" | "medium" | "high" | "xhigh" | "max";

export interface ReasoningCapability {
  supports_reasoning: boolean;
  supports_effort: boolean;
  supported_efforts: ReasoningEffort[] | null;
  default_effort: ReasoningEffort | null;
  mandatory: boolean | null;
  default_enabled: boolean | null;
  supports_max_tokens: boolean | null;
}

export interface MetricSet {
  n_runs: number;
  n_tasks: number;
  resolved_at_1: number | null;
  task_success_rate: number | null;
  visible_test_pass_rate: number | null;
  hidden_test_pass_rate: number | null;
  hidden_semantic_pass_rate: number | null;
  hidden_compat_pass_rate: number | null;
  hidden_pr_parity_pass_rate: number | null;
  patch_validity_rate: number | null;
  handoff_rate: number | null;
  repair_success_rate: number | null;
  regression_rate: number | null;
  mean_regressions: number | null;
  test_overfitting_rate: number | null;
  tool_use_validity_rate: number | null;
  policy_rejections_per_run: number | null;
  hallucinated_refs_per_run: number | null;
  mean_iterations: number | null;
  mean_steps: number | null;
  mean_duration_s: number | null;
  mean_llm_calls: number | null;
  mean_total_tokens: number | null;
  mean_cost_usd: number | null;
  total_cost_usd: number | null;
  cost_per_success_usd: number | null;
  mean_quality_score: number | null;
}

export interface RunRecord {
  task_id: string;
  run_id: string;
  session_id: string;
  agent_mode: string;
  provider: "openrouter" | "local" | string | null;
  action_transport: string | null;
  reasoning_effort: ReasoningEffort | null;
  reasoning_max_tokens: number | null;
  model_routes: Record<string, Record<string, unknown>>;
  reproducibility: Record<string, unknown>;
  run_fingerprint: string | null;
  model: string;
  status: AgentStatus;
  steps: number;
  iterations: number;
  test_passed: boolean;
  task_success: boolean | null;
  hidden_tests_passed: boolean | null;
  hidden_required_tests_passed: boolean | null;
  hidden_semantic_tests_passed: boolean | null;
  hidden_compat_tests_passed: boolean | null;
  hidden_pr_parity_tests_passed: boolean | null;
  hidden_suite_results: Record<string, boolean>;
  changed_files: string[];
  finished_at: string | null;
  duration_s: number | null;
  llm_calls: number | null;
  input_tokens: number | null;
  output_tokens: number | null;
  cached_input_tokens: number | null;
  cache_write_input_tokens: number | null;
  reasoning_tokens: number | null;
  total_tokens: number | null;
  cost_usd: number | null;
  effective_cost_usd: number | null;
  estimated_cost_usd: number | null;
  provider_reported_cost_usd: number | null;
  provider_cost_calls: number | null;
  provider_cost_complete: boolean;
  cost_source: string | null;
  model_policy: {
    type?: string;
    base_model?: string;
    roles?: Record<string, string>;
    developer_escalation_model?: string | null;
    developer_escalate_after_no_edit_episodes?: number;
    developer_escalate_after_failed_tests?: number;
  };
  role_usage: Record<string, Record<string, Record<string, number>>>;
  role_transports: Record<string, Record<string, string>>;
  role_steps: Record<string, number>;
  developer_escalations: number;
  max_cost_usd: number | null;
  action_counts: Record<string, number>;
  regressions: number | null;
  test_oracle_tampered: boolean | null;
  test_oracle_tamper_attempts: number | null;
  workspace: string | null;
  task_type: string | null;
  size: string | null;
  difficulty: string | null;
  difficulty_estimate: string | null;
  quality_score: number | null;
  quality_rationale: string | null;
  reviewer_model: string | null;
  task_set_id: string | null;
  campaign_id: string | null;
  experiment_fingerprint: string | null;
  docker_image: string | null;
  docker_image_id: string | null;
  agent_network_disabled_after_setup: boolean | null;
}

export interface TaskRecord {
  task_id: string;
  repo_path: string;
  task_file_path: string;
  size: string;
  task_type: string;
  repo_url: string;
  pr_number: string;
  pr_url: string;
  base_commit: string;
  merge_commit: string;
  hidden_tests_path: string;
  source_files: number | null;
  source_loc: number | null;
  source_loc_nonblank: number | null;
  patch_files: number | null;
  patch_loc: number | null;
  test_files: number | null;
  visible_test_command: string;
  hidden_test_command: string;
  hidden_semantic_test_command: string;
  hidden_compat_test_command: string;
  hidden_pr_parity_test_command: string;
  task_status: string;
  notes: string;
  large_by_spec: boolean;
  difficulty_estimate: string | null;
}

export interface TaskDifficulty {
  task_id: string;
  n_runs: number;
  n_models: number;
  solve_rate: number;
  difficulty: number;
  discrimination: number | null;
}

export interface MetricGroups {
  by_agent_mode: Record<string, MetricSet>;
  by_model_policy: Record<string, MetricSet>;
  by_size: Record<string, MetricSet>;
  by_task_type: Record<string, MetricSet>;
  by_difficulty: Record<string, MetricSet>;
  by_difficulty_estimate: Record<string, MetricSet>;
  by_model: Record<string, MetricSet>;
  by_action_transport: Record<string, MetricSet>;
}

export interface CallSummary {
  llm_calls: number;
  input_tokens: number;
  output_tokens: number;
  total_tokens: number;
  total_cost_usd: number | null;
  runs_with_usage: number;
}

export interface SessionOverview {
  session_id: string;
  runs: number;
  tasks: number;
  agents: string[];
  models: string[];
  active_jobs: number;
  created_at: string | null;
  first_finished_at: string | null;
  last_finished_at: string | null;
  metrics: MetricSet;
  groups: MetricGroups;
  task_difficulty: TaskDifficulty[];
  calls: CallSummary;
  actions: Record<string, number>;
}

export interface Overview {
  generated_at: string;
  summary: {
    runs: number;
    run_tasks: number;
    collection_tasks: number;
    verified_tasks: number;
    unverified_tasks: number;
    unknown_run_tasks: string[];
    last_finished_at: string | null;
  };
  sessions: SessionOverview[];
  metrics: MetricGroups & {
    overall: MetricSet | null;
    by_session: Record<string, MetricSet>;
  };
  task_difficulty: TaskDifficulty[];
  calls: CallSummary;
  actions: Record<string, number>;
  runs: RunRecord[];
  tasks: TaskRecord[];
}

export interface RunDetail {
  run: RunRecord;
  metrics_file: unknown;
  patch: string;
}

export interface JobRecord {
  job_id: string;
  status: "running" | "completed" | "failed" | "stopped" | string;
  returncode: number | null;
  kind?: "run" | "sweep" | string;
  task_id: string;
  provider: string;
  model: string | null;
  agent?: string | null;
  action_transport?: string | null;
  reasoning_effort?: ReasoningEffort | null;
  reasoning_max_tokens?: number | null;
  session?: string | null;
  dry_run: boolean;
  command: string[];
  pid: number | null;
  log_path: string;
  started_at: string;
  finished_at: string | null;
  log_tail?: string;
}

export interface LaunchPayload {
  task_id: string;
  provider: "openrouter" | "local";
  model?: string;
  agent?: string;
  action_transport?: ActionTransport;
  reasoning_effort?: ReasoningEffort;
  reasoning_max_tokens?: number;
  role_models?: string[];
  developer_escalation_model?: string;
  developer_escalate_after_no_edit_episodes?: number;
  developer_escalate_after_failed_tests?: number;
  max_cost_usd?: number;
  session?: string;
  max_steps?: number;
  max_iterations?: number;
  dry_run?: boolean;
  no_score?: boolean;
  no_regression?: boolean;
  enable_review?: boolean;
  no_network?: boolean;
  no_setup?: boolean;
}

export interface SweepPayload {
  tasks?: string;   // comma-separated task_ids, or "all"
  task_set?: string;
  matrix?: string;
  campaign?: string;
  models: string;  // comma-separated models ("" = provider default)
  agents?: string;  // comma-separated agents; manifest matrix can fix this
  action_transport?: ActionTransport;
  reasoning_effort?: ReasoningEffort;
  reasoning_max_tokens?: number;
  role_models?: string[];
  developer_escalation_model?: string;
  developer_escalate_after_no_edit_episodes?: number;
  developer_escalate_after_failed_tests?: number;
  max_cost_usd?: number;
  max_total_cost_usd?: number;
  resume?: boolean;
  session?: string;
  provider: "openrouter" | "local";
  concurrency?: number;  // parallel runs (1 = sequential)
  max_steps?: number;
  max_iterations?: number;
  no_score?: boolean;
  no_regression?: boolean;
  enable_review?: boolean;
  no_network?: boolean;
  no_setup?: boolean;
}

export interface SessionInfo {
  session_id: string;
  runs: number;
  last_finished_at: string | null;
}

export interface Meta {
  generated_at: string;
  agents: string[];
  action_transports: ActionTransport[];
  reasoning_efforts?: ReasoningEffort[];
  // Models that accept the `reasoning` config (OpenRouter catalog);
  // null/absent = catalog unavailable, treat support as unknown.
  reasoning_models?: string[] | null;
  reasoning_capabilities?: Record<string, ReasoningCapability>;
  reasoning_catalog_source?: "openrouter" | "openrouter_cache" | "model_profiles" | string;
  reasoning_catalog_complete?: boolean;
  providers: string[];
  model_presets: string[];
  models_seen: string[];
  sessions: SessionInfo[];
}
