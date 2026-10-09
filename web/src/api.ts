export type Wilson = { k: number; n: number; p: number; lo: number; hi: number };

export type Summary = {
  n_episodes: number;
  task_success: Wilson;
  object_clear_rate: Wilson;
  episode_outcomes: Record<string, number>;
  object_outcomes: Record<string, number>;
  per_object: Record<string, Wilson & { label: string; grasp: string; outcomes: Record<string, number> }>;
  picks_per_side: Record<string, number>;
  safety: {
    protective_stops: number;
    self_collision_steps: number;
    high_force_steps: number;
    workspace_steps: number;
    rate_limited_steps: number;
    max_contact_force_n: number;
  };
  mean_sim_time_s: number;
  mean_perception_ms: number;
  mean_picks: number;
};

export type Run = {
  id: number;
  created: number;
  name: string;
  controller: string;
  perception: string;
  randomization: string;
  n_episodes: number;
  status: string;
  done: number;
  summary: Summary | null;
  elapsed_s: number | null;
};

export type Episode = {
  id: number;
  run_id: number;
  run_name: string;
  seed: number;
  outcome: string;
  success: number;
  n_cleared: number;
  n_objects: number;
  steps: number;
  replay: string | null;
  object_outcomes: Record<string, string>;
  objects: string[];
};

export type Config = {
  controllers: { id: string; label: string; kind: string }[];
  perception: { id: string; label: string; available?: boolean }[];
  randomization: Record<string, Record<string, unknown>>;
  cameras: string[];
  objects: { id: string; label: string; grasp: string; mass: number }[];
  grasps: Record<string, { label: string; thumb_yaw: number; close: number[]; z_frac: number }>;
};

export const OUTCOME_COLORS: Record<string, string> = {
  success: "#3fb950",
  not_attempted: "#8b949e",
  grasp_fail: "#d29922",
  dropped: "#db6d28",
  misplaced: "#a371f7",
  knocked_over: "#f85149",
  timeout: "#6e7681",
  safety_stop: "#ff7b72",
  sim_error: "#484f58",
  running: "#58a6ff",
};

async function req<T>(method: string, url: string, body?: unknown): Promise<T> {
  const r = await fetch(url, {
    method,
    headers: body ? { "Content-Type": "application/json" } : undefined,
    body: body ? JSON.stringify(body) : undefined,
  });
  if (!r.ok) {
    const t = await r.text();
    throw new Error(`${r.status}: ${t}`);
  }
  return r.json() as Promise<T>;
}

export const api = {
  config: () => req<Config>("GET", "/api/config"),
  system: () => req<any>("GET", "/api/system"),
  liveStart: (cfg: Record<string, unknown>) => req("POST", "/api/live/start", cfg),
  liveStop: () => req("POST", "/api/live/stop"),
  liveCamera: (c: string) => req("POST", `/api/live/camera/${c}`),
  evalStart: (cfg: Record<string, unknown>) => req<any>("POST", "/api/eval", cfg),
  evalJob: () => req<any>("GET", "/api/eval/job"),
  runs: () => req<Run[]>("GET", "/api/runs"),
  deleteRun: (id: number) => req("DELETE", `/api/runs/${id}`),
  episodes: (runId?: number, outcome?: string) => {
    const q = new URLSearchParams();
    if (runId) q.set("run_id", String(runId));
    if (outcome) q.set("outcome", outcome);
    return req<Episode[]>("GET", `/api/episodes?${q}`);
  },
  replay: (id: number) => req("POST", `/api/episodes/${id}/replay`),
  models: () => req<any>("GET", "/api/models"),
};

export const pct = (x: number) => `${(100 * x).toFixed(0)}%`;
