import { useEffect, useState } from "react";
import { api, Config, OUTCOME_COLORS, pct, Run } from "./api";
import { CIBars, SimGapChart, StackedBars, Stat } from "./charts";

const CTRL_COLORS = ["#58a6ff", "#f0883e", "#3fb950", "#a371f7", "#ff7b72", "#d29922"];
const LEVELS = ["nominal", "low", "medium", "high"];

export function Evaluate({ config }: { config: Config | null }) {
  const [runs, setRuns] = useState<Run[]>([]);
  const [job, setJob] = useState<any>({});
  const [form, setForm] = useState({ controller: "expert", perception: "auto", randomization: "nominal", n: 20, seed0: 0 });
  const [err, setErr] = useState("");
  const [selected, setSelected] = useState<number | null>(null);
  const set = (k: string, v: any) => setForm((f) => ({ ...f, [k]: v }));

  const refresh = () => { api.runs().then(setRuns); api.evalJob().then(setJob); };
  useEffect(() => { refresh(); const t = setInterval(refresh, 2000); return () => clearInterval(t); }, []);

  const done = runs.filter((r) => r.summary && r.status === "done");
  const ctrls = [...new Set(done.map((r) => r.controller))];
  const colorOf = (c: string) => CTRL_COLORS[ctrls.indexOf(c) % CTRL_COLORS.length];
  const sel = runs.find((r) => r.id === selected) ?? done[0];

  // best (latest) run per controller x randomisation level for the sim-gap plot
  const gap = ctrls.map((c) => ({
    name: c,
    color: colorOf(c),
    points: LEVELS.map((lv) => {
      const r = done.find((x) => x.controller === c && x.randomization === lv);
      return r?.summary ? { label: lv, ...r.summary.object_clear_rate, color: colorOf(c) } : null;
    }),
  }));

  return (
    <div className="page">
      <div className="grid2">
        <div className="card">
          <h3>Run a batch evaluation</h3>
          <p className="muted">Episodes run in parallel worker processes (count capped by free RAM). Results are stored in SQLite with a replay file per episode.</p>
          <div className="row">
            <label>Controller<select value={form.controller} onChange={(e) => set("controller", e.target.value)}>
              {config?.controllers.map((c) => <option key={c.id} value={c.id}>{c.label}</option>)}</select></label>
            <label>Perception<select value={form.perception} onChange={(e) => set("perception", e.target.value)}>
              {config?.perception.map((p) => <option key={p.id} value={p.id} disabled={p.available === false}>{p.label}</option>)}</select></label>
          </div>
          <div className="row">
            <label>Randomisation<select value={form.randomization} onChange={(e) => set("randomization", e.target.value)}>
              {LEVELS.map((l) => <option key={l}>{l}</option>)}</select></label>
            <label>Episodes<input type="number" min={1} max={500} value={form.n} onChange={(e) => set("n", +e.target.value)} /></label>
            <label>First seed<input type="number" value={form.seed0} onChange={(e) => set("seed0", +e.target.value)} /></label>
          </div>
          <button className="primary" disabled={job.status === "running"}
            onClick={() => api.evalStart(form).then(() => { setErr(""); refresh(); }).catch((e) => setErr(String(e)))}>Start evaluation</button>
          {err && <div className="error">{err}</div>}
          {job.status && (
            <div className="progress">
              <div className="bar" style={{ width: `${(100 * job.done) / job.total}%` }} />
              <span>run #{job.run_id}: {job.done}/{job.total} · {job.status}</span>
            </div>
          )}
        </div>
        <div className="card">
          <h3>Sim-gap study</h3>
          <p className="muted">Objects cleared vs. randomisation level (friction, mass, perception noise, latency, finger gain), 95 % Wilson intervals.</p>
          {gap.length ? <SimGapChart levels={LEVELS} lines={gap} /> : <div className="empty">run evaluations at several randomisation levels</div>}
        </div>
      </div>

      <div className="card">
        <h3>Runs</h3>
        <table className="runs">
          <thead><tr><th>#</th><th>controller</th><th>perception</th><th>randomisation</th><th>episodes</th><th>task success</th><th>objects cleared</th><th>safety stops</th><th>status</th><th></th></tr></thead>
          <tbody>
            {runs.map((r) => (
              <tr key={r.id} className={sel?.id === r.id ? "sel" : ""} onClick={() => setSelected(r.id)}>
                <td>{r.id}</td><td><span className="dot" style={{ background: colorOf(r.controller) }} />{r.controller}</td>
                <td>{r.perception}</td><td>{r.randomization}</td><td>{r.done}/{r.n_episodes}</td>
                <td>{r.summary ? `${pct(r.summary.task_success.p)} [${pct(r.summary.task_success.lo)}–${pct(r.summary.task_success.hi)}]` : "…"}</td>
                <td>{r.summary ? `${pct(r.summary.object_clear_rate.p)} [${pct(r.summary.object_clear_rate.lo)}–${pct(r.summary.object_clear_rate.hi)}]` : "…"}</td>
                <td>{r.summary?.safety?.protective_stops ?? "…"}</td>
                <td>{r.status}</td>
                <td><button className="link" onClick={(e) => { e.stopPropagation(); if (confirm(`Delete run #${r.id}?`)) api.deleteRun(r.id).then(refresh); }}>delete</button></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {done.length > 0 && (
        <div className="grid2">
          <div className="card">
            <h3>Comparison: objects cleared</h3>
            <CIBars items={done.slice(0, 12).map((r) => ({ label: `#${r.id} ${r.controller}`, sub: `${r.perception} · ${r.randomization}`, ...r.summary!.object_clear_rate, color: colorOf(r.controller) }))} />
          </div>
          <div className="card">
            <h3>Failure taxonomy (per object)</h3>
            <StackedBars colors={OUTCOME_COLORS} rows={done.slice(0, 12).map((r) => ({ label: `#${r.id} ${r.controller} · ${r.randomization}`, counts: r.summary!.object_outcomes }))} />
          </div>
        </div>
      )}

      {sel?.summary && (
        <div className="card">
          <h3>Run #{sel.id}: {sel.name}</h3>
          <div className="stats">
            <Stat label="task success (all objects)" value={pct(sel.summary.task_success.p)} sub={`95% CI ${pct(sel.summary.task_success.lo)}–${pct(sel.summary.task_success.hi)} · n=${sel.summary.n_episodes}`} />
            <Stat label="objects cleared" value={pct(sel.summary.object_clear_rate.p)} sub={`${sel.summary.object_clear_rate.k}/${sel.summary.object_clear_rate.n}`} />
            <Stat label="mean episode time" value={`${sel.summary.mean_sim_time_s}s`} sub={`${sel.summary.mean_picks} picks / episode`} />
            <Stat label="perception latency" value={`${sel.summary.mean_perception_ms} ms`} sub={sel.perception} />
            <Stat label="protective stops" value={sel.summary.safety.protective_stops} sub={`max env. force ${sel.summary.safety.max_contact_force_n} N`} />
            <Stat label="picks per hand" value={Object.entries(sel.summary.picks_per_side).map(([k, v]) => `${k} ${v}`).join(" · ")} />
          </div>
          <h4>Per object / grasp type</h4>
          <CIBars items={Object.values(sel.summary.per_object).map((o) => ({ ...o, label: o.label, sub: o.grasp, color: "#3fb950" }))} />
          <p className="muted">Wall time {sel.elapsed_s?.toFixed(0)} s. Episodes of this run (with replays) are in the Episodes tab.</p>
        </div>
      )}
    </div>
  );
}
