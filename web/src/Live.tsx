import { useEffect, useRef, useState } from "react";
import { api, Config, OUTCOME_COLORS } from "./api";
import { LineChart, Stat } from "./charts";

type Snapshot = {
  mode: string;
  config: any;
  frame: string | null;
  head: string | null;
  telemetry: any;
  history: { t: number; grip_left: number; grip_right: number; cleared: number }[];
  log: any[];
  result: any;
  error: string | null;
};

export function useLive() {
  const [snap, setSnap] = useState<Snapshot | null>(null);
  const [connected, setConnected] = useState(false);
  useEffect(() => {
    let ws: WebSocket | null = null;
    let stop = false;
    const connect = () => {
      const proto = location.protocol === "https:" ? "wss" : "ws";
      ws = new WebSocket(`${proto}://${location.host}/ws/live`);
      ws.onopen = () => setConnected(true);
      ws.onclose = () => { setConnected(false); if (!stop) setTimeout(connect, 1000); };
      ws.onmessage = (e) => setSnap(JSON.parse(e.data));
    };
    connect();
    return () => { stop = true; ws?.close(); };
  }, []);
  return { snap, connected };
}

export function Viewer({ snap, connected }: { snap: Snapshot | null; connected: boolean }) {
  const tel = snap?.telemetry ?? {};
  const cur = tel.current;
  return (
    <div className="viewer">
      <div className="video">
        {snap?.frame ? <img src={`data:image/jpeg;base64,${snap.frame}`} /> : <div className="placeholder">{connected ? "Press ▶ Run to start an episode" : "connecting…"}</div>}
        <div className="overlay">
          <span className={`badge mode-${snap?.mode}`}>{snap?.mode ?? "…"}</span>
          {tel.t !== undefined && <span className="badge">t = {tel.t}s</span>}
          {tel.phase && <span className="badge">{tel.phase}</span>}
        </div>
        {cur && (
          <div className="overlay bottom">
            <span className="badge accent">{cur.side} hand → {cur.target.name}</span>
            <span className="badge">{cur.grasp}</span>
            {cur.attempt > 1 && <span className="badge warn">retry #{cur.attempt}</span>}
          </div>
        )}
      </div>
      <div className="side-view">
        <div className="panel-title">Head camera · detections</div>
        {snap?.head ? <img src={`data:image/jpeg;base64,${snap.head}`} /> : <div className="placeholder small">—</div>}
      </div>
    </div>
  );
}

export function Live({ config }: { config: Config | null }) {
  const { snap, connected } = useLive();
  const [form, setForm] = useState({ controller: "expert", perception: "auto", randomization: "nominal", seed: 0, n_objects: 0, camera: "front", speed: 1 });
  const logRef = useRef<HTMLDivElement>(null);
  const set = (k: string, v: any) => setForm((f) => ({ ...f, [k]: v }));
  useEffect(() => { logRef.current?.scrollTo(0, 1e9); }, [snap?.log?.length]);
  const tel = snap?.telemetry ?? {};
  const res = snap?.result;
  const hist = snap?.history ?? [];
  const safety = res?.safety ?? tel.safety;

  return (
    <div className="page live">
      <div className="controls card">
        <h3>Episode</h3>
        <label>Controller
          <select value={form.controller} onChange={(e) => set("controller", e.target.value)}>
            {config?.controllers.map((c) => <option key={c.id} value={c.id}>{c.label}</option>)}
          </select>
        </label>
        <label>Perception
          <select value={form.perception} onChange={(e) => set("perception", e.target.value)}>
            {config?.perception.map((p) => <option key={p.id} value={p.id} disabled={p.available === false}>{p.label}{p.available === false ? " (not trained)" : ""}</option>)}
          </select>
        </label>
        <label>Sim-gap randomisation
          <select value={form.randomization} onChange={(e) => set("randomization", e.target.value)}>
            {config && Object.keys(config.randomization).map((k) => <option key={k}>{k}</option>)}
          </select>
        </label>
        <div className="row">
          <label>Seed<input type="number" value={form.seed} onChange={(e) => set("seed", +e.target.value)} /></label>
          <label>Objects<select value={form.n_objects} onChange={(e) => set("n_objects", +e.target.value)}>
            <option value={0}>random 3-4</option>{[1, 2, 3, 4].map((n) => <option key={n} value={n}>{n}</option>)}
          </select></label>
        </div>
        <label>Speed ×{form.speed}
          <input type="range" min={0.25} max={4} step={0.25} value={form.speed} onChange={(e) => set("speed", +e.target.value)} />
        </label>
        <div className="row">
          <button className="primary" onClick={() => api.liveStart(form)}>▶ Run</button>
          <button onClick={() => api.liveStop()}>■ Stop</button>
          <button onClick={() => { set("seed", form.seed + 1); api.liveStart({ ...form, seed: form.seed + 1 }); }}>Next seed</button>
        </div>
        <label>Camera
          <div className="seg">
            {config?.cameras.map((c) => (
              <button key={c} className={form.camera === c ? "on" : ""} onClick={() => { set("camera", c); api.liveCamera(c); }}>{c}</button>
            ))}
          </div>
        </label>
        {snap?.error && <div className="error">{snap.error}</div>}
      </div>

      <div className="main">
        <Viewer snap={snap} connected={connected} />
        <div className="grid3">
          <div className="card">
            <div className="panel-title">Objects on the counter</div>
            <table className="mini">
              <tbody>
                {Object.entries(tel.objects ?? {}).map(([name, o]: any) => (
                  <tr key={name}>
                    <td>{name}</td>
                    <td>{o.in_bin ? <span className="ok">in bin ✓</span> : o.in_hand ? <span className="accent">in hand</span> : `lift ${(o.lift * 100).toFixed(1)} cm`}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            {res && (
              <div className="result" style={{ borderColor: OUTCOME_COLORS[res.outcome] }}>
                <b style={{ color: OUTCOME_COLORS[res.outcome] }}>{res.outcome}</b> · {res.n_cleared}/{res.n_objects} objects in the bin
                {res.sim_time && <> · {res.sim_time}s</>}
              </div>
            )}
          </div>
          <div className="card">
            <div className="panel-title">Grip force (N)</div>
            <LineChart height={150} xLabel="time (s)" series={[
              { name: "left hand", color: "#58a6ff", points: hist.map((h) => [h.t, h.grip_left]) },
              { name: "right hand", color: "#f0883e", points: hist.map((h) => [h.t, h.grip_right]) },
            ]} />
          </div>
          <div className="card">
            <div className="panel-title">Safety layer</div>
            {safety ? (
              <div className="stats2">
                <Stat label="protective stop" value={safety.protective_stop ? <span className="bad">YES</span> : <span className="ok">no</span>} sub={safety.stop_reason} />
                <Stat label="max env. force" value={`${safety.max_contact_force} N`} />
                <Stat label="rate-limited steps" value={safety.clipped_velocity} />
                <Stat label="self-collision steps" value={safety.self_collision} />
              </div>
            ) : <div className="empty">—</div>}
          </div>
        </div>
        <div className="card">
          <div className="panel-title">Planner log</div>
          <div className="log" ref={logRef}>
            {(snap?.log ?? []).map((e, i) => (
              <div key={i} className={`log-${e.event}`}>
                <span className="t">{(e.t * 0.04).toFixed(1)}s</span> {describe(e)}
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

function describe(e: any): string {
  switch (e.event) {
    case "pick": return `PICK ${e.target.name} with ${e.side} hand (${e.grasp}), attempt ${e.attempt}, est. (${e.target.pos[0]}, ${e.target.pos[1]})`;
    case "skill_done": return `${e.side} hand finished ${e.object}: ${e.status}`;
    case "table_clear": return "head camera sees no more objects on the counter";
    case "gave_up": return `gave up on: ${e.remaining.join(", ")}`;
    case "lifted": return `${e.object} lifted`;
    case "released": return `${e.object} released`;
    case "knocked_over": return `${e.object} knocked over`;
    case "safety_warning": return `⚠ safety warning: ${e.reason}`;
    case "protective_stop": return `⛔ protective stop: ${e.reason}`;
    case "sim_error": return "physics diverged";
    default: return JSON.stringify(e);
  }
}
