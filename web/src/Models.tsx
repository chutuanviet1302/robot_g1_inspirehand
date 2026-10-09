import { useEffect, useState } from "react";
import { api, Config } from "./api";
import { LineChart, Stat } from "./charts";

const COLORS: Record<string, string> = { act: "#58a6ff", diffusion: "#f0883e" };

export function Models({ config }: { config: Config | null }) {
  const [m, setM] = useState<any>(null);
  useEffect(() => { api.models().then(setM); }, []);
  if (!m) return <div className="page"><div className="empty">loading…</div></div>;
  const det = m.detector;
  return (
    <div className="page">
      <div className="grid2">
        <div className="card">
          <h3>Imitation learning: ACT vs Diffusion Policy</h3>
          <p className="muted">Both learn the pick-and-place skill from the scripted expert (state + perceived object pose → 13 joint targets, action chunks). Validation = action MSE on held-out demonstrations (normalised units). Task-level success is in the Evaluation tab.</p>
          <LineChart yLabel="val action MSE" xLabel="training step" series={m.policies.map((p: any) => ({
            name: p.name, color: COLORS[p.policy] ?? "#a371f7",
            points: p.history.map((h: any) => [h.step, h.val_action_mse]),
          }))} />
          <table className="runs">
            <thead><tr><th>model</th><th>params</th><th>steps</th><th>batch</th><th>device</th><th>train time</th><th>final val MSE</th></tr></thead>
            <tbody>{m.policies.map((p: any) => (
              <tr key={p.name}><td>{p.name}</td><td>{p.params_M} M</td><td>{p.steps}</td><td>{p.batch_size}</td><td>{p.device}</td>
                <td>{(p.train_seconds / 60).toFixed(1)} min</td><td>{p.history.at(-1)?.val_action_mse}</td></tr>))}</tbody>
          </table>
        </div>
        <div className="card">
          <h3>Perception: head camera detector</h3>
          {det ? (
            <>
              <div className="stats">
                <Stat label="box mAP50" value={det.box_map50.toFixed(3)} sub={`mAP50-95 ${det.box_map50_95.toFixed(3)}`} />
                <Stat label="mask mAP50" value={det.mask_map50.toFixed(3)} sub={`mAP50-95 ${det.mask_map50_95.toFixed(3)}`} />
              </div>
              {det.bench && (
                <table className="runs">
                  <thead><tr><th>detector</th><th>recall</th><th>pos. error mean</th><th>p90</th><th>wrist-yaw error</th><th>latency</th></tr></thead>
                  <tbody>{Object.values(det.bench).map((b: any) => (
                    <tr key={b.detector}><td>{b.detector}</td><td>{(100 * b.recall).toFixed(1)}%</td><td>{b.pos_err_mean_mm} mm</td>
                      <td>{b.pos_err_p90_mm} mm</td><td>{b.yaw_err_mean_deg}°</td><td>{b.detect_ms} ms</td></tr>))}</tbody>
                </table>
              )}
            </>
          ) : <div className="empty">not trained yet (`homehand perception train`)</div>}
          <h4>Datasets</h4>
          <table className="runs">
            <thead><tr><th>name</th><th>demos</th><th>frames</th><th>per object</th><th>left / right</th></tr></thead>
            <tbody>{m.datasets.map((d: any) => (
              <tr key={d.name}><td>{d.name}</td><td>{d.n_demos}</td><td>{d.n_frames}</td>
                <td>{Object.entries(d.per_object).map(([k, v]) => `${k} ${v}`).join(", ")}</td>
                <td>{d.per_side.left} / {d.per_side.right}</td></tr>))}</tbody>
          </table>
        </div>
      </div>
      <div className="card">
        <h3>Grasp library</h3>
        <table className="runs">
          <thead><tr><th>object</th><th>grasp type</th><th>mass</th><th>thumb opposition</th><th>closure (thumb, index, middle, ring, pinky)</th><th>grasp height</th></tr></thead>
          <tbody>{config?.objects.map((o) => {
            const g = Object.values(config.grasps).find((x) => x.label === o.grasp)!;
            return (<tr key={o.id}><td>{o.label}</td><td>{o.grasp}</td><td>{o.mass} kg</td><td>{g?.thumb_yaw} rad</td>
              <td>{g?.close.join(", ")}</td><td>{(g?.z_frac * 100).toFixed(0)}% of height</td></tr>);
          })}</tbody>
        </table>
      </div>
    </div>
  );
}
