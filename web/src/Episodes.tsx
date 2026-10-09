import { useEffect, useState } from "react";
import { api, Episode, OUTCOME_COLORS, Run } from "./api";
import { useLive, Viewer } from "./Live";

export function Episodes() {
  const [runs, setRuns] = useState<Run[]>([]);
  const [runId, setRunId] = useState<number | undefined>();
  const [outcome, setOutcome] = useState("");
  const [eps, setEps] = useState<Episode[]>([]);
  const { snap, connected } = useLive();

  useEffect(() => { api.runs().then(setRuns); }, []);
  useEffect(() => { api.episodes(runId, outcome || undefined).then(setEps); }, [runId, outcome]);

  return (
    <div className="page episodes">
      <div className="card list">
        <h3>Episodes</h3>
        <div className="row">
          <label>Run<select value={runId ?? ""} onChange={(e) => setRunId(e.target.value ? +e.target.value : undefined)}>
            <option value="">all</option>{runs.map((r) => <option key={r.id} value={r.id}>#{r.id} {r.name}</option>)}</select></label>
          <label>Outcome<select value={outcome} onChange={(e) => setOutcome(e.target.value)}>
            <option value="">all</option>{Object.keys(OUTCOME_COLORS).filter((k) => k !== "running").map((k) => <option key={k}>{k}</option>)}</select></label>
        </div>
        <div className="scroll">
          <table className="runs">
            <thead><tr><th>run</th><th>seed</th><th>outcome</th><th>cleared</th><th>objects</th><th></th></tr></thead>
            <tbody>
              {eps.map((e) => (
                <tr key={e.id}>
                  <td>#{e.run_id}</td><td>{e.seed}</td>
                  <td><span className="pill" style={{ background: OUTCOME_COLORS[e.outcome] }}>{e.outcome}</span></td>
                  <td>{e.n_cleared}/{e.n_objects}</td>
                  <td className="objs">{Object.entries(e.object_outcomes).map(([o, r]) => (
                    <span key={o} title={r} className="obj" style={{ borderColor: OUTCOME_COLORS[r] }}>{o}</span>))}</td>
                  <td><button disabled={!e.replay} onClick={() => api.replay(e.id)}>▶ replay</button></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
      <div className="main">
        <Viewer snap={snap} connected={connected} />
        {snap?.result && (
          <div className="card">
            <b style={{ color: OUTCOME_COLORS[snap.result.outcome] }}>{snap.result.outcome}</b> · {snap.result.n_cleared}/{snap.result.n_objects} objects ·{" "}
            {Object.entries(snap.result.object_outcomes ?? {}).map(([o, r]: any) => `${o}: ${r}`).join(" · ")}
          </div>
        )}
      </div>
    </div>
  );
}
