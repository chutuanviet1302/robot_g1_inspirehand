import { useEffect, useState } from "react";
import { api, Config } from "./api";
import { Episodes } from "./Episodes";
import { Evaluate } from "./Evaluate";
import { Live } from "./Live";
import { Models } from "./Models";

const TABS = ["Live", "Evaluation", "Episodes", "Models"] as const;
type Tab = (typeof TABS)[number];

export function App() {
  const [tab, setTab] = useState<Tab>(() => (localStorage.getItem("tab") as Tab) || "Live");
  const [config, setConfig] = useState<Config | null>(null);
  const [sys, setSys] = useState<any>(null);
  useEffect(() => { api.config().then(setConfig); }, []);
  useEffect(() => {
    const f = () => api.system().then(setSys).catch(() => {});
    f(); const t = setInterval(f, 5000); return () => clearInterval(t);
  }, []);
  const go = (t: Tab) => { setTab(t); try { localStorage.setItem("tab", t); } catch { /* private mode */ } };

  return (
    <div className="app">
      <header>
        <div className="brand">🤖 <b>HomeHand</b><span className="muted"> · Unitree G1 + 2× Inspire hand · tidy the counter</span></div>
        <nav>{TABS.map((t) => <button key={t} className={tab === t ? "on" : ""} onClick={() => go(t)}>{t}</button>)}</nav>
        {sys && (
          <div className="sys muted">
            RAM free {(sys.ram_available_mb / 1024).toFixed(1)} GB · workers ≤ {sys.safe_workers}
            {sys.gpu && <> · {sys.gpu.name} {(sys.gpu.free_mb / 1024).toFixed(1)}/{(sys.gpu.total_mb / 1024).toFixed(1)} GB</>}
          </div>
        )}
      </header>
      {tab === "Live" && <Live config={config} />}
      {tab === "Evaluation" && <Evaluate config={config} />}
      {tab === "Episodes" && <Episodes />}
      {tab === "Models" && <Models config={config} />}
    </div>
  );
}
