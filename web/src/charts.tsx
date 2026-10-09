// Small dependency-free SVG charts.
import React from "react";

const AXIS = "#8b949e";
const GRID = "#30363d";

type Series = { name: string; color: string; points: [number, number][] };

export function LineChart({ series, height = 180, yLabel, xLabel, yMax }: {
  series: Series[]; height?: number; yLabel?: string; xLabel?: string; yMax?: number;
}) {
  const W = 560, H = height, L = 44, R = 10, T = 10, B = 28;
  const all = series.flatMap((s) => s.points);
  if (!all.length) return <div className="empty">no data yet</div>;
  const xs = all.map((p) => p[0]), ys = all.map((p) => p[1]);
  const x0 = Math.min(...xs), x1 = Math.max(...xs) || 1;
  const y0 = 0, y1 = yMax ?? Math.max(1e-6, ...ys) * 1.1;
  const sx = (x: number) => L + ((x - x0) / (x1 - x0 || 1)) * (W - L - R);
  const sy = (y: number) => T + (1 - (y - y0) / (y1 - y0)) * (H - T - B);
  const ticks = [0, 0.25, 0.5, 0.75, 1].map((f) => y0 + f * (y1 - y0));
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="chart">
      {ticks.map((t) => (
        <g key={t}>
          <line x1={L} x2={W - R} y1={sy(t)} y2={sy(t)} stroke={GRID} />
          <text x={L - 6} y={sy(t) + 4} fill={AXIS} fontSize="10" textAnchor="end">{fmt(t)}</text>
        </g>
      ))}
      <text x={W / 2} y={H - 4} fill={AXIS} fontSize="10" textAnchor="middle">{xLabel}</text>
      <text x={12} y={H / 2} fill={AXIS} fontSize="10" textAnchor="middle" transform={`rotate(-90 12 ${H / 2})`}>{yLabel}</text>
      {series.map((s) => (
        <polyline key={s.name} fill="none" stroke={s.color} strokeWidth="2"
          points={s.points.map(([x, y]) => `${sx(x)},${sy(y)}`).join(" ")} />
      ))}
      {series.map((s, i) => (
        <g key={s.name + "l"} transform={`translate(${L + 8 + i * 140}, ${T + 4})`}>
          <rect width="10" height="3" y="4" fill={s.color} />
          <text x="14" y="10" fill="#c9d1d9" fontSize="10">{s.name}</text>
        </g>
      ))}
    </svg>
  );
}

const fmt = (v: number) => (Math.abs(v) >= 100 ? v.toFixed(0) : Math.abs(v) >= 1 ? v.toFixed(1) : v.toFixed(2));

export type CIItem = { label: string; sub?: string; p: number; lo: number; hi: number; color: string };

/** Horizontal bars with Wilson 95 % confidence whiskers. */
export function CIBars({ items }: { items: CIItem[] }) {
  const W = 560, rowH = 30, L = 210, R = 50;
  const H = items.length * rowH + 24;
  const sx = (v: number) => L + v * (W - L - R);
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="chart">
      {[0, 0.25, 0.5, 0.75, 1].map((t) => (
        <g key={t}>
          <line x1={sx(t)} x2={sx(t)} y1={0} y2={H - 18} stroke={GRID} />
          <text x={sx(t)} y={H - 4} fill={AXIS} fontSize="10" textAnchor="middle">{(t * 100).toFixed(0)}%</text>
        </g>
      ))}
      {items.map((it, i) => {
        const y = i * rowH + 6;
        return (
          <g key={it.label + i}>
            <text x={L - 8} y={y + 12} fill="#c9d1d9" fontSize="11" textAnchor="end">{it.label}</text>
            {it.sub && <text x={L - 8} y={y + 23} fill={AXIS} fontSize="9" textAnchor="end">{it.sub}</text>}
            <rect x={L} y={y + 3} width={Math.max(1, sx(it.p) - L)} height={14} fill={it.color} opacity={0.85} rx={2} />
            <line x1={sx(it.lo)} x2={sx(it.hi)} y1={y + 10} y2={y + 10} stroke="#f0f6fc" strokeWidth="1.5" />
            <line x1={sx(it.lo)} x2={sx(it.lo)} y1={y + 5} y2={y + 15} stroke="#f0f6fc" />
            <line x1={sx(it.hi)} x2={sx(it.hi)} y1={y + 5} y2={y + 15} stroke="#f0f6fc" />
            <text x={sx(it.hi) + 6} y={y + 14} fill="#c9d1d9" fontSize="10">{(it.p * 100).toFixed(0)}%</text>
          </g>
        );
      })}
    </svg>
  );
}

/** One stacked horizontal bar per row (failure taxonomy). */
export function StackedBars({ rows, colors }: { rows: { label: string; counts: Record<string, number> }[]; colors: Record<string, string> }) {
  const W = 560, rowH = 26, L = 210, R = 10;
  const keys = Object.keys(colors).filter((k) => rows.some((r) => (r.counts[k] ?? 0) > 0));
  const H = rows.length * rowH + 40;
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="chart">
      {rows.map((r, i) => {
        const tot = Object.values(r.counts).reduce((a, b) => a + b, 0) || 1;
        let x = L;
        return (
          <g key={r.label + i}>
            <text x={L - 8} y={i * rowH + 17} fill="#c9d1d9" fontSize="11" textAnchor="end">{r.label}</text>
            {keys.map((k) => {
              const w = ((r.counts[k] ?? 0) / tot) * (W - L - R);
              const el = w > 0 ? (
                <rect key={k} x={x} y={i * rowH + 5} width={w} height={16} fill={colors[k]}>
                  <title>{`${k}: ${r.counts[k]}`}</title>
                </rect>
              ) : null;
              x += w;
              return el;
            })}
          </g>
        );
      })}
      {keys.map((k, i) => (
        <g key={k} transform={`translate(${L + (i % 4) * 88}, ${rows.length * rowH + 8 + Math.floor(i / 4) * 14})`}>
          <rect width="9" height="9" fill={colors[k]} />
          <text x="12" y="8" fill="#c9d1d9" fontSize="9">{k}</text>
        </g>
      ))}
    </svg>
  );
}

/** Success vs. randomisation level, one line per controller, with CI whiskers. */
export function SimGapChart({ levels, lines }: {
  levels: string[]; lines: { name: string; color: string; points: (CIItem | null)[] }[];
}) {
  const W = 560, H = 220, L = 44, R = 12, T = 12, B = 30;
  const sx = (i: number) => L + (i / Math.max(1, levels.length - 1)) * (W - L - R);
  const sy = (v: number) => T + (1 - v) * (H - T - B);
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="chart">
      {[0, 0.25, 0.5, 0.75, 1].map((t) => (
        <g key={t}>
          <line x1={L} x2={W - R} y1={sy(t)} y2={sy(t)} stroke={GRID} />
          <text x={L - 6} y={sy(t) + 4} fill={AXIS} fontSize="10" textAnchor="end">{(t * 100).toFixed(0)}%</text>
        </g>
      ))}
      {levels.map((l, i) => (
        <text key={l} x={sx(i)} y={H - 10} fill={AXIS} fontSize="10" textAnchor="middle">{l}</text>
      ))}
      {lines.map((ln) => {
        const pts = ln.points.map((p, i) => (p ? [sx(i), sy(p.p), p] : null)).filter(Boolean) as [number, number, CIItem][];
        return (
          <g key={ln.name}>
            <polyline fill="none" stroke={ln.color} strokeWidth="2" points={pts.map(([x, y]) => `${x},${y}`).join(" ")} />
            {pts.map(([x, , p], j) => (
              <g key={j}>
                <line x1={x} x2={x} y1={sy(p.lo)} y2={sy(p.hi)} stroke={ln.color} strokeWidth="1.5" />
                <circle cx={x} cy={sy(p.p)} r={3.5} fill={ln.color} />
              </g>
            ))}
          </g>
        );
      })}
      {lines.map((ln, i) => (
        <g key={ln.name + "l"} transform={`translate(${L + 10 + i * 150}, ${T + 2})`}>
          <rect width="10" height="3" y="4" fill={ln.color} />
          <text x="14" y="10" fill="#c9d1d9" fontSize="10">{ln.name}</text>
        </g>
      ))}
    </svg>
  );
}

export function Stat({ label, value, sub }: { label: string; value: React.ReactNode; sub?: React.ReactNode }) {
  return (
    <div className="stat">
      <div className="stat-label">{label}</div>
      <div className="stat-value">{value}</div>
      {sub && <div className="stat-sub">{sub}</div>}
    </div>
  );
}
