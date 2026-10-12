"""Learning curves of the residual PPO run: share of picks still in the hand at the release point, and the reward
terms per pick, against environment steps (moving window over rollouts, the expert's level for reference).

    python scripts/plot_rl.py models/residual_ppo [--window 20] [--expert 0.75]
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("--window", type=int, default=20, help="rollouts per moving-window point")
    ap.add_argument("--expert", type=float, default=None, help="the expert's pick success, drawn for reference")
    a = ap.parse_args()
    rows = list(csv.DictReader(open(a.run / "progress.csv")))
    steps = np.array([float(r["steps"]) for r in rows])
    picks = np.array([float(r["picks"]) for r in rows])
    ok = np.array([float(r["pick_success"] or 0) for r in rows]) * picks
    w = min(a.window, len(rows))
    k = np.ones(w)
    succ = np.convolve(ok, k, "valid") / np.convolve(picks, k, "valid")
    n = np.convolve(picks, k, "valid")
    half = 1.96 * np.sqrt(succ * (1 - succ) / n)          # normal approximation, for the band only
    xs = steps[w - 1:]
    terms = [t for t in ("lift", "lifted", "carried", "neighbour", "target", "force", "action", "stop") if t in rows[0]]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(xs, succ, color="C0", label="expert + residual")
    ax1.fill_between(xs, succ - half, succ + half, color="C0", alpha=0.2)
    if a.expert is not None:
        ax1.axhline(a.expert, color="0.4", ls="--", label="expert alone")
    ax1.set(xlabel="environment steps", ylabel="picks carried to the bin", ylim=(0, 1),
            title=f"pick success (window {w} rollouts)")
    ax1.legend(loc="lower right")
    for i, t in enumerate(terms):
        v = np.array([float(r[t] or 0) for r in rows])
        ax2.plot(xs, np.convolve(v * picks, k, "valid") / n, label=t, color=f"C{i}")
    ax2.axhline(0, color="0.6", lw=0.5)
    ax2.set(xlabel="environment steps", ylabel="reward per pick", title="reward terms")
    ax2.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    out = a.run / "learning_curve.png"
    fig.savefig(out, dpi=120)
    print(f"-> {out}  (last window: {succ[-1]:.2f} over {int(n[-1])} picks)")


if __name__ == "__main__":
    main()
