"""How human-like are the reaching and carrying movements? Kinematic markers from human motor-control studies.

    python scripts/human_likeness.py --style human --seeds 0-23 --out data/human_likeness_human.json
    python scripts/human_likeness.py --style v1    --seeds 0-23 --out data/human_likeness_v1.json

Per reach-to-grasp movement (palm from the start of `reach` until the fingers start closing) and per carry
(`lift` + `transport`), from the palm trajectory at 25 Hz (expert, ground-truth poses):

  stops            interior samples where the palm is (almost) at rest, |v| < 2 cm/s   human: 0
  speed peaks      local maxima above 30 % of the peak speed                            human: 1
  time to peak     time of peak speed / movement time            reach-to-grasp human: ~0.35-0.45
  SPARC            spectral arc length of the speed profile (smoothness, closer to 0 = smoother)
                                                                 healthy reaching: about -1.4 to -1.8
  straightness     path length / straight-line distance          human: ~1.0-1.2 (gently curved)
  peak aperture    time of the widest hand opening / reach time  human: ~0.6-0.7

References: Flash & Hogan 1985; Marteniuk et al. 1987; Jeannerod 1984; Morasso 1981;
Balasubramanian et al. 2015 (SPARC).
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

FS = 25.0
REST_SPEED = 0.02
HUMAN = {"stops": (0, 0), "speed_peaks": (1, 1), "time_to_peak": (0.35, 0.45), "sparc": (-1.8, -1.4),
         "straightness": (1.0, 1.2), "peak_aperture_time": (0.6, 0.7)}


def sparc(speed: np.ndarray, fs: float = FS, padlevel: int = 4, fc: float = 10.0, amp_th: float = 0.05) -> float:
    """Spectral arc length (Balasubramanian et al. 2015)."""
    nfft = int(2 ** (np.ceil(np.log2(len(speed))) + padlevel))
    f = np.arange(0, fs, fs / nfft)
    mag = np.abs(np.fft.fft(speed, nfft))
    mag = mag / max(mag.max(), 1e-12)
    sel = f <= fc
    f, mag = f[sel], mag[sel]
    above = np.nonzero(mag >= amp_th)[0]
    f, mag = f[above[0]: above[-1] + 1], mag[above[0]: above[-1] + 1]
    if len(f) < 2:
        return 0.0
    return float(-np.sum(np.sqrt((np.diff(f) / (f[-1] - f[0])) ** 2 + np.diff(mag) ** 2)))


def movement_markers(pos: np.ndarray, aperture: np.ndarray | None = None) -> dict:
    # as in human motion-capture studies: low-pass the trajectory (~5 Hz, 5-sample moving average at 25 Hz)
    # and analyse the movement between onset and offset (speed above REST_SPEED), not the holds around it
    k = np.ones(5) / 5
    pos = np.stack([np.convolve(np.pad(pos[:, i], 2, mode="edge"), k, mode="valid") for i in range(3)], axis=1)
    v = np.linalg.norm(np.diff(pos, axis=0), axis=1) * FS
    moving = np.nonzero(v >= REST_SPEED)[0]
    if len(moving) < 4:
        return {}
    on, off = moving[0], moving[-1] + 1
    v, pos = v[on:off], pos[on: off + 1]
    if aperture is not None:
        aperture = aperture[on: off + 1]
    if len(v) < 4 or v.max() < 1e-4:
        return {}
    inner = v[2:-2]
    rest = inner < REST_SPEED
    stops = int(np.sum(rest[1:] & ~rest[:-1]) + (rest[0] if len(rest) else 0))
    peaks = [i for i in range(1, len(v) - 1) if v[i] >= v[i - 1] and v[i] > v[i + 1] and v[i] > 0.3 * v.max()]
    out = {"duration_s": round(len(v) / FS, 2), "stops": stops, "speed_peaks": len(peaks),
           "time_to_peak": round(float(np.argmax(v) + 0.5) / len(v), 3), "sparc": round(sparc(v), 3),
           "straightness": round(float(v.sum() / FS / max(np.linalg.norm(pos[-1] - pos[0]), 1e-6)), 3)}
    if aperture is not None:
        # the hand holds its widest opening for a while: time at which it first reaches 95 % of it
        lo, hi = float(aperture.min()), float(aperture.max())
        i = int(np.argmax(aperture >= lo + 0.95 * (hi - lo))) if hi - lo > 1e-3 else 0
        out["peak_aperture_time"] = round(i / max(len(aperture) - 1, 1), 3)
    return out


def run_seed(args) -> dict:
    seed, style = args
    os.environ["HOMEHAND_EXPERT_STYLE"] = style        # read when the skill module is imported
    os.environ.setdefault("MUJOCO_GL", "egl" if sys.platform.startswith("linux") else "glfw")
    from homehand.control.planner import TidyPlanner
    from homehand.env.kitchen_env import KitchenEnv
    from homehand.runner import run_episode
    env = KitchenEnv()
    planner = TidyPlanner(env, perception="ground_truth")
    log: list[tuple] = []

    def on_step(obs, action):
        if planner.current is None or planner.skill is None:
            log.append((None, None, None, None))
            return
        side = planner.current["side"]
        base = 0 if side == "left" else 6
        opening = -float(np.mean(obs["hand_q"][base + 2: base + 6]))      # fingers less flexed = wider open
        log.append((id(planner.skill), planner.skill.phase, obs["palm"][side].copy(), opening))

    ep = run_episode(env, planner, seed=seed, on_step=on_step)
    moves = []
    for kind, phases in (("reach", ("reach", "descend", "approach")), ("carry", ("lift", "transport"))):
        i = 0
        while i < len(log):
            sk, ph = log[i][0], log[i][1]
            if sk is not None and ph == phases[0]:
                j = i
                while j < len(log) and log[j][0] == sk and log[j][1] in phases:
                    j += 1
                pos = np.array([log[k][2] for k in range(i, j)])
                ap = np.array([log[k][3] for k in range(i, j)]) if kind == "reach" else None
                m = movement_markers(pos, ap)
                if m:
                    moves.append({"kind": kind, **m})
                i = j
            else:
                i += 1
    return {"seed": seed, "outcome": ep["outcome"], "cleared": ep["n_cleared"], "n": ep["n_objects"],
            "sim_time": ep["sim_time"], "moves": moves}


def summarize(res: list[dict]) -> dict:
    out = {"episodes": len(res), "clean": sum(r["outcome"] == "success" for r in res),
           "objects": f"{sum(r['cleared'] for r in res)}/{sum(r['n'] for r in res)}",
           "mean_episode_s": round(float(np.mean([r["sim_time"] for r in res])), 1)}
    for kind in ("reach", "carry"):
        ms = [m for r in res for m in r["moves"] if m["kind"] == kind]
        if not ms:
            continue
        s = {"n": len(ms)}
        for key in ("duration_s", "stops", "speed_peaks", "time_to_peak", "sparc", "straightness",
                    "peak_aperture_time"):
            vals = [m[key] for m in ms if key in m]
            if vals:
                s[key] = round(float(np.mean(vals)), 3)
                if key in HUMAN:
                    lo, hi = HUMAN[key]
                    s[key + "_in_human_range"] = round(float(np.mean([lo - 1e-9 <= v <= hi + 1e-9 for v in vals])), 2)
        out[kind] = s
    return out


def main() -> None:
    from homehand.resources import safe_workers
    ap = argparse.ArgumentParser()
    ap.add_argument("--style", default="human", choices=("human", "v1"))
    ap.add_argument("--seeds", default="0-23")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    lo, _, hi = a.seeds.partition("-")
    jobs = [(s, a.style) for s in range(int(lo), int(hi or lo) + 1)]
    with mp.get_context("spawn").Pool(safe_workers(a.workers)) as pool:
        res = sorted(pool.map(run_seed, jobs), key=lambda r: r["seed"])
    summary = {"style": a.style, **summarize(res)}
    print(json.dumps(summary, indent=1))
    if a.out:
        Path(a.out).write_text(json.dumps({"summary": summary, "episodes": res}, indent=1))


if __name__ == "__main__":
    main()
