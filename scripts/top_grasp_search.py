"""Search the top-grasp geometry for a lying object (palm orientation + where the object sits in the hand).

    python scripts/top_grasp_search.py --object soup_can

Each candidate: the hand comes down vertically (palm-down frame tilted by `tilt` about the thumb axis, so the
fingers point down by that much), closes, lifts 12 cm and holds. Success = the object is lifted >= 8 cm and still
in the hand. Only the hand / IK / physics are used (no planner), so a candidate takes ~1 s.
"""

from __future__ import annotations

import argparse
import itertools
import json
import multiprocessing as mp
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

_W: dict = {}


def _qmul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def _qaxis(axis, ang):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    return np.r_[np.cos(ang / 2), np.sin(ang / 2) * axis]


def hand_quat(side: str, yaw: float, tilt: float) -> np.ndarray:
    """Palm down, fingers along `yaw`, then tilted fingers-down by `tilt` about the thumb axis."""
    a = -np.pi / 2 if side == "right" else np.pi / 2
    q = _qmul(_qaxis((0, 0, 1), yaw), _qaxis((1, 0, 0), a))
    # local z (thumb axis); positive rotation turns the fingers (x) towards the palm normal (y) = downwards
    return _qmul(q, _qaxis((0, 0, 1), tilt if side == "right" else -tilt))


def _init():
    os.environ.setdefault("MUJOCO_GL", "egl" if sys.platform.startswith("linux") else "glfw")
    from homehand.control.ik import ArmIK
    from homehand.env.kitchen_env import KitchenEnv
    _W["env"], _W["ik"] = KitchenEnv(), ArmIK()


def trial(args) -> dict:
    import mujoco
    from homehand.control.skills import compose_action
    from homehand.env.kitchen_env import TABLE_Z
    from homehand.model.objects import _rot, lying_dims
    from homehand.model import spec
    cand, name, side, x, y = args
    env, ik = _W["env"], _W["ik"]
    tilt, align, dx, dy, dz, thumb_yaw, pre_thumb = (cand[k] for k in
                                                     ("tilt", "align", "dx", "dy", "dz", "thumb_yaw", "pre_thumb"))
    # object axis heading such that the hand yaw is 0 (fingers forward): thumb along the axis -> axis along y
    axis_yaw = np.pi / 2 if align == "along" else 0.0
    obs = env.reset("pick_place", seed=0, layout=[(name, x, y, axis_yaw, "lying")])
    c0 = obs["objects"][name]["centre"].copy()
    q_hand = hand_quat(side, 0.0, np.deg2rad(tilt))
    R = _rot(q_hand)
    palm = c0 - R @ np.array([dx, dy if side == "right" else -dy, dz])   # left palm normal = -y
    pre = palm + np.array([0, 0, 0.10])
    q, err = ik.solve_best(side, pre, q_hand)
    if err > 0.01:
        return {**cand, "ok": False, "why": "ik_pre", "lift": 0.0}
    hand = {s: np.array([0.0, 0.1, 0.3, 0.35, 0.4, 0.45]) for s in spec.SIDES}
    hand[side] = np.array([pre_thumb, 0, 0, 0, 0, 0], float)
    # teleport the arm to the pre-grasp pose (this search only judges the grasp itself)
    env.data.qpos[env.arm_qadr] = ik.arm_targets(q)
    env.last_action = compose_action(ik, q, hand)
    env.data.ctrl[env.act_ids] = env.last_action
    mujoco.mj_forward(env.model, env.data)
    for _ in range(10):
        env.step(compose_action(ik, q, hand))

    def move(target_pos, steps, hand_to=None):
        nonlocal q
        path = ik.plan_path(q, side, [(q_pos, q_hand) for q_pos in
                                       np.linspace(ik.palm_pose(q, side)[0], target_pos, 8)])
        h0 = hand[side].copy()
        for k in range(steps):
            a = min(1.0, (k + 1) / steps)
            i = min(len(path) - 1, int(a * (len(path) - 1)))
            q = path[i]
            if hand_to is not None:
                hand[side] = h0 + (np.asarray(hand_to) - h0) * a
            env.step(compose_action(ik, q, hand))

    _, e = ik.solve_best(side, palm, q_hand)
    if e > 0.01:
        return {**cand, "ok": False, "why": "ik_grasp", "lift": 0.0}
    move(palm, 25)
    moved = float(np.linalg.norm(env.observe()["objects"][name]["centre"][:2] - c0[:2]))
    move(palm, 20, hand_to=(thumb_yaw, 0.6, 1.47, 1.47, 1.47, 1.47))
    for _ in range(5):
        env.step(compose_action(ik, q, hand))
    move(palm + [0, 0, 0.12], 25)
    for _ in range(15):
        env.step(compose_action(ik, q, hand))
    ob = env.observe()
    c1 = ob["objects"][name]["centre"]
    lift = float(c1[2] - c0[2])
    Rp = env.data.site_xmat[env.palm_site[side]].reshape(3, 3)
    held = Rp.T @ (c1 - ob["palm"][side])
    if side == "left":
        held[1] = -held[1]                     # mirrored into the right hand's convention
    cand = {**cand, "held": [round(float(v), 4) for v in held]}
    in_hand = name in env.contacts_with_hands()
    return {**cand, "ok": bool(lift > 0.08 and in_hand), "why": "" if lift > 0.08 else "not lifted",
            "lift": round(lift, 3), "pushed_before_close": round(moved, 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--object", default="soup_can")
    ap.add_argument("--side", default="right")
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--out", default="")
    ap.add_argument("--grid", default="", help="JSON dict overriding grid entries")
    ap.add_argument("--absy", type=float, default=0.14)
    a = ap.parse_args()
    grid = dict(tilt=[30, 45, 55, 70], align=["along", "across"], dx=[0.04, 0.05, 0.06, 0.07],
                dy=[-0.01, 0.0, 0.01], dz=[0.01, 0.02, 0.03], thumb_yaw=[1.0], pre_thumb=[0.0])
    if a.grid:
        grid.update(json.loads(a.grid))
    cands = [dict(zip(grid, v)) for v in itertools.product(*grid.values())]
    sgn = -1 if a.side == "right" else 1
    jobs = [(c, a.object, a.side, 0.37, sgn * a.absy) for c in cands]
    from homehand.resources import safe_workers
    with mp.get_context("spawn").Pool(safe_workers(a.workers), initializer=_init) as pool:
        res = pool.map(trial, jobs)
    res.sort(key=lambda r: (-r["ok"], -r["lift"]))
    for r in res[:8]:
        print({k: r[k] for k in ("tilt", "align", "dx", "dy", "dz", "ok", "lift")})
    print(f"{sum(r['ok'] for r in res)}/{len(res)} candidates lift the {a.object}")
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
