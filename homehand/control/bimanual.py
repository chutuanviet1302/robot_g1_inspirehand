"""Scripted bimanual expert: lift a wide box by pressing both palms on its sides (`env.bimanual_env`).

Both palms keep the "handshake" orientation (fingers forward, palms facing each other), turned with the box's yaw.
Every move is planned before it runs: the two palm targets are interpolated in Cartesian space together (the
same offset for both, so the grip on the box does not change while it is carried), IK solves both arms at once
for every sample (warm-started), and a minimum-jerk profile times the joint path.

  reach      above and outside the box
  descend    down beside it
  squeeze    palms in by SQUEEZE past the box's sides (the position-servoed arms press on it)
  lift       up by LIFT_H, hold for the evaluation's HOLD_STEPS
  carry      sideways to the goal
  place      down to the counter
  release    palms out, up, back to the ready pose
"""

from __future__ import annotations

import numpy as np

from homehand.control.ik import ArmIK
from homehand.control.skills import compose_action, min_jerk, rot2, yaw_quat
from homehand.env.bimanual_env import HOLD_STEPS
from homehand.env.kitchen_env import TABLE_Z

SQUEEZE = 0.018            # m past each side of the box: the press that holds it
OUT = 0.05                 # m outside the box before / after the squeeze
ABOVE = 0.10               # m above the grasp height while reaching over
LIFT_H = 0.12
GRIP_Z = 0.05              # palm height above the counter (below the box's half height)
SPEED = 0.08               # m/s, palm speed of the Cartesian moves
STEP = 0.01                # m between IK samples
# thumbs up out of the way, fingers fully curled: they bend towards the palm, i.e. into the box's side, and add
# finger-pad contact to the palm's (straight fingers: 4/10 lifts, curled: 10/10, squeeze / speed tuned on 10 seeds)
HAND = (0.0, 0.0, 1.47, 1.47, 1.47, 1.47)


class BimanualExpert:
    kind = "bimanual_expert"

    def __init__(self, env, ik: ArmIK | None = None):
        self.env = env
        self.ik = ik or ArmIK()
        self.perception_mode = "ground_truth"
        self.skill_kind = "bimanual_expert"
        self.last_perception = None

    # ------------------------------------------------------------------ planning
    def _palms(self, centre: np.ndarray, yaw: float, half_y: float, z: float) -> dict:
        """Palm targets with the palms `half_y` from the box's centre plane, on its +-y sides."""
        q = yaw_quat(yaw)
        side_dir = rot2(yaw, (0.0, 1.0))
        return {s: (np.array([*(centre + sgn * half_y * side_dir), z]), q)
                for s, sgn in (("left", 1.0), ("right", -1.0))}

    def _move(self, q0: np.ndarray, start: dict, goal: dict, min_steps: int = 10) -> tuple[np.ndarray, list]:
        """Joint path for a straight move of both palms from `start` to `goal`, with its time profile."""
        dist = max(float(np.linalg.norm(goal[s][0] - start[s][0])) for s in goal)
        n = max(2, int(np.ceil(dist / STEP)))
        qs, q = [], q0
        for k in range(1, n + 1):
            u = k / n
            targets = {s: (start[s][0] + u * (goal[s][0] - start[s][0]), goal[s][1]) for s in goal}
            q, _ = self.ik.solve(q, targets, iters=40)
            qs.append(q)
        steps = max(min_steps, int(np.ceil(1.875 * dist / SPEED / 0.04)))
        traj = [qs[min(len(qs) - 1, int(min_jerk((i + 1) / steps) * len(qs)))] for i in range(steps)]
        return q, traj

    def reset(self, obs: dict) -> None:
        tote = obs["tote"]
        c, yaw, half = np.asarray(tote["pos"][:2], float), float(tote["yaw"]), tote["half"]
        goal_c = np.asarray(obs["goal"], float)
        hy = float(half[1])
        z, zt = TABLE_Z + GRIP_Z, TABLE_Z + GRIP_Z + LIFT_H
        ik = self.ik
        q = ik.q_home.copy()
        home = {s: ik.palm_pose(q, s) for s in ("left", "right")}
        keys = [
            ("reach", self._palms(c, yaw, hy + OUT, z + ABOVE), 20, 0),
            ("descend", self._palms(c, yaw, hy + OUT, z), 10, 0),
            ("squeeze", self._palms(c, yaw, hy - SQUEEZE, z), 12, 8),
            ("lift", self._palms(c, yaw, hy - SQUEEZE, zt), 15, HOLD_STEPS + 10),
            ("carry", self._palms(goal_c, yaw, hy - SQUEEZE, zt), 15, 5),
            ("place", self._palms(goal_c, yaw, hy - SQUEEZE, z + 0.004), 15, 5),
            ("release", self._palms(goal_c, yaw, hy + OUT, z + 0.004), 10, 0),
            ("up", self._palms(goal_c, yaw, hy + OUT, z + ABOVE), 10, 0),
        ]
        self.plan, cur = [], home
        for name, goal, min_steps, hold in keys:
            q, traj = self._move(q, cur, goal, min_steps)
            self.plan += [(name, qq) for qq in traj] + [(name, q)] * hold
            cur = goal
        # back to the ready pose in joint space (min-jerk)
        n = 30
        self.plan += [("return", q + min_jerk((i + 1) / n) * (ik.q_home - q)) for i in range(n)]
        self.k = 0
        self.log = [{"t": obs["t"], "event": "plan", "steps": len(self.plan)}]
        self.hand = {s: np.array(HAND, float) for s in ("left", "right")}

    # ------------------------------------------------------------------ execution
    @property
    def done(self) -> bool:
        return self.k >= len(self.plan)

    @property
    def phase(self) -> str:
        return self.plan[min(self.k, len(self.plan) - 1)][0]

    def act(self, obs: dict) -> np.ndarray:
        name, q = self.plan[min(self.k, len(self.plan) - 1)]
        self.k += 1
        self.env.context = {"phase": f"both:{name}", "target": "tote"}
        return compose_action(self.ik, q, self.hand)
