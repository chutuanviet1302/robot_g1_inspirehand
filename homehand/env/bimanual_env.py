"""Bimanual task: lift a box too wide for one hand with both palms, hold it, carry it aside, put it down.

Scene `generated/scene_bimanual.xml` = the kitchen scene + the box ("tote", see `model.build.TOTE_HALF`); the YCB
objects stay parked. Physics randomisation (friction, box mass, finger gains, latency, observation noise) uses the
same presets as the tidy-table task; the box pose is jittered per seed.

Outcomes: success (lifted >= LIFT_OK and held HOLD_STEPS, then put down upright within GOAL_TOL of the goal),
not_lifted, dropped (lost while lifted), misplaced (down, but off the goal or tilted), safety_stop, sim_error.
"""

from __future__ import annotations

import mujoco
import numpy as np

from homehand import paths
from homehand.env.kitchen_env import TABLE_Z, KitchenEnv, quat_yaw, yaw_of
from homehand.model import spec

TOTE_X = 0.32              # m; at x >= 0.35 the palms, low beside the box, are at the edge of the arms' reach
POS_JITTER = 0.02          # m, box position per seed
YAW_JITTER = np.deg2rad(10)
GOAL_SHIFT = 0.06          # m sideways (left or right, per seed)
LIFT_OK = 0.08
HOLD_STEPS = 50            # 2 s
GOAL_TOL = 0.03
TILT_OK = np.deg2rad(15)
HORIZON = 600
OUTCOMES = ["success", "not_lifted", "dropped", "misplaced", "safety_stop", "sim_error"]


class BimanualEnv(KitchenEnv):
    def __init__(self, record: bool = False):
        super().__init__(xml_path=paths.BIMANUAL_XML, record=record)
        m = self.model
        self.tote_body = m.body("tote").id
        self.tote_qadr = m.jnt_qposadr[m.joint("tote_free").id]
        self.tote_geom = m.geom("tote").id
        self.tote_half = m.geom_size[self.tote_geom].copy()
        self.nominal_tote = (m.geom_friction[self.tote_geom].copy(), m.body_mass[self.tote_body],
                             m.body_inertia[self.tote_body].copy())

    def reset(self, task: str = "bimanual_lift", seed: int = 0, objects=None, randomization=None, layout=None) -> dict:
        super().reset("pick_place", seed=seed, randomization=randomization, layout=[])   # every YCB object parked
        m, d = self.model, self.data
        fr, mass, inertia = self.nominal_tote
        m.geom_friction[self.tote_geom] = fr
        m.geom_friction[self.tote_geom, 0] = fr[0] * self.rand["friction"]
        m.body_mass[self.tote_body] = mass * self.rand["mass"]
        m.body_inertia[self.tote_body] = inertia * self.rand["mass"]
        # mass changed: recompute the derived constants (see KitchenEnv.reset); mj_setConst also puts the state
        # back to qpos0, so keep the ready pose the parent reset established
        state = d.qpos.copy(), d.qvel.copy(), d.ctrl.copy()
        mujoco.mj_setConst(m, d)
        d.qpos[:], d.qvel[:], d.ctrl[:] = state
        x = TOTE_X + self.rng.uniform(-POS_JITTER, POS_JITTER)
        y = self.rng.uniform(-POS_JITTER, POS_JITTER)
        yaw = self.rng.uniform(-YAW_JITTER, YAW_JITTER)
        a = self.tote_qadr
        d.qpos[a:a + 3] = [x, y, TABLE_Z + 0.001]
        d.qpos[a + 3:a + 7] = quat_yaw(yaw)
        d.qvel[m.jnt_dofadr[m.joint("tote_free").id]:][:6] = 0.0
        mujoco.mj_forward(m, d)
        for _ in range(60):
            mujoco.mj_step(m, d)
        self.goal = np.array([x, y + self.rng.choice([-1.0, 1.0]) * GOAL_SHIFT])
        self.task = task
        self.horizon = HORIZON
        self.layout = [("tote", float(x), float(y), float(yaw), "upright")]
        self.max_lift, self.held_steps, self.best_hold = 0.0, 0, 0
        self.was_lifted = self.dropped = False
        self.info.objects = ["tote"]
        return self.observe()

    # ------------------------------------------------------------------ state of the box
    def tote_pose(self) -> tuple[np.ndarray, np.ndarray]:
        a = self.tote_qadr
        return self.data.qpos[a:a + 3].copy(), self.data.qpos[a + 3:a + 7].copy()

    def tote_tilt(self) -> float:
        R = self.data.xmat[self.tote_body].reshape(3, 3)
        return float(np.arccos(np.clip(R[2, 2], -1.0, 1.0)))

    def hands_on_tote(self) -> set[str]:
        d, m = self.data, self.model
        out = set()
        for i in range(d.ncon):
            c = d.contact[i]
            b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
            for bo, bh in ((b1, b2), (b2, b1)):
                if bo == self.tote_body and bh in self.body_side:
                    out.add(self.body_side[bh])
        return out

    def observe(self) -> dict:
        obs = super().observe()
        pos, quat = self.tote_pose()
        noise = self.rand["obs_noise"] if hasattr(self, "rand") else 0.0
        if noise > 0:
            pos = pos + self.rng.normal(0, noise, 3)
        obs["tote"] = {"pos": pos, "yaw": yaw_of(quat), "half": self.tote_half.copy()}
        obs["goal"] = getattr(self, "goal", None)
        return obs

    # ------------------------------------------------------------------ bookkeeping (replaces the tidy-table ones)
    def _update_tracks(self) -> None:
        pos, _ = self.tote_pose()
        lift = float(pos[2] - TABLE_Z)
        self.max_lift = max(self.max_lift, lift)
        both = self.hands_on_tote() == set(spec.SIDES)
        self.held_steps = self.held_steps + 1 if both and lift >= LIFT_OK else 0
        self.best_hold = max(self.best_hold, self.held_steps)
        if lift >= LIFT_OK and both:
            self.was_lifted = True
        if self.was_lifted and lift > 0.02 and not self.hands_on_tote():
            if not self.dropped:
                self.events.append({"t": self.t, "event": "dropped", "object": "tote"})
            self.dropped = True

    def _check_done(self) -> bool:
        if self.t >= self.horizon:
            self._finish()
            return True
        return False

    def _finish(self) -> None:
        pos, _ = self.tote_pose()
        on_table = pos[2] - TABLE_Z < 0.01 and not self.hands_on_tote()
        at_goal = float(np.linalg.norm(pos[:2] - self.goal)) < GOAL_TOL
        upright = self.tote_tilt() < TILT_OK
        if self.best_hold < HOLD_STEPS:
            out = "dropped" if self.dropped else "not_lifted"
        elif self.dropped and not (on_table and at_goal and upright):
            out = "dropped"
        else:
            out = "success" if on_table and at_goal and upright else "misplaced"
        if self.safety.stats.protective_stop and out != "success":
            out = "safety_stop"
        self.info.object_outcomes = {"tote": out}
        self.info.outcome = out
        self.info.success = out == "success"
        self.info.steps = self.t
        self.info.details = {
            "max_lift": round(self.max_lift, 4), "held_steps": self.best_hold,
            "goal_error": round(float(np.linalg.norm(pos[:2] - self.goal)), 4),
            "tilt_deg": round(float(np.rad2deg(self.tote_tilt())), 1),
            "sampled": self.rand, "n_cleared": int(out == "success"), "n_objects": 1,
            "safety": self.safety.stats.to_dict(), "causes": {},
        }
