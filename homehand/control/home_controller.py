"""Home task controller: walk to the counter (or the conveyor) around obstacles, stop in front of it, do the job.

The job is the tidy-table planner at the counter, or the conveyor planner at the belt (`HomeConveyorEnv`: the
belt starts when the robot is locked in place).

Phases
1. navigate  the lidar scans at 5 Hz into an occupancy grid; A* plans from where the robot stands to a point
             `PRE_DOCK_BACK` m in front of the stop point; the path is replanned whenever a new obstacle shows up
             on it; pure pursuit turns it into velocity commands for the walking policy. Arms at the ready pose.
2. approach  the last half metre straight to the stop point and the counter's heading, on the walker's
             position loop (the counter itself is the goal, so the inflated map would refuse it).
3. settle    1.5 s standing at the stop point, then the arms' IK is told where the pelvis really stands.
4. tidy      the tidy-table planner, unchanged (perception, grasp choice, skills, safety), while the legs keep
             the robot at the stop point. It plans for the pelvis where the robot docked (`ArmIK.set_base`), so
             the docking error is absorbed; the walking policy keeps stepping in place, swaying the pelvis by up to
             +-7 cm, so every arm command is re-solved for where the pelvis is at that moment (`_compensate`).
"""

from __future__ import annotations

import numpy as np

from homehand.control.grasps import RELAXED_HAND
from homehand.control.ik import ArmIK
from homehand.control.planner import TidyPlanner
from homehand.control.skills import compose_action
from homehand.locomotion import nav
from homehand.locomotion.walker import wrap
from homehand.model import spec
from homehand.model.build import HOME_DOCK

PRE_DOCK_BACK = 0.55
APPROACH_V = 0.15
LOCK_BASE = __import__("os").environ.get("HOMEHAND_LOCK_BASE", "1") == "1"   # weld the pelvis at the stop point            # m/s of the moving hold target on the last half metre
SCAN_EVERY = 5               # control steps (5 Hz)
ARRIVE_TOL = 0.12            # m: close enough to the pre-dock point
DOCK_TOL = (0.03, np.deg2rad(4))
SETTLE_STEPS = 38            # 1.5 s
LOOKAHEAD_S = float(__import__("os").environ.get("HOMEHAND_SWAY_AHEAD", 0.1))
COMP_ITERS = 12              # warm-started IK iterations per control step for the sway compensation


class HomeController:
    kind = "home"

    def __init__(self, env, perception: str = "auto", ik: ArmIK | None = None):
        self.env = env
        self.ik = ik or ArmIK()
        self.conveyor = hasattr(env, "start_belt")
        if self.conveyor:      # the conveyor variant: pick off the belt once standing at it
            from homehand.control.conveyor_planner import ConveyorPlanner
            self.tidy = ConveyorPlanner(env, perception=perception, ik=self.ik)
        else:
            self.tidy = TidyPlanner(env, perception=perception, ik=self.ik)
        self.perception_mode = self.tidy.perception_mode
        self.skill_kind = "expert"
        self.pre_dock = np.array([HOME_DOCK[0] - PRE_DOCK_BACK, HOME_DOCK[1]])

    # ------------------------------------------------------------------ status
    @property
    def done(self) -> bool:
        return self.stage == "tidy" and self.tidy.done

    @property
    def phase(self) -> str:
        return f"tidy/{self.tidy.phase}" if self.stage == "tidy" else self.stage

    @property
    def last_perception(self):
        return getattr(self.tidy, "last_perception", None) if self.stage == "tidy" else None

    @property
    def current(self):
        return getattr(self.tidy, "current", None) if self.stage == "tidy" else None

    @property
    def log(self):
        return self.nav_log + (self.tidy.log if self.stage == "tidy" else [])

    # ------------------------------------------------------------------ loop
    def reset(self, obs: dict) -> None:
        self.stage = "navigate"
        self.grid = nav.OccupancyGrid()
        self.path = None
        self.replans = 0
        self.k = 0
        self.settle = 0
        self.nav_log: list[dict] = [{"t": obs["t"], "event": "walk_start"}]
        self.hand = {s: np.array(RELAXED_HAND, float) for s in spec.SIDES}
        self.t0 = obs["t"]

    def _hold_arms(self) -> np.ndarray:
        return compose_action(self.ik, self.ik.q_home, self.hand)

    def act(self, obs: dict) -> np.ndarray:
        env, walker = self.env, self.env.walker
        pos, yaw = walker.base_pose()
        t = obs["t"]
        if self.stage == "navigate":
            if t % SCAN_EVERY == 0:
                pts = nav.scan(env.model, env.data, env.pelvis, pos[:2], yaw)
                new = self.grid.mark(pts)
                if self.path is None or (new and nav.path_blocked(self.grid, self.path, self.k if self.path else 0)):
                    self.path = nav.astar(self.grid, pos[:2], self.pre_dock)
                    self.replans += 1
                    self.nav_log.append({"t": t, "event": "plan", "n": self.replans,
                                         "length_m": round(sum(np.linalg.norm(self.path[i + 1] - self.path[i])
                                                               for i in range(len(self.path) - 1)), 2)
                                         if self.path else None})
            if self.path is None:
                walker.set_velocity((0.0, 0.0, 0.3))          # look around until a way shows up
                return self._hold_arms()
            cmd, self.k = nav.follow(self.path, pos, yaw)
            walker.set_velocity(cmd)
            if np.linalg.norm(pos[:2] - self.pre_dock) < ARRIVE_TOL:
                self.stage = "approach"
                self.app = (t, pos[:2].copy(), yaw)
                self.nav_log.append({"t": t, "event": "approach"})
            return self._hold_arms()
        if self.stage == "approach":
            # the hold target moves to the stop point at APPROACH_V: holding the stop point itself from half a
            # metre away overshot it and swept the arms into the objects at the counter's edge
            t0, p0, y0 = self.app
            goal = np.array(HOME_DOCK[:2])
            u = min(1.0, (t - t0) * spec.CONTROL_DT * APPROACH_V / max(np.linalg.norm(goal - p0), 1e-6))
            g = p0 + (goal - p0) * u
            walker.hold(g[0], g[1], y0 + wrap(HOME_DOCK[2] - y0) * u)
            if u >= 1.0:
                self.stage = "settle"
            return self._hold_arms()
        if self.stage == "settle":
            walker.hold(*HOME_DOCK)          # keeps stepping in place (see locomotion.walker on the balance stand)
            near = np.linalg.norm(pos[:2] - np.array(HOME_DOCK[:2])) < DOCK_TOL[0]                 and abs(wrap(yaw - HOME_DOCK[2])) < DOCK_TOL[1]
            self.settle = self.settle + 1 if near else 0      # 1.5 s in a row within the tolerance
            if self.settle >= SETTLE_STEPS:
                err = float(np.linalg.norm(pos[:2] - np.array(HOME_DOCK[:2])))
                env.nav.update(arrived=True, dock_error_m=round(err, 3),
                               dock_yaw_error_deg=round(float(np.rad2deg(wrap(yaw - HOME_DOCK[2]))), 1),
                               walk_time_s=round((t - self.t0) * spec.CONTROL_DT, 1))
                self.nav_log.append({"t": t, "event": "docked", "error_cm": round(100 * err, 1)})
                env.safety.check_workspace = True
                if LOCK_BASE:
                    walker.lock()                      # see HomeEnv: the base-lock simplification
                    self.nav_log.append({"t": t, "event": "locked"})
                self.base_ref = self._base()           # the tidy planner plans for the pelvis standing here
                self.ik.set_base(*self.base_ref)
                self.q_comp = None
                self.stage = "tidy"
                if self.conveyor:
                    env.start_belt()                   # the belt runs from now on
                    obs = env.observe()
                self.tidy.reset(obs)
            return self._hold_arms()
        a = self.tidy.act(obs)
        return a if LOCK_BASE else self._compensate(a)

    def _base(self, ahead: float = 0.0):
        """Pelvis pose now, or predicted `ahead` s from now from its velocity (the arm reaches a commanded
        configuration a control step plus the actuator lag later)."""
        import mujoco
        d, w = self.env.data, self.env.walker
        pos = d.qpos[w.base_q:w.base_q + 3].copy()
        quat = d.qpos[w.base_q + 3:w.base_q + 7].copy()
        if ahead > 0:
            v = d.qvel[w.base_v:w.base_v + 6]
            pos = pos + v[:3] * ahead
            mujoco.mju_quatIntegrate(quat, v[3:6].copy(), ahead)   # (free joint: angular velocity in the body frame)
        return pos, quat

    def _compensate(self, action: np.ndarray) -> np.ndarray:
        """Stepping in place sways the pelvis by up to +-7 cm: keep each palm where the plan (made for the
        pelvis at the stop point) puts it in the world, by re-solving the arms' IK for where the pelvis is now."""
        ik = self.ik
        q_plan = (self.q_comp if self.q_comp is not None else ik.q_home).copy()
        q_plan[ik.arm_qadr["left"]] = action[:7]
        q_plan[ik.arm_qadr["right"]] = action[7:14]
        targets = {s: ik.palm_pose(q_plan, s) for s in spec.SIDES}        # (IK base = the stop point)
        self.last_targets = targets
        ik.set_base(*self._base(LOOKAHEAD_S))
        q0 = q_plan if self.q_comp is None else self.q_comp
        q, _ = ik.solve(q0, targets, iters=COMP_ITERS, posture_target=q_plan)
        ik.set_base(*self.base_ref)
        self.q_comp = q
        out = action.copy()
        out[:14] = ik.arm_targets(q)
        return out
