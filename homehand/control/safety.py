"""Safety layer between every controller (scripted expert, ACT, Diffusion Policy) and the robot.

Command filtering (every control step, before the command reaches the actuators)
  * joint position limits with a margin (3 % of the range away from each hard stop)
  * joint velocity limits: arm 2.0 rad/s, fingers 3.0 rad/s (rate limiting of the position targets)
Monitoring (after every physics step)
  * self-collision: arm/hand touching the torso or the other arm, palms closer than 8 cm
  * excessive contact force between the robot and the environment (> 80 N)
  * workspace: palm below the counter surface or outside the reachable box
Protective stop: a hard violation that persists for 5 control steps (0.2 s) freezes the robot at its current
configuration for the rest of the episode; the event is logged and the episode is labelled `safety_stop`.

Actuator torque limits come from the robot model itself (G1 arm 25 N m, wrist 5 N m, Inspire fingers 2 N m).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

from homehand.model import spec

ARM_MAX_VEL = 2.0       # rad/s
ARM_MAX_ACC = 10.0      # rad/s^2, used by the IK trajectory generator (smooth starts / stops)
FINGER_MAX_VEL = 3.0    # rad/s
LIMIT_MARGIN = 0.03     # fraction of the joint range kept free at each end
MAX_CONTACT_FORCE = 80.0  # N, robot vs. environment
MIN_PALM_GAP = 0.08     # m between the two palms
WORKSPACE = {"x": (-0.05, 0.75), "y": (-0.6, 0.6), "z": (0.80 + 0.01, 1.35)}  # palm site bounds
STOP_AFTER = 5          # control steps of continuous hard violation before a protective stop


@dataclass
class SafetyStats:
    clipped_position: int = 0
    clipped_velocity: int = 0
    self_collision: int = 0
    high_force: int = 0
    workspace: int = 0
    max_contact_force: float = 0.0
    protective_stop: bool = False
    stop_reason: str = ""
    events: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {k: (round(v, 2) if isinstance(v, float) else v) for k, v in self.__dict__.items() if k != "events"}


class SafetyLayer:
    def __init__(self, model: mujoco.MjModel, act_ids: np.ndarray, palm_sites: dict, dt: float = spec.CONTROL_DT):
        self.model = model
        lo = model.actuator_ctrlrange[act_ids, 0].copy()
        hi = model.actuator_ctrlrange[act_ids, 1].copy()
        margin = LIMIT_MARGIN * (hi - lo)
        self.lo, self.hi = lo + margin, hi - margin
        # fingers may close fully onto an object: keep their upper limit, only guard the opening side
        self.hi[14:] = hi[14:]
        self.lo[14:] = lo[14:]
        self.max_step = np.r_[np.full(14, ARM_MAX_VEL * dt), np.full(12, FINGER_MAX_VEL * dt)]
        self.palm_sites = palm_sites
        m = model
        robot_bodies = set()
        for side in spec.SIDES:
            root = m.body(f"{side}_shoulder_pitch_link").id
            robot_bodies |= {b for b in range(m.nbody) if _is_descendant(m, b, root)}
        self.arm_side = {}
        for side in spec.SIDES:
            root = m.body(f"{side}_shoulder_pitch_link").id
            for b in range(m.nbody):
                if _is_descendant(m, b, root):
                    self.arm_side[b] = side
        self.torso = {m.body("torso_link").id, m.body("pelvis").id}
        self.robot_bodies = robot_bodies
        self.reset()

    def reset(self) -> None:
        self.stats = SafetyStats()
        self.violation_steps = 0
        self.frozen: np.ndarray | None = None

    # ------------------------------------------------------------------ command filter
    def filter(self, action: np.ndarray, previous: np.ndarray) -> np.ndarray:
        if self.frozen is not None:
            return self.frozen
        a = np.clip(action, self.lo, self.hi)
        if np.any(a != action):
            self.stats.clipped_position += 1
        step = np.clip(a - previous, -self.max_step, self.max_step)
        if np.any(np.abs(step - (a - previous)) > 1e-9):
            self.stats.clipped_velocity += 1
        return previous + step

    # ------------------------------------------------------------------ monitor
    def monitor(self, data: mujoco.MjData, t: int, applied: np.ndarray) -> str | None:
        m = self.model
        hard = None
        fr = np.zeros(6)
        env_force = 0.0
        for i in range(data.ncon):
            c = data.contact[i]
            b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
            r1, r2 = b1 in self.robot_bodies, b2 in self.robot_bodies
            if r1 and r2 and self.arm_side.get(b1) != self.arm_side.get(b2):
                self.stats.self_collision += 1
                hard = "arm-arm contact"
            elif (r1 and b2 in self.torso) or (r2 and b1 in self.torso):
                self.stats.self_collision += 1
                hard = "arm-torso contact"
            elif r1 != r2:
                other = b2 if r1 else b1
                if m.body_dofnum[other] == 0:  # static environment (counter, bin), not a held object
                    mujoco.mj_contactForce(m, data, i, fr)
                    env_force += abs(fr[0])
        self.stats.max_contact_force = max(self.stats.max_contact_force, env_force)
        if env_force > MAX_CONTACT_FORCE:
            self.stats.high_force += 1
            hard = hard or f"contact force {env_force:.0f} N"
        palms = {s: data.site_xpos[sid] for s, sid in self.palm_sites.items()}
        if np.linalg.norm(palms["left"] - palms["right"]) < MIN_PALM_GAP:
            self.stats.self_collision += 1
            hard = hard or "palms too close"
        for s, p in palms.items():
            for k, ax in enumerate("xyz"):
                lo, hi = WORKSPACE[ax]
                if not lo <= p[k] <= hi:
                    self.stats.workspace += 1
                    hard = hard or f"{s} palm outside workspace ({ax}={p[k]:.2f})"
        self.violation_steps = self.violation_steps + 1 if hard else 0
        if hard and self.violation_steps == 1:
            self.stats.events.append({"t": t, "event": "safety_warning", "reason": hard})
        if self.violation_steps >= STOP_AFTER and self.frozen is None:
            self.frozen = applied.copy()
            self.stats.protective_stop = True
            self.stats.stop_reason = hard
            self.stats.events.append({"t": t, "event": "protective_stop", "reason": hard})
            return hard
        return None


def _is_descendant(m: mujoco.MjModel, b: int, root: int) -> bool:
    while b > 0:
        if b == root:
            return True
        b = m.body_parentid[b]
    return False
