"""HomeHand kitchen environment: fixed-base Unitree G1 with two Inspire hands at a kitchen counter.

Tasks
  tidy_table   3-4 household objects are scattered on the counter; the robot must put all of them in the bin
               (perception -> hand assignment -> pick & place, see `homehand.control.planner`)
  pick_place   a single object, used to train / evaluate the pick-and-place skill in isolation

Control: 25 Hz, action = 26 absolute joint-position targets (see `spec.ACTUATORS`).
Every episode ends with an outcome label used by the failure taxonomy (see `OUTCOMES`).
"""

from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass, field

import mujoco
import numpy as np

from homehand import paths
from homehand.control.safety import SafetyLayer
from homehand.env.randomize import Randomization, resolve
from homehand.model import spec
from homehand.model.objects import CLASS_NAMES, LYING, OBJECTS, body_pose_for, lying_dims, rest_of

TABLE_Z = 0.80
BIN_XY = np.array([0.56, 0.0])   # as close as the objects allow: the G1 palm reaches x ~ 0.46 m at drop height
BIN_HALF_INNER = np.array([0.135, 0.15])
BIN_WALL_TOP = TABLE_Z + 0.10
BIN_MAX_Z = TABLE_Z + 0.17   # an object counts as stored when its centre of mass rests inside the bin footprint

# Two objects per side of the counter: an "inner" one near the middle and an "outer" one further out and
# slightly further from the robot. Centres are >= 11 cm apart sideways, so every object can be approached
# along +x, and all of them are well in front of the chest (x >= 0.34 m).
SPAWN_ZONES = {
    "inner": {"x": (0.30, 0.35), "absy": (0.12, 0.14)},   # in front of the bin: >= 1 cm from its wall
    "outer": {"x": (0.34, 0.40), "absy": (0.25, 0.29)},
}
BOX_YAW_JITTER = np.deg2rad(35)
PARK_XY = (-1.2, 0.0)   # unused objects wait on the floor behind the robot, out of view

# Fraction of objects that start tipped over (lying on their side): those need a top grasp.
P_LYING = 0.5            # of the objects in the outer slots
# 18-19 cm long objects lying: outer slot only, a little further out, long axis across the counter (along y), so
# the palm-down hand can wrap them with the fingers pointing forward (the only direction the arm reaches well)
LONG_LYING = {"mustard", "sugar_box"}
LONG_LYING_ABSY = (0.30, 0.32)
LYING_X = (0.33, 0.37)        # the palm-down hand reaches lying objects up to x ~ 0.38 m
LYING_YAW_JITTER = np.deg2rad(20)
# Short lying objects: heading of the long axis such that the top grasp's fingers point within +-40 deg of
# straight ahead (the palm-down hand reaches nothing else): the can's axis across the counter, the meat can's along.
SHORT_LYING_YAW = {"soup_can": (np.pi / 2, np.deg2rad(40)), "meat_can": (0.0, np.deg2rad(40))}

TASKS = {
    "tidy_table": {"n_objects": (4, 4), "steps_per_object": 400},
    "pick_place": {"n_objects": (1, 1), "steps_per_object": 360},
}
TASK_NAMES = list(TASKS)

OUTCOMES = [
    "success",
    "not_attempted",  # object never touched (missed by perception or planner gave up)
    "grasp_fail",     # touched but never lifted
    "dropped",        # lifted, then lost away from the bin
    "misplaced",      # released near the bin but did not end up inside
    "knocked_over",   # tipped over / pushed off the counter before being grasped
    "timeout",        # still in hand when time ran out
    "safety_stop",    # the safety layer froze the robot (self-collision, excessive force, workspace)
    "sim_error",      # the physics diverged (reported, never hidden)
]


@dataclass
class ObjectTrack:
    name: str
    max_lift: float = 0.0
    touched: bool = False
    grasped_once: bool = False
    in_hand: bool = False
    last_release_xy: np.ndarray | None = None
    tipped: bool = False
    up0: np.ndarray | None = None

    def outcome(self, in_bin: bool) -> str:
        if in_bin and not self.in_hand:
            return "success"
        if self.tipped:
            return "knocked_over"
        if not self.touched:
            return "not_attempted"
        if not self.grasped_once:
            return "grasp_fail"
        if self.in_hand:
            return "timeout"
        near = self.last_release_xy is not None and np.linalg.norm(self.last_release_xy - BIN_XY) < 0.2
        return "misplaced" if near else "dropped"


@dataclass
class EpisodeInfo:
    task: str
    seed: int
    objects: list[str]
    randomization: dict
    outcome: str = "running"
    steps: int = 0
    success: bool = False
    object_outcomes: dict = field(default_factory=dict)
    details: dict = field(default_factory=dict)

    @property
    def n_cleared(self) -> int:
        return sum(v == "success" for v in self.object_outcomes.values())


def quat_yaw(yaw: float) -> np.ndarray:
    return np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])


def yaw_of(q: np.ndarray) -> float:
    w, x, y, z = q
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


class KitchenEnv:
    def __init__(self, xml_path=None, record: bool = False):
        self.model = mujoco.MjModel.from_xml_path(str(xml_path or paths.SCENE_XML))
        self.model.opt.noslip_iterations = 3  # markedly reduces in-hand slip of the light Inspire fingers
        self.data = mujoco.MjData(self.model)
        self.record = record
        self.obj_info = json.loads(paths.OBJECTS_JSON.read_text())
        m = self.model
        self.act_ids = np.array([m.actuator(n).id for n in spec.ACTUATORS])
        self.arm_qadr = np.array([m.jnt_qposadr[m.joint(j).id] for j in spec.ARM_JOINTS])
        hand_joints = [m.actuator(n).trnid[0] for n in spec.hand_actuators("left") + spec.hand_actuators("right")]
        self.hand_qadr = np.array([m.jnt_qposadr[j] for j in hand_joints])
        self.obj_qadr = {n: m.jnt_qposadr[m.joint(f"{n}_free").id] for n in OBJECTS}
        self.obj_body = {n: m.body(n).id for n in OBJECTS}
        self.obj_geoms = {n: [g for g in range(m.ngeom) if m.geom_bodyid[g] == self.obj_body[n]] for n in OBJECTS}
        self.palm_site = {s: m.site(spec.palm_site(s)).id for s in spec.SIDES}
        self.hand_bodies = {
            s: {b for b in range(m.nbody) if m.body(b).name.startswith(spec.PREFIX[s])} for s in spec.SIDES
        }
        self.body_side = {b: s for s, bs in self.hand_bodies.items() for b in bs}
        self.finger_geoms = np.array([g for g in range(m.ngeom) if m.geom_priority[g] == 1])
        self.nominal = {
            "friction": m.geom_friction.copy(),
            "mass": m.body_mass.copy(),
            "inertia": m.body_inertia.copy(),
            "gainprm": m.actuator_gainprm.copy(),
            "biasprm": m.actuator_biasprm.copy(),
        }
        self.safety = SafetyLayer(m, self.act_ids, self.palm_site)
        self.home_ctrl: np.ndarray | None = None
        self._renderers: dict = {}
        self.rng = np.random.default_rng(0)
        self.task = "tidy_table"
        self.tracks: dict[str, ObjectTrack] = {}

    # ------------------------------------------------------------------ reset
    def set_home(self, arm_targets: np.ndarray) -> None:
        """Arm joint targets of the ready pose (hands open)."""
        from homehand.control.grasps import RELAXED_HAND
        ctrl = np.zeros(spec.ACTION_DIM)
        ctrl[:14] = arm_targets
        ctrl[14:] = np.tile(RELAXED_HAND, 2)
        self.home_ctrl = ctrl

    def sample_layout(self, n: int, names: list[str] | None = None,
                      p_lying: float | None = None) -> list[tuple[str, float, float, float, str]]:
        """Random (name, x, y, yaw, rest) per object: up to two objects per side (inner + outer slot). A share of
        the objects lies on its side (`rest` = "lying", `yaw` = heading of its long axis)."""
        p_lying = P_LYING if p_lying is None else p_lying
        names = list(names) if names else list(self.rng.choice(CLASS_NAMES, size=n, replace=False))
        assert len(names) <= 4, "at most two objects per side"
        slots = [(side, kind) for side in (-1.0, 1.0) for kind in ("inner", "outer")]
        if len(names) < 4:
            slots = [slots[i] for i in self.rng.choice(4, size=len(names), replace=False)]
        self.rng.shuffle(names)
        # Tipped-over objects lie in the outer slots only: there the palm-down hand's fingertips (they point
        # forward and down) stay clear of the bin, which is in front of the inner slots.
        rests = ["lying" if kind == "outer" and self.rng.random() < p_lying else "upright" for _, kind in slots]
        out = []
        for nm, rest, (side, kind) in zip(names, rests, slots):
            z = SPAWN_ZONES[kind]
            x, y = float(self.rng.uniform(*(LYING_X if rest == "lying" else z["x"]))), \
                float(side * self.rng.uniform(*z["absy"]))
            if rest == "lying" and nm in LONG_LYING:
                y = float(side * self.rng.uniform(*LONG_LYING_ABSY))
                yaw = float(np.pi / 2 + self.rng.uniform(-LYING_YAW_JITTER, LYING_YAW_JITTER))
            elif rest == "lying":
                c, w = SHORT_LYING_YAW[nm]
                yaw = float(c + self.rng.uniform(-w, w))
            else:
                yaw = self._sample_yaw(nm)
            out.append((nm, x, y, yaw, rest))
        return out

    def _sample_yaw(self, name: str) -> float:
        """Boxes stand roughly square to the counter edge (+-35 deg), like people leave them; round objects
        take any yaw."""
        if OBJECTS[name].box:
            return float(self.rng.integers(4) * np.pi / 2 + self.rng.uniform(-BOX_YAW_JITTER, BOX_YAW_JITTER))
        return float(self.rng.uniform(-np.pi, np.pi))

    def reset(self, task: str = "tidy_table", seed: int = 0, objects: list[str] | None = None,
              randomization: str | dict | Randomization | None = None, layout=None) -> dict:
        if self.home_ctrl is None:
            self.set_home(default_home())
        self.task = task
        self.cfg = TASKS[task]
        self.rng = np.random.default_rng(seed)
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)   # also clears the warning counters

        # --- physics randomisation
        self.rand_cfg = resolve(randomization)
        self.rand = self.rand_cfg.sample(self.rng)
        m.geom_friction[:] = self.nominal["friction"]
        obj_geoms = [g for gs in self.obj_geoms.values() for g in gs]
        for g in obj_geoms + list(self.finger_geoms):
            m.geom_friction[g, 0] = self.nominal["friction"][g, 0] * self.rand["friction"]
        m.body_mass[:] = self.nominal["mass"]
        m.body_inertia[:] = self.nominal["inertia"]
        for b in self.obj_body.values():
            m.body_mass[b] = self.nominal["mass"][b] * self.rand["mass"]
            m.body_inertia[b] = self.nominal["inertia"][b] * self.rand["mass"]
        m.actuator_gainprm[:] = self.nominal["gainprm"]
        m.actuator_biasprm[:] = self.nominal["biasprm"]
        for a in self.act_ids[14:]:
            m.actuator_gainprm[a, 0] *= self.rand["finger_gain"]
            m.actuator_biasprm[a, 1] *= self.rand["finger_gain"]
        # Masses changed: recompute the derived constants (subtree masses, the inverse weights the contact
        # solver scales constraints with). Without this a +20 % object mass already made grasped objects slip
        # and the simulation diverge, i.e. the "sim gap" measured a bug instead of the physics.
        mujoco.mj_setConst(m, d)
        self.action_queue: deque = deque()
        self.safety.reset()

        # --- robot at the ready pose
        d.qpos[self.arm_qadr] = self.home_ctrl[:14]
        d.qpos[self.hand_qadr] = self.home_ctrl[14:]
        d.ctrl[self.act_ids] = self.home_ctrl
        self.last_action = self.home_ctrl.copy()

        # --- objects
        if layout is None:
            lo, hi = self.cfg["n_objects"]
            n = len(objects) if objects else int(self.rng.integers(lo, hi + 1))
            layout = self.sample_layout(n, objects)
        self.layout = [tuple(x) for x in layout]
        used = {nm for nm, *_ in self.layout}
        for k, name in enumerate(n for n in OBJECTS if n not in used):
            a = self.obj_qadr[name]
            d.qpos[a:a + 3] = [PARK_XY[0], PARK_XY[1] + 0.2 * k, 0.001]
            d.qpos[a + 3:a + 7] = [1, 0, 0, 0]
        self.layout = [tuple(e) + ("upright",) if len(e) == 4 else tuple(e) for e in self.layout]
        for name, x, y, yaw, rest in self.layout:
            a = self.obj_qadr[name]
            pos, quat = body_pose_for(name, rest, x, y, yaw, float(self.obj_info[name]["height"]), TABLE_Z)
            d.qpos[a:a + 3] = pos
            d.qpos[a + 3:a + 7] = quat
        mujoco.mj_forward(m, d)
        for _ in range(60):  # let objects settle
            mujoco.mj_step(m, d)

        self.tracks = {nm: ObjectTrack(nm) for nm, *_ in self.layout}
        self.rest0 = {nm: rest for nm, *_, rest in self.layout}
        self.info = EpisodeInfo(task=task, seed=seed, objects=list(self.tracks),
                                randomization=self.rand_cfg.to_dict())
        self.horizon = self.cfg["steps_per_object"] * len(self.layout) + 40
        self.success_hold = 0
        self.t = 0
        self.traj_qpos: list[np.ndarray] = []
        self.traj_act: list[np.ndarray] = []
        self.events: list[dict] = []
        self._log_qpos()
        return self.observe()

    # ------------------------------------------------------------------ step
    def step(self, action: np.ndarray) -> tuple[dict, bool]:
        action = np.asarray(action, dtype=float)
        self.action_queue.append(action)
        target = self.action_queue.popleft() if len(self.action_queue) > self.rand["latency"] else self.last_action
        applied = self.safety.filter(target, self.last_action)   # joint limits + rate limits
        self.last_action = applied
        self.data.ctrl[self.act_ids] = applied
        for _ in range(spec.N_SUBSTEPS):
            mujoco.mj_step(self.model, self.data)
        self.t += 1
        if self._diverged():
            self.events.append({"t": self.t, "event": "sim_error"})
            self._finish()
            self.info.outcome = "sim_error"
            self.info.success = False
            self._log_qpos()
            return self.observe(), True
        if self.safety.monitor(self.data, self.t, applied):
            self.events.append({"t": self.t, "event": "protective_stop", "reason": self.safety.stats.stop_reason})
        if self.record:
            self.traj_act.append(action.astype(np.float32))
        self._update_tracks()
        done = self._check_done()
        self._log_qpos()
        return self.observe(), done

    def _diverged(self) -> bool:
        w = self.data.warning
        n = w[mujoco.mjtWarning.mjWARN_BADQACC].number + w[mujoco.mjtWarning.mjWARN_BADQVEL].number \
            + w[mujoco.mjtWarning.mjWARN_BADQPOS].number
        return n > 0

    # ------------------------------------------------------------------ observation
    def object_pose(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        a = self.obj_qadr[name]
        return self.data.qpos[a:a + 3].copy(), self.data.qpos[a + 3:a + 7].copy()

    def observe(self) -> dict:
        """Proprioception + ground-truth object poses (with optional perception noise)."""
        d = self.data
        noise = self.rand["obs_noise"]
        objects = {}
        for name in self.tracks:
            pos, quat = self.object_pose(name)
            if noise > 0:
                pos = pos + self.rng.normal(0, noise, 3)
            rest, ryaw = rest_of(name, quat)
            centre = self.data.xpos[self.obj_body[name]] + self.data.xmat[self.obj_body[name]].reshape(3, 3) @ \
                np.array([0.0, 0.0, self.obj_info[name]["height"] / 2])
            if noise > 0:
                centre = centre + (pos - self.object_pose(name)[0])          # same noise as the body origin
            objects[name] = {"pos": pos, "yaw": yaw_of(quat) if rest == "upright" else ryaw, "quat": quat,
                             "rest": rest, "centre": centre,
                             "on_table": bool(pos[2] > TABLE_Z - 0.05 and not self.in_bin(name))}
        return {
            "task": self.task,
            "t": self.t,
            "arm_q": d.qpos[self.arm_qadr].copy(),
            "hand_q": d.qpos[self.hand_qadr].copy(),
            "palm": {s: d.site_xpos[self.palm_site[s]].copy() for s in spec.SIDES},
            "objects": objects,
            "bin": np.array([*BIN_XY, TABLE_Z]),
        }

    # ------------------------------------------------------------------ bookkeeping
    def contacts_with_hands(self) -> dict[str, set[str]]:
        """object name -> set of hand sides touching it."""
        d, m = self.data, self.model
        body_obj = {b: n for n, b in self.obj_body.items()}
        out: dict[str, set[str]] = {}
        for i in range(d.ncon):
            c = d.contact[i]
            b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
            for bo, bh in ((b1, b2), (b2, b1)):
                if bo in body_obj and bh in self.body_side:
                    out.setdefault(body_obj[bo], set()).add(self.body_side[bh])
        return out

    def in_bin(self, name: str) -> bool:
        """Centre of mass over the bin interior and resting low (objects may lean on a wall)."""
        com = self.data.xipos[self.obj_body[name]]
        return bool(np.all(np.abs(com[:2] - BIN_XY) < BIN_HALF_INNER) and com[2] < BIN_MAX_Z)

    def _update_tracks(self) -> None:
        touching_all = self.contacts_with_hands()
        for name, tr in self.tracks.items():
            pos, quat = self.object_pose(name)
            lift = pos[2] - TABLE_Z
            tr.max_lift = max(tr.max_lift, lift)
            touching = name in touching_all
            tr.touched |= touching
            if touching and lift > 0.03 and not tr.grasped_once:
                tr.grasped_once = True
                self.events.append({"t": self.t, "event": "lifted", "object": name})
            if tr.in_hand and not touching:
                tr.last_release_xy = pos[:2].copy()
                self.events.append({"t": self.t, "event": "released", "object": name})
            tr.in_hand = touching
            # tipped = its rest pose changed by > 60 deg (an object may start lying on its side)
            R = self.data.xmat[self.obj_body[name]].reshape(3, 3)
            if tr.up0 is None:
                tr.up0 = R.T @ np.array([0.0, 0.0, 1.0])          # world up, in the object frame, at the start
            same_rest = float((R @ tr.up0)[2]) > 0.5
            fell_off = pos[2] < TABLE_Z - 0.05 and not self.in_bin(name)
            if not tr.tipped and (fell_off or (not tr.grasped_once and not touching and not same_rest)):
                tr.tipped = True
                self.events.append({"t": self.t, "event": "knocked_over", "object": name})

    def _check_done(self) -> bool:
        if all(self.in_bin(n) and not tr.in_hand for n, tr in self.tracks.items()):
            self.success_hold += 1
            if self.success_hold >= 8:  # stable for 0.3 s
                self._finish()
                return True
        else:
            self.success_hold = 0
        if self.t >= self.horizon:
            self._finish()
            return True
        return False

    def _finish(self) -> None:
        outs = {n: tr.outcome(self.in_bin(n)) for n, tr in self.tracks.items()}
        failed = [o for o in outs.values() if o != "success"]
        self.info.object_outcomes = outs
        self.info.outcome = "success" if not failed else min(failed, key=OUTCOMES.index)
        if failed and self.safety.stats.protective_stop:
            self.info.outcome = "safety_stop"
        self.info.success = not failed
        self.info.steps = self.t
        self.info.details = {
            "max_lift": {n: round(float(t.max_lift), 4) for n, t in self.tracks.items()},
            "sampled": self.rand,
            "n_cleared": self.info.n_cleared,
            "n_objects": len(self.tracks),
            "safety": self.safety.stats.to_dict(),
        }

    def force_finish(self) -> None:
        """End an episode early (controller finished or gave up)."""
        if self.info.outcome == "running":
            self._finish()

    def _log_qpos(self) -> None:
        if self.record:
            self.traj_qpos.append(self.data.qpos.astype(np.float32).copy())

    # ------------------------------------------------------------------ telemetry & rendering
    def telemetry(self) -> dict:
        d, m = self.data, self.model
        grip = {s: 0.0 for s in spec.SIDES}
        fr = np.zeros(6)
        for i in range(d.ncon):
            c = d.contact[i]
            for b in (m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]):
                if b in self.body_side:
                    mujoco.mj_contactForce(m, d, i, fr)
                    grip[self.body_side[b]] += abs(fr[0])
                    break
        objs = {}
        for name, tr in self.tracks.items():
            pos, _ = self.object_pose(name)
            objs[name] = {
                "pos": [round(float(v), 4) for v in pos],
                "in_bin": self.in_bin(name),
                "in_hand": tr.in_hand,
                "lift": round(float(pos[2] - TABLE_Z), 4),
            }
        return {
            "t": round(self.t * spec.CONTROL_DT, 2),
            "step": self.t,
            "grip_force": {s: round(v, 2) for s, v in grip.items()},
            "objects": objs,
            "n_cleared": sum(o["in_bin"] for o in objs.values()),
            "outcome": self.info.outcome,
            "safety": self.safety.stats.to_dict(),
        }

    def renderer(self, width: int, height: int) -> mujoco.Renderer:
        key = (width, height)
        if key not in self._renderers:
            self._renderers[key] = mujoco.Renderer(self.model, height, width)
        return self._renderers[key]

    def render(self, camera: str = "front", width: int = 640, height: int = 480) -> np.ndarray:
        r = self.renderer(width, height)
        r.update_scene(self.data, camera=camera)
        return r.render()

    def close(self) -> None:
        for r in self._renderers.values():
            r.close()
        self._renderers.clear()

    def episode_arrays(self) -> dict:
        return {
            "qpos": np.asarray(self.traj_qpos, dtype=np.float32),
            "action": np.asarray(self.traj_act, dtype=np.float32),
        }


_HOME_CACHE: dict = {}


def default_home() -> np.ndarray:
    if "q" not in _HOME_CACHE:
        from homehand.control.ik import ArmIK
        ik = ArmIK()
        _HOME_CACHE["q"] = ik.arm_targets(ik.q_home)
    return _HOME_CACHE["q"]
