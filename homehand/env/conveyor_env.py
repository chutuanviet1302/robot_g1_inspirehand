"""Conveyor task: YCB objects ride a running belt across the counter; the robot picks them on the fly into the bin.

Scene `generated/scene_conveyor.xml` = the kitchen scene with a short belt along y in front of the chest (`model.build`,
BELT_X / BELT_HALF_W / BELT_HALF_L) and a small bin on a stand at each hip (`CONVEYOR_BINS`; each hand drops into
its own, neither reaches across the body). The belt is a long slab on a slide joint, driven at the belt
speed by a velocity actuator and shifted back by BELT_PERIOD whenever it has run that far. Objects ride it by real
friction, so a hand that grasps one works against exactly the belt's friction.

The objects are put on the belt at the upstream end (y = -Y_SPAWN, as from a chute; moving towards +y) one after the
other, `GAP` apart, upright
or lying, at any heading; one that reaches y = +Y_MISS on the belt is taken off and counted `missed`. With the belt
stopped (speed 0) they are laid out along the belt at once, inside the pick zones.

Outcomes per object: success, missed (rode past the robot), knocked_over, grasp_fail, dropped, misplaced,
timeout, not_spawned; plus the episode's safety_stop / sim_error.
"""

from __future__ import annotations

import mujoco
import numpy as np

from homehand import paths
from homehand.env.kitchen_env import TABLE_Z, KitchenEnv, ObjectTrack
from homehand.model import spec
from homehand.env.kitchen_env import BIN_MAX_Z
from homehand.model.build import (BELT_HALF_L, BELT_HALF_W, BELT_PERIOD, BELT_THICK, BELT_X, CONVEYOR_BIN_HALF,
                                  CONVEYOR_BINS)
from homehand.model.objects import CONVEYOR_NAMES, body_pose_for, lying_dims, rest_options

BELT_Z = TABLE_Z + 0.0005 + BELT_THICK  # belt surface
Y_SPAWN = BELT_HALF_L - 0.10            # objects appear at y = -Y_SPAWN (a 19 cm object lying along the belt fits)
Y_MISS = 0.45                           # ... and are taken off at y = +Y_MISS
GAP = (0.40, 0.55)                      # m between consecutive objects along the belt (13-18 s at 3 cm/s: one pick)
P_LYING = 0.5
SPAWN_X = BELT_X                         # footprint centre across the belt
X_JITTER = 0.01
STATIC_YS = (-0.28, -0.14, 0.14, 0.28)  # belt stopped: where the objects lie (inside the two hands' pick zones)
OUTCOMES = ["success", "missed", "not_attempted", "knocked_over", "grasp_fail", "dropped", "misplaced", "timeout",
            "not_spawned", "safety_stop", "sim_error"]


class ConveyorEnv(KitchenEnv):
    OBJECT_NAMES = list(CONVEYOR_NAMES)
    OBJECTS_JSON = paths.CONVEYOR_OBJECTS_JSON
    YOLO_WEIGHTS = paths.MODELS_DIR / "detector_conveyor" / "yolo_seg.pt"

    def __init__(self, record: bool = False, xml_path=None):
        super().__init__(xml_path=xml_path or paths.CONVEYOR_XML, record=record)
        self.belt_geom = self.model.geom("belt").id
        self.belt_act = self.model.actuator("belt_drive").id
        self.belt_qadr = self.model.jnt_qposadr[self.model.joint("belt_slide").id]
        self.belt_dof = self.model.jnt_dofadr[self.model.joint("belt_slide").id]
        self.safety.env_bodies.add(self.model.body("conveyor").id)   # pressing on the belt is contact force too
        self.obj_dof = {n: self.model.jnt_dofadr[self.model.joint(f"{n}_free").id] for n in self.OBJECT_NAMES}
        self.speed = 0.0

    # ------------------------------------------------------------------ reset / spawning
    def reset(self, task: str = "conveyor", seed: int = 0, objects=None, randomization=None, layout=None,
              speed: float = 0.03, n_objects: int = 4, gap=GAP) -> dict:
        """speed: belt speed (m/s, 0 = stopped); n_objects: how many (each class at most once); gap: (min, max) m
        between consecutive objects along the belt."""
        super().reset("pick_place", seed=seed, randomization=randomization, layout=[])   # all objects parked
        for n in self.OBJECT_NAMES:          # out of the cameras' view (behind them) until they come onto the belt
            self._park(n)
        self.task = task
        self.speed = float(speed)
        self.data.ctrl[self.belt_act] = self.speed
        self.data.qvel[self.belt_dof] = self.speed
        names = list(objects) if objects else list(self.rng.permutation(self.OBJECT_NAMES))[:n_objects]
        self.queue = []          # (spawn step, name, x, y, yaw, rest)
        if self.speed > 0:
            t = 0.0
            for n in names:
                self.queue.append((int(round(t / spec.CONTROL_DT)), n, *self._sample_pose(n, -Y_SPAWN)))
                t += self.rng.uniform(*gap) / self.speed
            travel = (Y_SPAWN + Y_MISS) / self.speed
            self.horizon = int((t + travel) / spec.CONTROL_DT) + 200
        else:
            ys = self.rng.permutation(STATIC_YS)[: len(names)]
            self.queue = [(0, n, *self._sample_pose(n, float(y))) for n, y in zip(names, ys)]
            self.horizon = 450 * len(names) + 40
        self.layout = []
        self.tracks = {}
        self.missed: dict[str, ObjectTrack] = {}   # rode past the robot (their tracks, kept for the verdict)
        self.info.objects = names
        self.info.task = task
        self._spawn_due()
        return self.observe()

    def _sample_pose(self, name: str, y: float) -> tuple[float, float, float, str]:
        opts = rest_options(name)
        rest = opts[0] if len(opts) == 1 else ("lying" if self.rng.random() < P_LYING else "upright")
        return (SPAWN_X + self.rng.uniform(-X_JITTER, X_JITTER), y, float(self.rng.uniform(-np.pi, np.pi)), rest)

    def _spawn_due(self) -> None:
        m, d = self.model, self.data
        while self.queue and self.queue[0][0] <= self.t:
            _, name, x, y, yaw, rest = self.queue.pop(0)
            a, v = self.obj_qadr[name], self.obj_dof[name]
            pos, quat = body_pose_for(name, rest, x, y, yaw, float(self.obj_info[name]["height"]), BELT_Z)
            d.qpos[a:a + 3] = pos + np.array([0.0, 0.0, 0.002])
            d.qpos[a + 3:a + 7] = quat
            d.qvel[v:v + 6] = 0.0
            d.qvel[v + 1] = self.speed
            mujoco.mj_forward(m, d)
            self.tracks[name] = ObjectTrack(name)
            self.layout.append((name, x, y, yaw, rest))
            self.events.append({"t": self.t, "event": "spawned", "object": name, "rest": rest})

    def step(self, action):
        # "disturbed" (failure attribution) means moved relative to where it rests: on the belt, where the belt
        # carries it, so the reference moves with the belt
        for tr in self.tracks.values():
            if tr.pos0 is not None and not tr.grasped_once:
                tr.pos0[1] += self.speed * spec.CONTROL_DT
        self.data.ctrl[self.belt_act] = self.speed
        obs, done = super().step(action)
        if self.data.qpos[self.belt_qadr] > BELT_PERIOD:          # the slab has run one period: shift it back
            self.data.qpos[self.belt_qadr] -= BELT_PERIOD
            mujoco.mj_forward(self.model, self.data)
        self._spawn_due()
        for n, tr in list(self.tracks.items()):
            p = self.object_pose(n)[0]
            if p[1] > Y_MISS and not tr.in_hand and abs(p[0] - BELT_X) < BELT_HALF_W + 0.05:
                self._park(n)                     # off the belt's end: out of the episode
                self.missed[n] = self.tracks.pop(n)
                self.events.append({"t": self.t, "event": "missed", "object": n})
        return (self.observe() if not done else obs), done

    def _park(self, name: str) -> None:
        a, v = self.obj_qadr[name], self.obj_dof[name]
        k = self.OBJECT_NAMES.index(name)
        self.data.qpos[a:a + 3] = [4.0, 0.3 * k, 0.001]
        self.data.qpos[a + 3:a + 7] = [1, 0, 0, 0]
        self.data.qvel[v:v + 6] = 0.0
        mujoco.mj_forward(self.model, self.data)

    # ------------------------------------------------------------------ bookkeeping
    def observe(self) -> dict:
        obs = super().observe()
        obs["belt"] = {"speed": self.speed, "x": BELT_X, "half_width": BELT_HALF_W, "y_miss": Y_MISS}
        return obs

    def in_bin(self, name: str) -> bool:
        """Centre of mass inside either bin, resting low."""
        com = self.data.xipos[self.obj_body[name]]
        return any(bool(np.all(np.abs(com[:2] - np.asarray(xy)) < CONVEYOR_BIN_HALF) and com[2] < BIN_MAX_Z)
                   for xy in CONVEYOR_BINS.values())

    def _resolved(self, name: str) -> bool:
        tr = self.tracks[name]
        return (self.in_bin(name) and not tr.in_hand) or tr.tipped

    def _check_done(self) -> bool:
        if not self.queue and all(self._resolved(n) for n in self.tracks):
            self.success_hold += 1
            if self.success_hold >= 8:
                self._finish()
                return True
        else:
            self.success_hold = 0
        if self.t >= self.horizon:
            self._finish()
            return True
        return False

    def _finish(self) -> None:
        outs = {}
        for n in self.info.objects:
            if n in self.missed:
                outs[n] = "missed" if not self.missed[n].touched else "knocked_over"
            elif n not in self.tracks:
                outs[n] = "not_spawned"
            else:
                outs[n] = self.tracks[n].outcome(self.in_bin(n))
        failed = [o for o in outs.values() if o != "success"]
        self.info.object_outcomes = outs
        self.info.outcome = "success" if not failed else min(failed, key=OUTCOMES.index)
        if failed and self.safety.stats.protective_stop:
            self.info.outcome = "safety_stop"
        self.info.success = not failed
        self.info.steps = self.t
        self.info.details = {
            "speed": self.speed, "sampled": self.rand,
            "n_cleared": sum(o == "success" for o in outs.values()), "n_objects": len(outs),
            "safety": self.safety.stats.to_dict(),
            "causes": {n: {"outcome": outs[n], "disturbed": t.disturbed, "release": t.release,
                           "tipped": t.tipped_info, "left_bin": t.left_bin, "last_toucher": t.last_toucher}
                       for n, t in {**self.tracks, **self.missed}.items()},
        }
