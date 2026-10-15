"""The flat-box grasp on the whole robot, in the kitchen scene: the YCB gelatin box (its mesh), the robot's full
collision meshes and contact settings (2 ms, 100 solver iterations, noslip), the other objects parked.

Same interface as `flat_grasp_env.FlatGraspSim` (reset(seed) -> obs, step(action | ctrl), success(), arm_q(),
box_seen()), so the expert and the trained policy run on it unchanged: the training model is this robot's right
arm and hand, so the 13 joint targets go to the robot as they are. The episode draws the same things as in
training (box pose, perception bias, grip friction, box mass); the arm first moves from the ready pose to the
pre-grasp the episode starts from.

    python -m homehand.rl.flat_grasp_deploy --build      # writes generated/scene_flat_grasp.xml (+ objects json)
"""

from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from homehand import paths
from homehand.model import spec
from homehand.rl import flat_grasp_env as E

DEPLOY_XML = paths.GENERATED_DIR / "scene_flat_grasp.xml"
DEPLOY_JSON = paths.GENERATED_DIR / "objects_flat_grasp.json"
FRAME_JSON = paths.GENERATED_DIR / "flat_grasp.json"
TRAIN_XML = paths.GENERATED_DIR / "flat_grasp_train.xml"
BOX = "gelatin_box"
SUBSTEPS = round(E.DT / spec.SIM_DT)        # 10 physics steps of the robot scene per 50 Hz step
REACH_S, SETTLE_S = 1.6, 0.4


def build_deploy_scene() -> None:
    """The tidy-table kitchen plus the YCB gelatin box (the tidy objects stay, parked); the bin moves to the left
    (its wall is in the right hand's way at this grasp)."""
    from homehand.model.build import _add_object
    from homehand.model.objects import OBJECTS
    tree = ET.parse(paths.SCENE_XML)
    root = tree.getroot()
    info = json.loads(paths.OBJECTS_JSON.read_text())
    info[BOX] = _add_object(root.find("asset"), root.find("worldbody"), OBJECTS[BOX], paths.GENERATED_DIR, k=9)
    for b in root.find("worldbody").findall("body"):
        if b.get("name") == "bin":
            b.set("pos", "0.50 0.36 0.8")
    ET.indent(tree)
    tree.write(DEPLOY_XML)
    DEPLOY_JSON.write_text(json.dumps(info, indent=2))
    print(f"wrote {DEPLOY_XML}")


class RobotTwin:
    def __init__(self, record: bool = False):
        from homehand.env.kitchen_env import KitchenEnv
        from homehand.model.objects import TIDY_OBJECTS, rest_quat

        class _Env(KitchenEnv):
            OBJECT_NAMES = list(TIDY_OBJECTS) + [BOX]
            OBJECTS_JSON = DEPLOY_JSON

        self.env = _Env(xml_path=DEPLOY_XML, record=record)
        m = self.model = self.env.model
        self.data = self.env.data
        # the training model's layout, for the limits / pre-grasp / observation; this robot's own indices below
        lay = E.Layout.of(mujoco.MjModel.from_xml_path(str(TRAIN_XML)), FRAME_JSON)
        self.act = np.r_[self.env.act_ids[7:14], self.env.act_ids[20:26]]          # right arm, right hand
        joints = m.actuator_trnid[self.act, 0]
        self.qadr = m.jnt_qposadr[joints]
        self.dadr = m.jnt_dofadr[joints]
        self.palm = m.site("R_palm").id
        self.tips = np.array([m.site(s).id for s in spec.fingertip_sites("right")])
        self.box_body = m.body(BOX).id
        self.box_geoms = set(self.env.obj_geoms[BOX])
        self.box_h = float(self.env.obj_info[BOX]["height"])
        part = {}
        for b in range(m.nbody):
            name = m.body(b).name
            for k, p in enumerate(E.PARTS):
                if name.startswith(f"R_{p}_"):
                    part[b] = k
            if name == "R_hand_base_link":
                part[b] = 5
        self.part_of_body = part
        Rr = np.zeros(9)
        mujoco.mju_quat2Mat(Rr, rest_quat(BOX, "lying", 0.0))
        self.R_rest = Rr.reshape(3, 3)
        # the observation function indexes the training model's arrays: a layout with this robot's indices
        n_site = max(self.palm, int(self.tips.max())) + 1
        from dataclasses import replace
        self.lay = replace(lay, act_qadr=self.qadr, act_dadr=self.dadr, palm_site=self.palm, tip_sites=self.tips)
        self.n_site = n_site
        self.frames: list[np.ndarray] = []

    # ------------------------------------------------------------------ the box as it is / as it is seen
    def _box_pose(self) -> tuple[np.ndarray, float]:
        d = self.data
        R = d.xmat[self.box_body].reshape(3, 3)
        centre = d.xpos[self.box_body] + R @ np.array([0.0, 0.0, self.box_h / 2])
        Rrel = R @ self.R_rest.T
        # the training model's box is this mesh in this body frame: its yaw is the body's, i.e. the layout's
        # yaw (the long axis' heading, `objects.rest_quat`) - pi/2
        return centre, float(np.arctan2(Rrel[1, 0], Rrel[0, 0]) - np.pi / 2)

    def box_true(self) -> np.ndarray:
        return self._box_pose()[0]

    def box_seen(self) -> tuple[np.ndarray, float]:
        pos, yaw = self._box_pose()
        b = self.ep["bias"]
        return pos + b[:3] + self.rng.normal(0, E.NOISE, 3), yaw + b[3]

    def arm_q(self) -> np.ndarray:
        return self.data.qpos[self.qadr[:E.N_ARM]].copy()

    def touch(self) -> np.ndarray:
        return E.touch_np(self.model, self.data, self.box_geoms, lambda g: self.part_of_body.get(self.model.geom_bodyid[g]))

    def observe(self) -> np.ndarray:
        d = self.data
        pos, yaw = self.box_seen()
        return E.observe(np, self.lay, d.qpos, d.qvel, d.site_xpos, d.site_xmat, self.ctrl, self.touch(), pos, yaw,
                         self.t / E.EPISODE_STEPS).astype(np.float32)

    # ------------------------------------------------------------------ episode
    def _apply(self, ctrl13: np.ndarray, renderer=None, cam=None) -> None:
        ctrl = self.env.home_ctrl.copy()
        ctrl[7:14] = ctrl13[:7]
        ctrl[20:26] = ctrl13[7:]
        self.data.ctrl[self.env.act_ids] = ctrl
        for _ in range(SUBSTEPS):
            mujoco.mj_step(self.model, self.data)
        if renderer is not None:
            renderer.update_scene(self.data, camera=cam)
            self.frames.append(renderer.render())

    def reset(self, seed: int, renderer=None, cam=None) -> np.ndarray:
        env, m, d, lay = self.env, self.model, self.data, self.lay
        self.rng = np.random.default_rng(seed)
        ep = self.ep = E.draw_episode(self.rng)
        x, y = lay.box_xy + ep["xy"]
        env.reset("tidy_table", seed=seed, layout=[(BOX, float(x), float(y), float(ep["yaw"] + np.pi / 2), "lying")])
        m.geom_friction[env.finger_geoms, 0] = ep["grip_friction"]
        m.body_mass[self.box_body] = env.nominal["mass"][self.box_body] * ep["mass"] / 0.10
        m.body_inertia[self.box_body] = env.nominal["inertia"][self.box_body] * ep["mass"] / 0.10
        state = d.qpos.copy(), d.qvel.copy(), d.ctrl.copy()
        mujoco.mj_setConst(m, d)                  # (it leaves the data at the model's default pose)
        d.qpos[:], d.qvel[:], d.ctrl[:] = state
        mujoco.mj_forward(m, d)
        # ready pose -> pre-grasp, in joint space
        q0 = np.r_[d.qpos[self.qadr[:E.N_ARM]], E.FINGER_OPEN]
        n = int(REACH_S / E.DT)
        for k in range(1, n + 1):
            f = 0.5 - 0.5 * np.cos(np.pi * k / n)
            self._apply(q0 + (lay.ctrl0 - q0) * f, renderer, cam)
        for _ in range(int(SETTLE_S / E.DT)):
            self._apply(lay.ctrl0, renderer, cam)
        self.ctrl = lay.ctrl0.copy()
        self.box_xy0 = self.box_true()[:2]
        self.t = 0
        return self.observe()

    def step(self, action: np.ndarray | None = None, ctrl: np.ndarray | None = None, renderer=None, cam=None):
        if ctrl is None:
            ctrl = self.ctrl + np.clip(action, -1, 1) * E.RATE * E.DT
        self.ctrl = np.clip(ctrl, self.lay.lo, self.lay.hi)
        self._apply(self.ctrl, renderer, cam)
        self.t += 1
        return self.observe(), 0.0, False, self.t >= E.EPISODE_STEPS

    def lift_cm(self) -> float:
        return 100 * (self.box_true()[2] - self.lay.z_rest)

    def success(self) -> bool:
        t = self.touch()
        return bool(t[0] and t[1:5].any() and self.lift_cm() / 100 > E.LIFT_OK)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    a = ap.parse_args()
    if a.build:
        build_deploy_scene()
