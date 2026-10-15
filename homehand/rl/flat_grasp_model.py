"""Model for the flat-box grasping RL task (MuJoCo / MJX): the G1's right arm and Inspire hand, as they are.

The expert takes the 3 cm flat gelatin box (9 x 10 x 3 cm) only about half of the time. This model keeps exactly
what that grasp involves: the robot (fixed base) with its right arm (7 joints, the robot's position servos, its
5 Nm wrist motors, gravity compensation) and right hand (6 actuators, the finger couplings), the counter and the box.
Everything else goes: the legs and the left arm, and every collision except the right hand's against the box and the
counter. The hand's collision meshes (the palm's alone has 28k vertices, far too slow for MJX) become small convex
hulls written into the file. An earlier version put the hand on an idealised 6-DoF "wrist" (slides and hinges):
on the whole robot its grasps went differently (the fingers, the wrist's compliance), so the policy now drives the
real arm joints and runs on the robot unchanged.

The arm starts in the expert's pre-grasp pose for the box at its nominal spot (`skills.top_grasp_plan`: palm 10 cm
over the grasp pose, fingers open, thumb turned in).

    python -m homehand.rl.flat_grasp_model    # writes generated/flat_grasp.xml (with the visual meshes, for viewing
                                              # / videos), flat_grasp_train.xml (MJX) and flat_grasp.json
"""

from __future__ import annotations

import json
import re

import mujoco
import numpy as np

from homehand import paths

OUT_XML = paths.GENERATED_DIR / "flat_grasp.xml"
OUT_JSON = paths.GENERATED_DIR / "flat_grasp.json"
OUT_TRAIN_XML = paths.GENERATED_DIR / "flat_grasp_train.xml"
BOX_HALF = (0.0445, 0.0505, 0.015)       # the YCB gelatin box, lying flat
BOX_MASS = 0.10
TABLE_Z = 0.80
NOMINAL_BOX_XY = (0.33, -0.18)           # where the box sits in the expert's reference grasp
HULL_VERTICES = 24
KEEP_ROOTS = ("right_shoulder_pitch_link",)
DROP_ROOTS = ("left_hip_pitch_link", "right_hip_pitch_link", "left_shoulder_pitch_link")
HAND_ENDS = ("R_thumb_distal", "R_index_intermediate", "R_middle_intermediate", "R_ring_intermediate",
             "R_pinky_intermediate", "R_hand_base_link")      # these also touch the counter


def pre_grasp() -> dict:
    """The expert's pre-grasp for the box at its nominal spot: palm pose and the arm's joints (IK)."""
    from homehand.control.ik import ArmIK
    from homehand.control.skills import Target, grasp_plan
    ik = ArmIK()
    t = Target("gelatin_box", np.array([*NOMINAL_BOX_XY, TABLE_Z]), 0.0, 2 * BOX_HALF[2], rest="lying")
    p = grasp_plan("right", t)
    q, err = ik.solve_best("right", p["pre"], p["quat"])
    assert err < 0.005, err
    return {"palm_pos": np.asarray(p["pre"], float).tolist(), "palm_quat": np.asarray(p["quat"], float).tolist(),
            "arm_q": q[ik.arm_qadr["right"]].tolist()}


def _support_points(v: np.ndarray, n: int) -> np.ndarray:
    """Up to n hull vertices: the farthest vertex along n directions spread over the sphere."""
    k = np.arange(n) + 0.5
    phi, z = np.pi * (1 + 5 ** 0.5) * k, 1 - 2 * k / n
    dirs = np.c_[np.sqrt(1 - z * z) * np.cos(phi), np.sqrt(1 - z * z) * np.sin(phi), z]
    return np.unique(v[np.argmax(v @ dirs.T, axis=0)], axis=0)


def gelatin_hull(n: int) -> np.ndarray:
    """The robot scene's collision hull of the gelatin box, as n support points in a frame at its centre (axes
    and z up, the box lying, its long side along y)."""
    from homehand.rl.flat_grasp_deploy import DEPLOY_XML
    m = mujoco.MjModel.from_xml_path(str(DEPLOY_XML))
    g = m.geom("gelatin_box").id
    k = m.geom_dataid[g]
    v = m.mesh_vert[m.mesh_vertadr[k]:m.mesh_vertadr[k] + m.mesh_vertnum[k]]
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, m.geom_quat[g])
    v = m.geom_pos[g] + v @ R.reshape(3, 3).T          # in the body frame (lying, yaw 0)
    v = v - (v.max(0) + v.min(0)) / 2
    return _support_points(v, n)


def _write(s: mujoco.MjSpec, path) -> None:
    # (attach / delete can leave an empty, unnamed default class that the XML parser rejects)
    path.write_text(re.sub(r"\n\s*<default/>", "", s.to_xml()), encoding="utf-8")


def build(timestep: float = 0.004, iterations: int = 10, ls_iterations: int = 20,
          hull_vertices: int = HULL_VERTICES, suffix: str = "", box_shape: str = "box") -> None:
    pre = pre_grasp()
    s = mujoco.MjSpec.from_file(str(paths.ROBOT_XML))
    s.modelname = "flat_grasp"
    # the robot's contact physics (elliptic cones, impratio 10, implicitfast) with a coarser step and fewer
    # solver iterations than its 2 ms / 100 / 50 (MJX: each costs on every step). Checked against the robot
    # scene (the expert on 60 episodes, scripts in RESULTS): 4 ms / 10 / 20 gives the same outcome in 52 of 60
    # (2 ms / 100 / 50: 53); fewer line-search iterations (8, 12) and grasped boxes slip out every time
    s.option.timestep = timestep
    s.option.iterations = iterations
    s.option.ls_iterations = ls_iterations
    for name in DROP_ROOTS:
        s.delete(s.body(name))
    joints = {j.name for j in s.joints}
    for a in list(s.actuators):
        if a.target not in joints:
            s.delete(a)
    for e in list(s.equalities):
        if e.name1 not in joints or e.name2 not in joints:
            s.delete(e)
    bodies = {b.name for b in s.bodies}
    for x in list(s.excludes):
        if x.bodyname1 not in bodies or x.bodyname2 not in bodies:
            s.delete(x)
    for k, g in enumerate(s.geoms):
        if not g.name:
            g.name = f"geom{k}"
    m0 = s.compile()
    # collisions: the right hand only (bit 2), against the box (bit 1); the hand's ends and palm also against the
    # counter's top (bit 4)
    for g in s.geoms:
        i = m0.geom(g.name).id
        body = m0.body(m0.geom_bodyid[i]).name
        hand = body.startswith("R_") and (g.contype or g.conaffinity)
        if not hand:
            g.contype = g.conaffinity = 0
            continue
        g.contype, g.conaffinity = 2, 1 | (4 if body in HAND_ENDS else 0)
        if g.type == mujoco.mjtGeom.mjGEOM_MESH:
            k = m0.geom_dataid[i]
            v = m0.mesh_vert[m0.mesh_vertadr[k]:m0.mesh_vertadr[k] + m0.mesh_vertnum[k]]
            s.add_mesh(name=f"{g.name}_hull", uservert=_support_points(v, hull_vertices).ravel().tolist())
            g.meshname = f"{g.name}_hull"
            g.pos, g.quat = m0.geom_pos[i], m0.geom_quat[i]
        elif g.type == mujoco.mjtGeom.mjGEOM_CYLINDER:     # MJX has no cylinder-box collision
            r, h = g.size[0], g.size[1]
            g.type, g.size = mujoco.mjtGeom.mjGEOM_BOX, [r, r, h]
    w = s.worldbody
    w.add_light(pos=[0.5, 0, 2.5], dir=[0, 0, -1], type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL)
    w.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[3, 3, 0.05], rgba=[0.7, 0.7, 0.7, 1],
               contype=0, conaffinity=0)
    w.add_geom(name="counter", type=mujoco.mjtGeom.mjGEOM_BOX, pos=[0.55, 0, TABLE_Z / 2],
               size=[0.35, 0.65, TABLE_Z / 2], rgba=[0.93, 0.92, 0.89, 1], contype=0, conaffinity=0)
    w.add_geom(name="table", type=mujoco.mjtGeom.mjGEOM_PLANE, pos=[0.55, 0, TABLE_Z], size=[0.35, 0.65, 0.01],
               rgba=[0, 0, 0, 0], contype=4, conaffinity=1, friction=[0.8, 0.01, 0.001], condim=3, group=3)
    # the box: the YCB gelatin box as the robot scene has it (`flat_grasp_deploy`; its hull, or by default the hull's
    # smallest bounding box), with the scene's object contact settings; body frame at the box's centre
    verts = gelatin_hull(hull_vertices)
    half = (verts.max(0) - verts.min(0)) / 2
    box = w.add_body(name="box", pos=[*NOMINAL_BOX_XY, TABLE_Z + half[2] + 0.001])
    box.add_freejoint(name="box_free")
    contact = dict(mass=BOX_MASS, rgba=[0.9, 0.3, 0.4, 1], friction=[1.0, 0.01, 0.001], condim=4, solref=[0.004, 1])
    if box_shape == "hull":
        s.add_mesh(name="box_hull", uservert=verts.ravel().tolist())
        box.add_geom(name="box", type=mujoco.mjtGeom.mjGEOM_MESH, meshname="box_hull", **contact)
    else:
        # the hull's oriented bounding box: the box is a box (MJX collides boxes far faster than meshes)
        def area(a):
            r = verts[:, :2] @ np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
            return np.prod(r.max(0) - r.min(0))
        theta = float(min(np.radians(np.arange(0, 90, 0.25)), key=area))      # the smallest rectangle around it
        c, s_ = np.cos(theta), np.sin(theta)
        local = verts @ np.array([[c, -s_, 0], [s_, c, 0], [0, 0, 1]])
        lo, hi = local.min(0), local.max(0)
        centre = np.array([[c, -s_, 0], [s_, c, 0], [0, 0, 1]]) @ ((lo + hi) / 2)
        box.add_geom(name="box", type=mujoco.mjtGeom.mjGEOM_BOX, size=((hi - lo) / 2).tolist(), pos=centre.tolist(),
                     quat=[np.cos(theta / 2), 0, 0, np.sin(theta / 2)], **contact)
    m1 = s.compile()
    _write(s, OUT_XML.with_name(f"flat_grasp{suffix}.xml"))
    # training copy: the visual meshes out (no asset files needed: Kaggle); inertias are explicit, checked
    for g in list(s.geoms):
        if g.type == mujoco.mjtGeom.mjGEOM_MESH and not g.contype:
            s.delete(g)
    for mesh in list(s.meshes):
        if not mesh.name.endswith("_hull"):
            s.delete(mesh)
    train = s.compile()
    assert np.allclose(train.body_mass, m1.body_mass) and np.allclose(train.body_inertia, m1.body_inertia)
    _write(s, OUT_TRAIN_XML.with_name(f"flat_grasp{suffix}_train.xml"))
    OUT_JSON.write_text(json.dumps({**pre, "box_xy": list(NOMINAL_BOX_XY), "box_half": half.tolist()}, indent=2))
    ncol = sum(1 for g in range(train.ngeom) if train.geom_contype[g] or train.geom_conaffinity[g])
    print(f"wrote flat_grasp{suffix}(_train).xml: nq {train.nq} nu {train.nu} neq {train.neq} "
          f"ngeom {train.ngeom} (colliding {ncol})")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--timestep", type=float, default=0.004)
    ap.add_argument("--iterations", type=int, default=10)
    ap.add_argument("--ls-iterations", type=int, default=20)
    ap.add_argument("--hull-vertices", type=int, default=HULL_VERTICES)
    ap.add_argument("--suffix", default="")
    ap.add_argument("--box-shape", default="box", choices=["box", "hull"])
    a = ap.parse_args()
    build(a.timestep, a.iterations, a.ls_iterations, a.hull_vertices, a.suffix, a.box_shape)
