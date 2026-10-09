"""Build the fixed-base robot from the user-provided `unitree_g1_inspire` model (G1 29 DoF + two Inspire hands).

That model already carries both hands attached to the wrist (same orientation as our own assembly, checked
numerically) with the high-resolution visual meshes. We keep its kinematics and meshes and adapt it:

  * floating base, legs and waist are welded (all their joints removed), only the 2x7 arm joints and the
    2x(6 actuated + 6 mimic) finger joints stay
  * their torque motors are replaced by position actuators (arms kp 400, fingers kp 20 with a torque cap)
  * joints / bodies are renamed to the HomeHand convention (R_/L_ prefix) used by the rest of the code
  * the collision geometry of the hand is the URDF collision primitives + simplified collision meshes (the
    visual meshes they also let collide are dropped), contact parameters tuned for grasping
  * palm / fingertip sites and the contact exclusions are added
The result is the same `generated/g1_inspire.xml` the Menagerie-based builder produces.
"""

from __future__ import annotations

import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

from homehand import paths
from homehand.model import spec
from homehand.model.urdf2mjcf import parse_urdf

ARM_KP = 400.0
FINGER_KP = 20.0
FINGER_FORCE = 0.8  # N m at the actuated joint (~13 N at the fingertip; the real RH56DFX tops out near 10 N)

WRIST_FLANGE_X = 0.038  # wrist_yaw_link mesh ends at x=0.042
HAND_MARK = {"right": "r_ih_", "left": "l_ih_"}
PALM_SITE = {
    "right": {"pos": "-0.035 -0.09 0", "quat": "0 0.7071068 -0.7071068 0"},
    "left": {"pos": "-0.035 -0.09 0", "quat": "0.7071068 0 0 -0.7071068"},
}
COLLISION = dict(group="3", contype="2", conaffinity="1", condim="4", friction="1.5 0.02 0.002",
                 solref="0.004 1", solimp="0.95 0.99 0.001", priority="1")


def _iter_with_parent(elem: ET.Element):
    for child in list(elem):
        yield elem, child
        yield from _iter_with_parent(child)


def _hand_side(name: str) -> str | None:
    for side, mark in HAND_MARK.items():
        if name.startswith(mark):
            return side
    return None


def new_body_name(name: str) -> str:
    side = _hand_side(name)
    if side is None:
        return name
    pre = spec.PREFIX[side]
    rest = name[len(HAND_MARK[side]):]
    return pre + ("hand_base_link" if rest == "hand_base" else rest)


def new_joint_name(name: str) -> str:
    side = _hand_side(name)
    return name if side is None else f"{spec.PREFIX[side]}{name[len(HAND_MARK[side]):]}_joint"


def build_from_repo(out_dir: Path | None = None) -> None:
    src_dir = paths.REPO_MODEL_DIR
    out_dir = out_dir or paths.GENERATED_DIR
    asset_out = out_dir / "assets" / "repo"
    asset_out.mkdir(parents=True, exist_ok=True)

    tree = ET.parse(src_dir / "g1_29dof_inspire_hand.xml")
    root = tree.getroot()
    root.set("model", "g1_inspire_fixed")
    root.find("compiler").attrib.update(meshdir="assets", texturedir="assets", angle="radian")

    for tag in ("sensor", "actuator", "keyframe"):
        for e in root.findall(tag):
            root.remove(e)
    opt = ET.Element("option", integrator="implicitfast", timestep=str(spec.SIM_DT), cone="elliptic", impratio="10")
    root.insert(list(root).index(root.find("compiler")) + 1, opt)

    # ---- joints: keep arms + fingers, weld everything else
    keep_arm = set(spec.ARM_JOINTS)
    for parent, child in list(_iter_with_parent(root.find("worldbody"))):
        if child.tag != "joint" and child.tag != "freejoint":
            continue
        name = child.get("name", "")
        if child.tag == "freejoint" or child.get("type") == "free":
            parent.remove(child)
        elif _hand_side(name) is None and name not in keep_arm:
            parent.remove(child)

    # ---- geoms
    bodies = {b.get("name"): b for b in root.iter("body")}
    moving = set()
    for side in spec.SIDES:
        moving |= {b.get("name") for b in bodies[f"{side}_shoulder_pitch_link"].iter("body")}
    for body in root.iter("body"):
        bname = body.get("name")
        side = _hand_side(bname)
        for g in list(body.findall("geom")):
            mesh = g.get("mesh", "")
            is_visual_twin = g.get("contype") == "0"
            if side is not None:
                is_collision_mesh = "_ih_c_" in mesh
                is_visual_mesh = bool(mesh) and not is_collision_mesh
                if is_visual_mesh and not is_visual_twin:
                    body.remove(g)            # their visual mesh used as a colliding geom: drop
                elif is_collision_mesh and is_visual_twin:
                    body.remove(g)            # invisible twin of the collision mesh: drop
                elif is_visual_mesh:          # visual
                    g.set("group", "2")
                    if bname.endswith("hand_base"):
                        # The palm's collision primitives sit up to 5 mm inside its visual shell: the convex
                        # hull of the visual mesh collides too, so the rendered palm never sinks into an object.
                        hull = ET.SubElement(body, "geom", type="mesh", mesh=mesh, **COLLISION)
                        hull.set("rgba", "0.8 0.3 0.3 1")
                        for k in ("pos", "quat"):
                            if g.get(k):
                                hull.set(k, g.get(k))
                else:                         # collision (primitive or simplified mesh)
                    g.attrib.pop("rgba", None)
                    g.attrib.update(COLLISION)
                    g.set("rgba", "0.8 0.3 0.3 1")
            else:
                if is_visual_twin:
                    g.set("group", "2")
                elif bname in moving or bname == "torso_link":
                    g.attrib.update(group="3")
                    g.set("rgba", "0.7 0.7 0.7 1")
                else:
                    body.remove(g)            # legs / pelvis never touch anything: visuals only

    # ---- renames
    for b in root.iter("body"):
        b.set("name", new_body_name(b.get("name")))
    world = root.find("worldbody")
    for j in world.iter("joint"):
        j.set("name", new_joint_name(j.get("name")))
    for e in root.find("equality"):
        # The source model lists (joint1 = actuated, joint2 = follower) with the URDF coefficients, but MuJoCo
        # computes q1 = poly(q2): that makes the *actuated* joint follow the others (the finger bends 25-45 %
        # too little). Swap them so that follower = offset + multiplier * actuated, as in the URDF <mimic>.
        j1, j2 = e.get("joint1"), e.get("joint2")
        e.set("joint1", new_joint_name(j2))
        e.set("joint2", new_joint_name(j1))
        e.set("solref", "0.005 1")
        e.attrib.pop("solimp", None)
    bodies = {b.get("name"): b for b in root.iter("body")}
    # The source puts the hand flange 8 mm beyond the end of the wrist link (visible gap): seat it 4 mm inside.
    for side in spec.SIDES:
        bodies[f"{spec.PREFIX[side]}hand_base_link"].set("pos", f"{WRIST_FLANGE_X} 0 0")

    # ---- joint dynamics
    for j in world.iter("joint"):
        name = j.get("name")
        j.attrib.pop("class", None)
        if name in keep_arm:
            j.set("armature", "0.01")
            j.set("damping", "0.3" if "wrist" in name else "1.0")
            j.set("frictionloss", "0.3")
        else:
            j.set("armature", "0.002")
            j.set("damping", "0.05")
            j.set("frictionloss", "0.001")
            j.attrib.pop("actuatorfrcrange", None)
    for b in bodies.values():
        if b.get("name") in moving or b.get("name", "").startswith(("L_", "R_")):
            b.set("gravcomp", "1")

    # ---- sites: palm + fingertips (offsets from the URDF)
    for side in spec.SIDES:
        pre = spec.PREFIX[side]
        urdf = parse_urdf(paths.INSPIRE_DIR / f"inspire_hand_{side}.urdf")
        ET.SubElement(bodies[f"{pre}hand_base_link"], "site", name=spec.palm_site(side), size="0.008",
                      rgba="0 1 0 1", group="4", **PALM_SITE[side])
        for j in urdf.joints:
            if j.child.endswith("_tip"):
                ET.SubElement(bodies[pre + j.parent], "site", name=pre + j.child, size="0.006", rgba="1 0.5 0 1",
                              group="4", pos=" ".join(f"{v:.6g}" for v in j.xyz))

    # ---- actuators (arms first, then hands: matches spec.ACTUATORS) and contact exclusions
    act = ET.SubElement(root, "actuator")
    joint_el = {j.get("name"): j for j in world.iter("joint")}
    for side in spec.SIDES:
        for jn in spec.arm_joints(side):
            ET.SubElement(act, "position", name=jn, joint=jn, kp=str(ARM_KP), dampratio="1", inheritrange="1")
    for side in spec.SIDES:
        for jn in spec.hand_actuators(side):
            lo, hi = joint_el[jn].get("range").split()
            ET.SubElement(act, "position", name=jn, joint=jn, kp=str(FINGER_KP), dampratio="1",
                          ctrlrange=f"{lo} {hi}", forcerange=f"{-FINGER_FORCE} {FINGER_FORCE}")
    contact = ET.SubElement(root, "contact")
    for side in spec.SIDES:
        pre = spec.PREFIX[side]
        ET.SubElement(contact, "exclude", body1=f"{side}_shoulder_roll_link", body2="torso_link")
        for other in ("wrist_yaw_link", "wrist_pitch_link", "wrist_roll_link"):
            ET.SubElement(contact, "exclude", body1=f"{pre}hand_base_link", body2=f"{side}_{other}")

    # ---- assets: copy only the meshes that are still used, flatten into assets/repo
    asset = root.find("asset")
    used = {g.get("mesh") for g in root.iter("geom") if g.get("mesh")}
    for tag in list(asset):
        if tag.tag == "mesh" and (tag.get("name") or Path(tag.get("file", "")).stem) not in used:
            asset.remove(tag)
    for mesh in asset.findall("mesh"):
        f = src_dir / "meshes" / mesh.get("file")
        f = f.resolve()
        if not f.exists():
            raise FileNotFoundError(f)
        mesh.set("name", mesh.get("name") or f.stem)
        target = f"{mesh.get('name')}{f.suffix}"
        shutil.copy2(f, asset_out / target)
        mesh.set("file", f"repo/{target}")
    for m in asset.findall("material"):
        if m.get("name") not in {g.get("material") for g in root.iter("geom")}:
            asset.remove(m)

    for d in root.findall("default"):  # the original motor classes are no longer referenced
        root.remove(d)
    ET.indent(tree)
    out_dir.mkdir(parents=True, exist_ok=True)
    tree.write(out_dir / paths.ROBOT_XML.name)
