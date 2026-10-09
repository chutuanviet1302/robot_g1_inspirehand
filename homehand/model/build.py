"""Build the fixed-base Unitree G1 + two Inspire hands model.

Pipeline (all done on the XML so the result is a plain, inspectable MJCF file):
  * G1 (Menagerie): drop the floating base and every joint except the 2x7 arm joints
    -> robot stands still in front of the counter, arms are gravity-compensated and position-servoed.
  * Inspire (dex-urdf): converted by `urdf2mjcf`, visual .glb dropped, collision .obj reused as visuals,
    URDF mimic joints -> equality constraints, armature + damping added (without them the light
    finger links blow up), mounted on each `*_wrist_yaw_link` in place of the rubber hands.
"""

from __future__ import annotations

import json
import shutil
import xml.etree.ElementTree as ET

import numpy as np

from homehand import paths
from homehand.model import spec
from homehand.model.objects import OBJECTS, ObjectSpec
from homehand.model.urdf2mjcf import build_body_tree, parse_urdf

# Hands are mounted so their fingers point along the forearm (+x of the wrist link), the palms face the
# body midline and the thumbs point up: a "handshake" pose, ideal for side grasps on a counter.
# Each quaternion maps the hand-root frame (x = palm normal, +-y = thumb side, z = fingers) onto the wrist.
HAND_MOUNT = {
    "right": {"pos": "0.0415 -0.003 0", "quat": "0.5 0.5 0.5 0.5"},
    "left": {"pos": "0.0415 0.003 0", "quat": "0.5 -0.5 0.5 -0.5"},
}
# Palm site in the hand_base_link frame, ~1.5 cm in front of the palm. Its frame is x = fingers, z = thumb, so
# a handshake grasp of an upright object is the identity orientation for both hands (palm faces +y on the
# right hand, -y on the left hand).
PALM_SITE = {
    "right": {"pos": "-0.035 -0.09 0", "quat": "0 0.7071068 -0.7071068 0"},
    "left": {"pos": "-0.035 -0.09 0", "quat": "0.7071068 0 0 -0.7071068"},
}

ARM_KP = 400.0
FINGER_KP = 20.0
FINGER_FORCE = 0.8  # N m at the actuated joint: ~13 N at the fingertip (the real RH56DFX tops out near 10 N per finger)


def _remove_children(parent: ET.Element, tags: set[str]) -> None:
    for child in list(parent):
        if child.tag in tags:
            parent.remove(child)


def _iter_with_parent(elem: ET.Element):
    for child in list(elem):
        yield elem, child
        yield from _iter_with_parent(child)


def build(out_dir=None, source: str = "auto") -> None:
    """source: "repo" (user-provided unitree_g1_inspire model), "menagerie" (Menagerie G1 + dex-urdf hands)
    or "auto" (repo model when present)."""
    out_dir = out_dir or paths.GENERATED_DIR
    repo_ok = (paths.REPO_MODEL_DIR / "g1_29dof_inspire_hand.xml").exists()
    if source == "repo" or (source == "auto" and repo_ok and paths.INSPIRE_DIR.exists()):
        from homehand.model.build_repo import build_from_repo
        build_from_repo(out_dir)
        build_scene(out_dir)
        print(f"[build] robot from {paths.REPO_MODEL_DIR.name}; wrote {out_dir / paths.ROBOT_XML.name}, scene and objects")
        return
    if not paths.G1_DIR.exists() or not paths.INSPIRE_DIR.exists():
        raise SystemExit("Robot assets missing - run `homehand fetch-assets` first.")
    asset_out = out_dir / "assets"
    (asset_out / "g1").mkdir(parents=True, exist_ok=True)
    (asset_out / "inspire").mkdir(parents=True, exist_ok=True)

    tree = ET.parse(paths.G1_DIR / "g1.xml")
    root = tree.getroot()
    root.set("model", "g1_inspire_fixed")
    root.find("compiler").attrib.update(meshdir="assets", texturedir="assets")

    # ---- options: stable stiff contacts for grasping
    opt = root.find("option")
    opt.attrib.update(
        integrator="implicitfast", timestep=str(spec.SIM_DT), cone="elliptic", impratio="10",
        noslip_iterations="0",
    )

    # ---- defaults for the hand
    defaults = root.find("default")
    hand = ET.SubElement(defaults, "default", attrib={"class": "inspire"})
    ET.SubElement(hand, "default", attrib={"class": "hand_joint"}).append(
        ET.Element("joint", armature="0.002", damping="0.05", frictionloss="0.001")
    )
    col = ET.SubElement(hand, "default", attrib={"class": "hand_col"})
    ET.SubElement(
        col, "geom", group="3", contype="2", conaffinity="1", condim="4", friction="1.5 0.02 0.002",
        solref="0.004 1", solimp="0.95 0.99 0.001", priority="1", rgba="0.8 0.3 0.3 1",
    )
    vis = ET.SubElement(hand, "default", attrib={"class": "hand_vis"})
    ET.SubElement(vis, "geom", group="2", contype="0", conaffinity="0", density="0", material="hand_mat")

    # ---- assets
    asset = root.find("asset")
    ET.SubElement(asset, "material", name="hand_mat", rgba="0.15 0.15 0.17 1", specular="0.3", shininess="0.5")
    for mesh in asset.findall("mesh"):
        src = paths.G1_DIR / "assets" / mesh.get("file")
        shutil.copy2(src, asset_out / "g1" / src.name)
        mesh.set("name", mesh.get("name") or src.stem)
        mesh.set("file", f"g1/{src.name}")

    # ---- strip G1 joints (fixed base, only the two arms move)
    _remove_children(root, {"sensor", "keyframe", "actuator"})
    for parent, child in list(_iter_with_parent(root.find("worldbody"))):
        if child.tag == "freejoint":
            parent.remove(child)
        elif child.tag == "joint" and child.get("name") not in spec.ARM_JOINTS:
            parent.remove(child)
    bodies = {b.get("name"): b for b in root.iter("body")}
    # passive joint damping on the arms (N m s/rad): smooths contact transients and fast commands
    for j in root.iter("joint"):
        name = j.get("name", "")
        if name in spec.ARM_JOINTS:
            j.set("damping", "0.3" if "wrist" in name else "1.0")

    # Static parts (legs, pelvis) never touch anything: keep their visuals, drop their collision meshes.
    moving = {b.get("name") for side in spec.SIDES for b in bodies[f"{side}_shoulder_pitch_link"].iter("body")}
    for parent, child in list(_iter_with_parent(root.find("worldbody"))):
        if child.tag != "geom" or child.get("class") != "collision" or parent.tag != "body":
            continue
        name = parent.get("name", "")
        if name not in moving and name != "torso_link":
            parent.remove(child)

    act = ET.SubElement(root, "actuator")
    eq = ET.SubElement(root, "equality")
    contact = ET.SubElement(root, "contact")
    hand_actuators: list[ET.Element] = []
    for side in spec.SIDES:
        pre = spec.PREFIX[side]
        for b in bodies[f"{side}_shoulder_pitch_link"].iter("body"):
            b.set("gravcomp", "1")

        # ---- Inspire hand
        urdf = parse_urdf(paths.INSPIRE_DIR / f"inspire_hand_{side}.urdf")
        assert [j.name for j in urdf.actuated_joints] == spec.HAND_URDF_JOINTS
        hand_body, meshes = build_body_tree(
            urdf, mesh_prefix=pre, name_prefix=pre,
            root_pos=HAND_MOUNT[side]["pos"], root_quat=HAND_MOUNT[side]["quat"],
        )
        hand_body.set("childclass", "inspire")
        for fname in meshes:
            src = paths.INSPIRE_DIR / "meshes" / "collision" / fname
            shutil.copy2(src, asset_out / "inspire" / fname)
            ET.SubElement(asset, "mesh", name=f"{pre}{src.stem}", file=f"inspire/{fname}")
        for b in hand_body.iter("body"):
            name = b.get("name")
            if name.endswith("_tip"):
                ET.SubElement(b, "site", name=name, size="0.006", rgba="1 0.5 0 1", group="4")
            if name == f"{pre}hand_base_link":
                ET.SubElement(b, "site", name=spec.palm_site(side), size="0.008", rgba="0 1 0 1", group="4",
                              **PALM_SITE[side])
        wrist = bodies[f"{side}_wrist_yaw_link"]
        for g in list(wrist.findall("geom")):
            if g.get("mesh") == f"{side}_rubber_hand":
                wrist.remove(g)
        wrist.append(hand_body)

        for j in spec.arm_joints(side):
            ET.SubElement(act, "position", name=j, joint=j, kp=str(ARM_KP), dampratio="1", inheritrange="1",
                          attrib={"class": "g1"})
        for j in urdf.actuated_joints:
            hand_actuators.append(ET.Element(
                "position", name=pre + j.name, joint=pre + j.name, kp=str(FINGER_KP), dampratio="1",
                ctrlrange=f"{j.lower:.6g} {j.upper:.6g}", forcerange=f"{-FINGER_FORCE} {FINGER_FORCE}",
            ))
        # mimic joints -> equality constraints
        for j in urdf.mimic_joints:
            master, mult, offset = j.mimic
            ET.SubElement(eq, "joint", joint1=pre + j.name, joint2=pre + master,
                          polycoef=f"{offset:.6g} {mult:.6g} 0 0 0", solref="0.005 1")
        # shoulder roll link grazes the torso collision mesh at some shoulder poses (mesh artefact; the real
        # robot has mechanical stops there)
        ET.SubElement(contact, "exclude", body1=f"{side}_shoulder_roll_link", body2="torso_link")
        # hand base vs wrist links overlap at the mount
        for other in ("wrist_yaw_link", "wrist_pitch_link", "wrist_roll_link"):
            for hb in ("hand_base_link", "hand_root"):
                ET.SubElement(contact, "exclude", body1=pre + hb, body2=f"{side}_{other}")
    act.extend(hand_actuators)  # arms first, then hands: matches spec.ACTUATORS

    # drop mesh assets nothing refers to any more (rubber hands, removed collision meshes)
    used = {g.get("mesh") for g in root.iter("geom") if g.get("mesh")}
    for mesh in list(asset.findall("mesh")):
        if mesh.get("name") not in used:
            asset.remove(mesh)
            (asset_out / mesh.get("file")).unlink(missing_ok=True)

    ET.indent(tree)
    out_dir.mkdir(parents=True, exist_ok=True)
    tree.write(out_dir / paths.ROBOT_XML.name)
    build_scene(out_dir)
    print(f"[build] wrote {out_dir / paths.ROBOT_XML.name}, {paths.SCENE_XML.name}, {paths.OBJECTS_JSON.name}")


def _obj_vertices(path) -> np.ndarray:
    with open(path) as f:
        return np.array([[float(v) for v in line.split()[1:4]] for line in f if line.startswith("v ")])


def _write_hull(v: np.ndarray, path) -> None:
    """Collision geometry: the convex hull only (MuJoCo would otherwise keep all 8k mesh vertices)."""
    from scipy.spatial import ConvexHull
    hull = ConvexHull(v)
    idx = {int(i): k + 1 for k, i in enumerate(hull.vertices)}
    with open(path, "w") as f:
        for i in hull.vertices:
            f.write("v {:.6f} {:.6f} {:.6f}\n".format(*v[i]))
        for tri in hull.simplices:
            f.write("f {} {} {}\n".format(*(idx[int(i)] for i in tri)))


def _add_object(asset: ET.Element, world: ET.Element, obj: ObjectSpec, out_dir, k: int = 0) -> dict:
    """Append one free-floating object whose body origin is at the centre of its footprint, on its bottom.

    Default poses are distinct parking spots on the floor behind the robot: MuJoCo resets to them after a
    numerical blow-up, so they must not overlap."""
    body = ET.SubElement(world, "body", name=obj.name, pos=f"-1.2 {0.25 * k - 0.4:.2f} 0.001")
    ET.SubElement(body, "freejoint", name=f"{obj.name}_free")
    src = paths.YCB_DIR / obj.ycb_id / "google_16k"
    if (src / "textured.obj").exists():
        dst = out_dir / "assets" / "ycb"
        dst.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src / "textured.obj", dst / f"{obj.name}.obj")
        # 2048^2 YCB textures cost ~50 MB each in every simulation process; 512^2 looks the same on screen
        from PIL import Image
        Image.open(src / "texture_map.png").convert("RGB").resize((512, 512), Image.LANCZOS).save(dst / f"{obj.name}.png")
        v = _obj_vertices(src / "textured.obj")
        lo, hi = v.min(0), v.max(0)
        centre = np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, lo[2]])
        ET.SubElement(asset, "texture", name=f"{obj.name}_tex", type="2d", file=f"ycb/{obj.name}.png")
        ET.SubElement(asset, "material", name=f"{obj.name}_mat", texture=f"{obj.name}_tex", specular="0.2")
        ET.SubElement(asset, "mesh", name=f"{obj.name}_mesh", file=f"ycb/{obj.name}.obj")
        _write_hull(_obj_vertices(src / "textured.obj"), dst / f"{obj.name}_hull.obj")
        ET.SubElement(asset, "mesh", name=f"{obj.name}_hull", file=f"ycb/{obj.name}_hull.obj", maxhullvert="64")
        pos = " ".join(f"{c:.5f}" for c in -centre)
        ET.SubElement(body, "geom", name=obj.name, type="mesh", mesh=f"{obj.name}_hull", pos=pos, mass=str(obj.mass),
                      group="3", rgba="1 1 1 0", attrib={"class": "object"})
        ET.SubElement(body, "geom", type="mesh", mesh=f"{obj.name}_mesh", pos=pos, material=f"{obj.name}_mat",
                      contype="0", conaffinity="0", density="0", group="2")
        half = (hi - lo) / 2
        return {"source": "ycb", "height": float(hi[2] - lo[2]), "half_xy": [float(half[0]), float(half[1])]}
    kind, *size = obj.primitive
    if kind == "cylinder":
        r, hz = size
        geom = dict(type="cylinder", size=f"{r} {hz}", pos=f"0 0 {hz}")
        info = {"height": 2 * hz, "half_xy": [r, r]}
    elif kind == "box":
        hx, hy, hz = size
        geom = dict(type="box", size=f"{hx} {hy} {hz}", pos=f"0 0 {hz}")
        info = {"height": 2 * hz, "half_xy": [hx, hy]}
    else:
        (r,) = size
        geom = dict(type="sphere", size=str(r), pos=f"0 0 {r}")
        info = {"height": 2 * r, "half_xy": [r, r]}
    ET.SubElement(body, "geom", name=obj.name, mass=str(obj.mass), rgba=obj.rgba, attrib={"class": "object", **geom})
    return {"source": "primitive", **info}


def build_scene(out_dir) -> None:
    tree = ET.parse(paths.ASSETS_DIR / "scene_kitchen.xml")
    root = tree.getroot()
    asset, world = root.find("asset"), root.find("worldbody")
    info = {name: _add_object(asset, world, obj, out_dir, k) for k, (name, obj) in enumerate(OBJECTS.items())}
    ET.indent(tree)
    tree.write(out_dir / paths.SCENE_XML.name)
    (out_dir / paths.OBJECTS_JSON.name).write_text(json.dumps(info, indent=2))
