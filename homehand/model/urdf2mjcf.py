"""Minimal URDF -> MJCF body-tree converter for the Inspire hand.

MuJoCo's own URDF importer drops `<mimic>` tags and cannot read `.glb` visuals, so we convert the
(small, simple) hand URDF ourselves: links become bodies, revolute joints become hinge joints,
mimic joints become `<equality joint=...>` constraints, collision meshes are reused as visuals.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


def rpy_to_quat(rpy: np.ndarray) -> np.ndarray:
    """URDF rpy (R = Rz(yaw) Ry(pitch) Rx(roll)) -> MuJoCo quaternion (w, x, y, z)."""
    r, p, y = rpy
    cr, sr = np.cos(r / 2), np.sin(r / 2)
    cp, sp = np.cos(p / 2), np.sin(p / 2)
    cy, sy = np.cos(y / 2), np.sin(y / 2)
    return np.array(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ]
    )


def _vec(text: str | None, default: str = "0 0 0") -> np.ndarray:
    return np.array([float(v) for v in (text or default).split()])


def _fmt(v) -> str:
    return " ".join(f"{float(x):.6g}" for x in np.atleast_1d(v))


@dataclass
class Joint:
    name: str
    type: str
    parent: str
    child: str
    xyz: np.ndarray
    rpy: np.ndarray
    axis: np.ndarray
    lower: float = 0.0
    upper: float = 0.0
    mimic: tuple[str, float, float] | None = None  # (master, multiplier, offset)


@dataclass
class HandDescription:
    root_link: str
    links: dict[str, ET.Element]
    joints: list[Joint]
    children: dict[str, list[Joint]] = field(default_factory=dict)

    @property
    def mimic_joints(self) -> list[Joint]:
        return [j for j in self.joints if j.mimic is not None]

    @property
    def actuated_joints(self) -> list[Joint]:
        return [j for j in self.joints if j.type == "revolute" and j.mimic is None]


def parse_urdf(path: Path) -> HandDescription:
    root = ET.parse(path).getroot()
    links = {link.get("name"): link for link in root.findall("link")}
    joints: list[Joint] = []
    for j in root.findall("joint"):
        origin = j.find("origin")
        limit = j.find("limit")
        mimic = j.find("mimic")
        joints.append(
            Joint(
                name=j.get("name"),
                type=j.get("type"),
                parent=j.find("parent").get("link"),
                child=j.find("child").get("link"),
                xyz=_vec(origin.get("xyz") if origin is not None else None),
                rpy=_vec(origin.get("rpy") if origin is not None else None),
                axis=_vec(j.find("axis").get("xyz")) if j.find("axis") is not None else np.array([0, 0, 1.0]),
                lower=float(limit.get("lower", 0)) if limit is not None else 0.0,
                upper=float(limit.get("upper", 0)) if limit is not None else 0.0,
                mimic=(
                    (mimic.get("joint"), float(mimic.get("multiplier", 1)), float(mimic.get("offset", 0)))
                    if mimic is not None
                    else None
                ),
            )
        )
    child_links = {j.child for j in joints}
    root_link = next(name for name in links if name not in child_links)
    desc = HandDescription(root_link=root_link, links=links, joints=joints)
    for j in joints:
        desc.children.setdefault(j.parent, []).append(j)
    return desc


def build_body_tree(
    desc: HandDescription,
    mesh_prefix: str,
    name_prefix: str,
    root_pos: str,
    root_quat: str,
    gravcomp: bool = True,
) -> tuple[ET.Element, list[str]]:
    """Return (<body> element for the root link, list of mesh file names referenced)."""
    meshes: list[str] = []

    def add_link_geoms(body: ET.Element, link: ET.Element) -> None:
        inertial = link.find("inertial")
        if inertial is not None:
            o = inertial.find("origin")
            m = inertial.find("mass")
            I = inertial.find("inertia")
            full = [float(I.get(k)) for k in ("ixx", "iyy", "izz", "ixy", "ixz", "iyz")]
            ET.SubElement(
                body,
                "inertial",
                pos=_fmt(_vec(o.get("xyz") if o is not None else None)),
                mass=f"{float(m.get('value')):.6g}",
                fullinertia=_fmt(full),
            )
        for col in link.findall("collision"):
            o = col.find("origin")
            pos = _vec(o.get("xyz") if o is not None else None)
            quat = rpy_to_quat(_vec(o.get("rpy") if o is not None else None))
            g = col.find("geometry")[0]
            attrs = {"pos": _fmt(pos), "quat": _fmt(quat)}
            if g.tag == "mesh":
                fname = Path(g.get("filename")).name
                mesh_name = f"{mesh_prefix}{Path(fname).stem}"
                if fname not in meshes:
                    meshes.append(fname)
                attrs.update(type="mesh", mesh=mesh_name)
            elif g.tag == "box":
                attrs.update(type="box", size=_fmt(_vec(g.get("size")) / 2))
            elif g.tag == "cylinder":
                attrs.update(type="cylinder", size=_fmt([float(g.get("radius")), float(g.get("length")) / 2]))
            elif g.tag == "sphere":
                attrs.update(type="sphere", size=g.get("radius"))
            else:
                raise ValueError(f"unsupported geometry {g.tag}")
            ET.SubElement(body, "geom", attrib={**attrs, "class": "hand_col"})
            ET.SubElement(body, "geom", attrib={**attrs, "class": "hand_vis"})

    def recurse(parent_body: ET.Element, link_name: str) -> None:
        for j in desc.children.get(link_name, []):
            body = ET.SubElement(
                parent_body, "body", name=name_prefix + j.child, pos=_fmt(j.xyz), quat=_fmt(rpy_to_quat(j.rpy))
            )
            if gravcomp:
                body.set("gravcomp", "1")
            if j.type == "revolute":
                ET.SubElement(
                    body, "joint", name=name_prefix + j.name, axis=_fmt(j.axis), range=_fmt([j.lower, j.upper]),
                    attrib={"class": "hand_joint"},
                )
            elif j.type != "fixed":
                raise ValueError(f"unsupported joint type {j.type}")
            add_link_geoms(body, desc.links[j.child])
            recurse(body, j.child)

    root_body = ET.Element("body", name=f"{name_prefix}hand_root", pos=root_pos, quat=root_quat)
    if gravcomp:
        root_body.set("gravcomp", "1")
    add_link_geoms(root_body, desc.links[desc.root_link])
    recurse(root_body, desc.root_link)
    return root_body, meshes
