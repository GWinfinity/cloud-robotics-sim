"""Convert the Franka FR3 v2 MJCF model to a URDF usable by cuRobo.

The ``franka_fr3_v2`` package ships MJCF only (``fr3v2.xml`` + STL collision
meshes). cuRobo requires a URDF, so this script walks the MJCF body tree and
emits an equivalent URDF:

- one URDF link per MJCF body (collision STL reused as visual+collision mesh);
- one revolute URDF joint per MJCF joint (origin = body pos/quat, axis z);
- one fixed URDF joint for bodies without a joint (``base``, flange ``link8``);
- inertial blocks copied from MJCF ``<inertial>`` when present.

Usage::

    uv run python tools/convert_fr3_mjcf_to_urdf.py \
        --mjcf "D:/githbi/awesome-robot-descriptions-main/robot_descriptions/Arms/franka_fr3_v2/fr3v2.xml" \
        --out assets_genesis/embodiments/franka-fr3-v2
"""

from __future__ import annotations

import argparse
import math
import shutil
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class BodyNode:
    """Parsed MJCF body."""

    name: str
    pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)  # w x y z
    joint_name: str | None = None
    joint_axis: tuple[float, float, float] = (0.0, 0.0, 1.0)
    joint_range: tuple[float, float] | None = None
    joint_effort: float = 87.0
    mesh_file: str | None = None
    inertial: dict[str, str] | None = None
    children: list["BodyNode"] = field(default_factory=list)


def _parse_vec(
    text: str | None, n: int, default: tuple[float, ...]
) -> tuple[float, ...]:
    if not text:
        return default
    parts = [float(x) for x in text.split()]
    if len(parts) != n:
        raise ValueError(f"expected {n} floats, got {text!r}")
    return tuple(parts)


def quat_wxyz_to_rpy(
    quat: tuple[float, float, float, float],
) -> tuple[float, float, float]:
    """Convert a ``(w, x, y, z)`` quaternion to roll/pitch/yaw."""
    w, x, y, z = quat
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm == 0.0:
        raise ValueError("zero quaternion")
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    sinp = 2.0 * (w * y - z * x)
    pitch = math.copysign(math.pi / 2.0, sinp) if abs(sinp) >= 1.0 else math.asin(sinp)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return (roll, pitch, yaw)


def parse_mjcf(mjcf_path: Path) -> tuple[BodyNode, dict[str, str]]:
    """Parse the MJCF file, returning the root body and mesh-name → file map."""
    tree = ET.parse(mjcf_path)
    root = tree.getroot()

    meshes: dict[str, str] = {}
    asset = root.find("asset")
    if asset is not None:
        for mesh in asset.findall("mesh"):
            name = mesh.get("name")
            file = mesh.get("file")
            if name and file:
                meshes[name] = file

    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError("MJCF has no <worldbody>")
    bodies = worldbody.findall("body")
    if len(bodies) != 1:
        raise ValueError(f"expected exactly 1 root body, got {len(bodies)}")

    def walk(elem: ET.Element) -> BodyNode:
        node = BodyNode(
            name=elem.get("name", "unnamed"),
            pos=_parse_vec(elem.get("pos"), 3, (0.0, 0.0, 0.0)),  # type: ignore[assignment]
            quat=_parse_vec(  # type: ignore[assignment]
                elem.get("quat"), 4, (1.0, 0.0, 0.0, 0.0)
            ),
        )
        joint = elem.find("joint")
        if joint is not None and joint.get("type", "hinge") in ("hinge", None):
            node.joint_name = joint.get("name")
            node.joint_axis = _parse_vec(  # type: ignore[assignment]
                joint.get("axis"), 3, (0.0, 0.0, 1.0)
            )
            rng = _parse_vec(joint.get("range"), 2, ())  # type: ignore[arg-type]
            if rng:
                node.joint_range = (rng[0], rng[1])
            effort = joint.get("actuatorfrcrange")
            if effort:
                node.joint_effort = max(abs(float(x)) for x in effort.split())
        for geom in elem.findall("geom"):
            mesh_ref = geom.get("mesh")
            if mesh_ref and mesh_ref in meshes:
                node.mesh_file = meshes[mesh_ref]
                break
        inertial = elem.find("inertial")
        if inertial is not None:
            node.inertial = dict(inertial.attrib)
        for child in elem.findall("body"):
            node.children.append(walk(child))
        return node

    return walk(bodies[0]), meshes


def _fmt(vals: tuple[float, ...]) -> str:
    return " ".join(f"{v:.10g}" for v in vals)


def _inertial_xml(node: BodyNode, indent: str) -> str:
    if node.inertial:
        mass = float(node.inertial.get("mass", "1.0"))
        ixx, iyy, izz = _parse_vec(
            node.inertial.get("diaginertia"), 3, (1e-3, 1e-3, 1e-3)
        )
        origin = _parse_vec(node.inertial.get("pos"), 3, (0.0, 0.0, 0.0))
    else:
        mass, ixx, iyy, izz, origin = 1.0, 1e-3, 1e-3, 1e-3, (0.0, 0.0, 0.0)
    return (
        f"{indent}<inertial>\n"
        f'{indent}  <origin xyz="{_fmt(origin)}"/>\n'
        f'{indent}  <mass value="{mass:.6g}"/>\n'
        f'{indent}  <inertia ixx="{ixx:.6g}" ixy="0" ixz="0" '
        f'iyy="{iyy:.6g}" iyz="0" izz="{izz:.6g}"/>\n'
        f"{indent}</inertial>\n"
    )


def _link_xml(node: BodyNode, mesh_dir: str) -> str:
    parts = [f'  <link name="{node.name}">\n']
    if node.mesh_file:
        mesh_path = f"{mesh_dir}/{node.mesh_file}"
        for tag in ("visual", "collision"):
            parts.append(
                f"    <{tag}>\n"
                f'      <geometry><mesh filename="{mesh_path}"/></geometry>\n'
                f"    </{tag}>\n"
            )
    parts.append(_inertial_xml(node, "    "))
    parts.append("  </link>\n")
    return "".join(parts)


def build_urdf(root_body: BodyNode, mesh_dir: str = "assets") -> str:
    """Build URDF text from the parsed body tree."""
    links: list[str] = []
    joints: list[str] = []

    def emit(node: BodyNode, parent: BodyNode | None) -> None:
        links.append(_link_xml(node, mesh_dir))
        if parent is not None:
            rpy = quat_wxyz_to_rpy(node.quat)
            origin = f'<origin xyz="{_fmt(node.pos)}" rpy="{_fmt(rpy)}"/>'
            if node.joint_name and node.joint_range:
                lo, hi = node.joint_range
                joints.append(
                    f'  <joint name="{node.joint_name}" type="revolute">\n'
                    f'    <parent link="{parent.name}"/>\n'
                    f'    <child link="{node.name}"/>\n'
                    f"    {origin}\n"
                    f'    <axis xyz="{_fmt(node.joint_axis)}"/>\n'
                    f'    <limit lower="{lo:.10g}" upper="{hi:.10g}" '
                    f'effort="{node.joint_effort:.6g}" velocity="2.0"/>\n'
                    f"  </joint>\n"
                )
            else:
                jname = f"fixed_{parent.name}_to_{node.name}"
                joints.append(
                    f'  <joint name="{jname}" type="fixed">\n'
                    f'    <parent link="{parent.name}"/>\n'
                    f'    <child link="{node.name}"/>\n'
                    f"    {origin}\n"
                    f"  </joint>\n"
                )
        for child in node.children:
            emit(child, node)

    emit(root_body, None)
    return (
        '<?xml version="1.0"?>\n'
        '<robot name="fr3v2">\n' + "".join(links) + "".join(joints) + "</robot>\n"
    )


def convert(mjcf_path: Path, out_dir: Path) -> Path:
    """Convert MJCF → URDF, copying referenced STL meshes. Returns URDF path."""
    root_body, meshes = parse_mjcf(mjcf_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    mesh_out = out_dir / "assets"
    mesh_out.mkdir(exist_ok=True)
    mesh_src_dir = mjcf_path.parent / "assets"
    copied = 0
    for file in meshes.values():
        if file.lower().endswith(".stl"):
            src = mesh_src_dir / file
            if src.exists():
                shutil.copy2(src, mesh_out / file)
                copied += 1
    urdf_path = out_dir / "fr3v2.urdf"
    urdf_path.write_text(build_urdf(root_body), encoding="utf-8")
    print(f"wrote {urdf_path} ({copied} STL meshes copied to {mesh_out})")
    return urdf_path


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mjcf", required=True, type=Path, help="Path to fr3v2.xml")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("assets_genesis/embodiments/franka-fr3-v2"),
        help="Output directory for fr3v2.urdf + assets/",
    )
    args = parser.parse_args()
    convert(args.mjcf, args.out)


if __name__ == "__main__":
    main()
