#!/usr/bin/env python3
"""Build the P1 (入住即买) gap assets as self-built parametric geometry.

Closes the six P1 entries of ``data/recipes/asset_gap_list.yaml`` with
license-clean, self-modeled stand-ins (registered Apache-2.0 per the
Track-B design doc §2 "自建" asset path). They are geometric placeholders:
later Objaverse / PartNet-Mobility imports can replace or augment instances
through the same library layout.

Rigid classes are written in the RoboTwin-OD GLB layout
(``collision/base<N>.glb`` + ``model_data<N>.json``); articulated furniture
uses the PartNet-Mobility layout (``<id>/mobility.urdf`` + ``model_data.json``)
with primitive box geometry only (no mesh files needed).

Tall tools (mop / broom) set ``keep_scale: true`` in their metadata so
``RoboTwinObjectLibrary.spawn_in_scene`` does not shrink them to grasp-toy
size — they are scene/working tools, not grasp targets.

Usage::

    uv run python tools/build_p1_assets.py                 # build all six
    uv run python tools/build_p1_assets.py --classes fridge wardrobe
    uv run python tools/build_p1_assets.py --force         # rebuild existing
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.import_generated_objects import resolve_class_name  # noqa: E402

logger = logging.getLogger(__name__)

GENERATOR = "tools/build_p1_assets.py"
LICENSE = "Apache-2.0"  # self-built assets, per design doc §2 自建 path
AUTHOR = "genesis-cloud-sim (self-built parametric)"


# ---------------------------------------------------------------------------
# mesh helpers
# ---------------------------------------------------------------------------


def _rod(p0, p1, radius: float, sections: int = 12) -> "trimesh.Trimesh":
    """Cylinder spanning two points (for wire frames like hangers/handles)."""
    import trimesh

    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    vec = p1 - p0
    length = float(np.linalg.norm(vec))
    rod = trimesh.creation.cylinder(radius=radius, height=length, sections=sections)
    direction = vec / length
    z = np.array([0.0, 0.0, 1.0])
    axis = np.cross(z, direction)
    if np.linalg.norm(axis) < 1e-9:
        if direction[2] < 0:
            rod.apply_transform(
                trimesh.transformations.rotation_matrix(math.pi, [1, 0, 0])
            )
    else:
        angle = math.acos(float(np.clip(np.dot(z, direction), -1.0, 1.0)))
        rod.apply_transform(trimesh.transformations.rotation_matrix(angle, axis))
    rod.apply_translation((p0 + p1) / 2.0)
    return rod


def _box(extents, center=(0.0, 0.0, 0.0)) -> "trimesh.Trimesh":
    import trimesh

    mesh = trimesh.creation.box(extents=extents)
    mesh.apply_translation(center)
    return mesh


def _frustum(
    r_bottom: float, r_top: float, height: float, sections: int = 48
) -> "trimesh.Trimesh":
    """Closed (solid) frustum in the Y-up frame, base at y=0."""
    import trimesh

    angles = np.linspace(0.0, 2.0 * math.pi, sections, endpoint=False)
    cos, sin = np.cos(angles), np.sin(angles)
    verts = np.vstack(
        [
            np.column_stack([r_bottom * cos, np.zeros(sections), r_bottom * sin]),
            np.column_stack([r_top * cos, np.full(sections, height), r_top * sin]),
            [[0.0, 0.0, 0.0]],
            [[0.0, height, 0.0]],
        ]
    )
    c_bot, c_top = 2 * sections, 2 * sections + 1
    faces = []
    for i in range(sections):
        j = (i + 1) % sections
        faces.append([i, j, sections + j])
        faces.append([i, sections + j, sections + i])
        faces.append([c_bot, j, i])
        faces.append([c_top, sections + i, sections + j])
    return trimesh.Trimesh(vertices=verts, faces=np.asarray(faces), process=True)


def _ground(mesh: "trimesh.Trimesh") -> "trimesh.Trimesh":
    """Recenter xz at origin, base at y=0 (library convention, no rescale)."""
    mesh = mesh.copy()
    bounds = mesh.bounds
    center_xz = (bounds[0] + bounds[1]) / 2.0
    mesh.apply_translation([-center_xz[0], -bounds[0][1], -center_xz[2]])
    return mesh


def _concat(parts) -> "trimesh.Trimesh":
    import trimesh

    return trimesh.util.concatenate(parts)


# ---------------------------------------------------------------------------
# rigid class builders -> list of (mesh, extra_metadata) per instance
# ---------------------------------------------------------------------------


def build_clothes_hanger() -> list:
    """Wire hangers: standard + slim variants (~0.42 m wide)."""

    def hanger(width: float, wire_r: float) -> "trimesh.Trimesh":
        half = width / 2.0
        apex = (0.0, 0.12, 0.0)
        parts = [
            _rod((-half, 0.0, 0.0), (half, 0.0, 0.0), wire_r),  # bottom bar
            _rod((-half, 0.0, 0.0), apex, wire_r),  # left slant
            _rod((half, 0.0, 0.0), apex, wire_r),  # right slant
        ]
        # hook: vertical stub + arc of 3 segments curving over
        pts = [
            (0.0, 0.12, 0.0),
            (0.0, 0.175, 0.0),
            (0.03, 0.20, 0.0),
            (0.055, 0.18, 0.0),
            (0.05, 0.15, 0.0),
        ]
        parts.extend(_rod(a, b, wire_r) for a, b in zip(pts, pts[1:]))
        return _concat(parts)

    return [
        (
            _ground(hanger(0.42, 0.004)),
            {"source": "parametric:hanger-standard", "keep_scale": True},
        ),
        (
            _ground(hanger(0.38, 0.003)),
            {"source": "parametric:hanger-slim", "keep_scale": True},
        ),
    ]


def build_underwear_grid_box() -> list:
    """Drawer organizer: 0.32x0.24x0.10 m box with a 3x2 divider grid."""
    lx, ly, h, t = 0.32, 0.24, 0.10, 0.004

    def grid(cols: int, rows: int) -> "trimesh.Trimesh":
        parts = [
            _box((lx, t, ly), (0, t / 2, 0)),  # bottom
            _box((lx, h, t), (0, h / 2, -ly / 2 + t / 2)),
            _box((lx, h, t), (0, h / 2, ly / 2 - t / 2)),
            _box((t, h, ly), (-lx / 2 + t / 2, h / 2, 0)),
            _box((t, h, ly), (lx / 2 - t / 2, h / 2, 0)),
        ]
        for c in range(1, cols):  # dividers along x (z-aligned in trimesh frame)
            x = -lx / 2 + lx * c / cols
            parts.append(_box((t, h * 0.85, ly - 2 * t), (x, h * 0.45, 0)))
        for r in range(1, rows):  # dividers along z
            z = -ly / 2 + ly * r / rows
            parts.append(_box((lx - 2 * t, h * 0.85, t), (0, h * 0.45, z)))
        return _concat(parts)

    # modeled Y-up (walls rise along +Y), matching the glTF library frame
    return [
        (_ground(grid(3, 2)), {"source": "parametric:grid-box-3x2"}),
        (_ground(grid(2, 2)), {"source": "parametric:grid-box-2x2"}),
    ]


def build_mop_broom_set() -> list:
    """Mop, broom and bucket (long tools keep real scale via keep_scale)."""
    # mop: handle + round head disk
    mop = _concat(
        [
            _rod((0, 0.06, 0), (0, 1.26, 0), 0.012),
            _frustum(0.075, 0.06, 0.07, sections=24),
        ]
    )
    # broom: handle + angled head block + bristle strip
    broom = _concat(
        [
            _rod((0, 0.10, 0.02), (0, 1.30, 0.0), 0.011),
            _box((0.30, 0.06, 0.05), (0, 0.06, 0.03)),
            _box((0.28, 0.05, 0.03), (0, 0.025, 0.05)),
        ]
    )
    # bucket: solid frustum + rim torus
    import trimesh

    rim = trimesh.creation.torus(0.16, 0.008)
    rim.apply_transform(trimesh.transformations.rotation_matrix(math.pi / 2, [1, 0, 0]))
    rim.apply_translation((0, 0.28, 0))
    bucket = _concat([_frustum(0.12, 0.16, 0.28), rim])
    keep = {"keep_scale": True}
    return [
        (_ground(mop), {"source": "parametric:mop", **keep}),
        (_ground(broom), {"source": "parametric:broom", **keep}),
        (_ground(bucket), {"source": "parametric:bucket"}),
    ]


def build_mop_holder() -> list:
    """Wall-mount tool holder: back plate + 3 clamp stubs (~0.30 m wide)."""
    parts = [_box((0.30, 0.06, 0.02), (0, 0.05, 0))]  # back plate (z thin)
    import trimesh

    for x in (-0.10, 0.0, 0.10):
        ring = trimesh.creation.torus(0.018, 0.005)
        ring.apply_transform(
            trimesh.transformations.rotation_matrix(math.pi / 2, [1, 0, 0])
        )
        ring.apply_translation((x, 0.05, 0.025))
        parts.append(ring)
        parts.append(_box((0.012, 0.05, 0.03), (x, 0.035, 0.012)))  # clamp arm
    return [(_ground(_concat(parts)), {"source": "parametric:mop-holder-3clamp"})]


# ---------------------------------------------------------------------------
# articulated furniture (PartNet-Mobility URDF layout, primitive geometry)
# ---------------------------------------------------------------------------

_URDF_TEMPLATE = """<?xml version="1.0"?>
<robot name="{name}">
  <link name="base">
    <visual><origin xyz="0 0 {body_cz}"/><geometry><box size="{body_size}"/></geometry></visual>
    <collision><origin xyz="0 0 {body_cz}"/><geometry><box size="{body_size}"/></geometry></collision>
    <inertial><mass value="{body_mass}"/><origin xyz="0 0 {body_cz}"/>
      <inertia ixx="10" ixy="0" ixz="0" iyy="10" iyz="0" izz="5"/></inertial>
  </link>
{door_blocks}
</robot>
"""

_DOOR_TEMPLATE = """  <link name="{door_name}">
    <visual><origin xyz="{door_geo_xyz}"/><geometry><box size="{door_size}"/></geometry></visual>
    <collision><origin xyz="{door_geo_xyz}"/><geometry><box size="{door_size}"/></geometry></collision>
    <inertial><mass value="6"/><origin xyz="{door_geo_xyz}"/>
      <inertia ixx="1" ixy="0" ixz="0" iyy="1" iyz="0" izz="1"/></inertial>
  </link>
  <joint name="{door_name}_joint" type="revolute">
    <parent link="base"/>
    <child link="{door_name}"/>
    <origin xyz="{hinge_xyz}" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="0" upper="2.0" effort="10" velocity="1.0"/>
  </joint>
"""


def _furniture_urdf(
    name: str, body_size: str, body_mass: float, doors: list[dict]
) -> str:
    """Compose a URDF string: box body + revolute door(s), Z-up, base at z=0."""
    sx, sy, sz = (float(v) for v in body_size.split())
    blocks = []
    for d in doors:
        blocks.append(
            _DOOR_TEMPLATE.format(
                door_name=d["name"],
                door_geo_xyz=d["geo_xyz"],
                door_size=d["size"],
                hinge_xyz=d["hinge_xyz"],
            )
        )
    return _URDF_TEMPLATE.format(
        name=name,
        body_size=body_size,
        body_cz=f"{sz / 2:.3f}",
        body_mass=body_mass,
        door_blocks="\n".join(blocks),
    )


def build_fridge_urdf() -> str:
    """0.6 x 0.6 x 1.8 m fridge, single left-hinged front door."""
    return _furniture_urdf(
        "fridge",
        "0.6 0.6 1.8",
        45.0,
        [
            {
                "name": "door",
                "size": "0.56 0.05 1.72",
                "geo_xyz": "0.27 0.025 0.0",
                "hinge_xyz": "-0.27 0.30 0.90",
            }
        ],
    )


def build_wardrobe_urdf() -> str:
    """1.2 x 0.6 x 2.0 m wardrobe, two side-hinged front doors."""
    return _furniture_urdf(
        "wardrobe",
        "1.2 0.6 2.0",
        60.0,
        [
            {
                "name": "door_left",
                "size": "0.58 0.04 1.9",
                "geo_xyz": "0.29 0.02 0.0",
                "hinge_xyz": "-0.58 0.30 1.0",
            },
            {
                "name": "door_right",
                "size": "0.58 0.04 1.9",
                "geo_xyz": "-0.29 0.02 0.0",
                "hinge_xyz": "0.58 0.30 1.0",
            },
        ],
    )


# ---------------------------------------------------------------------------
# writers
# ---------------------------------------------------------------------------


def _meta(extra: dict | None = None, generator: str = GENERATOR) -> dict:
    data = {
        "scale": [1.0, 1.0, 1.0],
        "stable": True,
        "generator": generator,
        "license": LICENSE,
        "author": AUTHOR,
        "imported_at": datetime.now(timezone.utc).isoformat(),
    }
    if extra:
        data.update(extra)
    return data


def write_glb_class(
    objects_dir: Path,
    class_name: str,
    instances: list,
    force: bool,
    generator: str = GENERATOR,
) -> str:
    """Write one rigid class (collision/base<N>.glb + model_data<N>.json)."""
    resolved = resolve_class_name(class_name, objects_dir)
    class_dir = objects_dir / resolved
    if class_dir.exists() and not force:
        logger.info("skip existing class %s (use --force to rebuild)", resolved)
        return resolved
    collision = class_dir / "collision"
    collision.mkdir(parents=True, exist_ok=True)
    for idx, (mesh, extra) in enumerate(instances):
        mesh.export(str(collision / f"base{idx}.glb"))
        bounds = mesh.bounds
        center = ((bounds[0] + bounds[1]) / 2.0).tolist()
        meta = _meta(
            {**extra, "center": center, "extents": mesh.extents.tolist()}, generator
        )
        (class_dir / f"model_data{idx}.json").write_text(
            json.dumps(meta, indent=4), encoding="utf-8"
        )
    logger.info("built %s (%d instances)", resolved, len(instances))
    return resolved


def write_urdf_class(
    objects_dir: Path,
    class_name: str,
    urdf: str,
    force: bool,
    generator: str = GENERATOR,
) -> str:
    """Write one articulated class (<id>/mobility.urdf + model_data.json)."""
    resolved = resolve_class_name(class_name, objects_dir)
    class_dir = objects_dir / resolved
    if class_dir.exists() and not force:
        logger.info("skip existing class %s (use --force to rebuild)", resolved)
        return resolved
    inst_dir = class_dir / "900001"
    inst_dir.mkdir(parents=True, exist_ok=True)
    (inst_dir / "mobility.urdf").write_text(urdf, encoding="utf-8")
    (inst_dir / "model_data.json").write_text(
        json.dumps(_meta({"source": f"parametric:{class_name}"}, generator), indent=4),
        encoding="utf-8",
    )
    logger.info("built %s (urdf, 1 instance)", resolved)
    return resolved


RIGID_BUILDERS = {
    "clothes_hanger": build_clothes_hanger,
    "underwear_grid_box": build_underwear_grid_box,
    "mop_broom_set": build_mop_broom_set,
    "mop_holder": build_mop_holder,
}
URDF_BUILDERS = {
    "fridge": build_fridge_urdf,
    "wardrobe": build_wardrobe_urdf,
}


def main(argv: list[str] | None = None) -> int:
    """CLI entry point; returns 0 on success."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--objects-dir",
        default="assets/robotwin/objects/objects",
        help="RoboTwin object library root (default: %(default)s)",
    )
    parser.add_argument(
        "--classes",
        nargs="*",
        default=sorted(RIGID_BUILDERS) + sorted(URDF_BUILDERS),
        help="subset of classes to build (default: all six)",
    )
    parser.add_argument("--force", action="store_true", help="rebuild existing classes")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    objects_dir = Path(args.objects_dir)
    objects_dir.mkdir(parents=True, exist_ok=True)

    built: list[str] = []
    for name in args.classes:
        if name in RIGID_BUILDERS:
            built.append(
                write_glb_class(objects_dir, name, RIGID_BUILDERS[name](), args.force)
            )
        elif name in URDF_BUILDERS:
            built.append(
                write_urdf_class(objects_dir, name, URDF_BUILDERS[name](), args.force)
            )
        else:
            logger.error(
                "unknown class %s (have %s)",
                name,
                sorted(RIGID_BUILDERS | URDF_BUILDERS),
            )
            return 1
    print(f"built {len(built)} class(es): {', '.join(built)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
