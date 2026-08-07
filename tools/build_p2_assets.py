#!/usr/bin/env python3
"""Build the P2 (1-3 个月补齐) gap assets as self-built parametric geometry.

Second tier of ``data/recipes/asset_gap_list.yaml``: 14 rigid / wall-mount
GLB classes + 3 articulated URDF classes (slim cabinets / pull-out rack /
rolling cart with prismatic drawers and continuous wheels). Same self-built
Apache-2.0 registration path as ``tools/build_p1_assets.py``; writers and
mesh helpers are reused from it.

Usage::

    uv run python tools/build_p2_assets.py                 # build all 17
    uv run python tools/build_p2_assets.py --classes pot_rack slim_cabinet
    uv run python tools/build_p2_assets.py --force
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.build_p1_assets import (  # noqa: E402
    _box,
    _concat,
    _frustum,
    _ground,
    _rod,
    write_glb_class,
    write_urdf_class,
)

logger = logging.getLogger(__name__)

GENERATOR = "tools/build_p2_assets.py"


# ---------------------------------------------------------------------------
# shared composite helpers
# ---------------------------------------------------------------------------


def _tier_rack(
    width: float, depth: float, heights: list[float], top: float
) -> "trimesh.Trimesh":
    """N-tier rack: 4 corner posts + thin shelf boards at ``heights``."""
    post_r, shelf_t = 0.006, 0.015
    parts = []
    for x in (-width / 2 + post_r, width / 2 - post_r):
        for z in (-depth / 2 + post_r, depth / 2 - post_r):
            parts.append(_rod((x, 0.0, z), (x, top, z), post_r))
    for h in heights:
        parts.append(_box((width, shelf_t, depth), (0.0, h, 0.0)))
    return _concat(parts)


def _open_box(
    lx: float,
    lz: float,
    h: float,
    t: float = 0.004,
    dividers_x: int = 0,
    dividers_z: int = 0,
    front_h: float | None = None,
) -> "trimesh.Trimesh":
    """Open-top organizer box (Y-up), optional divider grid / lower front."""
    fh = h if front_h is None else front_h
    parts = [
        _box((lx, t, lz), (0.0, t / 2, 0.0)),  # bottom
        _box((lx, h, t), (0.0, h / 2, -lz / 2 + t / 2)),  # back
        _box((lx, fh, t), (0.0, fh / 2, lz / 2 - t / 2)),  # front (maybe low)
        _box((t, h, lz), (-lx / 2 + t / 2, h / 2, 0.0)),
        _box((t, h, lz), (lx / 2 - t / 2, h / 2, 0.0)),
    ]
    for c in range(1, dividers_x + 1):
        x = -lx / 2 + lx * c / (dividers_x + 1)
        parts.append(_box((t, h * 0.85, lz - 2 * t), (x, h * 0.45, 0.0)))
    for r in range(1, dividers_z + 1):
        z = -lz / 2 + lz * r / (dividers_z + 1)
        parts.append(_box((lx - 2 * t, h * 0.85, t), (0.0, h * 0.45, z)))
    return _concat(parts)


def _hook(
    points: list[tuple[float, float, float]], r: float = 0.004
) -> "trimesh.Trimesh":
    """Bent wire hook through ``points``."""
    return _concat([_rod(a, b, r) for a, b in zip(points, points[1:])])


# ---------------------------------------------------------------------------
# rigid class builders
# ---------------------------------------------------------------------------


def build_bathroom_shelf() -> list:
    """3-tier bathroom rack (0.30 x 0.15 x 0.55 m)."""
    return [
        (
            _ground(_tier_rack(0.30, 0.15, [0.03, 0.25, 0.47], top=0.55)),
            {"source": "parametric:bath-shelf-3tier", "keep_scale": True},
        )
    ]


def build_cable_box() -> list:
    """Cable management box with notched ends (0.25 x 0.09 x 0.09 m)."""
    t = 0.004
    body = _open_box(0.25, 0.09, 0.09, t=t)
    # notch slots on both end walls: two partial-height segments each
    notch = _concat(
        [
            _box((t, 0.03, 0.025), (-0.125 + t / 2, 0.075, 0.0)),
            _box((t, 0.03, 0.025), (0.125 - t / 2, 0.075, 0.0)),
        ]
    )
    lid = _box((0.25, t, 0.09), (0.0, 0.092, 0.0))
    return [(_ground(_concat([body, notch, lid])), {"source": "parametric:cable-box"})]


def build_corner_shelf() -> list:
    """2-tier countertop corner rack (0.24 x 0.20 x 0.33 m)."""
    return [
        (
            _ground(_tier_rack(0.24, 0.20, [0.03, 0.28], top=0.33)),
            {"source": "parametric:corner-shelf-2tier"},
        )
    ]


def build_cosmetic_organizer() -> list:
    """Stepped cosmetics organizer: 3 rows of decreasing height."""
    t = 0.004
    parts = [_box((0.20, t, 0.18), (0.0, t / 2, 0.0))]
    # three rows (z = -0.06 front low, 0.0 mid, +0.06 back tall)
    for z, h in ((-0.06, 0.04), (0.0, 0.07), (0.06, 0.10)):
        parts.append(
            _box((0.20, h, t), (0.0, h / 2, z - 0.03 + t / 2))
        )  # row front wall
    parts.append(_box((0.20, 0.10, t), (0.0, 0.05, 0.09 - t / 2)))  # back wall
    for x in (-0.10 + t / 2, 0.10 - t / 2):  # stepped side panels
        for z, h in ((-0.06, 0.04), (0.0, 0.07), (0.06, 0.10)):
            parts.append(_box((t, h, 0.06), (x, h / 2, z)))
        parts.append(_box((t, 0.10, 0.06), (x, 0.05, 0.06)))
    return [(_ground(_concat(parts)), {"source": "parametric:cosmetic-3row"})]


def build_drawer_dividers() -> list:
    """Flat partition panels, long + short (kitchen/wardrobe drawers)."""
    return [
        (
            _ground(_box((0.40, 0.08, 0.005), (0.0, 0.04, 0.0))),
            {"source": "parametric:divider-40cm", "keep_scale": True},
        ),
        (
            _ground(_box((0.25, 0.08, 0.005), (0.0, 0.04, 0.0))),
            {"source": "parametric:divider-25cm"},
        ),
    ]


def build_file_box() -> list:
    """Magazine file box: full back, half-height front (0.10 x 0.28 x 0.30 m)."""
    return [
        (
            _ground(_open_box(0.10, 0.28, 0.30, t=0.004, front_h=0.15)),
            {"source": "parametric:magazine-file"},
        )
    ]


def build_medicine_box() -> list:
    """Medicine chest with carry handle and one divider."""
    body = _open_box(0.28, 0.18, 0.16, t=0.005, dividers_x=1)
    handle = _hook(
        [(-0.10, 0.16, 0.0), (-0.10, 0.21, 0.0), (0.10, 0.21, 0.0), (0.10, 0.16, 0.0)],
        r=0.006,
    )
    return [(_ground(_concat([body, handle])), {"source": "parametric:medicine-chest"})]


def build_pants_hanger() -> list:
    """Multi-tier pants hanger: hook + frame with 5 rungs (~0.45 m tall)."""
    half, top, wire = 0.18, 0.40, 0.004
    parts = [
        _rod((-half, top, 0.0), (half, top, 0.0), wire),  # top bar
        _rod((-half, 0.05, 0.0), (-half, top, 0.0), wire),  # left side
        _rod((half, 0.05, 0.0), (half, top, 0.0), wire),  # right side
    ]
    for y in (0.05, 0.12, 0.19, 0.26, 0.33):
        parts.append(_rod((-half, y, 0.0), (half, y, 0.0), wire))  # rungs
    pts = [
        (0.0, top, 0.0),
        (0.0, top + 0.05, 0.0),
        (0.03, top + 0.075, 0.0),
        (0.05, top + 0.055, 0.0),
    ]
    parts.append(_hook(pts, wire))
    return [
        (
            _ground(_concat(parts)),
            {"source": "parametric:pants-hanger-5rung", "keep_scale": True},
        )
    ]


def build_pot_rack() -> list:
    """Vertical pot/lid rack: base rail + 5 upright dividers + top bars."""
    parts = [
        _box((0.35, 0.02, 0.15), (0.0, 0.01, 0.0)),
        _rod((-0.14, 0.17, -0.06), (0.14, 0.17, -0.06), 0.005),
        _rod((-0.14, 0.17, 0.06), (0.14, 0.17, 0.06), 0.005),
    ]
    for x in (-0.14, -0.07, 0.0, 0.07, 0.14):
        parts.append(_rod((x, 0.02, -0.06), (x, 0.17, -0.06), 0.005))
        parts.append(_rod((x, 0.02, 0.06), (x, 0.17, 0.06), 0.005))
    return [(_ground(_concat(parts)), {"source": "parametric:pot-rack-5slot"})]


def build_remote_caddy() -> list:
    """3-slot remote control caddy (0.20 x 0.10 x 0.08 m)."""
    return [
        (
            _ground(_open_box(0.20, 0.10, 0.08, t=0.004, dividers_x=2)),
            {"source": "parametric:remote-caddy-3slot"},
        )
    ]


def build_toilet_brush() -> list:
    """Toilet brush (long handle, keep_scale) + holder cup."""
    brush = _concat(
        [
            _rod((0.0, 0.05, 0.0), (0.0, 0.42, 0.0), 0.010),
            _frustum(0.045, 0.035, 0.06, sections=24),
        ]
    )
    holder = _frustum(0.05, 0.06, 0.14, sections=32)
    return [
        (_ground(brush), {"source": "parametric:toilet-brush", "keep_scale": True}),
        (_ground(holder), {"source": "parametric:brush-holder"}),
    ]


def build_wall_hooks() -> list:
    """Adhesive hook strip: back plate + 4 J-hooks (0.30 m wide)."""
    parts = [_box((0.30, 0.05, 0.012), (0.0, 0.025, 0.0))]
    for x in (-0.11, -0.04, 0.04, 0.11):
        parts.append(
            _hook([(x, 0.02, 0.006), (x, 0.02, 0.035), (x, 0.05, 0.035)], r=0.004)
        )
    return [(_ground(_concat(parts)), {"source": "parametric:hook-strip-4j"})]


def build_wall_rack_kitchen() -> list:
    """Kitchen wall rail: plate + rail bar + 3 S-hooks (0.35 m wide)."""
    parts = [
        _box((0.35, 0.04, 0.010), (0.0, 0.02, 0.0)),
        _rod((-0.16, 0.02, 0.030), (0.16, 0.02, 0.030), 0.005),  # rail
    ]
    for x in (-0.10, 0.0, 0.10):
        parts.append(
            _hook(
                [
                    (x, 0.02, 0.030),
                    (x, -0.005, 0.030),
                    (x, -0.020, 0.042),
                    (x, -0.008, 0.052),
                ],
                r=0.0035,
            )
        )
    return [
        (
            _ground(_concat(parts)),
            {"source": "parametric:wall-rail-3hook", "keep_scale": True},
        )
    ]


def build_wardrobe_dividers() -> list:
    """Wardrobe partition panels: vertical divider + horizontal shelf board."""
    return [
        (
            _ground(_box((0.45, 0.50, 0.005), (0.0, 0.25, 0.0))),
            {"source": "parametric:wardrobe-divider-v", "keep_scale": True},
        ),
        (
            _ground(_box((0.50, 0.35, 0.005), (0.0, 0.0025, 0.0))),
            {"source": "parametric:wardrobe-shelf-h", "keep_scale": True},
        ),
    ]


# ---------------------------------------------------------------------------
# articulated URDF (prismatic drawers / continuous wheels)
# ---------------------------------------------------------------------------

_BASE_TEMPLATE = """<?xml version="1.0"?>
<robot name="{name}">
  <link name="base">
    <visual><origin xyz="0 0 {body_cz}"/><geometry><box size="{body_size}"/></geometry></visual>
    <collision><origin xyz="0 0 {body_cz}"/><geometry><box size="{body_size}"/></geometry></collision>
    <inertial><mass value="{body_mass}"/><origin xyz="0 0 {body_cz}"/>
      <inertia ixx="5" ixy="0" ixz="0" iyy="5" iyz="0" izz="2"/></inertial>
  </link>
{sub_blocks}
</robot>
"""

_SUB_TEMPLATE = """  <link name="{link_name}">
    <visual><origin xyz="{geo_xyz}" rpy="{geo_rpy}"/><geometry>{geometry}</geometry></visual>
    <collision><origin xyz="{geo_xyz}" rpy="{geo_rpy}"/><geometry>{geometry}</geometry></collision>
    <inertial><mass value="{mass}"/><origin xyz="{geo_xyz}"/>
      <inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/></inertial>
  </link>
  <joint name="{link_name}_joint" type="{joint_type}">
    <parent link="base"/>
    <child link="{link_name}"/>
    <origin xyz="{joint_xyz}" rpy="0 0 0"/>
    <axis xyz="{axis}"/>
{limit_block}
  </joint>
"""


def compose_urdf(
    name: str, body_size: str, body_mass: float, sublinks: list[dict]
) -> str:
    """Compose a URDF: box body + sub-links with revolute/prismatic/continuous joints."""
    sx, sy, sz = (float(v) for v in body_size.split())
    blocks = []
    for s in sublinks:
        limit = ""
        if s["joint_type"] != "continuous":
            limit = f'    <limit lower="0" upper="{s["upper"]}" effort="10" velocity="0.5"/>'
        blocks.append(
            _SUB_TEMPLATE.format(
                link_name=s["link_name"],
                geometry=s["geometry"],
                geo_xyz=s["geo_xyz"],
                geo_rpy=s.get("geo_rpy", "0 0 0"),
                mass=s.get("mass", 1.0),
                joint_type=s["joint_type"],
                joint_xyz=s["joint_xyz"],
                axis=s["axis"],
                limit_block=limit,
            )
        )
    return _BASE_TEMPLATE.format(
        name=name,
        body_size=body_size,
        body_cz=f"{sz / 2:.3f}",
        body_mass=body_mass,
        sub_blocks="\n".join(blocks),
    )


def _drawer(name: str, z: float, size: str, upper: float = 0.30) -> dict:
    """Prismatic drawer sliding out along +y (front)."""
    return {
        "link_name": name,
        "geometry": f'<box size="{size}"/>',
        "geo_xyz": "0 0 0",
        "mass": 1.5,
        "joint_type": "prismatic",
        "joint_xyz": f"0 0 {z}",
        "axis": "0 1 0",
        "upper": upper,
    }


def build_slim_cabinet_urdf() -> str:
    """0.20 x 0.45 x 0.80 m gap cabinet with two drawers."""
    return compose_urdf(
        "slim_cabinet",
        "0.20 0.45 0.80",
        15.0,
        [
            _drawer("drawer_low", 0.20, "0.16 0.40 0.22"),
            _drawer("drawer_high", 0.58, "0.16 0.40 0.22"),
        ],
    )


def build_pullout_slim_rack_urdf() -> str:
    """0.15 x 0.50 x 0.70 m pull-out gap rack, one sliding basket."""
    return compose_urdf(
        "pullout_slim_rack",
        "0.15 0.50 0.70",
        8.0,
        [_drawer("basket", 0.45, "0.12 0.45 0.20", upper=0.35)],
    )


def build_rolling_slim_cart_urdf() -> str:
    """0.18 x 0.40 x 0.75 m rolling cart, four continuous wheels."""
    wheels = []
    for i, (x, y) in enumerate(
        ((-0.07, -0.16), (0.07, -0.16), (-0.07, 0.16), (0.07, 0.16))
    ):
        wheels.append(
            {
                "link_name": f"wheel_{i}",
                "geometry": '<cylinder radius="0.04" length="0.02"/>',
                "geo_xyz": "0 0 0",
                "geo_rpy": f"0 {math.pi / 2:.4f} 0",  # cylinder z-axis -> wheel x-axis
                "mass": 0.2,
                "joint_type": "continuous",
                "joint_xyz": f"{x} {y} 0.04",
                "axis": "1 0 0",
            }
        )
    return compose_urdf("rolling_slim_cart", "0.18 0.40 0.75", 6.0, wheels)


RIGID_BUILDERS = {
    "bathroom_shelf": build_bathroom_shelf,
    "cable_box": build_cable_box,
    "corner_shelf": build_corner_shelf,
    "cosmetic_organizer": build_cosmetic_organizer,
    "drawer_dividers": build_drawer_dividers,
    "file_box": build_file_box,
    "medicine_box": build_medicine_box,
    "pants_hanger": build_pants_hanger,
    "pot_rack": build_pot_rack,
    "remote_caddy": build_remote_caddy,
    "toilet_brush": build_toilet_brush,
    "wall_hooks": build_wall_hooks,
    "wall_rack_kitchen": build_wall_rack_kitchen,
    "wardrobe_dividers": build_wardrobe_dividers,
}
URDF_BUILDERS = {
    "pullout_slim_rack": build_pullout_slim_rack_urdf,
    "rolling_slim_cart": build_rolling_slim_cart_urdf,
    "slim_cabinet": build_slim_cabinet_urdf,
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
        help="subset of classes to build (default: all 17)",
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
                write_glb_class(
                    objects_dir,
                    name,
                    RIGID_BUILDERS[name](),
                    args.force,
                    generator=GENERATOR,
                )
            )
        elif name in URDF_BUILDERS:
            built.append(
                write_urdf_class(
                    objects_dir,
                    name,
                    URDF_BUILDERS[name](),
                    args.force,
                    generator=GENERATOR,
                )
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
