#!/usr/bin/env python3
"""Build the P3 (按需升级) gap assets as self-built parametric geometry.

Third tier of ``data/recipes/asset_gap_list.yaml``: 14 rigid / wall-mount
GLB classes + 6 articulated URDF classes (rotating trays, lidded boxes,
drawer cabinet, wheeled desk cart). Soft-body and electronic items stay
deferred per the gap list. Mesh helpers come from ``build_p1_assets``,
composite helpers and ``compose_urdf`` from ``build_p2_assets``.

Content companions are bundled as extra instances of their class (same
precedent as ``122_mop_broom_set``): ``hairdryer_set`` (holder + dryer),
``toothbrush_set`` (holder + brush), ``umbrella_set`` (stand + folded
umbrella), ``kitchen_utensil_set`` (spatula / ladle / whisk — the hanging
contents for ``139_wall_rack_kitchen``).

Usage::

    uv run python tools/build_p3_assets.py                 # build all 20
    uv run python tools/build_p3_assets.py --classes pegboard desk_cart
    uv run python tools/build_p3_assets.py --force
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

from tools.build_p1_assets import (  # noqa: E402  # noqa: E402
    _box,
    _concat,
    _frustum,
    _ground,
    _rod,
    write_glb_class,
    write_urdf_class,
)
from tools.build_p2_assets import (  # noqa: E402
    _drawer,
    _hook,
    _open_box,
    compose_urdf,  # noqa: E402
)

logger = logging.getLogger(__name__)

GENERATOR = "tools/build_p3_assets.py"


def _rot_x(mesh: "trimesh.Trimesh", angle: float) -> "trimesh.Trimesh":
    """Rotate a mesh about the X axis (for slanted shelves etc.)."""
    import trimesh

    mesh = mesh.copy()
    mesh.apply_transform(
        trimesh.transformations.rotation_matrix(angle, [1.0, 0.0, 0.0])
    )
    return mesh


# ---------------------------------------------------------------------------
# rigid class builders
# ---------------------------------------------------------------------------


def build_bedside_wall_shelf() -> list:
    """Bedside wall shelf with front lip (0.30 x 0.15 m board)."""
    parts = [
        _box((0.30, 0.015, 0.15), (0.0, 0.0075, 0.0)),
        _box((0.30, 0.03, 0.008), (0.0, 0.022, 0.071)),  # front lip
        _rod((-0.13, 0.0, -0.06), (-0.13, -0.08, -0.06), 0.005),  # brackets
        _rod((0.13, 0.0, -0.06), (0.13, -0.08, -0.06), 0.005),
    ]
    return [(_ground(_concat(parts)), {"source": "parametric:bedside-shelf"})]


def build_bookend_stand() -> list:
    """L-shaped bookends (left/right pair as two instances)."""

    def bookend(mirror: float) -> "trimesh.Trimesh":
        return _concat(
            [
                _box((0.12, 0.14, 0.005), (0.0, 0.07, mirror * 0.05)),  # upright plate
                _box((0.12, 0.005, 0.10), (0.0, 0.0025, 0.0)),  # base tongue
            ]
        )

    return [
        (_ground(bookend(1.0)), {"source": "parametric:bookend-left"}),
        (_ground(bookend(-1.0)), {"source": "parametric:bookend-right"}),
    ]


def build_detergent_caddy() -> list:
    """Laundry detergent caddy: open basket + carry handle."""
    body = _open_box(0.30, 0.20, 0.15, t=0.005)
    handle = _hook(
        [(-0.12, 0.15, 0.0), (-0.12, 0.21, 0.0), (0.12, 0.21, 0.0), (0.12, 0.15, 0.0)],
        r=0.006,
    )
    return [
        (_ground(_concat([body, handle])), {"source": "parametric:detergent-caddy"})
    ]


def build_fridge_side_rack() -> list:
    """2-tier rack hanging over the fridge side wall (0.25 x 0.10 m shelves)."""
    parts = []
    for x in (-0.115, 0.115):
        parts.append(_rod((x, 0.0, -0.04), (x, 0.30, -0.04), 0.005))  # back posts
    for y in (0.05, 0.20):
        parts.append(_box((0.25, 0.008, 0.10), (0.0, y, 0.0)))  # shelves
        parts.append(
            _rod((-0.125, y + 0.04, 0.045), (0.125, y + 0.04, 0.045), 0.004)
        )  # rails
    for x in (-0.10, 0.10):  # hang-over hooks at top
        parts.append(
            _hook(
                [
                    (x, 0.30, -0.04),
                    (x, 0.34, -0.04),
                    (x, 0.34, -0.09),
                    (x, 0.26, -0.09),
                ],
                r=0.005,
            )
        )
    return [(_ground(_concat(parts)), {"source": "parametric:fridge-side-rack"})]


def build_hairdryer_set() -> list:
    """Hair dryer (pistol grip + barrel) + wall ring holder."""
    dryer = _concat(
        [
            _rod((0.0, 0.0, 0.0), (0.0, 0.16, 0.02), 0.022),  # handle
            _rod((0.0, 0.16, 0.02), (0.0, 0.22, 0.14), 0.045),  # barrel
            _rod((0.0, 0.22, 0.14), (0.0, 0.225, 0.16), 0.05),  # nozzle rim
        ]
    )
    import trimesh

    ring = trimesh.creation.torus(0.055, 0.006)
    ring.apply_transform(
        trimesh.transformations.rotation_matrix(math.pi / 2, [1, 0, 0])
    )
    ring.apply_translation((0.0, 0.10, 0.02))
    holder = _concat(
        [
            _box((0.10, 0.12, 0.012), (0.0, 0.10, 0.0)),  # wall plate
            ring,
        ]
    )
    return [
        (_ground(dryer), {"source": "parametric:hairdryer"}),
        (_ground(holder), {"source": "parametric:dryer-ring-holder"}),
    ]


def build_kitchen_utensil_set() -> list:
    """Hanging utensils for the wall rail: spatula / ladle / whisk."""
    spatula = _concat(
        [
            _rod((0.0, 0.10, 0.0), (0.0, 0.36, 0.0), 0.008),  # handle
            _box((0.07, 0.10, 0.004), (0.0, 0.05, 0.0)),  # blade
        ]
    )
    ladle = _concat(
        [
            _rod((0.0, 0.07, 0.0), (0.0, 0.35, 0.0), 0.007),
            _frustum(0.035, 0.045, 0.03, sections=24),  # bowl (concave-ish cup)
        ]
    )
    whisk = _concat(
        [
            _rod((0.0, 0.12, 0.0), (0.0, 0.34, 0.0), 0.009),  # handle
            _frustum(0.004, 0.032, 0.11, sections=16),  # cage silhouette
        ]
    )
    keep = {"keep_scale": True}
    return [
        (_ground(spatula), {"source": "parametric:spatula", **keep}),
        (_ground(ladle), {"source": "parametric:ladle", **keep}),
        (_ground(whisk), {"source": "parametric:whisk", **keep}),
    ]


def build_magnetic_knife_rack() -> list:
    """Magnetic knife bar with two mounting blocks (0.35 m)."""
    parts = [
        _box((0.35, 0.045, 0.012), (0.0, 0.0225, 0.0)),
        _box((0.03, 0.06, 0.02), (-0.14, 0.03, -0.008)),
        _box((0.03, 0.06, 0.02), (0.14, 0.03, -0.008)),
    ]
    return [
        (
            _ground(_concat(parts)),
            {"source": "parametric:magnetic-bar-35", "keep_scale": True},
        )
    ]


def build_over_toilet_shelf() -> list:
    """Over-toilet rack: tall frame, bottom clearance, 2 shelves (1.5 m)."""
    parts = []
    for x in (-0.28, 0.28):
        for z in (-0.115, 0.115):
            parts.append(_rod((x, 0.0, z), (x, 1.50, z), 0.008))
    parts.append(
        _rod((-0.28, 0.15, -0.115), (0.28, 0.15, -0.115), 0.006)
    )  # rear crossbar
    for y in (0.85, 1.25):
        parts.append(_box((0.60, 0.012, 0.25), (0.0, y, 0.0)))
    return [
        (
            _ground(_concat(parts)),
            {"source": "parametric:over-toilet-2tier", "keep_scale": True},
        )
    ]


def build_pegboard() -> list:
    """Pegboard panel with a 6x4 peg grid (0.40 x 0.30 m)."""
    parts = [_box((0.40, 0.30, 0.008), (0.0, 0.15, 0.0))]
    for i in range(6):
        for j in range(4):
            x = -0.15 + i * 0.06
            y = 0.06 + j * 0.06
            parts.append(_rod((x, y, 0.004), (x, y, 0.028), 0.003))
    return [
        (
            _ground(_concat(parts)),
            {"source": "parametric:pegboard-6x4", "keep_scale": True},
        )
    ]


def build_picture_book_rack() -> list:
    """Front-display picture-book rack: 3 slanted shelves."""
    parts = []
    for y, z in ((0.12, 0.10), (0.30, 0.16), (0.48, 0.22)):
        shelf = _rot_x(_box((0.50, 0.012, 0.16), (0.0, y, z)), -0.35)
        parts.append(shelf)
        parts.append(_rod((-0.24, y - 0.06, z - 0.05), (-0.24, y, z), 0.006))
        parts.append(_rod((0.24, y - 0.06, z - 0.05), (0.24, y, z), 0.006))
    parts.append(_box((0.52, 0.02, 0.30), (0.0, 0.01, 0.14)))  # base
    return [
        (
            _ground(_concat(parts)),
            {"source": "parametric:picture-book-3tier", "keep_scale": True},
        )
    ]


def build_toothbrush_set() -> list:
    """Wall-mount toothbrush holder (cup + plate) + toothbrush."""
    holder = _concat(
        [
            _box((0.10, 0.08, 0.010), (0.0, 0.10, 0.0)),  # wall plate
            _frustum(0.030, 0.035, 0.09, sections=24),  # cup
        ]
    )
    brush = _concat(
        [
            _box((0.015, 0.15, 0.010), (0.0, 0.075, 0.0)),  # handle
            _box((0.018, 0.030, 0.014), (0.0, 0.160, 0.0)),  # head
            _box((0.014, 0.012, 0.008), (0.0, 0.152, 0.008)),  # bristles
        ]
    )
    return [
        (_ground(holder), {"source": "parametric:brush-wall-holder"}),
        (_ground(brush), {"source": "parametric:toothbrush"}),
    ]


def build_trash_bag_dispenser() -> list:
    """Trash-bag roll dispenser: box + slotted lid (two lid halves)."""
    body = _open_box(0.20, 0.12, 0.12, t=0.004)
    lid = _concat(
        [
            _box((0.20, 0.004, 0.045), (0.0, 0.122, -0.035)),
            _box((0.20, 0.004, 0.045), (0.0, 0.122, 0.035)),
        ]
    )  # 0.03 m dispensing slot between the halves
    return [(_ground(_concat([body, lid])), {"source": "parametric:bag-dispenser"})]


def build_tube_organizer() -> list:
    """Upright organizer for tube toiletries (3 narrow slots)."""
    return [
        (
            _ground(_open_box(0.15, 0.10, 0.12, t=0.004, dividers_x=2)),
            {"source": "parametric:tube-organizer-3slot"},
        )
    ]


def build_umbrella_set() -> list:
    """Umbrella drain stand (open frustum) + folded umbrella."""
    stand = _frustum(0.10, 0.12, 0.30, sections=32)
    umbrella = _concat(
        [
            _frustum(0.008, 0.035, 0.42, sections=16),  # folded canopy
            _rod((0.0, 0.42, 0.0), (0.0, 0.52, 0.0), 0.006),  # tip
            _hook(
                [
                    (0.0, 0.0, 0.0),
                    (0.0, -0.06, 0.0),
                    (0.03, -0.09, 0.0),
                    (0.055, -0.06, 0.0),
                ],
                r=0.006,
            ),
        ]
    )
    return [
        (_ground(stand), {"source": "parametric:umbrella-stand"}),
        (
            _ground(umbrella),
            {"source": "parametric:folded-umbrella", "keep_scale": True},
        ),
    ]


# ---------------------------------------------------------------------------
# articulated URDF builders (compose_urdf from build_p2_assets)
# ---------------------------------------------------------------------------


def _lid(
    name: str, size: str, hinge_xyz: str, geo_xyz: str, upper: float = 1.9
) -> dict:
    """Top lid hinged at the back top edge (revolute about X, opens backward)."""
    return {
        "link_name": name,
        "geometry": f'<box size="{size}"/>',
        "geo_xyz": geo_xyz,
        "mass": 1.0,
        "joint_type": "revolute",
        "joint_xyz": hinge_xyz,
        "axis": "1 0 0",
        "upper": upper,
    }


def build_desk_cart_urdf() -> str:
    """0.40 x 0.30 x 0.60 m desktop storage cart, four continuous wheels."""
    wheels = []
    for i, (x, y) in enumerate(
        ((-0.16, -0.11), (0.16, -0.11), (-0.16, 0.11), (0.16, 0.11))
    ):
        wheels.append(
            {
                "link_name": f"wheel_{i}",
                "geometry": '<cylinder radius="0.035" length="0.02"/>',
                "geo_xyz": "0 0 0",
                "geo_rpy": f"0 {math.pi / 2:.4f} 0",
                "mass": 0.2,
                "joint_type": "continuous",
                "joint_xyz": f"{x} {y} 0.035",
                "axis": "1 0 0",
            }
        )
    return compose_urdf("desk_cart", "0.40 0.30 0.60", 5.0, wheels)


def build_jewelry_box_urdf() -> str:
    """0.20 x 0.14 x 0.08 m jewelry box with a hinged lid."""
    return compose_urdf(
        "jewelry_box",
        "0.20 0.14 0.08",
        0.8,
        [
            _lid(
                "lid", "0.20 0.14 0.02", hinge_xyz="0 -0.07 0.08", geo_xyz="0 0.07 0.01"
            )
        ],
    )


def _rotating_tray(name: str, radius: float) -> str:
    """Pedestal base + freely rotating top disk (continuous joint about Z)."""
    return compose_urdf(
        name,
        f"{radius * 2:.2f} {radius * 2:.2f} 0.03",
        0.8,
        [
            {
                "link_name": "tray",
                "geometry": f'<cylinder radius="{radius}" length="0.025"/>',
                "geo_xyz": "0 0 0",
                "mass": 0.6,
                "joint_type": "continuous",
                "joint_xyz": "0 0 0.045",
                "axis": "0 0 1",
            }
        ],
    )


def build_rotating_desk_tray_urdf() -> str:
    """Rotating desktop organizer tray (radius 0.12 m)."""
    return _rotating_tray("rotating_desk_tray", 0.12)


def build_rotating_spice_tray_urdf() -> str:
    """Rotating spice turntable (radius 0.15 m)."""
    return _rotating_tray("rotating_spice_tray", 0.15)


def build_side_table_cabinet_urdf() -> str:
    """0.40 x 0.35 x 0.55 m side-table cabinet with one drawer."""
    return compose_urdf(
        "side_table_cabinet",
        "0.40 0.35 0.55",
        12.0,
        [_drawer("drawer", 0.30, "0.34 0.30 0.16", upper=0.25)],
    )


def build_storage_ottoman_urdf() -> str:
    """0.45 x 0.45 x 0.35 m storage ottoman with a hinged lid."""
    return compose_urdf(
        "storage_ottoman",
        "0.45 0.45 0.35",
        8.0,
        [
            _lid(
                "lid",
                "0.45 0.45 0.04",
                hinge_xyz="0 -0.225 0.35",
                geo_xyz="0 0.225 0.02",
            )
        ],
    )


RIGID_BUILDERS = {
    "bedside_wall_shelf": build_bedside_wall_shelf,
    "bookend_stand": build_bookend_stand,
    "detergent_caddy": build_detergent_caddy,
    "fridge_side_rack": build_fridge_side_rack,
    "hairdryer_set": build_hairdryer_set,
    "kitchen_utensil_set": build_kitchen_utensil_set,
    "magnetic_knife_rack": build_magnetic_knife_rack,
    "over_toilet_shelf": build_over_toilet_shelf,
    "pegboard": build_pegboard,
    "picture_book_rack": build_picture_book_rack,
    "toothbrush_set": build_toothbrush_set,
    "trash_bag_dispenser": build_trash_bag_dispenser,
    "tube_organizer": build_tube_organizer,
    "umbrella_set": build_umbrella_set,
}
URDF_BUILDERS = {
    "desk_cart": build_desk_cart_urdf,
    "jewelry_box": build_jewelry_box_urdf,
    "rotating_desk_tray": build_rotating_desk_tray_urdf,
    "rotating_spice_tray": build_rotating_spice_tray_urdf,
    "side_table_cabinet": build_side_table_cabinet_urdf,
    "storage_ottoman": build_storage_ottoman_urdf,
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
        help="subset of classes to build (default: all 20)",
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
