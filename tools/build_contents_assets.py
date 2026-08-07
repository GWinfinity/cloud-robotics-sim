#!/usr/bin/env python3
"""Build the daily-use contents assets (收纳内容物批) as parametric geometry.

Fills the "half-full" content gaps of the storage assets (see
``data/recipes/asset_gap_list.yaml`` notes): the small rigid daily items
that live inside the organizers — chopsticks/spoons for drawer dividers,
keys for the pegboard/key tray, glasses for the bedside shelf, pot lids
for the pot rack, eggs for the fridge, etc. All rigid GLB classes,
self-built and registered Apache-2.0.

Usage::

    uv run python tools/build_contents_assets.py              # build all 10
    uv run python tools/build_contents_assets.py --classes chopsticks egg_set
    uv run python tools/build_contents_assets.py --force
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
    write_glb_class,  # noqa: E402
)
from tools.build_p2_assets import _hook, _open_box  # noqa: E402

logger = logging.getLogger(__name__)

GENERATOR = "tools/build_contents_assets.py"


def _flat_sphere(radius: float, y_stretch: float = 1.3) -> "trimesh.Trimesh":
    """Ellipsoid (icosphere stretched along Y) — eggs and round knobs."""
    import trimesh

    mesh = trimesh.creation.icosphere(subdivisions=2, radius=radius)
    mesh.apply_transform(
        trimesh.transformations.scale_matrix(y_stretch, [0, 0, 0], [0, 1, 0])
    )
    return mesh


def _flatten_z(mesh: "trimesh.Trimesh", factor: float) -> "trimesh.Trimesh":
    """Squash a mesh along Z (spoon bowls, spatula-like curves)."""
    import trimesh

    mesh = mesh.copy()
    mesh.apply_transform(
        trimesh.transformations.scale_matrix(factor, [0, 0, 0], [0, 0, 1])
    )
    return mesh


# ---------------------------------------------------------------------------
# builders
# ---------------------------------------------------------------------------


def build_chopsticks() -> list:
    """Chopstick pairs (two parallel tapered-look rods)."""

    def pair(length: float, r: float) -> "trimesh.Trimesh":
        return _concat(
            [
                _rod((-0.005, 0.0, 0.0), (-0.005, length, 0.0), r),
                _rod((0.005, 0.0, 0.0), (0.005, length, 0.0), r),
            ]
        )

    return [
        (_ground(pair(0.24, 0.0035)), {"source": "parametric:chopsticks-bamboo"}),
        (_ground(pair(0.27, 0.0035)), {"source": "parametric:chopsticks-wood-long"}),
    ]


def build_comb() -> list:
    """Comb: spine + 14 teeth (0.16 m)."""
    parts = [_box((0.16, 0.015, 0.003), (0.0, 0.0375, 0.0))]  # spine
    for i in range(14):
        x = -0.072 + i * 0.011
        parts.append(_box((0.003, 0.030, 0.002), (x, 0.015, 0.0)))  # tooth
    return [(_ground(_concat(parts)), {"source": "parametric:comb-14tooth"})]


def build_egg_set() -> list:
    """Two eggs (regular / large) + a 6-cell egg carton."""
    carton = _open_box(0.15, 0.10, 0.045, t=0.003, dividers_x=2, dividers_z=1)
    return [
        (_ground(_flat_sphere(0.022)), {"source": "parametric:egg-regular"}),
        (_ground(_flat_sphere(0.024, 1.25)), {"source": "parametric:egg-large"}),
        (_ground(carton), {"source": "parametric:egg-carton-6cell"}),
    ]


def build_glasses() -> list:
    """Eyeglasses: two lens rims + bridge + folding temples."""
    import trimesh

    def rim(cx: float) -> "trimesh.Trimesh":
        ring = trimesh.creation.torus(0.024, 0.0018)
        ring.apply_translation((cx, 0.03, 0.0))  # lens plane XY, normal +z
        return ring

    parts = [
        rim(-0.028),
        rim(0.028),
        _rod((-0.006, 0.032, 0.0), (0.006, 0.032, 0.0), 0.0018),  # bridge
    ]
    for side in (-1, 1):  # temples: from lens edge backwards, then bending down
        x = side * 0.052
        parts.append(_rod((x, 0.032, 0.0), (x, 0.034, -0.11), 0.0018))
        parts.append(_rod((x, 0.034, -0.11), (x, 0.014, -0.135), 0.0018))
    return [(_ground(_concat(parts)), {"source": "parametric:glasses-round"})]


def build_handbag() -> list:
    """Handbag: structured body + twin top handles (0.30 m)."""
    body = _box((0.30, 0.22, 0.10), (0.0, 0.11, 0.0))
    handles = _concat(
        [
            _hook(
                [
                    (-0.08, 0.22, -0.02),
                    (-0.08, 0.30, -0.02),
                    (0.08, 0.30, -0.02),
                    (0.08, 0.22, -0.02),
                ],
                r=0.008,
            ),
            _hook(
                [
                    (-0.08, 0.22, 0.02),
                    (-0.08, 0.30, 0.02),
                    (0.08, 0.30, 0.02),
                    (0.08, 0.22, 0.02),
                ],
                r=0.008,
            ),
        ]
    )
    return [(_ground(_concat([body, handles])), {"source": "parametric:handbag-tote"})]


def build_key_ring() -> list:
    """Key ring with two keys hanging from it."""
    import trimesh

    ring = trimesh.creation.torus(0.016, 0.0025)
    ring.apply_translation((0.0, 0.075, 0.0))  # ring in XY plane

    def key(x: float, z: float, length: float) -> "trimesh.Trimesh":
        head = trimesh.creation.cylinder(radius=0.011, height=0.002, sections=16)
        head.apply_transform(
            trimesh.transformations.rotation_matrix(math.pi / 2, [1, 0, 0])
        )
        head.apply_translation((x, 0.062, z))  # flat in XY, hanging from ring
        shaft = _box((0.008, length, 0.002), (x, 0.062 - 0.011 - length / 2, z))
        teeth = _concat(
            [
                _box(
                    (0.004, 0.004, 0.002),
                    (x + 0.005, 0.062 - 0.011 - length + 0.004, z),
                ),
                _box(
                    (0.004, 0.003, 0.002),
                    (x + 0.005, 0.062 - 0.011 - length + 0.011, z),
                ),
            ]
        )
        return _concat([head, shaft, teeth])

    parts = [ring, key(-0.008, 0.0, 0.038), key(0.008, 0.003, 0.045)]
    return [(_ground(_concat(parts)), {"source": "parametric:keyring-2keys"})]


def build_pot_lid() -> list:
    """Pot lids (0.28 / 0.32 m diameter) with knob handles."""
    import trimesh

    def lid(r: float) -> "trimesh.Trimesh":
        disk = _frustum(r * 0.96, r, 0.012, sections=40)
        rim = trimesh.creation.torus(r, 0.004)
        rim.apply_transform(
            trimesh.transformations.rotation_matrix(math.pi / 2, [1, 0, 0])
        )
        rim.apply_translation((0.0, 0.010, 0.0))
        knob = _flat_sphere(0.018, 0.8)
        knob.apply_translation((0.0, 0.024, 0.0))
        stem = _rod((0.0, 0.010, 0.0), (0.0, 0.020, 0.0), 0.006)
        return _concat([disk, rim, knob, stem])

    return [
        (_ground(lid(0.14)), {"source": "parametric:pot-lid-28cm"}),
        (_ground(lid(0.16)), {"source": "parametric:pot-lid-32cm"}),
    ]


def build_power_strip() -> list:
    """Power strip: body + 3 socket faces + switch + cable stub (0.25 m)."""
    parts = [_box((0.25, 0.030, 0.070), (0.0, 0.015, 0.0))]
    for i in range(3):  # socket plates
        x = -0.075 + i * 0.06
        parts.append(_box((0.040, 0.004, 0.050), (x, 0.032, 0.0)))
        parts.append(_box((0.004, 0.006, 0.012), (x - 0.008, 0.034, 0.0)))  # slots
        parts.append(_box((0.004, 0.006, 0.012), (x + 0.008, 0.034, 0.0)))
    parts.append(_box((0.020, 0.006, 0.020), (0.105, 0.033, 0.0)))  # switch
    parts.append(_rod((0.125, 0.015, 0.0), (0.20, 0.012, 0.0), 0.005))  # cable stub
    return [(_ground(_concat(parts)), {"source": "parametric:power-strip-3way"})]


def build_scissors() -> list:
    """Scissors, slightly open: 2 blades + 2 finger rings + pivot."""
    import trimesh

    def half(angle: float) -> "trimesh.Trimesh":
        blade = _box((0.095, 0.012, 0.002), (0.055, 0.0, 0.0))  # pivot at origin
        ring = trimesh.creation.torus(0.016, 0.0035)
        ring.apply_translation((-0.045, 0.0, 0.0))  # ring in XY plane
        neck = _rod((-0.028, 0.0, 0.0), (-0.008, 0.0, 0.0), 0.004)
        half_mesh = _concat([blade, neck, ring])
        half_mesh.apply_transform(
            trimesh.transformations.rotation_matrix(angle, [0, 0, 1])
        )
        return half_mesh

    pivot = trimesh.creation.cylinder(radius=0.006, height=0.006, sections=12)
    pivot.apply_transform(
        trimesh.transformations.rotation_matrix(math.pi / 2, [1, 0, 0])
    )
    parts = [half(0.10), half(-0.10), pivot]
    return [(_ground(_concat(parts)), {"source": "parametric:scissors-19cm"})]


def build_spoon() -> list:
    """Spoons: soup spoon + teaspoon (handle + flattened bowl)."""

    def spoon(handle: float, bowl_r: float) -> "trimesh.Trimesh":
        bowl = _flatten_z(_frustum(bowl_r * 0.55, bowl_r, 0.014, sections=24), 0.7)
        grip = _rod((0.0, 0.010, 0.0), (0.0, handle, 0.0), 0.004)
        return _concat([bowl, grip])

    return [
        (_ground(spoon(0.16, 0.022)), {"source": "parametric:soup-spoon"}),
        (_ground(spoon(0.13, 0.016)), {"source": "parametric:teaspoon"}),
    ]


BUILDERS = {
    "chopsticks": build_chopsticks,
    "comb": build_comb,
    "egg_set": build_egg_set,
    "glasses": build_glasses,
    "handbag": build_handbag,
    "key_ring": build_key_ring,
    "pot_lid": build_pot_lid,
    "power_strip": build_power_strip,
    "scissors": build_scissors,
    "spoon": build_spoon,
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
        default=sorted(BUILDERS),
        help="subset of classes to build (default: all 10)",
    )
    parser.add_argument("--force", action="store_true", help="rebuild existing classes")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    objects_dir = Path(args.objects_dir)
    objects_dir.mkdir(parents=True, exist_ok=True)

    built: list[str] = []
    for name in args.classes:
        if name not in BUILDERS:
            logger.error("unknown class %s (have %s)", name, sorted(BUILDERS))
            return 1
        built.append(
            write_glb_class(
                objects_dir, name, BUILDERS[name](), args.force, generator=GENERATOR
            )
        )
    print(f"built {len(built)} class(es): {', '.join(built)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
