#!/usr/bin/env python3
"""Build household-physics trimesh batches as RoboTwin-OD assets.

The eight classes come from ``data/recipes/home_physics_assets.yaml``. Meshes use
real-world metric dimensions, Y-up coordinates and a grounded base. They are
self-built geometric stand-ins registered as Apache-2.0 through the common GLB
writer; no network access or external assets are required. Default output is an
isolated library under ``outputs/home_physics_assets/objects`` so a partial batch
never masquerades as the complete RoboTwin reference library. To merge into a
populated library, pass ``--objects-dir assets/robotwin/objects/objects``.

The clothes clip is a closed rigid approximation, not an articulated spring.
The sponge likewise uses the recipe's small-object rigid approximation.

Usage::

    uv run python tools/build_home_physics_assets.py
    uv run python tools/build_home_physics_assets.py --classes glass_cup sponge
    uv run python tools/build_home_physics_assets.py --force
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.build_p1_assets import (  # noqa: E402
    _box,
    _concat,
    _ground,
    _rod,
    write_glb_class,
)
from tools.build_p2_assets import _hook, _open_box  # noqa: E402

logger = logging.getLogger(__name__)

GENERATOR = "tools/build_home_physics_assets.py"
GLASS_WALL_THICKNESS_M = 0.002
DIMS_Y_UP = {
    "glass_cup": (0.07, 0.12, 0.07),
    "dustpan": (0.30, 0.12, 0.28),
    "mop_bucket": (0.48, 0.30, 0.28),
    "sponge": (0.11, 0.04, 0.07),
    "hanger": (0.42, 0.23, 0.10),
    "clothes_clip": (0.085, 0.035, 0.025),
    "cutting_board": (0.38, 0.02, 0.25),
    "laundry_basket": (0.45, 0.55, 0.35),
}


def _revolve(profile: list[tuple[float, float]], sections: int = 512):
    """Revolve a closed (radius, height) profile around the Y axis."""
    import trimesh

    profile_array = np.asarray(profile, dtype=float)
    n_profile = len(profile_array)
    angles = np.linspace(0.0, 2.0 * math.pi, sections, endpoint=False)
    radii = profile_array[:, 0]
    heights = profile_array[:, 1]
    vertices = np.column_stack(
        [
            np.tile(radii, sections) * np.repeat(np.cos(angles), n_profile),
            np.tile(heights, sections),
            np.tile(radii, sections) * np.repeat(np.sin(angles), n_profile),
        ]
    )
    faces = []
    for section in range(sections):
        next_section = (section + 1) % sections
        for point in range(n_profile):
            next_point = (point + 1) % n_profile
            a = section * n_profile + point
            b = section * n_profile + next_point
            c = next_section * n_profile + next_point
            d = next_section * n_profile + point
            faces.extend(([a, b, c], [a, c, d]))
    mesh = trimesh.Trimesh(
        vertices=vertices, faces=np.asarray(faces, dtype=np.int64), process=True
    )
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.remove_unreferenced_vertices()
    return mesh


def _fit_extents(mesh, extents: tuple[float, float, float]):
    """Apply small per-axis corrections so compound meshes match recipe dims."""
    transform = np.eye(4)
    for axis, target in enumerate(extents):
        transform[axis, axis] = target / mesh.extents[axis]
    mesh = mesh.copy()
    mesh.apply_transform(transform)
    return _ground(mesh)


def _metadata(source: str, physics_kind: str, **extra) -> dict:
    return {
        "source": source,
        "physics_kind": physics_kind,
        "keep_scale": True,
        **extra,
    }


def build_glass_cup() -> list:
    """Open glass tumbler with a 2 mm wall and a 6 mm solid base."""
    radius = DIMS_Y_UP["glass_cup"][0] / 2.0
    height = DIMS_Y_UP["glass_cup"][1]
    inner_radius = radius - GLASS_WALL_THICKNESS_M
    cup = _revolve(
        [
            (0.0, 0.0),
            (radius - 0.006, 0.0),
            (radius, 0.004),
            (radius, height),
            (inner_radius, height),
            (inner_radius, 0.006),
            (0.0, 0.006),
        ]
    )
    return [
        (
            _ground(cup),
            _metadata(
                "parametric:glass-tumbler-70x120mm",
                "fragile_rigid",
                wall_thickness_m=GLASS_WALL_THICKNESS_M,
            ),
        )
    ]


def build_dustpan() -> list:
    """Open dustpan with a low front lip and a raised rear handle."""
    body = _open_box(0.30, 0.18, 0.045, t=0.003, front_h=0.006)
    handle = _concat(
        [
            _rod((0.0, 0.040, -0.085), (0.0, 0.112, -0.182), 0.008),
            _rod((-0.05, 0.112, -0.182), (0.05, 0.112, -0.182), 0.004),
        ]
    )
    mesh = _fit_extents(_concat([body, handle]), DIMS_Y_UP["dustpan"])
    return [(mesh, _metadata("parametric:dustpan-low-lip", "rigid"))]


def build_mop_bucket() -> list:
    """Two-compartment mop bucket with a hollow wringer basket and handle."""
    body = _open_box(0.48, 0.28, 0.20, t=0.006, dividers_x=1)
    basket = _revolve(
        [
            (0.0, 0.10),
            (0.060, 0.10),
            (0.090, 0.26),
            (0.084, 0.26),
            (0.055, 0.106),
            (0.0, 0.106),
        ],
        sections=48,
    )
    basket.apply_translation((-0.12, 0.0, 0.0))
    handle = _hook(
        [
            (0.15, 0.19, -0.10),
            (0.15, 0.29, -0.10),
            (0.21, 0.29, -0.10),
            (0.21, 0.19, -0.10),
        ],
        r=0.005,
    )
    spinner_stem = _rod((-0.12, 0.10, 0.0), (-0.12, 0.275, 0.0), 0.005)
    mesh = _fit_extents(
        _concat([body, basket, handle, spinner_stem]), DIMS_Y_UP["mop_bucket"]
    )
    return [(mesh, _metadata("parametric:mop-bucket-double", "rigid"))]


def build_sponge() -> list:
    """Two-layer sponge and scouring pad, using the recipe's rigid proxy."""
    sponge = _box((0.11, 0.032, 0.07), (0.0, 0.016, 0.0))
    scour_pad = _box((0.11, 0.008, 0.07), (0.0, 0.036, 0.0))
    return [
        (
            _concat([sponge, scour_pad]),
            _metadata(
                "parametric:sponge-two-layer",
                "rigid",
                approximation="rigid_proxy_for_deformable_sponge",
            ),
        )
    ]


def build_hanger() -> list:
    """Triangular tubular hanger with a perpendicular swivel hook."""
    triangle = _hook(
        [(-0.207, 0.0, 0.0), (0.0, 0.14, 0.0), (0.207, 0.0, 0.0), (-0.207, 0.0, 0.0)],
        r=0.003,
    )
    swivel_hook = _hook(
        [
            (0.0, 0.14, 0.0),
            (0.0, 0.17, 0.0),
            (0.0, 0.215, 0.03),
            (0.0, 0.195, 0.075),
            (0.0, 0.16, 0.08),
        ],
        r=0.003,
    )
    mesh = _fit_extents(_concat([triangle, swivel_hook]), DIMS_Y_UP["hanger"])
    return [(mesh, _metadata("parametric:hanger-swivel-hook", "rigid"))]


def build_clothes_clip() -> list:
    """Closed clothespin proxy with two jaws and a visible spring coil."""
    import trimesh

    lower = _box((0.085, 0.012, 0.025), (0.0, 0.006, 0.0))
    lower.apply_transform(
        trimesh.transformations.rotation_matrix(-0.045, [0.0, 0.0, 1.0])
    )
    upper = _box((0.085, 0.012, 0.025), (0.0, 0.022, 0.0))
    upper.apply_transform(
        trimesh.transformations.rotation_matrix(0.045, [0.0, 0.0, 1.0])
    )
    coils = []
    for z in (-0.006, 0.0, 0.006):
        coil = trimesh.creation.torus(major_radius=0.009, minor_radius=0.0015)
        coil.apply_translation((0.0, 0.014, z))
        coils.append(coil)
    axle = _rod((0.0, 0.014, -0.011), (0.0, 0.014, 0.011), 0.002)
    mesh = _fit_extents(
        _concat([lower, upper, axle, *coils]), DIMS_Y_UP["clothes_clip"]
    )
    return [
        (
            mesh,
            _metadata(
                "parametric:clothes-clip-closed",
                "rigid",
                approximation="closed_rigid_no_spring_joint",
            ),
        )
    ]


def build_cutting_board() -> list:
    """Rounded board with a neck and an open circular hanging handle."""
    import trimesh

    parts = [
        _box((0.28, 0.02, 0.25), (-0.03, 0.01, 0.0)),
        _box((0.32, 0.02, 0.21), (-0.03, 0.01, 0.0)),
        _box((0.08, 0.02, 0.04), (0.13, 0.01, 0.0)),
    ]
    for x in (-0.17, 0.11):
        for z in (-0.105, 0.105):
            parts.append(_rod((x, 0.0, z), (x, 0.02, z), 0.02, sections=24))
    handle = trimesh.creation.torus(major_radius=0.014, minor_radius=0.004)
    handle.apply_transform(
        trimesh.transformations.rotation_matrix(math.pi / 2, [1.0, 0.0, 0.0])
    )
    handle.apply_translation((0.172, 0.01, 0.0))
    parts.append(handle)
    mesh = _fit_extents(_concat(parts), DIMS_Y_UP["cutting_board"])
    return [(mesh, _metadata("parametric:cutting-board-ring-handle", "rigid"))]


def build_laundry_basket() -> list:
    """Open laundry basket built from a base, vertical ribs and horizontal rims."""
    parts = [_box((0.45, 0.006, 0.35), (0.0, 0.003, 0.0))]
    for x in np.linspace(-0.20, 0.20, 7):
        for z in (-0.17, 0.17):
            parts.append(_rod((x, 0.004, z), (x, 0.52, z), 0.003))
    for z in np.linspace(-0.15, 0.15, 5):
        for x in (-0.22, 0.22):
            parts.append(_rod((x, 0.004, z), (x, 0.52, z), 0.003))
    for y, radius in (
        (0.04, 0.003),
        (0.18, 0.003),
        (0.32, 0.003),
        (0.50, 0.003),
        (0.54, 0.005),
    ):
        parts.extend(
            [
                _rod((-0.22, y, -0.17), (0.22, y, -0.17), radius),
                _rod((-0.22, y, 0.17), (0.22, y, 0.17), radius),
                _rod((-0.22, y, -0.17), (-0.22, y, 0.17), radius),
                _rod((0.22, y, -0.17), (0.22, y, 0.17), radius),
            ]
        )
    for x in (-0.22, 0.22):
        parts.append(
            _hook(
                [
                    (x, 0.47, -0.07),
                    (x, 0.545, -0.07),
                    (x, 0.545, 0.07),
                    (x, 0.47, 0.07),
                ],
                r=0.005,
            )
        )
    mesh = _fit_extents(_concat(parts), DIMS_Y_UP["laundry_basket"])
    return [
        (
            mesh,
            _metadata(
                "parametric:laundry-basket-open-grid",
                "rigid",
                construction="open_wire_grid",
            ),
        )
    ]


BUILDERS = {
    "clothes_clip": build_clothes_clip,
    "cutting_board": build_cutting_board,
    "dustpan": build_dustpan,
    "glass_cup": build_glass_cup,
    "hanger": build_hanger,
    "laundry_basket": build_laundry_basket,
    "mop_bucket": build_mop_bucket,
    "sponge": build_sponge,
}


def main(argv: list[str] | None = None) -> int:
    """Build all requested classes; return 1 for an unknown class name."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--objects-dir",
        default="outputs/home_physics_assets/objects",
        help="output object library root (default: %(default)s)",
    )
    parser.add_argument(
        "--classes",
        nargs="*",
        default=sorted(BUILDERS),
        help="subset of classes to build (default: all 8)",
    )
    parser.add_argument("--force", action="store_true", help="rebuild existing classes")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    objects_dir = Path(args.objects_dir)
    objects_dir.mkdir(parents=True, exist_ok=True)

    built = []
    for name in args.classes:
        if name not in BUILDERS:
            logger.error("unknown class %s (have %s)", name, sorted(BUILDERS))
            return 1
        existing = [
            path
            for path in objects_dir.iterdir()
            if path.is_dir() and path.name[:3].isdigit() and path.name[3:] == f"_{name}"
        ]
        if len(existing) > 1:
            logger.error(
                "duplicate class %s: %s", name, sorted(p.name for p in existing)
            )
            return 1
        class_name = existing[0].name if existing else name
        built.append(
            write_glb_class(
                objects_dir,
                class_name,
                BUILDERS[name](),
                args.force,
                generator=GENERATOR,
            )
        )
    print(f"built {len(built)} class(es): {', '.join(built)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
