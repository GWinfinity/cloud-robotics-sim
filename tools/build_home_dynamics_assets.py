#!/usr/bin/env python3
"""Build household fracture, PBD cloth and articulated URDF assets.

Usage::

    uv run python tools/build_home_dynamics_assets.py
    uv run python tools/build_home_dynamics_assets.py --classes glass_cup_fragments
    uv run python tools/build_home_dynamics_assets.py --force

Fragment generation requires ``uv sync --extra fracture``. Output defaults to
``outputs/home_physics_assets`` alongside the rigid household batches.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cloud_robotics_sim.robotwin.cloth import (  # noqa: E402
    CLOTH_SPECS,
    export_cloth_mesh,
)
from cloud_robotics_sim.robotwin.fracture import voronoi_fragments  # noqa: E402
from tools.build_home_physics_assets import build_glass_cup  # noqa: E402
from tools.build_p1_assets import write_glb_class, write_urdf_class  # noqa: E402
from tools.build_p2_assets import compose_urdf  # noqa: E402

logger = logging.getLogger(__name__)
GENERATOR = "tools/build_home_dynamics_assets.py"
BUILDERS = {"glass_cup_fragments", "washing_machine", *CLOTH_SPECS}


def build_washing_machine_urdf() -> str:
    """Build a fixed front-loading cabinet with a left-hinged circular door."""
    return compose_urdf(
        "washing_machine",
        "0.60 0.60 0.85",
        45.0,
        [
            {
                "link_name": "door",
                "geometry": '<cylinder radius="0.20" length="0.035"/>',
                "geo_xyz": "0.20 0.015 0.0",
                "geo_rpy": "1.570796 0 0",
                "mass": 4.0,
                "joint_type": "revolute",
                "joint_xyz": "-0.20 0.315 0.45",
                "axis": "0 0 1",
                "upper": 1.9,
            }
        ],
    )


def _existing_class(objects_dir: Path, name: str) -> str:
    matches = [
        path.name
        for path in objects_dir.iterdir()
        if path.is_dir() and path.name[:3].isdigit() and path.name[3:] == f"_{name}"
    ]
    if len(matches) > 1:
        raise ValueError(f"duplicate generated class {name}: {sorted(matches)}")
    return matches[0] if matches else name


def _build_fragments(output_dir: Path, force: bool) -> str:
    objects_dir = output_dir / "objects"
    objects_dir.mkdir(parents=True, exist_ok=True)
    cup = build_glass_cup()[0][0]
    fragments = voronoi_fragments(cup)
    instances = [
        (
            fragment,
            {
                "source": "parametric:glass-cup-voronoi-12",
                "physics_kind": "fragile_rigid",
                "keep_scale": True,
                "fragment_index": index,
                "fragment_center": fragment.center_mass.tolist(),
                "fragment_volume_m3": fragment.volume,
                "parent_volume_m3": cup.volume,
            },
        )
        for index, fragment in enumerate(fragments)
    ]
    return write_glb_class(
        objects_dir,
        _existing_class(objects_dir, "glass_cup_fragments"),
        instances,
        force,
        generator=GENERATOR,
    )


def main(argv: list[str] | None = None) -> int:
    """Build selected dynamics assets; return 1 for unknown classes."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/home_physics_assets",
        help="asset output root (default: %(default)s)",
    )
    parser.add_argument(
        "--classes",
        nargs="*",
        default=sorted(BUILDERS),
        help=f"subset of assets to build (default: all {len(BUILDERS)})",
    )
    parser.add_argument("--force", action="store_true", help="rebuild existing assets")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    unknown = set(args.classes) - BUILDERS
    if unknown:
        logger.error("unknown classes %s (have %s)", sorted(unknown), sorted(BUILDERS))
        return 1
    output_dir = Path(args.output_dir)
    built = []
    try:
        for name in args.classes:
            if name == "glass_cup_fragments":
                built.append(_build_fragments(output_dir, args.force))
            elif name == "washing_machine":
                objects_dir = output_dir / "objects"
                objects_dir.mkdir(parents=True, exist_ok=True)
                built.append(
                    write_urdf_class(
                        objects_dir,
                        _existing_class(objects_dir, name),
                        build_washing_machine_urdf(),
                        args.force,
                        generator=GENERATOR,
                    )
                )
            else:
                path = output_dir / "cloth" / f"{name}.obj"
                if path.exists() and not args.force:
                    logger.info("skip existing cloth %s (use --force to rebuild)", path)
                else:
                    export_cloth_mesh(CLOTH_SPECS[name], path)
                    logger.info("built cloth %s", path)
                built.append(name)
    except (ImportError, ValueError, RuntimeError) as exc:
        logger.error("%s", exc)
        return 1
    print(f"built {len(built)} asset(s): {', '.join(built)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
