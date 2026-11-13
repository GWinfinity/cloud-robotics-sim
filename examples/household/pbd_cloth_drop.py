"""Settle selected household cloths with Genesis's native PBD cloth solver.

Build the requested OBJ files with ``tools/build_home_dynamics_assets.py`` first.
The optional ``--pin-corner`` holds one rag corner as a fixed grasp-point proxy.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from cloud_robotics_sim.robotwin.cloth import (  # noqa: E402
    CLOTH_SPECS,
    particle_positions,
    spawn_pbd_cloth,
)
from cloud_robotics_sim.utils.genesis_compat import get_genesis_backend  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    """Simulate selected cloth assets and validate finite, non-penetrating state."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-dir", default="outputs/home_physics_assets/cloth")
    parser.add_argument(
        "--classes",
        nargs="+",
        choices=sorted(CLOTH_SPECS),
        default=["rag", "towel"],
        help="cloth assets to simulate (default: rag towel)",
    )
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--backend", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--pin-corner", action="store_true")
    args = parser.parse_args(argv)
    if args.pin_corner and "rag" not in args.classes:
        parser.error("--pin-corner requires rag in --classes")

    import genesis as gs

    get_genesis_backend(args.backend)
    gs.init(
        backend=gs.gpu if args.backend == "cuda" else gs.cpu,
        logging_level="warning",
    )
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.004, substeps=10),
        pbd_options=gs.options.PBDOptions(
            particle_size=0.01,
            max_stretch_solver_iterations=8,
            max_bending_solver_iterations=4,
        ),
        show_viewer=False,
    )
    scene.add_entity(gs.morphs.Plane())
    entities = {}
    spawn_positions = {}
    xs = np.linspace(-0.25, 0.25, len(args.classes))
    for name, x in zip(args.classes, xs, strict=True):
        pos = (float(x), 0.0, 0.25 if name == "rag" else 0.35)
        spawn_positions[name] = pos
        entities[name] = spawn_pbd_cloth(
            scene,
            CLOTH_SPECS[name],
            Path(args.asset_dir) / f"{name}.obj",
            pos,
        )
    scene.build()
    if args.pin_corner:
        rag = entities["rag"]
        spec = CLOTH_SPECS["rag"]
        x, y, z = spawn_positions["rag"]
        corner = rag.find_closest_particle(
            (x - spec.width / 2.0, y - spec.depth / 2.0, z)
        )
        rag.fix_particles(corner)

    for _ in range(args.steps):
        scene.step()
    result = {}
    ok = True
    for name, entity in entities.items():
        positions = particle_positions(entity)
        finite = bool(np.isfinite(positions).all())
        min_z = float(positions[:, 2].min())
        max_z = float(positions[:, 2].max())
        valid = finite and min_z > -0.02 and max_z < 0.5
        result[name] = {
            "particles": len(positions),
            "finite": finite,
            "min_z": min_z,
            "max_z": max_z,
            "ok": valid,
        }
        ok = ok and valid
    print(json.dumps(result))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
