"""Spawn smoke test for imported generated objects (GPU, one-off verification).

Loads every instance of a class from an objects dir via
``RoboTwinObjectLibrary.spawn_in_scene``, lets the objects settle on a plane,
and checks that all of them rest at a finite, plausible height.

Usage:
    python scripts/smoke_imported_objects.py \
        --objects-dir outputs/import_smoke/objects --class-name 001_smoke-gen
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root / "src"))

from cloud_robotics_sim.robotwin.object_library import (  # noqa: E402
    RoboTwinObjectLibrary,
)
from cloud_robotics_sim.utils.genesis_compat import get_genesis_backend  # noqa: E402


def main() -> int:
    """Spawn all instances, settle, and check resting heights."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--objects-dir", required=True)
    parser.add_argument("--class-name", required=True)
    parser.add_argument("--steps", type=int, default=300)
    args = parser.parse_args()

    import genesis as gs  # noqa: E402

    get_genesis_backend("cuda")
    gs.init(backend=gs.gpu, logging_level="warning")

    lib = RoboTwinObjectLibrary(args.objects_dir)
    indices = lib.instance_indices(args.class_name)
    print(f"class {args.class_name}: instances {indices}")

    scene = gs.Scene(show_viewer=False)
    scene.add_entity(gs.morphs.Plane())
    entities = []
    for i, idx in enumerate(indices):
        pos = (0.3 * i - 0.3 * (len(indices) - 1) / 2, 0.0, 0.25)
        entities.append(lib.spawn_in_scene(scene, args.class_name, index=idx, pos=pos))
    scene.build()

    for _ in range(args.steps):
        scene.step()

    ok = True
    for idx, ent in zip(indices, entities):
        qpos = ent.get_qpos()
        qpos = (
            qpos.detach().cpu().numpy() if hasattr(qpos, "detach") else np.asarray(qpos)
        )
        z = float(qpos[2])
        finite = bool(np.isfinite(qpos).all())
        # qpos is reported at the link (CoM) frame: tall items (keep_scale)
        # legitimately rest standing upright (CoM up to ~full height), so
        # the plausible range is "not sunk, not flung away": [-0.05, size].
        inst = lib.get_instance(args.class_name, index=idx)
        z_limit = 0.20
        if inst.metadata.get("keep_scale"):
            z_limit = max(inst.scaled_extents)
        resting = -0.05 < z < z_limit
        status = "OK" if (finite and resting) else "FAIL"
        ok = ok and finite and resting
        print(
            f"instance {idx}: z={z:.4f} (limit {z_limit:.2f}) finite={finite} -> {status}"
        )
    print("SUCCESS" if ok else "FAILURE")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
