"""Open the parametric washing-machine door through its revolute URDF joint.

Run ``tools/build_home_dynamics_assets.py --classes washing_machine`` first.
Genesis is initialized once per process.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from cloud_robotics_sim.robotwin.object_library import (  # noqa: E402
    RoboTwinObjectLibrary,
)
from cloud_robotics_sim.utils.genesis_compat import get_genesis_backend  # noqa: E402


def _array(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def main(argv: list[str] | None = None) -> int:
    """Drive the door joint and verify its settled position."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--objects-dir", default="outputs/home_physics_assets/objects")
    parser.add_argument(
        "--target", type=float, default=1.6, help="door angle in radians"
    )
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--backend", choices=["cuda", "cpu"], default="cuda")
    args = parser.parse_args(argv)
    if not 0.0 <= args.target <= 1.9:
        parser.error("--target must be between 0 and 1.9 radians")

    import genesis as gs

    get_genesis_backend(args.backend)
    gs.init(
        backend=gs.gpu if args.backend == "cuda" else gs.cpu,
        logging_level="warning",
    )
    lib = RoboTwinObjectLibrary(args.objects_dir)
    classes = [name for name in lib.list_classes() if name.endswith("_washing_machine")]
    if len(classes) != 1:
        raise ValueError(f"expected one washing_machine class, found {classes}")
    inst = lib.get_instance(classes[0])
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.004, substeps=10), show_viewer=False
    )
    scene.add_entity(gs.morphs.Plane())
    machine = scene.add_entity(
        gs.morphs.URDF(file=str(inst.asset_path), fixed=True),
        surface=gs.surfaces.Default(roughness=0.6),
    )
    scene.build()

    dofs = np.asarray(machine.get_joint("door_joint").dofs_idx_local, dtype=np.int64)
    machine.control_dofs_position(
        np.array([args.target], dtype=np.float64), dofs_idx_local=dofs
    )
    for _ in range(args.steps):
        scene.step()
    position = float(_array(machine.get_dofs_position(dofs)).reshape(-1)[0])
    ok = bool(np.isfinite(position) and abs(position - args.target) < 0.05)
    print(json.dumps({"target": args.target, "position": position, "ok": ok}))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
