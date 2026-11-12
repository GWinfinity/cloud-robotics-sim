"""Drop a glass cup and rebuild it as Voronoi rigid fragments on impact.

Run ``tools/build_home_dynamics_assets.py`` first. Use ``--expect-intact`` for
a low-drop negative control. Genesis is initialized once per process.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from cloud_robotics_sim.robotwin.fracture import (  # noqa: E402
    FractureTrigger,
    fragment_velocities,
)
from cloud_robotics_sim.robotwin.object_library import (  # noqa: E402
    RoboTwinObjectLibrary,
)
from cloud_robotics_sim.utils.genesis_compat import get_genesis_backend  # noqa: E402


def _array(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _class_by_suffix(lib: RoboTwinObjectLibrary, suffix: str) -> str:
    matches = [name for name in lib.list_classes() if name.endswith(f"_{suffix}")]
    if len(matches) != 1:
        raise ValueError(f"expected one {suffix} class, found {matches}")
    return matches[0]


def _spawn_instance(
    scene,
    lib: RoboTwinObjectLibrary,
    class_name: str,
    index: int,
    pos: np.ndarray,
    quat: np.ndarray | None = None,
):
    return lib.spawn_in_scene(
        scene,
        class_name,
        index=index,
        pos=tuple(pos),
        quat=None if quat is None else tuple(quat),
    )


def _spawn_fragment(scene, gs, inst, pos: np.ndarray, quat: np.ndarray):
    return scene.add_entity(
        gs.morphs.Mesh(
            file=str(inst.asset_path),
            pos=tuple(pos),
            quat=tuple(quat),
            convexify=True,
            decimate=False,
            decompose_object_error_threshold=float("inf"),
        ),
        material=gs.materials.Rigid(rho=2500.0),
    )


def main(argv: list[str] | None = None) -> int:
    """Run intact impact and, on fracture, a fragment settlement phase."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--objects-dir", default="outputs/home_physics_assets/objects")
    parser.add_argument("--height", type=float, default=0.35)
    parser.add_argument("--impulse-threshold", type=float, default=0.08)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--settle-steps", type=int, default=300)
    parser.add_argument("--backend", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--expect-intact", action="store_true")
    args = parser.parse_args(argv)

    import genesis as gs

    get_genesis_backend(args.backend)
    gs.init(
        backend=gs.gpu if args.backend == "cuda" else gs.cpu,
        logging_level="warning",
    )
    lib = RoboTwinObjectLibrary(args.objects_dir)
    cup_class = _class_by_suffix(lib, "glass_cup")
    dt = 0.004
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=dt, substeps=10), show_viewer=False
    )
    scene.add_entity(gs.morphs.Plane())
    cup = _spawn_instance(
        scene,
        lib,
        cup_class,
        0,
        np.array([0.0, 0.0, args.height]),
    )
    scene.build()
    trigger = FractureTrigger(args.impulse_threshold)
    fractured = False
    for _ in range(args.steps):
        scene.step()
        if trigger.update(cup.get_contacts(), dt, cup.geom_start, cup.geom_end):
            fractured = True
            break

    result = {
        "fractured": fractured,
        "fracture_step": trigger.fracture_step,
        "peak_impulse": trigger.peak_impulse,
    }
    if not fractured or args.expect_intact:
        print(json.dumps(result))
        return 0 if fractured != args.expect_intact else 1

    pos = _array(cup.get_pos()).reshape(3)
    quat = _array(cup.get_quat()).reshape(4)
    linear = _array(cup.get_vel()).reshape(3)
    angular = _array(cup.get_ang()).reshape(3)
    scene.destroy()

    fragment_class = _class_by_suffix(lib, "glass_cup_fragments")
    instances = [
        lib.get_instance(fragment_class, i)
        for i in lib.instance_indices(fragment_class)
    ]
    centers = [inst.metadata["fragment_center"] for inst in instances]
    velocities = fragment_velocities(linear, angular, centers)
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=dt, substeps=10), show_viewer=False
    )
    scene.add_entity(gs.morphs.Plane())
    fragments = [_spawn_fragment(scene, gs, inst, pos, quat) for inst in instances]
    scene.build()
    for fragment, velocity in zip(fragments, velocities):
        fragment.set_dofs_velocity(np.concatenate([velocity, angular]))
    for _ in range(args.settle_steps):
        scene.step()
    qpos = [_array(fragment.get_qpos()) for fragment in fragments]
    finite = all(np.isfinite(value).all() for value in qpos)
    min_z = min(float(value[..., 2].min()) for value in qpos)
    result.update(
        {"fragments": len(fragments), "finite": finite, "min_fragment_z": min_z}
    )
    print(json.dumps(result))
    return 0 if finite and min_z > -0.05 else 1


if __name__ == "__main__":
    sys.exit(main())
