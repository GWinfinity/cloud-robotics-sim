"""Run a closed-loop grasp simulation and compare full vs hierarchical QP."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

# Make src importable when running as a script.
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dexterous_gnn_qp.core.backends import create_backend
from dexterous_gnn_qp.core.controllers.pd_controller import (
    hand_desired_acceleration,
    object_desired_acceleration,
)
from dexterous_gnn_qp.core.metrics.benchmark import BenchmarkLogger, rotation_angle_between
from dexterous_gnn_qp.core.qp.full_qp import FullQP
from dexterous_gnn_qp.core.qp.hierarchical_qp import HierarchicalQP
from dexterous_gnn_qp.core.skeleton.rule_based import select_skeleton
from dexterous_gnn_qp.core.utils.config import load_config


def main():
    parser = argparse.ArgumentParser(description="Phase-1 benchmark: full vs hierarchical QP")
    parser.add_argument(
        "--config",
        "-c",
        default=str(Path(__file__).parent.parent / "configs" / "leap_sphere_grasp.yaml"),
    )
    parser.add_argument("--output", "-o", default=None)
    parser.add_argument("--headless", action="store_true", help="Disable interactive renderer")
    parser.add_argument(
        "--backend",
        "-b",
        default=None,
        choices=["mujoco", "genesis"],
        help="Override the backend specified in the config.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.backend is not None:
        cfg.sim.backend = args.backend
    if args.output:
        cfg.benchmark.log_path = args.output

    backend = create_backend(cfg)
    backend.set_initial_state(cfg)

    full_qp = FullQP(backend, cfg)
    hier_qp = HierarchicalQP(backend, cfg)
    logger = BenchmarkLogger(cfg.benchmark.log_path)

    obj_target_pos = np.array(cfg.object.position, dtype=float)
    obj_target_quat = np.array(cfg.object.quaternion, dtype=float)

    warm_full = None
    warm_hier = None

    n_steps = int(cfg.sim.n_steps)
    render = bool(cfg.sim.render) and not args.headless

    # MuJoCo-specific passive viewer; Genesis uses its own show_viewer flag.
    viewer_ctx = None
    if render and cfg.sim.backend == "mujoco":
        import mujoco.viewer

        viewer_ctx = mujoco.viewer.launch_passive(backend.model, backend.data)

    pbar = tqdm(range(n_steps), desc="simulating")
    for step in pbar:
        state = backend.compute_sim_state(
            mu=cfg.qp.mu,
            only_fingertips=cfg.contacts.only_fingertips,
        )

        qddot_des = hand_desired_acceleration(backend, state, cfg)
        a_obj_des = object_desired_acceleration(backend, state, cfg)

        warmup_steps = int(cfg.benchmark.get("warmup_steps", 0))
        if step < warmup_steps or len(state.contacts) == 0:
            # Warm-up / pre-contact: use pure PD computed torque to close the hand.
            tau = state.M_hand @ qddot_des + state.C_hand
            backend.apply_control(tau)
            if viewer_ctx is not None:
                viewer_ctx.sync()
            backend.step()
            continue

        # Full QP baseline.
        res_full = full_qp.solve(state, qddot_des, a_obj_des, warm_start=warm_full)
        warm_full = res_full.x

        # Rule-based skeleton selection using the current full QP forces.
        skeleton_indices, edge_indices = select_skeleton(state, res_full.forces, cfg)
        edge_forces = res_full.forces if len(edge_indices) > 0 else None

        # Hierarchical QP.
        res_hier = hier_qp.solve(
            state,
            skeleton_indices,
            edge_forces,
            qddot_des,
            a_obj_des,
            warm_start=warm_hier,
        )
        warm_hier = res_hier.x

        # Apply the hierarchical torque to the simulation.
        backend.apply_control(res_hier.tau)

        # Logging.
        obj_pos, obj_quat = backend.get_object_pose()
        pos_err = float(np.linalg.norm(obj_pos - obj_target_pos))
        rot_err = rotation_angle_between(obj_quat, obj_target_quat)
        logger.log(
            sim_time=backend.get_time(),
            n_contacts=len(state.contacts),
            n_skeleton=len(skeleton_indices),
            full_res=res_full,
            hier_res=res_hier,
            obj_pos_error=pos_err,
            obj_rot_error=rot_err,
        )

        if viewer_ctx is not None and step % int(cfg.sim.render_every) == 0:
            viewer_ctx.sync()

        backend.step()

        if step % 50 == 0:
            print(f"step={step} time={backend.get_time():.4f} obj_z={obj_pos[2]:.4f}")
        pbar.set_postfix(
            {
                "contacts": len(state.contacts),
                "skeleton": len(skeleton_indices),
                "full_ms": f"{res_full.solve_time * 1000:.2f}",
                "hier_ms": f"{res_hier.solve_time * 1000:.2f}",
            }
        )

    if viewer_ctx is not None:
        viewer_ctx.close()

    logger.save()
    summary = logger.summarize()
    obj_pos, _ = backend.get_object_pose()
    print(f"\nFinal sim time: {backend.get_time()}, obj_z: {obj_pos[2]}")
    print("\n=== Median metrics ===")
    for k, v in summary.items():
        print(f"{k:30s}: {v:.4f}")


if __name__ == "__main__":
    main()
