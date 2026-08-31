"""Single-step demo comparing the hierarchical QP to the full QP baseline."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dexterous_gnn_qp.core.backends import create_backend
from dexterous_gnn_qp.core.controllers.pd_controller import (
    hand_desired_acceleration,
    object_desired_acceleration,
)
from dexterous_gnn_qp.core.qp.full_qp import FullQP
from dexterous_gnn_qp.core.qp.hierarchical_qp import HierarchicalQP
from dexterous_gnn_qp.core.skeleton.rule_based import select_skeleton
from dexterous_gnn_qp.core.utils.config import load_config


def main():
    parser = argparse.ArgumentParser(description="Hierarchical QP single-step demo")
    parser.add_argument(
        "--config",
        "-c",
        default=str(Path(__file__).parent.parent / "configs" / "leap_sphere_grasp.yaml"),
    )
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

    backend = create_backend(cfg)
    backend.set_initial_state(cfg)

    state = backend.compute_sim_state(
        mu=cfg.qp.mu,
        only_fingertips=cfg.contacts.only_fingertips,
    )
    qddot_des = hand_desired_acceleration(backend, state, cfg)
    a_obj_des = object_desired_acceleration(backend, state, cfg)

    res_full = FullQP(backend, cfg).solve(state, qddot_des, a_obj_des)
    skeleton_indices, _ = select_skeleton(state, res_full.forces, cfg)
    res_hier = HierarchicalQP(backend, cfg).solve(
        state, skeleton_indices, res_full.forces, qddot_des, a_obj_des
    )

    f_err = np.linalg.norm(res_full.forces - res_hier.full_forces) / np.linalg.norm(
        res_full.forces
    )
    a_err = np.linalg.norm(res_full.a_obj - res_hier.a_obj) / np.linalg.norm(res_full.a_obj)

    print(f"Contacts: {len(state.contacts)} | Skeleton: {len(skeleton_indices)}")
    print(f"Full QP:    {res_full.solve_time * 1000:.3f} ms")
    print(f"Hier QP:    {res_hier.solve_time * 1000:.3f} ms")
    print(f"Speed-up:   {res_full.solve_time / res_hier.solve_time:.2f}x")
    print(f"Force err:  {f_err * 100:.2f}%")
    print(f"Accel err:  {a_err * 100:.2f}%")


if __name__ == "__main__":
    main()
