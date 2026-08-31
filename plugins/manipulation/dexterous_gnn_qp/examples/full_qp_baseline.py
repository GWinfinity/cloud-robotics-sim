"""Single-step demo of the full QP baseline on the grasp configuration."""
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
from dexterous_gnn_qp.core.utils.config import load_config


def main():
    parser = argparse.ArgumentParser(description="Full QP baseline single-step demo")
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

    res = FullQP(backend, cfg).solve(state, qddot_des, a_obj_des)

    print(f"Contacts: {len(state.contacts)}")
    print(f"QP status: {res.status}")
    print(f"Solve time: {res.solve_time * 1000:.3f} ms")
    print(f"Objective:  {res.objective:.6f}")
    print(f"Total contact force norm: {np.linalg.norm(res.forces):.4f} N")
    print(f"Hand torque norm:         {np.linalg.norm(res.tau):.4f} Nm")
    print(f"Predicted object accel:   {res.a_obj}")


if __name__ == "__main__":
    main()
