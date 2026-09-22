"""Flow past an immersed obstacle with the standalone 3D CFD core.

Headless demo of the obstacle feature (no genesis scene needed):

* default: analytic box obstacle inside the duct;
* ``--stl wing.stl``: rasterize an external mesh (e.g. a CATIA export)
  instead, optionally placed with ``--position`` / ``--rotation`` /
  ``--scale``.

Prints mass conservation, the fore/aft pressure difference (a drag proxy)
and the solid volume fraction; optionally saves the centre-slice fields.

Run::

    python plugins/solvers/cfd_coupling/examples/run_obstacle_flow.py
    python plugins/solvers/cfd_coupling/examples/run_obstacle_flow.py \
        --stl wing.stl --position 0.1 0.05 0.05 --rotation 0 0 0 \
        --steps 1500 --save outputs/obstacle_flow.npz
"""

# ruff: noqa: E402, I001
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from plugins.solvers.cfd_coupling import CFD3D, CFDOptions
from plugins.solvers.cfd_coupling.core.obstacles import mask_from_box


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--stl", type=str, default=None, help="mesh file (STL/OBJ/GLB)")
    p.add_argument(
        "--position",
        type=float,
        nargs=3,
        default=None,
        metavar=("X", "Y", "Z"),
        help="world placement of the mesh [m]",
    )
    p.add_argument(
        "--rotation",
        type=float,
        nargs=3,
        default=None,
        metavar=("RX", "RY", "RZ"),
        help="Euler angles xyz [deg]",
    )
    p.add_argument(
        "--scale",
        type=float,
        default=None,
        help="uniform scale applied to the mesh before placement",
    )
    p.add_argument("--steps", type=int, default=1200)
    p.add_argument("--dt", type=float, default=1.0e-3)
    p.add_argument("--u-in", type=float, default=1.0)
    p.add_argument("--save", type=str, default=None, help="npz output path")
    return p.parse_args()


def main() -> None:
    """Run the duct-with-obstacle demo."""
    args = parse_args()
    domain = (0.2, 0.1, 0.1)
    cells = (20, 10, 10)
    cfd = CFD3D(
        CFDOptions(
            domain=domain,
            cells=cells,
            viscosity=1.0e-4,
            inlet_patch=((0.4, 0.6), (0.4, 0.6)),
            advect_temperature=False,
            cg_tol=1e-6,
        )
    )

    if args.stl:
        from plugins.solvers.cfd_coupling.solver import (
            _add_obstacle_to_core,  # reusable bridge: path -> mask
        )

        mask = _add_obstacle_to_core(
            cfd,
            args.stl,
            tuple(args.position) if args.position else None,
            tuple(args.rotation) if args.rotation else None,
            args.scale,
        )
        print(f"mesh obstacle: {args.stl} ({int(mask.sum())} solid cells)")
    else:
        mask = mask_from_box((0.1, 0.05, 0.05), (0.04, 0.04, 0.04), domain, cells)
        cfd.set_solid_mask(mask)
        print(f"box obstacle: {int(mask.sum())} solid cells")

    cfd.set_inlet(args.u_in)
    t0 = time.perf_counter()
    for _ in range(args.steps):
        cfd.step(args.dt)
    elapsed = time.perf_counter() - t0

    q_in = cfd.inlet_flow()
    q_out = cfd.outlet_flow()
    p = cfd.p.cpu().numpy()
    u_cell = (0.5 * (cfd.u[:-1] + cfd.u[1:])).cpu().numpy()
    fluid = ~mask
    d_fore = p[: mask.shape[0] // 3][fluid[: mask.shape[0] // 3]].mean()
    d_aft = p[2 * mask.shape[0] // 3 :][fluid[2 * mask.shape[0] // 3 :]].mean()

    print(
        f"steps={args.steps}  wall={elapsed:.1f}s  ({1e3*elapsed/args.steps:.1f} ms/step)"
    )
    print(f"solid volume fraction: {mask.mean():.3f}")
    print(f"q_in  = {q_in:.3e} m^3/s (target {args.u_in * cfd.inlet_patch_area():.3e})")
    print(f"q_out = {q_out:.3e} m^3/s  (rel err {abs(q_out - q_in) / q_in:.2%})")
    print(f"mean p fore/aft: {d_fore:.3e} / {d_aft:.3e} m^2/s^2 (drag proxy)")
    print(f"max |u| inside solid: {np.abs(u_cell[mask]).max():.1e}")
    print(
        f"jet core upstream / wake downstream: "
        f"{u_cell[3, 5, 5]:.3f} / {u_cell[16, 5, 5]:.3f} m/s"
    )

    if args.save:
        out = Path(args.save)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            out,
            u=u_cell,
            p=p,
            solid=mask,
            domain=np.asarray(domain),
            cells=np.asarray(cells),
        )
        print(f"saved {out}")


if __name__ == "__main__":
    main()
