"""Basic usage of the qd_rbd plugin.

Builds a small 2-dof arm directly from raw arrays (no pinocchio needed), runs
FK / RNEA / CRBA / ABA over a batch, then optionally compares against
pinocchio when it is installed (e.g. inside WSL on Windows).

Run from the repository root:

    python plugins/solvers/qd_rbd/examples/basic_usage.py
"""

import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from plugins.solvers.qd_rbd import (  # noqa: E402
    HAS_PINOCCHIO,
    Data,
    Model,
    aba,
    crba,
    crba_apply,
    forward_kinematics,
    non_linear_effects,
    rnea,
)

G = np.array([0.0, 0.0, -9.81])


def two_link_model():
    """2-dof revolute arm from raw arrays: base(universe) + j1 + j2."""
    parents = [0, 0, 1]
    jtype = [2, 0, 0]  # base fixed, j1/j2 revolute
    axis = [[0.0, 0.0, 1.0]] * 3
    Rpl = np.tile(np.eye(3), (3, 1, 1))  # all placements identity rotation
    ppl = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.1], [0.0, 0.0, 0.1]]
    mass = [0.0, 1.0, 0.5]
    lever = [[0.0, 0.0, 0.0]] * 3  # com at each joint origin
    Icom = np.tile(np.diag([1e-3, 1e-3, 1e-3]), (3, 1, 1)).astype(float)
    iv = [-1, 0, 1]  # dof index, -1 = no dof
    return Model(parents, jtype, axis, Rpl, ppl, mass, lever, Icom, iv, G)


def main():
    """Run FK/RNEA/CRBA/ABA over a batch and cross-check the identities."""
    os.environ.setdefault("QD_THREADS", "4")
    model = two_link_model()
    B, nv = 8, model.nv
    data = Data(model, B)

    rng = np.random.default_rng(0)
    q = rng.uniform(-1.0, 1.0, size=(B, nv))
    v = rng.normal(scale=0.2, size=(B, nv))
    a = rng.normal(scale=1.0, size=(B, nv))

    tau = rnea(model, data, q, v, a)
    nle = non_linear_effects(model, data, q, v)
    M = crba(model, data, q)
    Mx = crba_apply(model, data, q, v)
    qdd = aba(model, data, q, v, tau)
    R, p = forward_kinematics(model, data, q)

    print(f"model: {model!r}, batch B={B}")
    print(f"tau[0]   = {tau[0]}")
    print(f"nle[0]   = {nle[0]}")
    print(f"M[0]     =\n{M[0]}")
    print(f"M@v[0]   = {Mx[0]}  (vs M@x: {M[0] @ v[0]})")
    print(f"qdd[0]   = {qdd[0]}  (should equal a[0]={a[0]})")
    print(f"tip pos  = {p[0, -1]}")

    # ABA must invert RNEA: M(q) qdd + nle == tau.
    lhs = np.einsum("bij,bj->bi", M, qdd) + nle
    print(f"|M qdd + nle - tau|_max = {np.abs(lhs - tau).max():.3e}")

    if HAS_PINOCCHIO:
        import pinocchio as pin

        # The same 2-dof arm built natively in pinocchio: two RZ joints at
        # z=0.1, point masses at the joint origins, gravity (0, 0, -9.81).
        pm = pin.Model()
        pm.gravity.linear = G
        parent, pos = 0, 0.0
        for name, m in (("j1", 1.0), ("j2", 0.5)):
            jid = pm.addJoint(
                parent,
                pin.JointModelRZ(),
                pin.SE3(np.eye(3), np.array([0.0, 0.0, pos])),
                name,
            )
            pm.appendBodyToJoint(
                jid, pin.Inertia(m, np.zeros(3), np.eye(3) * 1e-3), pin.SE3.Identity()
            )
            parent, pos = jid, 0.1
        pm_ref = Model.from_pinocchio(pm)
        err = max(
            np.abs(
                rnea(pm_ref, Data(pm_ref, 1), q[i : i + 1], v[i : i + 1], a[i : i + 1])[
                    0
                ]
                - pin.rnea(pm, pm.createData(), q[i], v[i], a[i])
            ).max()
            for i in range(B)
        )
        print(f"vs pinocchio rnea max err = {err:.3e}")
    else:
        print(
            "pinocchio not installed -- skipped cross-check "
            "(on Windows use WSL: pip install pin)"
        )


if __name__ == "__main__":
    main()
