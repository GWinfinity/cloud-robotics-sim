"""Reference kinematic trees built with Pinocchio (used to generate ground truth).

pinocchio is optional and imported lazily so that this module can be imported
on machines without it (e.g. native Windows -- use WSL there).
"""

import numpy as np


def _pin():
    try:
        import pinocchio as pin
    except ImportError as e:  # pragma: no cover - environment dependent
        raise ImportError(
            "qd_rbd.models needs the optional 'pin' package (pinocchio); "
            "on Windows use WSL"
        ) from e
    return pin


def _rand_inertia(pin, rng):
    mass = float(rng.uniform(0.4, 4.0))
    com = rng.normal(scale=0.04, size=3)
    inertia = np.diag(rng.uniform(4e-3, 6e-2, size=3))
    return pin.Inertia(mass, com, inertia), mass


def _add(pin, pm, parent, axis, R, p, rng, name):
    jm = pin.JointModelRevoluteUnaligned(
        np.asarray(axis, dtype=float) / np.linalg.norm(axis)
    )
    jid = pm.addJoint(parent, jm, pin.SE3(R, np.asarray(p, dtype=float)), name)
    inertia, _ = _rand_inertia(pin, rng)
    pm.appendBodyToJoint(jid, inertia, pin.SE3.Identity())
    return jid


def _rot(pin, axis, ang):
    a = np.asarray(axis, dtype=float)
    a = a / np.linalg.norm(a)
    return pin.AngleAxis(ang, a).matrix()


def chain(n=7, seed=0):
    """Serial arm, n revolute joints."""
    pin = _pin()
    rng = np.random.default_rng(seed)
    pm = pin.Model()
    parent = 0
    for i in range(n):
        axis = rng.normal(size=3)
        R = _rot(pin, rng.normal(size=3), rng.uniform(-0.4, 0.4))
        p = np.array([0.0, 0.0, 0.12]) + rng.normal(scale=0.02, size=3)
        parent = _add(pin, pm, parent, axis, R, p, rng, f"j{i}")
    return pm


def biped(seed=1):
    """15 dof: torso(3) + two 6-dof legs, branching tree."""
    pin = _pin()
    rng = np.random.default_rng(seed)
    pm = pin.Model()
    torso = 0
    for i in range(3):
        torso = _add(
            pin, pm, torso, [0, 0, 1], np.eye(3), [0, 0, 0.15], rng, f"torso{i}"
        )
    for side in range(2):
        parent = torso
        for k in range(6):
            axis = [1, 0, 0] if k % 3 == 0 else ([0, 1, 0] if k % 3 == 1 else [0, 0, 1])
            p = [0, 0.1 * (1 if side else -1), 0] if k == 0 else [0, 0, -0.15]
            parent = _add(pin, pm, parent, axis, np.eye(3), p, rng, f"leg{side}_{k}")
    return pm


def quadruped_arms(n_arms=4, per_arm=15, seed=2):
    """Large tree: n_arms serial chains hanging off a 3-dof trunk."""
    pin = _pin()
    rng = np.random.default_rng(seed)
    pm = pin.Model()
    trunk = 0
    for i in range(3):
        trunk = _add(pin, pm, trunk, [0, 0, 1], np.eye(3), [0, 0, 0.1], rng, f"t{i}")
    for a in range(n_arms):
        parent = trunk
        for k in range(per_arm):
            axis = rng.normal(size=3)
            p = (
                [
                    0.05 * np.cos(2 * np.pi * a / n_arms),
                    0.05 * np.sin(2 * np.pi * a / n_arms),
                    0,
                ]
                if k == 0
                else [0, 0, 0.08]
            )
            parent = _add(pin, pm, parent, axis, np.eye(3), p, rng, f"a{a}_{k}")
    return pm


MODELS = {
    "chain7": lambda: chain(7, 0),
    "biped15": lambda: biped(1),
    "arms60": lambda: quadruped_arms(4, 15, 2),
}
