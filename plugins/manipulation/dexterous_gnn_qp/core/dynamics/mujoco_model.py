"""MuJoCo-based dynamics helpers (Jacobians, grasp maps, wrench projections)."""
from __future__ import annotations

import mujoco
import numpy as np

from dexterous_gnn_qp.core.env.loader import HandScene
from dexterous_gnn_qp.core.env.sim import Contact


def hand_jacobian(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    scene: HandScene,
    contact: Contact,
) -> np.ndarray:
    """Return the 3 x n_hand_dof translational Jacobian of the contact point."""
    jacp = np.zeros((3, model.nv), dtype=float)
    jacr = np.zeros((3, model.nv), dtype=float)
    mujoco.mj_jac(model, data, jacp, jacr, contact.pos, contact.hand_body_id)
    return jacp[:, scene.hand_dof_indices]


def object_grasp_wrench(contact: Contact, obj_com: np.ndarray, force: np.ndarray) -> np.ndarray:
    """Map a 3D contact force on the object to a 6D spatial wrench about COM.

    Uses Featherstone spatial-vector convention [angular; linear].
    """
    r = contact.pos - obj_com
    torque = np.cross(r, force)
    return np.concatenate([torque, force])


def project_force_onto_contact_basis(force: np.ndarray, contact: Contact) -> np.ndarray:
    """Return [f_n, f_t1, f_t2] for a force expressed in world frame."""
    return np.array(
        [
            float(np.dot(contact.normal, force)),
            float(np.dot(contact.tangent1, force)),
            float(np.dot(contact.tangent2, force)),
        ]
    )
