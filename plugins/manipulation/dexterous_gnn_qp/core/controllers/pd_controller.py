"""PD controllers that produce desired hand joint and object spatial accelerations."""
from __future__ import annotations

import numpy as np
import scipy.spatial.transform as stf

from dexterous_gnn_qp.core.backends.base import DynamicsBackend
from dexterous_gnn_qp.core.env.sim import SimState
from dexterous_gnn_qp.core.utils.config import DotDict


def _quat_to_mat(q: np.ndarray) -> np.ndarray:
    """Convert MuJoCo quaternion [w, x, y, z] to a 3x3 rotation matrix."""
    return stf.Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()


def _quat_difference(q_des: np.ndarray, q_cur: np.ndarray) -> np.ndarray:
    """Return quaternion rotating q_cur to q_des (MuJoCo [w,x,y,z])."""
    # scipy uses [x,y,z,w]
    r_des = stf.Rotation.from_quat([q_des[1], q_des[2], q_des[3], q_des[0]])
    r_cur = stf.Rotation.from_quat([q_cur[1], q_cur[2], q_cur[3], q_cur[0]])
    r_err = r_des * r_cur.inv()
    return r_err.as_quat()  # [x,y,z,w]


def _axis_angle_from_quat(q: np.ndarray) -> np.ndarray:
    """q is [x,y,z,w] (scipy convention)."""
    rot = stf.Rotation.from_quat(q)
    return rot.as_rotvec()


def hand_desired_acceleration(
    backend: DynamicsBackend,
    state: SimState,
    cfg: DotDict,
) -> np.ndarray:
    """PD on hand joint positions -> desired joint acceleration."""
    q_target = np.array(cfg.controller.hand_target, dtype=float) if cfg.controller.hand_target is not None else np.zeros(backend.n_hand_dof)
    if len(q_target) != backend.n_hand_dof:
        raise ValueError(
            f"controller.hand_target length {len(q_target)} != hand DoF {backend.n_hand_dof}"
        )
    kp = float(cfg.controller.hand_kp)
    kv = float(cfg.controller.hand_kv)
    return kp * (q_target - state.q_hand) - kv * state.v_hand


def object_desired_acceleration(
    backend: DynamicsBackend,
    state: SimState,
    cfg: DotDict,
) -> np.ndarray:
    """PD on object pose -> desired spatial acceleration [angular; linear]."""
    obj_cfg = cfg.object
    p_des = np.array(obj_cfg.position, dtype=float)
    q_des = np.array(obj_cfg.quaternion, dtype=float)

    p_err = p_des - state.x_obj
    q_err_xyzw = _quat_difference(q_des, state.quat_obj)
    rot_err = _axis_angle_from_quat(q_err_xyzw)

    kp_lin = float(cfg.controller.object_kp)
    kv_lin = float(cfg.controller.object_kv)
    kp_rot = float(cfg.controller.object_kp)
    kv_rot = float(cfg.controller.object_kv)

    # v_obj is [linear; angular] matching MuJoCo free-joint qvel ordering.
    v = state.v_obj[:3]
    w = state.v_obj[3:]

    a_linear = kp_lin * p_err - kv_lin * v
    a_angular = kp_rot * rot_err - kv_rot * w
    return np.concatenate([a_linear, a_angular])
