"""Differential IK tasks (frax.ik.tasks port).

A task is static configuration: which frames/dofs to track, gains, weights,
and which keys of the per-frame ``ref`` dict carry the targets.  The solver
compiles the task set into its QP-build kernel, so all per-frame data flows
as plain numpy arrays and every quantity stays batch-parallel.

Velocity-level least squares, as in frax: each task contributes rows

    || weight * (J v - v_des) ||^2

to the QP objective.
"""

from collections.abc import Sequence

import numpy as np


def _per_row(x, n, name):
    """Scalar or per-row array -> (n,) float array."""
    a = np.broadcast_to(np.asarray(x, dtype=np.float64), (n,)).copy()
    return a


class FrameTask:
    """Track world-frame positions and/or orientations of joint frames.

    Args:
        model: :class:`~qd_ik.model.Model`
        joints: joint names or indices to track, shape (n,)
        gain_pos, gain_rot: scalar or (n,) tracking gains (1/s)
        weight_pos, weight_rot: scalar or (n,) row weights
        pos_key: ref dict key for target positions, (B, n, 3); None disables
        rot_key: ref dict key for target rotation matrices, (B, n, 3, 3);
            None disables
    """

    def __init__(
        self,
        model,
        joints: Sequence,
        gain_pos=1.0,
        gain_rot=1.0,
        weight_pos=1.0,
        weight_rot=1.0,
        pos_key: str | None = "frame_pos",
        rot_key: str | None = "frame_rmat",
    ):
        if pos_key is None and rot_key is None:
            raise ValueError("track position, rotation, or both")
        self.model = model
        idx = []
        for j in joints:
            idx.append(model.joint_index(j) if isinstance(j, str) else int(j))
        self.joints = idx
        self.n = len(idx)
        self.pos_key = pos_key
        self.rot_key = rot_key
        self.gain_pos = _per_row(gain_pos, self.n, "gain_pos")
        self.gain_rot = _per_row(gain_rot, self.n, "gain_rot")
        if pos_key is not None:
            self.weight_pos = _per_row(weight_pos, self.n, "weight_pos")
        if rot_key is not None:
            self.weight_rot = _per_row(weight_rot, self.n, "weight_rot")
        self.num_rows = self.n * (3 * (pos_key is not None) + 3 * (rot_key is not None))

    def ref_keys(self):
        return tuple(k for k in (self.pos_key, self.rot_key) if k is not None)


class JointTask:
    """Track actuated joints to reference positions.

    Args:
        model: :class:`~qd_ik.model.Model`
        joints: joint names or indices to track
        ref_key: ref dict key for target joint positions, (B, n)
        gain: scalar or (n,) tracking gain (1/s)
        weight: scalar or (n,) row weight
    """

    def __init__(self, model, joints, ref_key: str, gain=1.0, weight=1.0):
        self.model = model
        idx = []
        for j in joints:
            i = model.joint_index(j) if isinstance(j, str) else int(j)
            if model.floating_base and model.iv[i] < 6:
                raise ValueError("JointTask only supports actuated joints")
            idx.append(i)
        self.joints = idx
        self.n = len(idx)
        self.ref_key = ref_key
        self.gain = _per_row(gain, self.n, "gain")
        self.weight = _per_row(weight, self.n, "weight")
        self.num_rows = self.n

    def ref_keys(self):
        return (self.ref_key,)
