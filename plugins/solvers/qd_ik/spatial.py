"""Rotation / orientation helpers.

Kernel-side helpers are ``@qd.func`` (usable inside Quadrants kernels), ported
from frax/utils/rotation_utils.py so that the FrameTask orientation error is
numerically the same function frax uses.  Host-side numpy copies live at the
bottom (integration and preprocessing run on the host).
"""

import numpy as np
import quadrants as qd


@qd.func
def skew(v):
    """Cross-product matrix, i.e. ``skew(v) @ x == v.cross(x)``."""
    S = qd.Matrix.zero(qd.f64, 3, 3)
    S[0, 1] = -v[2]
    S[0, 2] = v[1]
    S[1, 0] = v[2]
    S[1, 2] = -v[0]
    S[2, 0] = -v[1]
    S[2, 1] = v[0]
    return S


@qd.func
def rot_axis_angle(axis, ang):
    """Rodrigues rotation, axis assumed unit length."""
    c = qd.cast(qd.cos(ang), qd.f64)
    s = qd.cast(qd.sin(ang), qd.f64)
    t = qd.cast(1.0, qd.f64) - c
    x = axis[0]
    y = axis[1]
    z = axis[2]
    R = qd.Matrix.zero(qd.f64, 3, 3)
    R[0, 0] = t * x * x + c
    R[0, 1] = t * x * y - s * z
    R[0, 2] = t * x * z + s * y
    R[1, 0] = t * x * y + s * z
    R[1, 1] = t * y * y + c
    R[1, 2] = t * y * z - s * x
    R[2, 0] = t * x * z - s * y
    R[2, 1] = t * y * z + s * x
    R[2, 2] = t * z * z + c
    return R


@qd.func
def quat_wxyz_to_rmat(q):
    """WXYZ quaternion -> rotation matrix (frax quat_wxyz_to_rmat)."""
    w = q[0]
    x = q[1]
    y = q[2]
    z = q[3]
    n = qd.sqrt(w * w + x * x + y * y + z * z)
    w = w / n
    x = x / n
    y = y / n
    z = z / n
    R = qd.Matrix.zero(qd.f64, 3, 3)
    R[0, 0] = 1.0 - 2.0 * y * y - 2.0 * z * z
    R[0, 1] = 2.0 * x * y - 2.0 * z * w
    R[0, 2] = 2.0 * x * z + 2.0 * y * w
    R[1, 0] = 2.0 * x * y + 2.0 * z * w
    R[1, 1] = 1.0 - 2.0 * x * x - 2.0 * z * z
    R[1, 2] = 2.0 * y * z - 2.0 * x * w
    R[2, 0] = 2.0 * x * z - 2.0 * y * w
    R[2, 1] = 2.0 * y * z + 2.0 * x * w
    R[2, 2] = 1.0 - 2.0 * x * x - 2.0 * y * y
    return R


@qd.func
def rmat_to_quat_wxyz(R):
    """Rotation matrix -> WXYZ quaternion (frax rmat_to_quat_wxyz).

    Picks the numerically largest of the four component cases; the result is
    un-normalized up to a positive scale, so it is normalized here.
    """
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    ww = 1.0 + tr
    xx = 1.0 + 2.0 * R[0, 0] - tr
    yy = 1.0 + 2.0 * R[1, 1] - tr
    zz = 1.0 + 2.0 * R[2, 2] - tr
    wx = R[2, 1] - R[1, 2]
    wy = R[0, 2] - R[2, 0]
    wz = R[1, 0] - R[0, 1]
    xy = R[0, 1] + R[1, 0]
    xz = R[0, 2] + R[2, 0]
    yz = R[1, 2] + R[2, 1]
    # frax uses nested jnp.where; if/elif on the scalar argmax is equivalent
    # and avoids materializing all four candidates.
    qw = qd.cast(0.0, qd.f64)
    qx = qd.cast(0.0, qd.f64)
    qy = qd.cast(0.0, qd.f64)
    qz = qd.cast(0.0, qd.f64)
    if (ww >= xx) and (ww >= yy) and (ww >= zz):
        qw, qx, qy, qz = ww, wx, wy, wz
    else:
        if (xx >= yy) and (xx >= zz):
            qw, qx, qy, qz = wx, xx, xy, xz
        else:
            if yy >= zz:
                qw, qx, qy, qz = wy, xy, yy, yz
            else:
                qw, qx, qy, qz = wz, xz, yz, zz
    n = qd.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    return qd.Vector([qw / n, qx / n, qy / n, qz / n], dt=qd.f64)


@qd.func
def orientation_error(R_cur, R_des):
    """Orientation error vector between two rotation matrices.

    Port of frax orientation_error_quat: the rotvec of ``R_cur @ R_des.T``
    (world-frame relative rotation), resolved through the shortest quaternion
    path.
    """
    quat = rmat_to_quat_wxyz(R_cur @ R_des.transpose())
    s = qd.cast(2.0, qd.f64)
    if quat[0] < 0.0:
        s = qd.cast(-2.0, qd.f64)
    return qd.Vector([s * quat[1], s * quat[2], s * quat[3]], dt=qd.f64)


# --------------------------------------------------------------------- host
def quat_wxyz_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """WXYZ quaternion product, numpy broadcast (host integrate)."""
    w1, x1, y1, z1 = np.moveaxis(q1, -1, 0)
    w2, x2, y2, z2 = np.moveaxis(q2, -1, 0)
    return np.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        axis=-1,
    )


def quat_wxyz_exp(rotvec: np.ndarray) -> np.ndarray:
    """Exponential map (rotation vector -> WXYZ quaternion), numpy batch."""
    theta_sq = np.einsum("...i,...i->...", rotvec, rotvec)
    small = theta_sq < 1e-8
    theta = np.sqrt(np.where(small, 1.0, theta_sq))
    w = np.where(small, 1.0 - theta_sq / 8.0, np.cos(theta / 2.0))
    s = np.where(small, 0.5 - theta_sq / 48.0, np.sin(theta / 2.0) / theta)
    return np.concatenate([w[..., None], s[..., None] * rotvec], axis=-1)
