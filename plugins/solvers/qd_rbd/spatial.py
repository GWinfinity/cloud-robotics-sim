"""Spatial (6D) algebra primitives.

Conventions (Featherstone, "Rigid Body Dynamics Algorithms"):

  motion vector   v = (omega, v_lin)        -- angular first
  force vector    f = (n, f_lin)            -- moment first

A placement ``M_AB = (R, p)`` maps coordinates expressed in frame B into
frame A, i.e. ``x_A = R x_B + p``.  The corresponding motion transform
``X_BA`` maps a motion vector expressed in A into B:

    X_BA = [[ R^T,        0 ],
            [ -R^T p^,    R^T ]]

and the force transform dual to it is

    X*_BA = [[ R,    p^ R ],
             [ 0,    R   ]]

Note that ``X*_BA == X_BA^T``, which lets the inertia transform be written as a
congruence (used all over ``algorithms.py``).

v1 stored these as dense 6x6 matrices; v2 never materialises one.  The block
form of every operator lives in ``algorithms.py`` -- see that module's docstring
for why.  Everything is f64 so results can be compared against Pinocchio at
1e-12.
"""

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
