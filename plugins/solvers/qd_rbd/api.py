"""Public API, deliberately close to Pinocchio's function names.

    model = Model.from_pinocchio(pin_model)
    data  = Data(model, batch=1024)
    tau   = rnea(model, data, q, v, a)      # q,v,a: (B, nv)
    M     = crba(model, data, q)            # (B, nv, nv)
    qdd   = aba(model, data, q, v, tau)     # (B, nv)

Inputs are handed to the kernel as they are (a contiguous f64 array is passed
by pointer, not copied), and each call allocates its own output unless ``out=``
is given.

``out=`` is not just "avoid an allocation": Quadrants keys its launch-context
cache on the identity of *every* argument, so one freshly allocated output
invalidates the whole tuple and re-marshals all of them -- 13 us at B=1, against
1.0 us of real allocation.  A simulation loop that reuses its buffers therefore
gets a ~22% cheaper call for free.
"""

import numpy as np

from . import ckern
from .algorithms import Data  # re-export
from .model import Model, ensure_init  # re-export

__all__ = [
    "Model",
    "Data",
    "ensure_init",
    "rnea",
    "non_linear_effects",
    "crba",
    "crba_apply",
    "aba",
    "forward_kinematics",
]


def _check(data, *arrays):
    B, nv = data.B, data.model.nv
    out = []
    for a in arrays:
        a = np.ascontiguousarray(a, dtype=np.float64)
        if a.shape != (B, nv):
            raise ValueError(f"expected shape {(B, nv)}, got {a.shape}")
        out.append(a)
    return out


def _out(data, out, shape):
    if out is None:
        return np.empty(shape)
    if not (
        out.flags["C_CONTIGUOUS"] and out.dtype == np.float64 and out.shape == shape
    ):
        raise ValueError(
            "out must be a C-contiguous float64 array of shape " f"{shape}"
        )
    return out


def _c_path(data):
    """True when this call should go to the C kernel (REPORT.md §12)."""
    return data.B >= ckern.MIN_BATCH and ckern.available()


def rnea(model, data, q, v, a, out=None):
    """Inverse dynamics: tau = M(q) a + b(q, v).  Reuse ``out`` in loops."""
    q, v, a = _check(data, q, v, a)
    tau = _out(data, out, (data.B, model.nv))
    if _c_path(data):
        model.cmodel().rnea(q, v, a, tau)
    else:
        data.kernels()["rnea"](q, v, a, tau, data.P, data.S, data.B)
    return tau


def non_linear_effects(model, data, q, v, out=None):
    """b(q, v) = RNEA at a = 0.  Cheaper than ``rnea`` by the qdd term only."""
    q, v = _check(data, q, v)
    tau = _out(data, out, (data.B, model.nv))
    if _c_path(data):
        model.cmodel().rnea(q, v, None, tau, bias=True)
    else:
        data.kernels()["rnea_bias"](q, v, tau, data.P, data.S, data.B)
    return tau


def crba(model, data, q, out=None):
    """Joint-space mass matrix.  Every entry is written by the kernel."""
    (q,) = _check(data, q)
    nv = model.nv
    out = _out(data, out, (data.B, nv, nv))
    data.kernels()["crba"](q, out, data.P, data.IC, data.B)
    return out


def crba_apply(model, data, q, x, out=None):
    """``M(q) x`` without forming M.

    Two sweeps, like RNEA with qd = 0 and g = 0 -- no (B, nv, nv) buffer is
    touched at all.  Use this whenever the caller only needs M times a vector.
    """
    q, x = _check(data, q, x)
    out = _out(data, out, (data.B, model.nv))
    data.kernels()["crba_apply"](q, x, out, data.P, data.S, data.B)
    return out


def aba(model, data, q, v, tau, out=None):
    """Forward dynamics: qdd = M(q)^-1 (tau - b(q, v))."""
    q, v, tau = _check(data, q, v, tau)
    qdd = _out(data, out, (data.B, model.nv))
    data.kernels()["aba"](q, v, tau, qdd, data.P, data.S, data.IC, data.D2, data.B)
    return qdd


def forward_kinematics(model, data, q):
    """World-frame orientation and translation of every joint, per instance.

    Returns (R, p) with R: (B, nj, 3, 3) and p: (B, nj, 3).
    """
    (q,) = _check(data, q)
    B, nj = data.B, model.nj
    R = np.empty((B, nj, 3, 3))
    p4 = np.empty((B, nj, 1, 3))
    data.kernels()["fk"](q, R, p4, data.P, data.B)
    return R, p4[:, :, 0]
