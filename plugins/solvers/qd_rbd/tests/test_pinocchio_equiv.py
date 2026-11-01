"""Numerical equivalence against Pinocchio, plus internal consistency.

Requires the optional ``pin`` (pinocchio) package; skipped where it is not
installed (native Windows -- run inside WSL there).

Run:  pytest plugins/solvers/qd_rbd/tests/test_pinocchio_equiv.py -q
"""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

pin = pytest.importorskip(
    "pinocchio", reason="pinocchio not installed (use WSL on Windows)"
)

import plugins.solvers.qd_rbd as rbd  # noqa: E402
from plugins.solvers.qd_rbd import models  # noqa: E402

TOL = 1e-10
B = 8


@pytest.fixture(scope="module", params=list(models.MODELS))
def case(request):
    """One (pinocchio model, qd_rbd model, data, random state) per tree."""
    pm = models.MODELS[request.param]()
    m = rbd.Model.from_pinocchio(pm)
    rng = np.random.default_rng(5)
    nv = pm.nv
    q = rng.normal(scale=0.6, size=(B, nv))
    v = rng.normal(size=(B, nv))
    a = rng.normal(size=(B, nv))
    return request.param, pm, m, rbd.Data(m, B), q, v, a


def test_rnea(case):
    """Batched RNEA matches serial pinocchio."""
    _, pm, m, d, q, v, a = case
    ref = np.stack([pin.rnea(pm, pm.createData(), q[i], v[i], a[i]) for i in range(B)])
    assert np.abs(rbd.rnea(m, d, q, v, a) - ref).max() < TOL


def test_non_linear_effects(case):
    """Batched nonLinearEffects matches serial pinocchio."""
    _, pm, m, d, q, v, _ = case
    ref = np.stack(
        [pin.nonLinearEffects(pm, pm.createData(), q[i], v[i]) for i in range(B)]
    )
    assert np.abs(rbd.non_linear_effects(m, d, q, v) - ref).max() < TOL


def test_crba(case):
    """Batched CRBA matches serial pinocchio."""
    _, pm, m, d, q, _, _ = case
    ref = np.stack([pin.crba(pm, pm.createData(), q[i]) for i in range(B)])
    assert np.abs(rbd.crba(m, d, q) - ref).max() < TOL


def test_crba_writes_every_entry(case):
    """The zeroing pass is gone: every (dof, dof) entry must be written.

    Handing the kernel a NaN-poisoned output buffer makes a missing write fail
    loudly instead of silently returning stale scratch from a previous call.
    """
    _, pm, m, d, q, _, _ = case
    out = np.full((B, m.nv, m.nv), np.nan)
    res = rbd.crba(m, d, q, out=out)
    assert res is out
    assert np.isfinite(res).all()


def test_crba_apply(case):
    """Matrix-free M x must match the explicitly formed matrix."""
    _, pm, m, d, q, _, _ = case
    x = np.random.default_rng(7).normal(size=(B, pm.nv))
    M = np.stack([pin.crba(pm, pm.createData(), q[i]) for i in range(B)])
    ref = np.einsum("bij,bj->bi", M, x)
    assert np.abs(rbd.crba_apply(m, d, q, x) - ref).max() < TOL
    assert (
        np.abs(
            rbd.crba_apply(m, d, q, x) - np.einsum("bij,bj->bi", rbd.crba(m, d, q), x)
        ).max()
        < TOL
    )


def test_aba(case):
    """Batched ABA matches serial pinocchio."""
    _, pm, m, d, q, v, _ = case
    tau = np.stack(
        [pin.rnea(pm, pm.createData(), q[i], v[i], np.zeros(pm.nv)) for i in range(B)]
    )
    ref = np.stack([pin.aba(pm, pm.createData(), q[i], v[i], tau[i]) for i in range(B)])
    assert np.abs(rbd.aba(m, d, q, v, tau) - ref).max() < TOL


def test_fk(case):
    """World placements match pinocchio's forwardKinematics per instance."""
    _, pm, m, d, q, _, _ = case
    R, p = rbd.forward_kinematics(m, d, q)
    for i in range(B):
        dd = pm.createData()
        pin.forwardKinematics(pm, dd, q[i])
        j = pm.njoints - 1
        assert np.abs(R[i, j] - np.array(dd.oMi[j].rotation)).max() < TOL
        assert np.abs(p[i, j] - np.array(dd.oMi[j].translation)).max() < TOL


def test_forward_dynamics_consistency(case):
    """M(q) qdd + nle(q, v) == tau, checked on the qd side only."""
    _, pm, m, d, q, v, _ = case
    tau = np.random.default_rng(9).normal(size=(B, pm.nv))
    qdd = rbd.aba(m, d, q, v, tau)
    lhs = np.einsum("bij,bj->bi", rbd.crba(m, d, q), qdd) + rbd.non_linear_effects(
        m, d, q, v
    )
    assert np.abs(lhs - tau).max() < 1e-9


def test_batch_independence(case):
    """Batching must not couple instances."""
    _, pm, m, d, q, v, a = case
    full = rbd.rnea(m, d, q, v, a)
    d1 = rbd.Data(m, 1)
    for i in (0, 1, B - 1):
        single = rbd.rnea(m, d1, q[i : i + 1], v[i : i + 1], a[i : i + 1])
        assert np.abs(single[0] - full[i]).max() < TOL


def test_out_buffer_is_reused_and_agrees(case):
    """``out=`` must write the caller's buffer, not a fresh one."""
    _, pm, m, d, q, v, a = case
    nv = pm.nv
    fresh = rbd.rnea(m, d, q, v, a)
    buf = np.zeros((B, nv))
    res = rbd.rnea(m, d, q, v, a, out=buf)
    assert res is buf
    assert np.abs(buf - fresh).max() < 1e-14

    # a second call must fully overwrite, leaving no stale entry behind
    buf[:] = np.nan
    rbd.rnea(m, d, q, v, a, out=buf)
    assert np.isfinite(buf).all()


def test_out_buffer_contract(case):
    """Wrong shape/dtype/layout is rejected instead of silently mangled."""
    _, pm, m, d, q, v, a = case
    nv = pm.nv
    with pytest.raises(ValueError):
        rbd.rnea(m, d, q, v, a, out=np.empty((B, nv + 1)))
    with pytest.raises(ValueError):
        rbd.rnea(m, d, q, v, a, out=np.empty((B, nv), dtype=np.float32))
    with pytest.raises(ValueError):
        rbd.crba(m, d, q, out=np.empty((B, nv, nv), dtype=np.float32))


def test_crba_row_coverage_flags():
    """The rows CRBA skips zeroing must really be covered by the fill.

    On a chain every row is covered; on a branched tree only the joints that
    are ancestors of everything below them are.  If this precondition is ever
    wrong the NaN test above fails, but a wrong *flag* would be silent.
    """
    chain = rbd.Model.from_pinocchio(models.MODELS["chain7"]())
    assert chain.row_full[1:].all()
    biped = rbd.Model.from_pinocchio(models.MODELS["biped15"]())
    # joint 0 is the fixed base: it has no dof, hence no row at all
    assert biped.row_full[1:4].all() and not biped.row_full[4:].any()
