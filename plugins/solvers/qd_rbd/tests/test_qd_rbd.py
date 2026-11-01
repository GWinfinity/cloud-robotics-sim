"""Behavioral tests for the qd_rbd plugin that need **no** pinocchio.

Models are built directly from raw arrays; correctness is anchored by
analytic results (1-dof pendulum) and cross-algorithm consistency
(RNEA <-> CRBA <-> ABA).  Equivalence against pinocchio lives in
test_pinocchio_equiv.py and runs only where pinocchio is installed.

Run:  pytest plugins/solvers/qd_rbd/tests -q
"""

import sys
from pathlib import Path

import numpy as np
import pytest

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

TOL = 1e-10
G = np.array([0.0, 0.0, -9.81])


def _rod(axis, ang):
    """Rodrigues rotation of unit ``axis`` by ``ang`` -- vectorized over ``ang``.

    ``ang`` may be a scalar or an array of shape (B,); the result is (..., 3, 3).
    """
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    ang = np.asarray(ang, dtype=float)
    c, s = np.cos(ang), np.sin(ang)
    t = 1.0 - c
    R = np.empty(ang.shape + (3, 3))
    R[..., 0, 0] = t * x * x + c
    R[..., 0, 1] = t * x * y - s * z
    R[..., 0, 2] = t * x * z + s * y
    R[..., 1, 0] = t * x * y + s * z
    R[..., 1, 1] = t * y * y + c
    R[..., 1, 2] = t * y * z - s * x
    R[..., 2, 0] = t * x * z - s * y
    R[..., 2, 1] = t * y * z + s * x
    R[..., 2, 2] = t * z * z + c
    return R


def pendulum_model():
    """1-dof revolute pendulum: pivot at origin, com at distance d along x.

    Analytic results (gravity (0,-g,0), PE = -m*g*d*sin(q)):
      rnea(q,0,0)      = -dPE/dq = +m*g*d*cos(q)   (actuator holding torque)
      crba(q)          = Izz + m*d^2                 (constant)
      aba(q,0,tau)     = (tau - m*g*d*cos(q))/M
    """
    g = np.array([0.0, -9.81, 0.0])
    m, d = 2.0, 0.3
    Izz = 0.01
    parents = [0, 0]
    jtype = [2, 0]  # base fixed, joint revolute about z
    axis = [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]
    Rpl = np.tile(np.eye(3), (2, 1, 1))
    ppl = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    mass = [0.0, m]
    lever = [[0.0, 0.0, 0.0], [d, 0.0, 0.0]]
    Icom = np.zeros((2, 3, 3))
    Icom[1] = np.diag([Izz, Izz, Izz])
    iv = [-1, 0]
    model = Model(parents, jtype, axis, Rpl, ppl, mass, lever, Icom, iv, g)
    return model, m, d, Izz


def chain_model(n=7, seed=0):
    """Serial chain of n revolute joints, mimics the archive chain7 shape."""
    rng = np.random.default_rng(seed)
    nj = n + 1
    parents = [0] + list(range(n))
    jtype = [2] + [0] * n
    axis = np.zeros((nj, 3))
    axis[1:] = rng.normal(size=(n, 3))
    axis[1:] /= np.linalg.norm(axis[1:], axis=1, keepdims=True)
    Rpl = np.tile(np.eye(3), (nj, 1, 1))
    ppl = np.zeros((nj, 3))
    ppl[1:] = np.array([0.0, 0.0, 0.12]) + rng.normal(scale=0.02, size=(n, 3))
    mass = np.zeros(nj)
    mass[1:] = rng.uniform(0.4, 4.0, size=n)
    lever = np.zeros((nj, 3))
    lever[1:] = rng.normal(scale=0.04, size=(n, 3))
    Icom = np.zeros((nj, 3, 3))
    for i in range(1, nj):
        Icom[i] = np.diag(rng.uniform(4e-3, 6e-2, size=3))
    iv = [-1] + list(range(n))
    return Model(parents, jtype, axis, Rpl, ppl, mass, lever, Icom, iv, G)


def branched_model():
    """Branching tree with a fixed joint: torso(2) + two 2-dof legs + fixed tip."""
    parents = [0, 0, 1, 2, 2, 4, 4, 3]
    jtype = [2, 0, 0, 0, 0, 2, 0, 0]  # joint 5 is fixed (no dof)
    axis = np.array(
        [
            [0, 0, 1],
            [0, 0, 1],
            [0, 0, 1],
            [0, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
            [0, 1, 0],
            [1, 0, 0],
        ],
        dtype=float,
    )
    Rpl = np.tile(np.eye(3), (8, 1, 1))
    ppl = np.array(
        [
            [0, 0, 0],
            [0, 0, 0.1],
            [0, 0, 0.15],
            [0, 0, 0.1],
            [0, 0.1, 0],
            [0, 0, 0.1],
            [0, 0, -0.1],
            [0, 0, 0.05],
        ]
    )
    mass = np.array([0, 1.5, 1.0, 0.5, 1.0, 0.3, 0.4, 0.2])
    lever = np.zeros((8, 3))
    lever[1:] = 0.02
    Icom = np.zeros((8, 3, 3))
    for i in range(1, 8):
        Icom[i] = np.diag([0.01, 0.01, 0.01])
    iv = [-1, 0, 1, 2, 3, -1, 4, 5]
    return Model(parents, jtype, axis, Rpl, ppl, mass, lever, Icom, iv, G)


@pytest.fixture(scope="module")
def chain():
    """Shared 7-dof chain model (compiled once per module)."""
    return chain_model()


@pytest.fixture(scope="module")
def branched():
    """Shared branching-tree model with a fixed joint."""
    return branched_model()


def _rand_state(model, B, seed=5):
    rng = np.random.default_rng(seed)
    q = rng.normal(scale=0.6, size=(B, model.nv))
    v = rng.normal(size=(B, model.nv))
    a = rng.normal(size=(B, model.nv))
    return q, v, a


# --------------------------------------------------------------- analytic anchor


def test_pendulum_rnea_matches_analytic():
    """RNEA(q, 0, 0) equals the analytic gravity holding torque."""
    model, m, d, _ = pendulum_model()
    data = Data(model, 16)
    rng = np.random.default_rng(3)
    q = rng.uniform(-np.pi, np.pi, size=(16, 1))
    tau = rnea(model, data, q, np.zeros((16, 1)), np.zeros((16, 1)))
    ref = m * 9.81 * d * np.cos(q)  # actuator torque balancing gravity
    assert np.abs(tau - ref).max() < TOL


def test_pendulum_crba_is_constant_inertia():
    """The 1-dof mass matrix is the constant inertia about the pivot."""
    model, m, d, Izz = pendulum_model()
    data = Data(model, 16)
    rng = np.random.default_rng(4)
    q = rng.uniform(-np.pi, np.pi, size=(16, 1))
    M = crba(model, data, q)
    assert np.abs(M - (Izz + m * d * d)).max() < TOL


def test_pendulum_aba_matches_analytic():
    """ABA inverts the scalar analytic dynamics."""
    model, m, d, Izz = pendulum_model()
    data = Data(model, 16)
    rng = np.random.default_rng(5)
    q = rng.uniform(-np.pi, np.pi, size=(16, 1))
    tau = rng.normal(size=(16, 1))
    qdd = aba(model, data, q, np.zeros((16, 1)), tau)
    M = Izz + m * d * d
    ref = (tau - m * 9.81 * d * np.cos(q)) / M  # M qdd = tau - holding torque
    assert np.abs(qdd - ref).max() < TOL


# ------------------------------------------------------ cross-algorithm consistency


@pytest.mark.parametrize("model_name", ["chain", "branched"])
def test_rnea_bias_equals_rnea_at_zero_acc(model_name, request):
    """non_linear_effects is RNEA evaluated at a = 0."""
    model = request.getfixturevalue(model_name)
    B = 8
    q, v, _ = _rand_state(model, B)
    d = Data(model, B)
    bias = non_linear_effects(model, d, q, v)
    zero_a = np.zeros((B, model.nv))
    assert np.abs(rnea(model, d, q, v, zero_a) - bias).max() < TOL


@pytest.mark.parametrize("model_name", ["chain", "branched"])
def test_crba_columns_match_rnea(model_name, request):
    """M[:, :, j] == RNEA(q, 0, e_j) - bias, column by column.

    Both calls must use the same velocity: with v != 0 the bias would keep a
    Coriolis term that RNEA(q, 0, e_j) does not have.
    """
    model = request.getfixturevalue(model_name)
    B, nv = 8, model.nv
    q, _, _ = _rand_state(model, B)
    d = Data(model, B)
    z = np.zeros((B, nv))
    bias = non_linear_effects(model, d, q, z)
    M = crba(model, d, q)
    for j in range(nv):
        e = np.zeros((B, nv))
        e[:, j] = 1.0
        col = rnea(model, d, q, z, e) - bias
        assert np.abs(M[:, :, j] - col).max() < 1e-9
    # mass matrix is symmetric
    assert np.abs(M - np.transpose(M, (0, 2, 1))).max() < 1e-9


@pytest.mark.parametrize("model_name", ["chain", "branched"])
def test_crba_apply_matches_crba_matvec(model_name, request):
    """The matrix-free M·x equals the explicit matrix applied to x."""
    model = request.getfixturevalue(model_name)
    B = 8
    q, v, _ = _rand_state(model, B)
    d = Data(model, B)
    M = crba(model, d, q)
    ref = np.einsum("bij,bj->bi", M, v)
    assert np.abs(crba_apply(model, d, q, v) - ref).max() < 1e-9


@pytest.mark.parametrize("model_name", ["chain", "branched"])
def test_aba_solves_mass_system(model_name, request):
    """M(q) qdd + nle(q, v) == tau, using ABA's qdd."""
    model = request.getfixturevalue(model_name)
    B = 8
    q, v, _ = _rand_state(model, B)
    tau = np.random.default_rng(9).normal(size=(B, model.nv))
    d = Data(model, B)
    qdd = aba(model, d, q, v, tau)
    M = crba(model, d, q)
    nle = non_linear_effects(model, d, q, v)
    lhs = np.einsum("bij,bj->bi", M, qdd) + nle
    assert np.abs(lhs - tau).max() < 1e-8


# ------------------------------------------------------------------ forward kinematics


@pytest.mark.parametrize("model_name", ["chain", "branched"])
def test_fk_matches_numpy_composition(model_name, request):
    """FK agrees with an independent numpy frame composition."""
    model = request.getfixturevalue(model_name)
    B = 8
    q, _, _ = _rand_state(model, B)
    d = Data(model, B)
    R, p = forward_kinematics(model, d, q)

    nj = model.nj
    R_ref = np.tile(np.eye(3), (B, nj, 1, 1))
    p_ref = np.zeros((B, nj, 3))
    for i in range(1, nj):
        par = model.parents[i]
        if model.jtype[i] == 0:  # revolute
            Rj = model.Rpl[i] @ _rod(model.axis[i], q[:, model.iv[i]])
            pj = model.ppl[i]
        elif model.jtype[i] == 1:  # prismatic
            Rj = model.Rpl[i]
            pj = model.ppl[i] + q[:, model.iv[i], None] * (model.Rpl[i] @ model.axis[i])
        else:  # fixed
            Rj = model.Rpl[i]
            pj = model.ppl[i]
        # oMi = oMi[par] * liMi  (column-vector world map: x_w = R x_l + p)
        R_ref[:, i] = R_ref[:, par] @ Rj
        p_ref[:, i] = (R_ref[:, par] @ pj[..., None])[:, :, 0] + p_ref[:, par]
    assert np.abs(R - R_ref).max() < 1e-12
    assert np.abs(p - p_ref).max() < 1e-12


# ---------------------------------------------------------------------- api contracts


@pytest.mark.parametrize("model_name", ["chain", "branched"])
def test_batch_independence(model_name, request):
    """Per-instance results must not depend on batch membership."""
    model = request.getfixturevalue(model_name)
    B = 8
    q, v, a = _rand_state(model, B)
    d = Data(model, B)
    full = rnea(model, d, q, v, a)
    d1 = Data(model, 1)
    for i in (0, 1, B - 1):
        single = rnea(model, d1, q[i : i + 1], v[i : i + 1], a[i : i + 1])
        assert np.abs(single[0] - full[i]).max() < TOL


@pytest.mark.parametrize("model_name", ["chain", "branched"])
def test_out_buffer_is_reused_and_fully_written(model_name, request):
    """out= writes the caller's buffer and leaves no stale entries."""
    model = request.getfixturevalue(model_name)
    B, nv = 8, model.nv
    q, v, a = _rand_state(model, B)
    d = Data(model, B)
    fresh = rnea(model, d, q, v, a)
    buf = np.zeros((B, nv))
    res = rnea(model, d, q, v, a, out=buf)
    assert res is buf
    assert np.abs(buf - fresh).max() < 1e-14
    buf[:] = np.nan  # a second call must not leave stale entries
    rnea(model, d, q, v, a, out=buf)
    assert np.isfinite(buf).all()

    M = np.full((B, nv, nv), np.nan)
    res = crba(model, d, q, out=M)
    assert res is M
    assert np.isfinite(M).all()


@pytest.mark.parametrize("model_name", ["chain", "branched"])
def test_out_buffer_contract(model_name, request):
    """Wrong shape/dtype/layout of out= is rejected."""
    model = request.getfixturevalue(model_name)
    B, nv = 8, model.nv
    q, v, a = _rand_state(model, B)
    d = Data(model, B)
    with pytest.raises(ValueError):
        rnea(model, d, q, v, a, out=np.empty((B, nv + 1)))
    with pytest.raises(ValueError):
        rnea(model, d, q, v, a, out=np.empty((B, nv), dtype=np.float32))
    with pytest.raises(ValueError):
        rnea(model, d, q, v, a, out=np.empty((B, nv), order="F"))
    with pytest.raises(ValueError):
        crba(model, d, q, out=np.empty((B, nv, nv), dtype=np.float32))


@pytest.mark.parametrize("B", [1, 3, 5, 7, 15, 63, 65])
def test_odd_batch_sizes(chain, B):
    """B not a multiple of any lane count must give identical per-instance results."""
    q, v, a = _rand_state(chain, B)
    d = Data(chain, B)
    full = rnea(chain, d, q, v, a)
    d1 = Data(chain, 1)
    for i in range(B):
        single = rnea(chain, d1, q[i : i + 1], v[i : i + 1], a[i : i + 1])
        assert np.abs(single[0] - full[i]).max() < TOL


def test_zero_state_gives_gravity_compensation(chain):
    """RNEA at q=v=a=0 is the gravity holding torque, not zero."""
    B = 4
    d = Data(chain, B)
    z = np.zeros((B, chain.nv))
    tau = rnea(chain, d, z, z, z)
    assert np.abs(tau - non_linear_effects(chain, d, z, z)).max() < 1e-14
    assert np.abs(tau).max() > 0.0  # gravity is nonzero for this model


def test_from_pinocchio_error_message_without_pin():
    """Without pinocchio, from_pinocchio raises a helpful ImportError."""
    if HAS_PINOCCHIO:
        pytest.skip("pinocchio installed; the no-pin path is not in force")
    with pytest.raises(ImportError, match="pin"):
        Model.from_pinocchio(object())
