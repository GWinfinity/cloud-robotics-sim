"""The C kernel must be a transliteration, not a reinterpretation (REPORT.md §12).

Every test here diffs the two implementations against each other on the same
buffers.  FMA contraction and the fused `sincos` mean the two are not bitwise
equal, so the tolerance is 1e-11 -- about three orders above the 3e-13 the pair
actually shows, and far below any difference a wrong formula would produce.

Run:  pytest plugins/solvers/qd_rbd/tests/test_c_kernel.py -q
"""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

pin = pytest.importorskip("pinocchio", reason="C-kernel ground truth needs pinocchio")

import plugins.solvers.qd_rbd as rbd  # noqa: E402
from plugins.solvers.qd_rbd import ckern, models  # noqa: E402

TOL = 1e-11
TREE = list(models.MODELS)


@pytest.fixture(scope="module")
def have_c():
    """Skip the whole module when the C kernel cannot build."""
    if not ckern.available():
        pytest.skip(f"C kernel unavailable: {ckern.error()}")


def _pair(name, B, seed=5):
    pm = models.MODELS[name]()
    m = rbd.Model.from_pinocchio(pm)
    d = rbd.Data(m, B)
    rng = np.random.default_rng(seed)
    q = rng.normal(scale=0.6, size=(B, pm.nv))
    v = rng.normal(size=(B, pm.nv))
    a = rng.normal(size=(B, pm.nv))
    return pm, m, d, q, v, a


@pytest.mark.parametrize("name", TREE)
def test_rnea_matches_quadrants(have_c, name):
    """C RNEA equals the Quadrants kernel to FMA-roundoff."""
    _, m, d, q, v, a = _pair(name, 64)
    tq = np.empty((64, m.nv))
    d.kernels()["rnea"](q, v, a, tq, d.P, d.S, 64)
    assert np.abs(m.cmodel().rnea(q, v, a, np.empty_like(tq)) - tq).max() < TOL


@pytest.mark.parametrize("name", TREE)
def test_bias_matches_quadrants(have_c, name):
    """C bias variant equals the Quadrants kernel."""
    _, m, d, q, v, _ = _pair(name, 64)
    tq = np.empty((64, m.nv))
    d.kernels()["rnea_bias"](q, v, tq, d.P, d.S, 64)
    assert (
        np.abs(m.cmodel().rnea(q, v, None, np.empty_like(tq), bias=True) - tq).max()
        < TOL
    )


@pytest.mark.parametrize("B", [1, 3, 5, 7, 15, 63, 65])
def test_tail_batches(have_c, B):
    """B not a multiple of the lane count still runs the remainder correctly."""
    _, m, d, q, v, a = _pair("chain7", B)
    tq = np.empty((B, m.nv))
    d.kernels()["rnea"](q, v, a, tq, d.P, d.S, B)
    assert np.abs(m.cmodel().rnea(q, v, a, np.empty_like(tq)) - tq).max() < TOL


@pytest.mark.parametrize("name", TREE)
def test_api_dispatches_to_c_and_agrees_with_pinocchio(have_c, name):
    """The default API path (C backend) still matches serial pinocchio."""
    pm, m, d, q, v, a = _pair(name, 32)
    tau = rbd.rnea(m, d, q, v, a)
    ref = np.stack([pin.rnea(pm, pm.createData(), q[i], v[i], a[i]) for i in range(32)])
    assert np.abs(tau - ref).max() < 1e-10


def test_lane_count_is_honoured(have_c):
    """Every lane count produces the same answer.

    The K loop is a schedule, not part of the algorithm.
    """
    _, m, d, q, v, a = _pair("biped15", 40)
    cm = m.cmodel()
    ref = cm.rnea(q, v, a, np.empty((40, m.nv)), K=1)
    for K in (2, 4, 8):
        got = cm.rnea(q, v, a, np.empty_like(ref), K=K)
        assert np.abs(got - ref).max() < TOL


def test_out_buffer_contract_with_c(have_c):
    """out= with the C backend honours the same buffer contract."""
    _, m, d, q, v, a = _pair("chain7", 16)
    buf = np.empty((16, m.nv))
    got = rbd.rnea(m, d, q, v, a, out=buf)
    assert got is buf
    with pytest.raises(ValueError):
        rbd.rnea(m, d, q, v, a, out=np.empty((16, m.nv), order="F"))


def test_qd_rbd_c_zero_falls_back(have_c):
    """QD_RBD_C=0 must still work, from a fresh interpreter.

    The build is cached per process, so the env var needs a new interpreter.
    """
    root = str(ROOT)
    code = (
        "import numpy as np, plugins.solvers.qd_rbd as r;"
        "from plugins.solvers.qd_rbd import models;"
        "from plugins.solvers.qd_rbd import ckern;"
        "pm=models.MODELS['chain7']();m=r.Model.from_pinocchio(pm);"
        "d=r.Data(m,4);q=np.zeros((4,pm.nv));"
        "tau=r.rnea(m,d,q,q,q);"
        "print(ckern.available(), float(abs(tau).max()))"
    )
    env = dict(os.environ, PYTHONPATH=root, QD_RBD_C="0")
    p = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env, cwd=root
    )
    assert p.returncode == 0, p.stderr[-800:]
    avail, _ = p.stdout.strip().splitlines()[-1].split()
    assert avail == "False"
