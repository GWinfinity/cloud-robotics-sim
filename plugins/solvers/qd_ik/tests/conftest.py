"""Conftest for qd_ik tests: a shared synthetic floating-base 3R arm."""

import numpy as np
import pytest

from plugins.solvers.qd_ik.model import Model


@pytest.fixture(scope="module")
def toy_model():
    """Floating-base 3R arm with shoulder/elbow/wrist frames."""
    nj = 4
    return Model(
        parents=np.array([0, 0, 1, 2]),
        jtype=np.array([2, 0, 0, 0]),  # FIX, REV, REV, REV
        axis=np.array([[1.0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 0]]),
        Rpl=np.tile(np.eye(3), (nj, 1, 1)),
        ppl=np.array([[0.0, 0, 0], [0, 0, 0.05], [0.2, 0, 0], [0.2, 0, 0]]),
        q_min=np.array([-2.5, -2.5, -2.5]),
        q_max=np.array([2.5, 2.5, 2.5]),
        v_max=np.array([10.0, 10.0, 10.0]),
        names=["universe", "shoulder", "elbow", "wrist"],
        floating_base=True,
    )


@pytest.fixture(scope="module")
def toy_config():
    """A batch of two non-trivial configurations."""
    q = np.zeros((2, 10))
    q[:, 3] = 1.0
    q[:, 2] = 0.3
    q[:, 7:] = [[0.3, -0.6, 0.4], [0.2, -0.4, 0.3]]
    return q
