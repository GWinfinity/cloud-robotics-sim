"""Tests for the Genesis-backed dynamics backend."""
from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np

try:
    import genesis as gs

    HAS_GENESIS = True
except Exception:  # pragma: no cover - Genesis may not be installed.
    HAS_GENESIS = False
    gs = None

try:
    import cloud_robotics_sim

    HAS_CLOUD_ROBOTICS_SIM = True
except Exception:  # pragma: no cover - cloud_robotics_sim may not be installed.
    HAS_CLOUD_ROBOTICS_SIM = False
    cloud_robotics_sim = None

REPO_ROOT = Path(__file__).parent.parent


@unittest.skipUnless(
    HAS_GENESIS and HAS_CLOUD_ROBOTICS_SIM,
    "Genesis or cloud_robotics_sim is not installed",
)
class TestGenesisBackend(unittest.TestCase):
    def _backend(self):
        sys.path.insert(0, str(REPO_ROOT.parent))
        from dexterous_gnn_qp.core.backends import create_backend
        from dexterous_gnn_qp.core.utils.config import load_config

        cfg = load_config(REPO_ROOT / "configs" / "leap_sphere_grasp.yaml")
        cfg.sim.backend = "genesis"
        return create_backend(cfg)

    def test_build_scene(self):
        backend = self._backend()
        self.assertEqual(backend.n_hand_dof, 16)

    def test_mass_matrix_shape(self):
        backend = self._backend()
        backend.set_initial_state(backend._cfg)
        M = backend._hand_backend.get_mass_matrix()
        self.assertEqual(M.shape, (backend.n_hand_dof, backend.n_hand_dof))
        self.assertTrue(np.all(np.linalg.eigvalsh(M) > 0))

    def test_jacobian_shape(self):
        backend = self._backend()
        backend.set_initial_state(backend._cfg)
        link = backend._hand_entity.get_link("if_ds")
        J = backend._hand_backend.get_jacobian(link)
        self.assertEqual(J.shape, (6, backend.n_hand_dof))

    def test_bias_force_sign(self):
        """A free-floating box should have bias force ~= -gravity on the z DoF."""
        gs.init(backend=gs.cpu)
        scene = gs.Scene()
        box = scene.add_entity(gs.morphs.Box(size=(0.1, 0.1, 0.1), pos=(0, 0, 0.5)))
        scene.build()
        for _ in range(3):
            scene.step()
        from cloud_robotics_sim.backends.genesis_backend import (
            GenesisArticulationBackend,
        )

        b = GenesisArticulationBackend(box._morph, name="box")
        b.bind(box)
        C = b.get_bias_force()
        self.assertEqual(C.shape, (6,))
        # Following the MuJoCo convention M*qddot + C = tau + J^T*f, gravity
        # produces a positive bias force C_z = m*g for a floating body at rest.
        self.assertGreater(C[2], 0.01)


class TestGenesisSmoke(unittest.TestCase):
    def test_full_qp_baseline_runs(self):
        result = subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "examples" / "full_qp_baseline.py"),
                "--backend",
                "genesis",
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore",
            timeout=180,
        )
        # If Genesis or cloud_robotics_sim is not available the import fails
        # gracefully and the test is skipped.
        if any(
            msg in result.stderr
            for msg in (
                "No module named 'genesis'",
                "No module named 'cloud_robotics_sim'",
            )
        ):
            self.skipTest("Genesis/cloud_robotics_sim not installed")
        self.assertEqual(
            result.returncode,
            0,
            msg=f"Genesis full QP baseline failed\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}",
        )
        self.assertIn("QP status:", result.stdout)


if __name__ == "__main__":
    unittest.main()
