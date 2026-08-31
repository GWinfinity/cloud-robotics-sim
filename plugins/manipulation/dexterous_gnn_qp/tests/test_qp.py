"""Unit and smoke tests for the Phase-1 QP pipeline."""
from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dexterous_gnn_qp.core.dynamics.mujoco_model import object_grasp_wrench
from dexterous_gnn_qp.core.env.sim import Contact
from dexterous_gnn_qp.core.skeleton.rule_based import select_skeleton
from dexterous_gnn_qp.core.utils.config import DotDict
from dexterous_gnn_qp.core.utils.math import skew


class TestMathHelpers(unittest.TestCase):
    def test_skew_self_is_zero(self):
        v = np.array([1.0, 2.0, 3.0])
        self.assertTrue(np.allclose(skew(v) @ v, 0.0))

    def test_skew_cross_equivalence(self):
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([0.0, 1.0, 0.0])
        self.assertTrue(np.allclose(skew(a) @ b, np.cross(a, b)))


class TestGraspMap(unittest.TestCase):
    def test_force_pure_z(self):
        cnt = Contact(
            id=0,
            pos=np.array([0.1, 0.0, 0.0]),
            normal=np.array([0.0, 0.0, 1.0]),
            tangent1=np.array([1.0, 0.0, 0.0]),
            tangent2=np.array([0.0, 1.0, 0.0]),
            hand_body_id=1,
            hand_body_name="finger",
            object_body_id=2,
            mu=0.8,
        )
        obj_com = np.zeros(3)
        force = np.array([0.0, 0.0, 1.0])
        wrench = object_grasp_wrench(cnt, obj_com, force)
        # Spatial wrench [torque; force]
        expected = np.array([0.0, -0.1, 0.0, 0.0, 0.0, 1.0])
        self.assertTrue(np.allclose(wrench, expected))


class TestSkeletonSelection(unittest.TestCase):
    def _make_cfg(self):
        return DotDict(
            {
                "skeleton": {
                    "min_size": 3,
                    "max_size": 8,
                    "target_size": 5,
                    "force_threshold": 0.05,
                    "support_polygon_check": False,
                    "force_closure_check": False,
                }
            }
        )

    def test_falls_back_to_all_when_no_history(self):
        cfg = self._make_cfg()
        contacts = [self._contact(i) for i in range(4)]
        state = self._state(contacts)
        sel, edge = select_skeleton(state, None, cfg)
        self.assertEqual(sel, [0, 1, 2, 3])
        self.assertEqual(edge, [])

    def test_selects_top_forces(self):
        cfg = self._make_cfg()
        contacts = [self._contact(i) for i in range(7)]
        state = self._state(contacts)
        forces = np.zeros(21)
        # Set high forces on contacts 3, 1, 5.
        forces[3 * 3 : 3 * 3 + 3] = [0.0, 0.0, 2.0]
        forces[3 * 1 : 3 * 1 + 3] = [0.0, 0.0, 1.5]
        forces[3 * 5 : 3 * 5 + 3] = [0.0, 0.0, 1.0]
        sel, edge = select_skeleton(state, forces, cfg)
        self.assertIn(3, sel)
        self.assertIn(1, sel)
        self.assertIn(5, sel)
        self.assertLessEqual(len(sel), cfg.skeleton.max_size)

    @staticmethod
    def _contact(i: int):
        return Contact(
            id=i,
            pos=np.array([0.0, 0.0, 0.0]),
            normal=np.array([0.0, 0.0, 1.0]),
            tangent1=np.array([1.0, 0.0, 0.0]),
            tangent2=np.array([0.0, 1.0, 0.0]),
            hand_body_id=i + 1,
            hand_body_name=f"f{i}",
            object_body_id=100,
            mu=0.8,
        )

    @staticmethod
    def _state(contacts):
        return DotDict(
            {
                "contacts": contacts,
                "x_obj": np.zeros(3),
            }
        )


class TestSmoke(unittest.TestCase):
    def test_benchmark_script_runs(self):
        repo_root = Path(__file__).parent.parent
        result = subprocess.run(
            [
                sys.executable,
                str(repo_root / "examples" / "benchmark.py"),
                "--config",
                str(repo_root / "configs" / "leap_sphere_grasp.yaml"),
                "--headless",
                "--output",
                str(repo_root / "data" / "test_benchmark.csv"),
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore",
            timeout=120,
        )
        self.assertEqual(
            result.returncode,
            0,
            msg=f"benchmark.py failed\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}",
        )
        csv_path = repo_root / "data" / "test_benchmark.csv"
        self.assertTrue(csv_path.exists())
        text = csv_path.read_text(encoding="utf-8")
        self.assertIn("full_solve_time_ms", text)


if __name__ == "__main__":
    unittest.main()
