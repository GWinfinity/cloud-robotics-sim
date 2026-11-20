"""Unit tests for the RoboWits ports (no engine).

The slow engine smoke needs the asset mirror (absent in CI); these tests
exercise the same geometry offline: the footprint clip algorithm, the
dominos overlap predicate, the stand_pages upright/bounds checks and the
stack_cubes AABB relations on a stub world.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

REPO_ROOT = Path(__file__).resolve().parents[2]

WXYZ = np.array([1.0, 0.0, 0.0, 0.0])


class _StubWorld:
    """Poses by link name; contacts always empty."""

    def __init__(self, poses: dict[str, tuple]):
        self._poses = poses

    def link_names(self) -> list[str]:
        return list(self._poses)

    def get_link_pos(self, name: str) -> np.ndarray:
        return np.asarray(self._poses[name][0], dtype=float)

    def get_link_quat(self, name: str) -> np.ndarray:
        return np.asarray(self._poses[name][1], dtype=float)

    def get_qpos(self) -> np.ndarray:
        return np.zeros(3)

    def get_self_contacts(self) -> dict:
        return {"geom_a": [], "geom_b": [], "link_a": [], "link_b": []}


def _scene(poses: dict[str, tuple]) -> SimpleNamespace:
    far = (np.array([5.0, 5.0, 5.0]), WXYZ)
    merged = {"Gripper_Tip_R": far, "Gripper_Tip_L": far, **poses}
    return SimpleNamespace(entities={"world": _StubWorld(merged)})


def _task(rw: dict):
    from cloud_robotics_sim.core.task import TaskConfig
    from cloud_robotics_sim.runtime.efe import EfeRwTask

    return EfeRwTask(TaskConfig(name="efe", max_episode_steps=100), world={}, rw=rw)


# --------------------------------------------------------------------------- geometry


def test_clip_square_over_square() -> None:
    """Half-overlapping unit squares clip to a 1x2 region."""
    from cloud_robotics_sim.runtime.efe import _rw_clip_polygon, _rw_poly_area

    subject = [(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)]
    clip = [(1.0, 0.0), (3.0, 0.0), (3.0, 2.0), (1.0, 2.0)]
    inter = _rw_clip_polygon(subject, clip)
    assert _rw_poly_area(inter) == pytest.approx(2.0, abs=1e-9)
    assert _rw_poly_area(subject) == pytest.approx(4.0, abs=1e-9)


def test_footprint_rotated_box() -> None:
    """A box yawed 90 deg projects to its rotated corner rectangle."""
    from cloud_robotics_sim.runtime.efe import _rw_footprint_poly

    quat = _yaw_quat(np.pi / 2)
    poly = _rw_footprint_poly(np.zeros(3), quat, np.array([0.02, 0.05, 0.09]))
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    assert max(xs) - min(xs) == pytest.approx(0.05, abs=1e-9)
    assert max(ys) - min(ys) == pytest.approx(0.02, abs=1e-9)


def _yaw_quat(yaw: float) -> np.ndarray:
    import math

    return np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])


# --------------------------------------------------------------------------- predicates


def _dominos_rw() -> dict:
    return {
        "family": "dominos",
        "objects": {
            "red block": {"size": [0.02, 0.05, 0.09]},
            "target area": {"size": [0.06, 0.10, 0.002]},
        },
        "graded": {"rest": ["red block"], "release_also": []},
        "settle": {
            "settle_steps": 30,
            "check_steps": 60,
            "check_stride": 6,
            "rest_m": 0.002,
        },
        "robot_joints": ["Joint1_R"],
        "ee_links": {"right": "Gripper_Tip_R", "left": "Gripper_Tip_L"},
        "table_z": 0.76,
    }


def test_dominos_overlap_threshold() -> None:
    """>= 50 % of the red block's footprint over the target area."""
    rw = _dominos_rw()
    target_pos = np.array([0.96, 0.0, 0.761])
    on_target = _scene(
        {
            "red block": (np.array([0.96, 0.0, 0.805]), WXYZ),
            "target area": (target_pos, WXYZ),
        }
    )
    task = _task(rw)
    assert task._predicate_dominos(on_target) is True
    # shift so only 25 % of the block remains over the target
    quarter = _scene(
        {
            "red block": (np.array([1.015, 0.0, 0.805]), WXYZ),
            "target area": (target_pos, WXYZ),
        }
    )
    assert task._predicate_dominos(quarter) is False


def _stand_rw() -> dict:
    return {
        "family": "stand_pages",
        "objects": {
            "page A with eyelets": {"size": [0.2, 0.28, 0.005]},
            "page B with eyelets": {"size": [0.2, 0.28, 0.005]},
        },
        "graded": {
            "rest": ["page A with eyelets", "page B with eyelets"],
            "release_also": ["stabilizing bar"],
        },
        "settle": {
            "settle_steps": 30,
            "check_steps": 60,
            "check_stride": 6,
            "rest_m": 0.002,
        },
        "robot_joints": ["Joint1_R"],
        "ee_links": {"right": "Gripper_Tip_R", "left": "Gripper_Tip_L"},
        "table_z": 0.76,
        "table_bounds": {"x": [0.21, 1.0], "y": [-0.69, 0.69]},
    }


def test_stand_pages_clauses() -> None:
    """Table-bounds, on-table and not-held clauses.

    Upstream semantics: euler ex/ey near 0/180 passes for a flat page
    too - the goal gate is the settle window, not the tilt.
    """
    rw = _stand_rw()
    task = _task(rw)
    lying = _scene(
        {
            "page A with eyelets": (np.array([0.5, 0, 0.76]), WXYZ),
            "page B with eyelets": (np.array([0.5, 0.3, 0.76]), WXYZ),
        }
    )
    assert task._predicate_stand_pages(lying) is True
    # a page outside the table bounds fails
    outside = _scene(
        {
            "page A with eyelets": (np.array([2.0, 0, 0.76]), WXYZ),
            "page B with eyelets": (np.array([0.5, 0.3, 0.76]), WXYZ),
        }
    )
    assert task._predicate_stand_pages(outside) is False
    # a gripper tip inside a page's AABB fails the not-held clause
    held = _scene(
        {
            "Gripper_Tip_R": (np.array([0.5, 0.0, 0.76]), WXYZ),
            "page A with eyelets": (np.array([0.5, 0, 0.76]), WXYZ),
            "page B with eyelets": (np.array([0.5, 0.3, 0.76]), WXYZ),
        }
    )
    assert task._predicate_stand_pages(held) is False


def test_stack_cubes_relations() -> None:
    """Bases on the table touching; apex on top over the dot."""
    rw = {
        "family": "stack_cubes",
        "objects": {
            "base cube 1": {"size": [0.04, 0.04, 0.04]},
            "base cube 2": {"size": [0.04, 0.04, 0.04]},
            "apex cube": {"size": [0.04, 0.04, 0.04]},
            "apex target": {"size": [0.05, 0.05, 0.002]},
        },
        "graded": {
            "rest": ["base cube 1", "base cube 2", "apex cube"],
            "release_also": [],
        },
        "settle": {
            "settle_steps": 30,
            "check_steps": 60,
            "check_stride": 6,
            "rest_m": 0.002,
        },
        "robot_joints": ["Joint1_R"],
        "ee_links": {"right": "Gripper_Tip_R", "left": "Gripper_Tip_L"},
        "table_z": 0.76,
    }
    task = _task(rw)

    def cube(x, y, z):
        return (np.array([x, y, z]), WXYZ)

    # two bases side by side on the table, apex on top over the dot
    ok = _scene(
        {
            "base cube 1": cube(0.90, -0.02, 0.78),
            "base cube 2": cube(0.90, 0.02, 0.78),
            "apex cube": cube(0.90, 0.0, 0.82),
            "apex target": cube(0.90, 0.0, 0.761),
        }
    )
    assert task._predicate_stack_cubes(ok) is True
    # apex beside the dot: distance 0.05 > 0.025
    off = _scene(
        {
            "base cube 1": cube(0.90, -0.02, 0.78),
            "base cube 2": cube(0.90, 0.02, 0.78),
            "apex cube": cube(0.95, 0.0, 0.82),
            "apex target": cube(0.90, 0.0, 0.761),
        }
    )
    assert task._predicate_stack_cubes(off) is False
