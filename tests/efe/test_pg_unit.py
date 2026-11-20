"""Unit tests for the MuJoCo Playground ports (no engine).

Covers the formula-level logic of ``EfePgTask`` that the slow engine smoke
cannot reach in CI (no asset mirror there): the gait Bezier curve, the
quadruped check aggregation (segment means, tracking bars, horizon rule),
the joystick termination box, and the peg-insertion axial-parameter check
on a stub world.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

REPO_ROOT = Path(__file__).resolve().parents[2]


class _StubEntity:
    """World-entity stub: poses by link name, no contacts, qpos given."""

    def __init__(self, poses: dict[str, tuple], qpos: np.ndarray):
        self._poses = poses
        self._qpos = np.asarray(qpos, dtype=float)

    def link_names(self) -> list[str]:
        return list(self._poses)

    def get_link_pos(self, name: str) -> np.ndarray:
        return np.asarray(self._poses[name][0], dtype=float)

    def get_link_quat(self, name: str) -> np.ndarray:
        return np.asarray(self._poses[name][1], dtype=float)

    def get_qpos(self) -> np.ndarray:
        return self._qpos

    def get_self_contacts(self) -> dict:
        return {"geom_a": [], "geom_b": [], "link_a": [], "link_b": []}


def _task(pg: dict):
    from cloud_robotics_sim.core.task import TaskConfig
    from cloud_robotics_sim.runtime.efe import EfePgTask

    return EfePgTask(TaskConfig(name="efe", max_episode_steps=1000), world={}, pg=pg)


# --------------------------------------------------------------------------- gait


def test_gait_rz_matches_bezier_curve() -> None:
    """get_rz: cubic Bezier up then down; the curve peaks at phase 0."""
    task = _task({"task_kind": "getup"})
    # phi = 0 -> x = 0.5, the first-half boundary: Bez(0, h, 1) = h
    assert float(task._gait_rz(np.array([0.0]), 0.04)[0]) == pytest.approx(
        0.04, abs=1e-12
    )
    # phi = pi -> x = 1.0: Bez(h, 0, 1) = 0
    assert float(task._gait_rz(np.array([np.pi]), 0.04)[0]) == pytest.approx(
        0.0, abs=1e-12
    )
    # phase pi/2 -> x = 0.75 in the second half: Bez(h, 0, 0.5) = h/2
    assert float(task._gait_rz(np.array([np.pi / 2]), 0.04)[0]) == pytest.approx(
        0.02, abs=1e-12
    )


# --------------------------------------------------------------------------- quadruped check


def _joystick_pg() -> dict:
    return {
        "task_kind": "joystick",
        "horizon": 10,
        "control_dt": 0.02,
        "constants": {
            "settle_steps": 2,
            "lin_tol": 0.2,
            "yaw_tol": 0.3,
            "gait_tol": 0.05,
            "hold_steps": 25,
            "tilt_max_deg": 10.0,
            "pose_hold_steps": 100,
            "up_z_min": 0.0,
        },
        "commands": [
            {"start_step": 0, "vx": 1.0, "vy": 0.0, "wz": 0.0},
            {"start_step": 5, "vx": 0.5, "vy": 0.0, "wz": 0.0},
        ],
    }


def test_joystick_check_aggregates_segment_means() -> None:
    """Segment means over steps after each segment's settle window."""
    """Segment means over steps after each segment's settle window."""
    task = _task(_joystick_pg())
    # 10 steps; segment 0 scores steps [2, 5), segment 1 steps [7, 10)
    task._log = [(0.1, 0.0, 5.0, 0.3, 1.0)] * 10
    task._steps = 10
    task._fell_at = None
    out = task._quadruped_check()
    assert out["success"] is True
    assert len(out["segments"]) == 2
    assert out["segments"][0]["lin_err"] == 0.1
    assert out["segments"][1]["lin_err"] == 0.1


def test_joystick_check_requires_full_horizon() -> None:
    """Success needs the full horizon, not just clean tracking."""
    task = _task(_joystick_pg())
    task._log = [(0.1, 0.0, 5.0, 0.3, 1.0)] * 3
    task._steps = 3
    task._fell_at = None
    out = task._quadruped_check()
    assert out["success"] is False
    assert out["tracking_ok"] is True  # only the reachability of the horizon fails


def test_joystick_check_fall_disqualifies() -> None:
    """A recorded fall disqualifies the episode even with clean tracking."""
    task = _task(_joystick_pg())
    task._log = [(0.1, 0.0, 5.0, 0.3, 1.0)] * 10
    task._steps = 10
    task._fell_at = 7
    out = task._quadruped_check()
    assert out["success"] is False


def test_terminated_joint_box() -> None:
    """Barkour's joint box flags qpos outside the tiled bounds."""
    pg = _joystick_pg()
    pg["constants"]["joint_box"] = [[-0.7, -1.0, 0.05], [0.52, 2.1, 2.1]]
    task = _task(pg)
    qpos = np.zeros(19)
    qpos[7:] = 0.1
    scene = SimpleNamespace(entities={"world": _StubEntity({}, qpos)})
    assert task._terminated(scene, up_z=1.0) is False
    qpos[7 + 3] = 3.0  # abduction beyond 0.52
    assert task._terminated(scene, up_z=1.0) is True
    qpos[7 + 3] = 0.1
    assert task._terminated(scene, up_z=-0.1) is True  # up_z below 0


# --------------------------------------------------------------------------- peg check


def _peg_scene() -> SimpleNamespace:
    """Socket along +x at origin; peg tip 20 mm inside, on the axis."""
    wxyz = np.array([1.0, 0.0, 0.0, 0.0])
    poses = {
        "socket": (np.zeros(3), wxyz),
        "peg": (np.zeros(3), wxyz),
    }
    world = _StubEntity(poses, np.zeros(8))
    return SimpleNamespace(entities={"world": world})


def test_peg_axial_parameter_check() -> None:
    """The tip's axial parameter t stays within [t_min, depth_t_max]."""
    pg = {
        "task_kind": "AlohaSinglePegInsertion",
        "sites": {
            "socket_rear": {
                "body": "socket",
                "local": [0.0, 0, 0],
                "local_quat": [1, 0, 0, 0],
            },
            "socket_entrance": {
                "body": "socket",
                "local": [0.048, 0, 0],
                "local_quat": [1, 0, 0, 0],
            },
            "peg_end2": {
                "body": "peg",
                "local": [-0.028, 0, 0],
                "local_quat": [1, 0, 0, 0],
            },
        },
        "check": {
            "sites": {
                "socket_rear": "socket_rear",
                "socket_entrance": "socket_entrance",
                "peg_tip": "peg_end2",
            },
            "t_min": -0.875,
            "depth_t_max": 0.75,
            "line_tol_m": 0.005,
        },
    }
    scene = _peg_scene()
    info = _task(pg)._manip_check(scene)
    assert info["success"] is True  # t = 0.583.. within [-0.875, 0.75], dist 0
    assert info["axial_t"] == pytest.approx(
        -0.5833, abs=1e-3
    )  # t<0: behind the entrance, inside the cavity
