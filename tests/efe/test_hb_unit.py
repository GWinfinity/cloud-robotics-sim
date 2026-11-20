"""Unit tests for the HumanoidBench ports (no engine).

The slow engine smoke needs the asset mirror (absent in CI); these tests
exercise the same formula-level logic offline: the dm_control tolerance
port against hand-computed reference values, and the per-family reward /
termination ports on a MuJoCo-free stub world whose poses/velocities are
set by hand. The stub mirrors the shim used by the probe-side formula
equivalence check (`.efe_stage/hb_formula_check.py`).
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

REPO_ROOT = Path(__file__).resolve().parents[2]


# --------------------------------------------------------------------------- tolerance


def test_tolerance_gaussian_at_margin() -> None:
    """Gaussian sigmoid: value_at_margin at d=1, d=2 gives value^4."""
    from cloud_robotics_sim.runtime.efe import _hb_tolerance

    # gaussian: value_at_margin 0.1 at d=1 -> exp(-0.5 * scale^2) with
    # scale = sqrt(-2 ln 0.1); d = 2 -> 0.1**4
    v1 = _hb_tolerance(1.0, bounds=(0.0, 0.0), margin=1.0)
    assert v1 == pytest.approx(0.1, abs=1e-12)
    v2 = _hb_tolerance(2.0, bounds=(0.0, 0.0), margin=1.0)
    assert v2 == pytest.approx(0.1**4, abs=1e-12)


def test_tolerance_inside_and_linear() -> None:
    """Inside bounds is 1; linear/quadratic margins decay as documented."""
    from cloud_robotics_sim.runtime.efe import _hb_tolerance

    assert _hb_tolerance(0.5, bounds=(0.0, 1.0), margin=1.0) == 1.0
    # linear, value_at_margin 0 at d=1: d=0.25 -> 0.75
    v = _hb_tolerance(
        1.25, bounds=(0.0, 1.0), margin=1.0, sigmoid="linear", value_at_margin=0.0
    )
    assert v == pytest.approx(0.75, abs=1e-12)
    # quadratic, value_at_margin 0 at d=1: d=0.5 -> 0.75
    v = _hb_tolerance(
        1.5, bounds=(0.0, 1.0), margin=1.0, sigmoid="quadratic", value_at_margin=0.0
    )
    assert v == pytest.approx(0.75, abs=1e-12)
    # margin 0: binary
    assert _hb_tolerance(1.5, bounds=(0.0, 1.0), margin=0.0) == 0.0


# --------------------------------------------------------------------------- stub world


class _Entity:
    """Genesis-entity-shaped stub over hand-set poses/velocities."""

    def __init__(self, bodies: dict[str, dict]):
        # bodies: name -> {pos, quat, lin, ang, mass}
        self.links = [SimpleNamespace(name=n) for n in bodies]
        self._b = bodies

    def get_links_vel(self, links_idx_local=None, envs_idx=None, ref=None):
        idx = links_idx_local if links_idx_local is not None else range(len(self.links))
        return np.array([self._b[self.links[i].name]["lin"] for i in idx])

    def get_links_COM(self, links_idx_local=None, envs_idx=None):  # noqa: N802
        """Link-COM positions (name mirrors the Genesis API)."""
        idx = links_idx_local if links_idx_local is not None else range(len(self.links))
        return np.array([self._b[self.links[i].name]["pos"] for i in idx])

    def get_links_ang(self, links_idx_local=None, envs_idx=None):
        idx = links_idx_local if links_idx_local is not None else range(len(self.links))
        return np.array([self._b[self.links[i].name]["ang"] for i in idx])

    def get_links_mass(self, links_idx_local=None, envs_idx=None):
        idx = links_idx_local if links_idx_local is not None else range(len(self.links))
        return np.array([self._b[self.links[i].name].get("mass", 1.0) for i in idx])

    def get_dofs_actuator_force(self):
        return np.zeros(1)  # unused: _actuator_forces is overridden below


class _World:
    def __init__(self, bodies: dict[str, dict], qpos=None):
        self._entity = _Entity(bodies)
        self._qpos = np.asarray(qpos, dtype=float) if qpos is not None else np.zeros(3)

    def link_names(self):
        return list(self._entity._b)

    def get_link_pos(self, name):
        return np.asarray(self._entity._b[name]["pos"], dtype=float)

    def get_link_quat(self, name):
        return np.asarray(self._entity._b[name]["quat"], dtype=float)

    def get_qpos(self):
        return self._qpos

    def set_qpos(self, q):
        self._qpos = np.asarray(q, dtype=float)

    def set_qvel(self, v):
        pass

    def get_self_contacts(self):
        return {"geom_a": [], "geom_b": [], "link_a": [], "link_b": []}

    def _resolve_entity(self):
        return self._entity


def _standing_bodies(root_z: float = 0.98) -> dict:
    quat = np.array([1.0, 0.0, 0.0, 0.0])
    return {
        "pelvis": {
            "pos": [0, 0, root_z],
            "quat": quat,
            "lin": [0, 0, 0],
            "ang": [0, 0, 0],
            "mass": 5.0,
        },
        "torso_link": {
            "pos": [0, 0, root_z + 0.3],
            "quat": quat,
            "lin": [0, 0, 0],
            "ang": [0, 0, 0],
            "mass": 20.0,
        },
        "head": {
            "pos": [0, 0, root_z + 0.7],
            "quat": quat,
            "lin": [0, 0, 0],
            "ang": [0, 0, 0],
            "mass": 1.0,
        },
    }


def _task(hb: dict, monkeypatch=None):
    from cloud_robotics_sim.core.task import TaskConfig
    from cloud_robotics_sim.runtime.efe import EfeHbTask

    task = EfeHbTask(TaskConfig(name="efe", max_episode_steps=1000), world={}, hb=hb)
    if monkeypatch is not None:
        monkeypatch.setattr(task, "_actuator_forces", lambda scene: np.zeros(19))
    return task


# --------------------------------------------------------------------------- walk reward


def _walk_hb() -> dict:
    return {
        "task_short": "walk",
        "success_bar": 700.0,
        "class_consts": {"_move_speed": 1.0},
        "module_globals": {
            "humanoid_bench.envs.basic_locomotion_envs": "_STAND_HEIGHT = 1.65"
        },
        "robot_links": [
            {"name": "pelvis", "mass": 5.0},
            {"name": "torso_link", "mass": 20.0},
            {"name": "head", "mass": 1.0},
        ],
        "actuated": [{"dofadr": 6 + i, "moment": [[6 + i, 1.0]]} for i in range(19)],
        "nu": 19,
        "termination": {"kind": "qpos_z", "qpos_z_lt": 0.2},
        "sites": {
            "head": {"body": "head", "local": [0, 0, 0], "local_quat": [1, 0, 0, 0]},
            "imu": {"body": "pelvis", "local": [0, 0, 0], "local_quat": [1, 0, 0, 0]},
            "left_hand": {
                "body": "torso_link",
                "local": [0, 0.3, 0],
                "local_quat": [1, 0, 0, 0],
            },
            "right_hand": {
                "body": "torso_link",
                "local": [0, -0.3, 0],
                "local_quat": [1, 0, 0, 0],
            },
        },
    }


def test_walk_reward_standing_still(monkeypatch) -> None:
    """Standing at rest: move raw 0 -> reward = small_control * (1/6)."""
    """Standing at rest: move raw 0 -> reward = small_control * 1 * (1/6)."""
    task = _task(_walk_hb(), monkeypatch)
    bodies = _standing_bodies()
    scene = SimpleNamespace(
        entities={"world": _World(bodies, qpos=[0, 0, 0.98, 1, 0, 0, 0])}
    )
    task.reset(scene, None, 0)
    reward, detail = task._reward_walk(scene)
    assert reward == pytest.approx((4 + 1.0) / 5 * 1.0 * (1.0 / 6.0), abs=1e-12)
    assert detail["standing"] == 1.0


def test_termination_qpos_z(monkeypatch) -> None:
    """The fall termination reads the free-joint z from qpos."""
    task = _task(_walk_hb(), monkeypatch)
    bodies = _standing_bodies(root_z=0.1)
    scene = SimpleNamespace(
        entities={"world": _World(bodies, qpos=[0, 0, 0.1, 1, 0, 0, 0])}
    )
    task.reset(scene, None, 0)
    assert task._check_terminated(scene) is True
    bodies2 = _standing_bodies()
    scene2 = SimpleNamespace(
        entities={"world": _World(bodies2, qpos=[0, 0, 0.98, 1, 0, 0, 0])}
    )
    assert task._check_terminated(scene2) is False


def test_door_openness_and_passage(monkeypatch) -> None:
    """Openness clamps at 1; passage reads the imu site x."""
    hb = {
        "task_short": "door",
        "module_globals": {"humanoid_bench.envs.door": "_STAND_HEIGHT = 1.65"},
        "robot_links": [{"name": "pelvis", "mass": 25.0}],
        "actuated": [{"dofadr": 6 + i, "moment": [[6 + i, 1.0]]} for i in range(19)],
        "nu": 19,
        "termination": {"kind": "qpos_z", "qpos_z_lt": 0.58},
        "sites": {
            "head": {"body": "head", "local": [0, 0, 0], "local_quat": [1, 0, 0, 0]},
            "imu": {"body": "pelvis", "local": [0, 0, 0], "local_quat": [1, 0, 0, 0]},
            "left_hand": {
                "body": "torso_link",
                "local": [0, 0.3, 0],
                "local_quat": [1, 0, 0, 0],
            },
            "right_hand": {
                "body": "torso_link",
                "local": [0, -0.3, 0],
                "local_quat": [1, 0, 0, 0],
            },
        },
    }
    task = _task(hb, monkeypatch)
    quat = np.array([1.0, 0, 0, 0])
    bodies = {
        "pelvis": {
            "pos": [1.5, 0, 0.98],
            "quat": quat,
            "lin": [0, 0, 0],
            "ang": [0, 0, 0],
            "mass": 25.0,
        },
        "torso_link": {
            "pos": [0, 0, 1.0],
            "quat": quat,
            "lin": [0, 0, 0],
            "ang": [0, 0, 0],
            "mass": 1.0,
        },
        "head": {
            "pos": [0, 0, 1.7],
            "quat": quat,
            "lin": [0, 0, 0],
            "ang": [0, 0, 0],
            "mass": 1.0,
        },
        "door_hatch": {
            "pos": [0.5, 0.3, 1.0],
            "quat": quat,
            "lin": [0, 0, 0],
            "ang": [0, 0, 0],
            "mass": 1.0,
        },
    }
    qpos = np.zeros(28)
    qpos[2] = 0.98
    qpos[-2] = 2.0  # openness > 1 -> clamped
    qpos[-1] = 1.0
    scene = SimpleNamespace(entities={"world": _World(bodies, qpos=qpos)})
    task.reset(scene, None, 0)
    reward, detail = task._reward_door(scene)
    assert 0.45 * 1.0 <= reward <= 1.0
    assert detail["door_openness"] == 1.0
    assert detail["passage"] == 1.0  # imu x = 1.5 > 1.2


def test_bookshelf_latch_advances(monkeypatch) -> None:
    """Dist < 0.15 increments task_index and adds the 100*k bonus."""
    hb = {
        "task_short": "bookshelf_simple",
        "module_globals": {"humanoid_bench.envs.bookshelf": "_STAND_HEIGHT = 1.65"},
        "robot_links": [{"name": "pelvis", "mass": 25.0}],
        "actuated": [{"dofadr": 6 + i, "moment": [[6 + i, 1.0]]} for i in range(19)],
        "nu": 19,
        "termination": {"kind": "bookshelf", "qpos_z_lt": 0.58, "min_obj_z_lt": 0.5},
        "task_state": {
            "bookshelf_objects": ["obj0", "obj1"],
            "placement_goals": [[0.5, 0.0, 0.8], [0.0, 0.0, 0.8]],
            "task_index": 0,
        },
        "sites": {
            "head": {"body": "head", "local": [0, 0, 0], "local_quat": [1, 0, 0, 0]},
            "imu": {"body": "pelvis", "local": [0, 0, 0], "local_quat": [1, 0, 0, 0]},
            "left_hand": {
                "body": "torso_link",
                "local": [0, 0.3, 0],
                "local_quat": [1, 0, 0, 0],
            },
            "right_hand": {
                "body": "torso_link",
                "local": [0, -0.3, 0],
                "local_quat": [1, 0, 0, 0],
            },
        },
    }
    task = _task(hb, monkeypatch)
    quat = np.array([1.0, 0, 0, 0])
    bodies = {
        "pelvis": {
            "pos": [0, 0, 0.98],
            "quat": quat,
            "lin": [0, 0, 0],
            "ang": [0, 0, 0],
            "mass": 25.0,
        },
        "torso_link": {
            "pos": [0, 0, 1.0],
            "quat": quat,
            "lin": [0, 0, 0],
            "ang": [0, 0, 0],
            "mass": 1.0,
        },
        "head": {
            "pos": [0, 0, 1.7],
            "quat": quat,
            "lin": [0, 0, 0],
            "ang": [0, 0, 0],
            "mass": 1.0,
        },
        "obj0": {
            "pos": [0.51, 0.0, 0.8],
            "quat": quat,
            "lin": [0, 0, 0],
            "ang": [0, 0, 0],
            "mass": 0.1,
        },
    }
    scene = SimpleNamespace(
        entities={"world": _World(bodies, qpos=[0, 0, 0.98, 1, 0, 0, 0])}
    )
    task.reset(scene, None, 0)
    reward, detail = task._reward_bookshelf_simple(scene)
    assert task._task_index == 1
    assert detail["success_subtasks"] == 1
    assert reward >= 100.0
