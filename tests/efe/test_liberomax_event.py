"""Tests for the LIBERO-MAX online-event task (runtime/efe.py)."""

from types import SimpleNamespace

import numpy as np
import pytest

from cloud_robotics_sim.core.task import TaskConfig
from cloud_robotics_sim.runtime.efe import EfeLiberoMaxTask


class _Entity:
    def __init__(self, pos):
        self.pos = np.asarray(pos, dtype=float)
        self.set_calls = []

    def get_pos(self):
        return self.pos

    def set_pos(self, pos):
        self.set_calls.append(np.asarray(pos, dtype=float).copy())
        self.pos = np.asarray(pos, dtype=float)


def _robot(ee_pos):
    entity = SimpleNamespace(get_link_pos=lambda link: np.asarray(ee_pos, dtype=float))
    return SimpleNamespace(entity=entity)


def _task(event):
    """Build an EfeLiberoMaxTask with the given event block."""
    return EfeLiberoMaxTask(TaskConfig(name="efe", max_episode_steps=10), event=event)


def _scene(**entities):
    """Wrap stub entities in a scene-like namespace."""
    return SimpleNamespace(entities=entities)


class TestSetup:
    """Setup-phase behaviour (off-scene removals)."""

    def test_reset_parks_removed_objects(self):
        bowl = _Entity((0.5, 0.0, 0.9))
        task = _task(
            {
                "setup": [{"operation": "remove_object", "object": "bowl"}],
                "trigger": {
                    "type": "on_proximity",
                    "value": "bowl",
                    "distance_m": 0.18,
                },
                "change": {
                    "operation": "move_object",
                    "object": "bowl",
                    "delta_position_m": [0.1, 0.0, 0.0],
                },
            }
        )
        scene = _scene(bowl=bowl)
        task.reset(scene, _robot((0, 0, 0)), seed=0)
        assert bowl.pos[2] == -5.0
        assert "bowl" in task._parked

    def test_reset_without_event_is_noop(self):
        task = _task(None)
        bowl = _Entity((0.5, 0.0, 0.9))
        scene = _scene(bowl=bowl)
        task.reset(scene, _robot((0, 0, 0)), seed=0)
        task.step(scene, _robot((0, 0, 0)), np.zeros(7))
        assert bowl.pos[2] == 0.9


class TestTrigger:
    """on_proximity trigger behaviour."""

    def _event(self):
        """A minimal on_proximity move_object event."""
        return {
            "trigger": {"type": "on_proximity", "value": "bowl", "distance_m": 0.18},
            "change": {
                "operation": "move_object",
                "object": "bowl",
                "delta_position_m": [0.1, -0.05, 0.0],
            },
        }

    def test_fires_when_ee_within_distance(self):
        bowl = _Entity((0.5, 0.0, 0.9))
        task = _task(self._event())
        scene = _scene(bowl=bowl)
        task.reset(scene, _robot((0, 0, 0)), seed=0)
        _, _, _, info = task.step(scene, _robot((0.4, 0.0, 0.9)), np.zeros(7))
        assert bowl.pos[0] == pytest.approx(0.6)
        assert bowl.pos[1] == pytest.approx(-0.05)
        assert info["libero_max_event"]["step"] == 1

    def test_applies_exactly_once(self):
        bowl = _Entity((0.5, 0.0, 0.9))
        task = _task(self._event())
        scene = _scene(bowl=bowl)
        task.reset(scene, _robot((0, 0, 0)), seed=0)
        task.step(scene, _robot((0.4, 0.0, 0.9)), np.zeros(7))
        _, _, _, info = task.step(scene, _robot((0.4, 0.0, 0.9)), np.zeros(7))
        assert "libero_max_event" not in info
        assert len(bowl.set_calls) == 1

    def test_no_fire_when_far(self):
        bowl = _Entity((0.5, 0.0, 0.9))
        task = _task(self._event())
        scene = _scene(bowl=bowl)
        task.reset(scene, _robot((0, 0, 0)), seed=0)
        _, _, _, info = task.step(scene, _robot((-0.5, 0.0, 0.9)), np.zeros(7))
        assert "libero_max_event" not in info
        assert bowl.pos[0] == pytest.approx(0.5)

    def test_unknown_trigger_entity_raises(self):
        bowl = _Entity((0.5, 0.0, 0.9))
        task = _task(self._event())
        scene = _scene(bowl=bowl)
        task.reset(scene, _robot((0, 0, 0)), seed=0)
        task._event["trigger"]["value"] = "ghost"
        with pytest.raises(KeyError, match="ghost"):
            task.step(scene, _robot((0.4, 0.0, 0.9)), np.zeros(7))


class TestOperations:
    """Change-operation semantics (physical vs observation-space)."""

    def test_insert_obstacle_uses_parked_z_and_path(self):
        obstacle = _Entity((0.0, 0.0, -5.0))
        soup = _Entity((0.6, 0.0, 0.9))
        task = _task(
            {
                "setup": [{"operation": "remove_object", "object": "obstacle"}],
                "trigger": {
                    "type": "on_proximity",
                    "value": "soup",
                    "distance_m": 0.18,
                },
                "change": {
                    "operation": "insert_obstacle",
                    "object": "obstacle",
                    "path_target": "soup",
                    "path_fraction": 0.5,
                    "lateral_offset_m": 0.1,
                },
            }
        )
        scene = _scene(obstacle=obstacle, soup=soup)
        task.reset(scene, _robot((0.2, 0.0, 0.9)), seed=0)
        # parked z recorded at setup, even though the dump starts off-world
        assert task._parked["obstacle"][2] == -5.0
        _, _, _, info = task.step(scene, _robot((0.5, 0.0, 0.9)), np.zeros(7))
        # position = eef + 0.5*(soup-eef), lateral +0.1 in y, z = parked z
        assert obstacle.pos[0] == pytest.approx(0.55)
        assert obstacle.pos[1] == pytest.approx(0.1)
        assert obstacle.pos[2] == pytest.approx(-5.0)
        assert info["libero_max_event"]["result"]["operation"] == "insert_obstacle"

    def test_insert_distractors_places_each(self):
        target = _Entity((0.1, 0.1, 0.9))
        a = _Entity((0.0, 0.0, -5.0))
        b = _Entity((0.0, 0.0, -5.0))
        task = _task(
            {
                "trigger": {
                    "type": "on_proximity",
                    "value": "target",
                    "distance_m": 0.18,
                },
                "change": {
                    "operation": "insert_distractors",
                    "placements": [
                        {"object": "a", "position_m": [0.3, 0.1, 0.9]},
                        {
                            "object": "b",
                            "relative_to": "a",
                            "offset_m": [0.0, 0.2, 0.0],
                        },
                    ],
                },
            }
        )
        scene = _scene(target=target, a=a, b=b)
        task.reset(scene, _robot((0, 0, 0)), seed=0)
        task.step(scene, _robot((0.1, 0.1, 0.9)), np.zeros(7))
        # placement order: a teleports first, b anchors to a's NEW position
        assert a.pos[0] == pytest.approx(0.3)
        assert b.pos[1] == pytest.approx(0.3)

    def test_observation_ops_change_no_physics(self):
        stove = _Entity((-0.26, 0.2, 0.9))
        task = _task(
            {
                "trigger": {
                    "type": "on_proximity",
                    "value": "stove",
                    "distance_m": 0.18,
                },
                "change": {
                    "operation": "set_visual_theme",
                    "rgb_permutation": [2, 0, 1],
                },
            }
        )
        scene = _scene(stove=stove)
        task.reset(scene, _robot((0, 0, 0)), seed=0)
        before = stove.pos.copy()
        _, _, _, info = task.step(scene, _robot((-0.2, 0.2, 0.9)), np.zeros(7))
        assert stove.pos[0] == pytest.approx(before[0])
        result = info["libero_max_event"]["result"]
        assert result == {"operation": "set_visual_theme", "simulator_changed": False}

    def test_unsupported_operation_raises(self):
        bowl = _Entity((0.5, 0.0, 0.9))
        task = _task(
            {
                "trigger": {
                    "type": "on_proximity",
                    "value": "bowl",
                    "distance_m": 0.18,
                },
                "change": {"operation": "replace_instruction"},
            }
        )
        scene = _scene(bowl=bowl)
        task.reset(scene, _robot((0, 0, 0)), seed=0)
        with pytest.raises(ValueError, match="operation"):
            task.step(scene, _robot((0.4, 0.0, 0.9)), np.zeros(7))
