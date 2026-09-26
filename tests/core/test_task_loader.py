"""Tests for the task-level YAML loader (core/task_loader.py)."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import yaml

from cloud_robotics_sim.backend import SceneBackend
from cloud_robotics_sim.core.composer import ComposedEnvironment
from cloud_robotics_sim.core.config_loader import ConfigError
from cloud_robotics_sim.core.registry import AssetRegistry
from cloud_robotics_sim.core.success import (
    DistanceThresholdCondition,
)
from cloud_robotics_sim.core.task import PickPlaceTask, TaskConfig
from cloud_robotics_sim.core.task_loader import (
    ConfigurableTask,
    EvaluationSpec,
    apply_layout,
    build_task_components,
    compose_task,
    load_task_spec,
    make_apply_seed_fn,
    parse_task_spec,
    sample_layout,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_TASK = REPO_ROOT / "configs" / "tasks" / "pick_place_cube.yaml"

VALID_TASK_YAML = """
version: 1
task:
  name: pick_place_cube
  type: pick_place
  robot: franka_panda
  scene: empty_room
  max_episode_steps: 200
  object_name: red_cube
scene:
  size: [10.0, 10.0, 3.0]
assets:
  task_relevant:
    - name: red_cube
      shape_type: box
      size: [0.05, 0.05, 0.05]
      position: [0.5, 0.0, 0.05]
      color: [0.9, 0.2, 0.2]
      static: false
  distractors:
    - name: blue_cube
      shape_type: box
      size: [0.04, 0.04, 0.04]
      position: [0.7, 0.15, 0.04]
init_distribution:
  object_poses:
    red_cube:
      position: [[0.4, 0.6], [-0.2, 0.2], [0.05, 0.05]]
  clutter_layout:
    region: [0.6, 0.9, -0.3, 0.3, 0.04]
  articulation_states: {}
randomization:
  friction: [0.3, 0.8]
cameras:
  - name: eval_cam
    pos: [1.5, 0.0, 1.2]
    look_at: [0.5, 0.0, 0.0]
    resolution: [640, 480]
success:
  type: distance_threshold
  params:
    object: red_cube
    target_position: [0.5, 0.0, 0.1]
    threshold: 0.05
    failure_stage: place_fail
evaluation:
  seeds: [0, 1, 2]
  episodes_per_seed: 2
simulation:
  headless: true
"""


def _valid_spec():
    return parse_task_spec(yaml.safe_load(VALID_TASK_YAML), source="test")


def _make_mock_backend(scene_backend: SceneBackend | None = None):
    """Create a mock SimulatorBackend returning the given scene_backend."""
    backend = MagicMock()
    backend.name = "genesis"
    if scene_backend is None:
        scene_backend = MagicMock(spec=SceneBackend)
        scene_backend.backend = backend
    backend.create_scene.return_value = scene_backend
    backend.create_box.return_value = MagicMock()
    backend.create_sphere.return_value = MagicMock()
    backend.create_cylinder.return_value = MagicMock()
    backend.create_mesh.return_value = MagicMock()
    backend.load_mjcf.return_value = MagicMock()
    backend.load_urdf.return_value = MagicMock()
    backend.create_light.return_value = MagicMock()
    return backend, scene_backend


def _mock_scene():
    scene = MagicMock()
    scene.config = SimpleNamespace(name="empty_room")
    scene.get_spawn_positions.return_value = [(0.0, 0.0, 0.1)]
    return scene


def _mock_robot():
    robot = MagicMock()
    robot.config = SimpleNamespace(name="franka_panda")
    robot.cameras = {}
    return robot


def _registry_with(scene=None, robot=None, task=None):
    registry = AssetRegistry()
    registry.scenes.register("empty_room")(lambda **kwargs: scene)
    registry.robots.register("franka_panda")(lambda **kwargs: robot)
    registry.tasks.register("pick_place")(lambda **kwargs: task)
    return registry


class TestEvaluationSpec:
    """Tests for the evaluation protocol expansion."""

    def test_episodes_expansion(self):
        """Seeds x episodes expands to the full evaluation grid."""
        spec = EvaluationSpec(seeds=[0, 1], episodes_per_seed=3)
        assert spec.episodes() == [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]

    def test_single_episode_default(self):
        """episodes_per_seed defaults to one episode per seed."""
        spec = EvaluationSpec(seeds=[7])
        assert spec.episodes() == [(7, 0)]


class TestLoadTaskSpec:
    """Tests for YAML parsing and validation."""

    def test_valid_round_trip(self, tmp_path):
        """A valid YAML file parses into the expected TaskSpec."""
        path = tmp_path / "task.yaml"
        path.write_text(VALID_TASK_YAML, encoding="utf-8")

        spec = load_task_spec(path)

        assert spec.name == "pick_place_cube"
        assert spec.task_name == "pick_place"
        assert spec.robot_name == "franka_panda"
        assert spec.scene_name == "empty_room"
        assert spec.max_episode_steps == 200
        assert spec.scene_kwargs == {"size": [10.0, 10.0, 3.0]}
        assert spec.task_kwargs == {"object_name": "red_cube"}
        assert [spawn.name for spawn in spec.object_spawns] == ["red_cube", "blue_cube"]
        assert spec.distractor_names == ["blue_cube"]
        assert len(spec.cameras) == 1
        assert spec.cameras[0].name == "eval_cam"
        assert spec.evaluation.seeds == [0, 1, 2]
        assert spec.evaluation.episodes_per_seed == 2
        assert spec.simulation == {"headless": True}
        assert isinstance(spec.success_condition(), DistanceThresholdCondition)

    def test_repo_example_loads(self):
        """The shipped example task config parses (regression test)."""
        spec = load_task_spec(EXAMPLE_TASK)
        assert spec.name == "pick_place_cube"
        assert spec.evaluation.episodes() == [
            (s, e) for s in range(5) for e in range(2)
        ]

    def test_task_type_defaults_to_name(self):
        """task.type defaults to task.name for simple registry layouts."""
        data = yaml.safe_load(VALID_TASK_YAML)
        del data["task"]["type"]
        spec = parse_task_spec(data, source="test")
        assert spec.task_name == "pick_place_cube"

    def test_missing_file(self, tmp_path):
        """A missing config file raises ConfigError."""
        with pytest.raises(ConfigError, match="not found"):
            load_task_spec(tmp_path / "nope.yaml")

    def test_problems_aggregated(self):
        """All validation problems are reported in one error."""
        data = yaml.safe_load(VALID_TASK_YAML)
        del data["task"]["name"]
        del data["evaluation"]
        data["assets"]["task_relevant"][0]["bogus_key"] = 1
        data["simulation"] = {"warp_drive": True}

        with pytest.raises(ConfigError) as excinfo:
            parse_task_spec(data, source="test")

        message = str(excinfo.value)
        assert "task.name is required" in message
        assert "evaluation.seeds" in message
        assert "bogus_key" in message
        assert "warp_drive" in message
        assert "invalid test" in message

    def test_mesh_requires_path(self):
        """A mesh asset without mesh_path is rejected."""
        data = yaml.safe_load(VALID_TASK_YAML)
        data["assets"]["task_relevant"][0]["shape_type"] = "mesh"
        with pytest.raises(ConfigError, match="mesh_path is required"):
            parse_task_spec(data, source="test")

    def test_invalid_success_block_reported(self):
        """A malformed success block is reported during parsing."""
        data = yaml.safe_load(VALID_TASK_YAML)
        data["success"]["type"] = "telepathy"
        with pytest.raises(ConfigError, match="invalid success block"):
            parse_task_spec(data, source="test")


class TestSampleLayout:
    """Tests for deterministic reset sampling."""

    def test_same_seed_same_layout(self):
        """The same seed always produces the same layout."""
        spec = _valid_spec()
        assert sample_layout(spec, 42) == sample_layout(spec, 42)

    def test_different_seeds_differ(self):
        """Different seeds sample different layouts."""
        spec = _valid_spec()
        layouts = {seed: sample_layout(spec, seed) for seed in range(10)}
        positions = {
            tuple(layout["red_cube"]["position"]) for layout in layouts.values()
        }
        assert len(positions) > 1

    def test_fixed_pose_without_distribution(self):
        """Objects without an object_poses entry keep their asset pose."""
        spec = _valid_spec()
        layout = sample_layout(spec, 0)
        assert layout["red_cube"]["position"] != [0.5, 0.0, 0.05]  # sampled
        spec.init_distribution = {}
        layout = sample_layout(spec, 0)
        assert layout["red_cube"]["position"] == [0.5, 0.0, 0.05]  # fixed
        assert layout["blue_cube"]["position"] == [0.7, 0.15, 0.04]

    def test_sampled_ranges_respected(self):
        """Sampled positions stay inside the configured ranges."""
        spec = _valid_spec()
        for seed in range(20):
            layout = sample_layout(spec, seed)
            x, y, z = layout["red_cube"]["position"]
            assert 0.4 <= x <= 0.6
            assert -0.2 <= y <= 0.2
            assert 0.05 <= z <= 0.05 + 1e-9
            bx, by, bz = layout["blue_cube"]["position"]
            assert 0.6 <= bx <= 0.9
            assert -0.3 <= by <= 0.3
            assert bz == pytest.approx(0.04)

    def test_yaw_range_orientation(self):
        """yaw_deg ranges produce valid unit quaternions about +Z."""
        spec = _valid_spec()
        spec.init_distribution = {
            "object_poses": {"red_cube": {"orientation": {"yaw_deg": [-90.0, 90.0]}}}
        }
        for seed in range(10):
            quat = np.asarray(sample_layout(spec, seed)["red_cube"]["orientation"])
            assert np.linalg.norm(quat) == pytest.approx(1.0)
            assert abs(quat[1]) < 1e-9 and abs(quat[2]) < 1e-9

    def test_articulation_states_carried(self):
        """Articulation values are sampled and carried in the layout."""
        spec = _valid_spec()
        spec.init_distribution = {
            "articulation_states": {"red_cube": {"drawer_joint": [0.0, 0.3]}}
        }
        layout = sample_layout(spec, 0)
        assert 0.0 <= layout["red_cube"]["articulation"]["drawer_joint"] <= 0.3

    def test_invalid_position_spec(self):
        """Malformed pose specs raise ConfigError."""
        spec = _valid_spec()
        spec.init_distribution = {
            "object_poses": {"red_cube": {"position": [1.0, 2.0]}}
        }
        with pytest.raises(ConfigError, match="position"):
            sample_layout(spec, 0)


class TestApplyLayout:
    """Tests for applying sampled layouts to scenes."""

    def test_apply_sets_poses(self):
        """Layout positions and orientations are written to entities."""
        entity = MagicMock()
        scene = SimpleNamespace(entities={"red_cube": entity})
        layout = {
            "red_cube": {
                "position": [0.1, 0.2, 0.3],
                "orientation": [1.0, 0.0, 0.0, 0.0],
            }
        }
        apply_layout(scene, layout)

        entity.set_pos.assert_called_once()
        np.testing.assert_allclose(entity.set_pos.call_args[0][0], [0.1, 0.2, 0.3])
        entity.set_quat.assert_called_once()

    def test_missing_entity_warns_not_raises(self):
        """Unknown objects are skipped instead of crashing."""
        scene = SimpleNamespace(entities={})
        apply_layout(scene, {"ghost": {"position": [0, 0, 0]}})  # no exception


class TestMakeApplySeedFn:
    """Tests for the batched_seed_search callback adapter."""

    def test_same_seed_same_layout_across_envs(self):
        """One seed maps to one layout in every env (cached)."""
        spec = _valid_spec()
        scenes = [SimpleNamespace(entities={}) for _ in range(2)]
        apply_seed = make_apply_seed_fn(spec, scenes)
        apply_seed(7, 0)
        apply_seed(7, 1)
        # entities absent -> no writes, but no crash and cached layout reused
        apply_seed(7, 0)

    def test_env_idx_out_of_range(self):
        """An out-of-range env index raises IndexError."""
        apply_seed = make_apply_seed_fn(_valid_spec(), [])
        with pytest.raises(IndexError):
            apply_seed(0, 0)


class TestConfigurableTask:
    """Tests for the ConfigurableTask wrapper."""

    def _base(self, *, reward=0.0, terminated=False, truncated=False, step_count=1):
        base = MagicMock()
        base.config = TaskConfig(name="pick_place", max_episode_steps=10)
        base.step.return_value = (reward, terminated, truncated, {"base": True})
        base.step_count = step_count
        base.succeeded = False
        base.reset.return_value = {}
        return base

    def _scene(self, pos):
        entity = MagicMock()
        entity.get_pos.return_value = np.asarray(pos)
        return SimpleNamespace(entities={"obj": entity})

    def test_condition_success_terminates(self):
        """A satisfied condition terminates the episode and labels success."""
        condition = DistanceThresholdCondition(
            object_name="obj", target_position=(0.5, 0.0, 0.1), threshold=0.05
        )
        task = ConfigurableTask(self._base(), condition, name="t", max_episode_steps=10)
        reward, terminated, truncated, info = task.step(
            self._scene((0.5, 0.0, 0.1)), None, np.zeros(1)
        )
        assert terminated is True
        assert truncated is False
        assert info["success"] is True
        assert info["failure_stage"] is None
        assert reward == pytest.approx(1.0)  # success_reward from TaskConfig
        assert task.succeeded is True
        assert info["base"] is True  # base info preserved

    def test_condition_failure_reports_stage(self):
        """A missed condition reports the configured failure stage."""
        condition = DistanceThresholdCondition(
            object_name="obj",
            target_position=(0.5, 0.0, 0.1),
            threshold=0.05,
            failure_stage="place_fail",
        )
        task = ConfigurableTask(self._base(), condition)
        _, terminated, _, info = task.step(
            self._scene((0.9, 0.0, 0.1)), None, np.zeros(1)
        )
        assert terminated is False
        assert info["success"] is False
        assert info["failure_stage"] == "place_fail"

    def test_base_termination_skips_condition(self):
        """When the base task terminates, the condition is not re-evaluated."""
        condition = DistanceThresholdCondition(
            object_name="obj", target_position=(0.5, 0.0, 0.1), threshold=0.05
        )
        task = ConfigurableTask(self._base(terminated=True), condition)
        _, terminated, _, info = task.step(
            self._scene((0.9, 0.0, 0.1)), None, np.zeros(1)
        )
        assert terminated is True
        assert "failure_stage" not in info

    def test_max_episode_steps_override(self):
        """The YAML episode budget truncates even when the base task differs."""
        base = self._base(step_count=7)
        task = ConfigurableTask(base, None, max_episode_steps=5)
        reward, terminated, truncated, _ = task.step(
            self._scene((0.9, 0.0, 0.1)), None, np.zeros(1)
        )
        assert terminated is False
        assert truncated is True
        assert reward == pytest.approx(-0.1)  # timeout_penalty

    def test_no_double_timeout_penalty(self):
        """A base task that already truncated is not penalized twice."""
        base = self._base(truncated=True, step_count=7)
        task = ConfigurableTask(base, None, max_episode_steps=5)
        reward, _, truncated, _ = task.step(
            self._scene((0.9, 0.0, 0.1)), None, np.zeros(1)
        )
        assert truncated is True
        assert reward == pytest.approx(0.0)

    def test_reset_delegates(self):
        """Reset clears wrapper state and delegates to the base task."""
        base = self._base()
        task = ConfigurableTask(base, None)
        info = task.reset(self._scene((0, 0, 0)), None, seed=3)
        assert info == {}
        assert task.step_count == 0
        base.reset.assert_called_once()


class TestBuildTaskComponents:
    """Tests for registry-based component assembly."""

    def test_objects_injected_and_task_wrapped(self):
        """Spawns are added to the scene; the task is wrapped and configured."""
        spec = _valid_spec()
        scene = _mock_scene()
        robot = _mock_robot()
        base_task = PickPlaceTask(
            TaskConfig(max_episode_steps=200),
            object_name="red_cube",
            target_position=(0.5, 0.0, 0.1),
        )
        registry = _registry_with(scene=scene, robot=robot, task=base_task)

        out_scene, out_robot, task = build_task_components(spec, registry=registry)

        assert out_scene is scene
        assert out_robot is robot
        assert scene.add_object.call_count == 2
        assert isinstance(task, ConfigurableTask)
        assert task.base_task is base_task
        assert task.config.name == "pick_place_cube"
        assert task.config.max_episode_steps == 200
        assert isinstance(task.success_condition, DistanceThresholdCondition)

    def test_unknown_component_lists_available(self):
        """Missing registry components raise ConfigError with available names."""
        spec = _valid_spec()
        with pytest.raises(ConfigError, match="scene 'empty_room' is not registered"):
            build_task_components(spec, registry=AssetRegistry())

    def test_non_task_factory_rejected(self):
        """A task factory returning a non-Task is rejected."""
        spec = _valid_spec()
        registry = _registry_with(
            scene=_mock_scene(), robot=_mock_robot(), task=MagicMock()
        )
        with pytest.raises(TypeError, match="expected a Task subclass"):
            build_task_components(spec, registry=registry)


class TestComposeTask:
    """Tests for full environment composition from a spec."""

    def test_compose_task(self, monkeypatch):
        """compose_task wires spec -> registry -> mocked backend -> LoadedTask."""
        spec = _valid_spec()
        scene = _mock_scene()
        robot = _mock_robot()
        base_task = PickPlaceTask(
            TaskConfig(max_episode_steps=200), object_name="red_cube"
        )
        registry = _registry_with(scene=scene, robot=robot, task=base_task)

        backend, scene_backend = _make_mock_backend()
        monkeypatch.setattr(
            "cloud_robotics_sim.core.composer.get_backend",
            MagicMock(return_value=backend),
        )

        loaded = compose_task(spec, registry=registry)

        assert isinstance(loaded.env, ComposedEnvironment)
        assert loaded.env.scene is scene
        assert loaded.success_condition is loaded.env.task.success_condition
        scene.build.assert_called_once_with(scene_backend)
        scene_backend.build.assert_called_once()
        assert loaded.evaluation_episodes() == [
            (0, 0),
            (0, 1),
            (1, 0),
            (1, 1),
            (2, 0),
            (2, 1),
        ]

    def test_reset_episode_applies_layout(self, monkeypatch):
        """reset_episode samples the seed layout before env.reset."""
        spec = _valid_spec()
        scene = _mock_scene()
        robot = _mock_robot()
        robot.get_observation.return_value = {}
        base_task = PickPlaceTask(
            TaskConfig(max_episode_steps=200), object_name="red_cube"
        )
        registry = _registry_with(scene=scene, robot=robot, task=base_task)

        backend, _ = _make_mock_backend()
        monkeypatch.setattr(
            "cloud_robotics_sim.core.composer.get_backend",
            MagicMock(return_value=backend),
        )

        loaded = compose_task(spec, registry=registry)
        obs, info = loaded.reset_episode(seed=5)

        assert info["seed"] == 5
        # apply_layout uses entities.get(...); MagicMock returns the same child
        # mock for every object, so check that *some* set_pos call matches the
        # red_cube layout (the other call belongs to the blue_cube distractor).
        entity = scene.entities.get.return_value
        entity.set_pos.assert_called()
        expected = np.asarray(sample_layout(spec, 5)["red_cube"]["position"])
        applied = [call[0][0] for call in entity.set_pos.call_args_list]
        assert any(np.allclose(position, expected) for position in applied)
