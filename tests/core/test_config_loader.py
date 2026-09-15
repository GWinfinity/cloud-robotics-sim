"""Tests for the simulation config loader."""

from __future__ import annotations

from pathlib import Path

import pytest

from cloud_robotics_sim.core.config_loader import ConfigError, load_sim_config

VALID = """
experiment:
  name: demo
environment:
  scene:
    type: empty_room
  robot:
    type: franka_panda
    urdf_path: null
  task:
    type: pick_place
  simulation:
    dt: 0.01
    substeps: 10
"""


class TestLoadSimConfig:
    """Tests for load_sim_config."""

    def test_valid_config(self, tmp_path: Path) -> None:
        path = tmp_path / "sim.yaml"
        path.write_text(VALID, encoding="utf-8")
        config = load_sim_config(path)
        assert config["environment"]["robot"]["type"] == "franka_panda"
        assert config["environment"]["robot"]["urdf_path"] is None

    def test_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError, match="not found"):
            load_sim_config(tmp_path / "nope.yaml")

    def test_invalid_yaml(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.yaml"
        path.write_text("a: [unclosed", encoding="utf-8")
        with pytest.raises(ConfigError, match="invalid YAML"):
            load_sim_config(path)

    def test_non_mapping(self, tmp_path: Path) -> None:
        path = tmp_path / "list.yaml"
        path.write_text("- just\n- a\n- list\n", encoding="utf-8")
        with pytest.raises(ConfigError, match="mapping"):
            load_sim_config(path)

    def test_missing_environment(self, tmp_path: Path) -> None:
        path = tmp_path / "noenv.yaml"
        path.write_text("experiment: {}\n", encoding="utf-8")
        with pytest.raises(ConfigError, match="environment"):
            load_sim_config(path)

    def test_missing_types(self, tmp_path: Path) -> None:
        path = tmp_path / "notypes.yaml"
        path.write_text(
            "environment:\n"
            "  scene: {}\n"
            "  robot: {}\n"
            "  task: {}\n"
            "  simulation: {dt: 0.01, substeps: 10}\n",
            encoding="utf-8",
        )
        with pytest.raises(ConfigError) as excinfo:
            load_sim_config(path)
        message = str(excinfo.value)
        assert "scene.type" in message
        assert "robot.type" in message
        assert "task.type" in message

    def test_missing_simulation_keys(self, tmp_path: Path) -> None:
        path = tmp_path / "nosim.yaml"
        path.write_text(
            "environment:\n"
            "  scene: {type: empty_room}\n"
            "  robot: {type: franka_panda}\n"
            "  task: {type: pick_place}\n"
            "  simulation: {}\n",
            encoding="utf-8",
        )
        with pytest.raises(ConfigError, match="simulation.dt"):
            load_sim_config(path)


class TestRepoConfig:
    """The shipped example config must load cleanly."""

    def test_franka_pickplace_yaml(self) -> None:
        from cloud_robotics_sim.core.robot_assets import repo_root

        config = load_sim_config(repo_root() / "configs" / "franka_pickplace.yaml")
        robot = config["environment"]["robot"]
        # No absolute machine-specific paths may ship in the config.
        assert robot["urdf_path"] is None
        assert config["environment"]["scene"]["type"] == "empty_room"
