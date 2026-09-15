"""Simulation configuration loading and validation.

Public entry point for the YAML configs in ``configs/`` (e.g.
``configs/franka_pickplace.yaml``). Consumed by
:func:`cloud_robotics_sim.runtime.main.make_env_from_config` (via the
improvement loop) and available for future train/eval entry points.

Required structure::

    environment:
      scene:       {type: empty_room, ...}
      robot:       {type: franka_panda, urdf_path: null, ...}
      task:        {type: pick_place, ...}
      simulation:  {dt: 0.01, substeps: 10, headless: true, resolution: [640, 480]}

``robot.urdf_path`` may be null — the model is then resolved automatically
through :mod:`cloud_robotics_sim.core.robot_assets` (bundled assets ->
auto-cloned awesome-robot-descriptions -> Genesis built-in lookup).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class ConfigError(Exception):
    """Raised when a simulation configuration is missing or invalid."""


_ENV_KEYS = ("scene", "robot", "task", "simulation")


def load_sim_config(path: str | Path) -> dict[str, Any]:
    """Load and validate a simulation config YAML.

    Args:
        path: Path to the YAML file.

    Returns:
        The parsed config dictionary.

    Raises:
        ConfigError: If the file is missing, is not a mapping, or lacks
            the required ``environment`` structure.
    """
    config_path = Path(path)
    if not config_path.exists():
        raise ConfigError(f"config file not found: {config_path}")
    try:
        data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML in {config_path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"config {config_path} must be a YAML mapping")
    validate_sim_config(data, source=str(config_path))
    return data


def validate_sim_config(config: dict[str, Any], source: str = "config") -> None:
    """Validate the required environment structure in-place consumer contract.

    Raises:
        ConfigError: With a message listing every problem found.
    """
    problems: list[str] = []
    env = config.get("environment")
    if not isinstance(env, dict):
        problems.append("missing 'environment' mapping")
    else:
        for key in _ENV_KEYS:
            section = env.get(key)
            if not isinstance(section, dict):
                problems.append(f"environment.{key} must be a mapping")
            elif key != "simulation" and not section.get("type"):
                problems.append(f"environment.{key}.type is required")
        sim = env.get("simulation")
        if isinstance(sim, dict):
            for key in ("dt", "substeps"):
                if key not in sim:
                    problems.append(f"environment.simulation.{key} is required")
    if problems:
        raise ConfigError(f"invalid {source}: " + "; ".join(problems))
