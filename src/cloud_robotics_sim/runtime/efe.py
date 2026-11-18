"""EFE (Embodied First Exam) suite registrations for the runtime registry.

P1 single-arm suites (MetaWorld+ / LIBERO-MAX / VLABench / RoboPaint / KinDER)
run on the generic stack: a neutral base task whose success comes entirely from
the configured :class:`~cloud_robotics_sim.core.success.SuccessCondition`
(EFE's graders are final-state predicates, not reward-shaping tasks), the
suite's robot (Sawyer / Franka Panda / TidyBot++), and a tabletop scene whose
objects the task YAML declares as primitive assets.

Importing this module registers the factories; ``runtime.main`` imports it so
the built-ins exist once the improvement-loop entry point is loaded.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from cloud_robotics_sim.core.embodiment import EmbodimentConfig, FrankaPanda
from cloud_robotics_sim.core.registry import (
    register_robot,
    register_scene,
    register_task,
)
from cloud_robotics_sim.core.scene import Scene, SceneConfig
from cloud_robotics_sim.core.task import Task, TaskConfig

#: Repo-rooted default model paths (upstream mirrors under assets/).
_REPO_ROOT = Path(__file__).resolve().parents[3]
SAWYER_MJCF = (
    _REPO_ROOT / "assets" / "metaworld" / "envs" / "assets" / "objects" / "xyz_base.xml"
)
TIDYBOT_URDF_DIR = _REPO_ROOT / "assets" / "kindergarden"
VLB_PANDA_MJCF = (
    _REPO_ROOT / "assets" / "vlabench" / "robots" / "franka_emika_panda" / "panda.xml"
)


class EfeBaseTask(Task):
    """Neutral base task for EFE ports: no shaping, success via condition.

    EFE grades the final state with a boolean predicate; there is no reward
    curriculum. ``ConfigurableTask`` (task_loader) wraps this base with the
    YAML-configured success condition, which is where every EFE goal check
    lives. The base task itself never terminates and never rewards.
    """

    def reset(self, scene: Any, robot: Any, seed: int) -> dict:
        """Reset episode counters; nothing else to initialise."""
        self.step_count = 0
        self.succeeded = False
        return {"task": "efe", "seed": seed}

    def step(self, scene: Any, robot: Any, action: np.ndarray):
        """Advance one step with zero reward and no self-termination."""
        self.step_count += 1
        info: dict = {"task": "efe"}
        return 0.0, False, False, info


class EfeTabletopScene(Scene):
    """Tabletop workspace for EFE single-arm suites.

    A small room whose floor plane stands in for the benchmark's table top
    (MetaWorld: table top at z = 0; VLABench/LIBERO: their own tables arrive
    as static assets in the task YAML). The robot mounts beside the workspace.
    """

    def __init__(self, size: tuple[float, float, float] = (3.0, 3.0, 2.0)):
        config = SceneConfig(
            name="efe_tabletop",
            size=size,
            default_camera_pos=(1.2, -1.2, 1.0),
        )
        super().__init__(config)

    def _build_custom(self):
        """No built-in furniture; task YAMLs declare the workspace assets."""


@register_scene("efe_tabletop")
def _make_efe_tabletop(**kwargs: Any) -> Any:
    kwargs.pop("type", None)
    return EfeTabletopScene(**kwargs)


@register_task("efe_static")
def _make_efe_static(**kwargs: Any) -> EfeBaseTask:
    """Neutral EFE base task; success comes from the YAML success block."""
    kwargs.pop("type", None)
    max_steps = int(kwargs.pop("max_episode_steps", 500))
    return EfeBaseTask(TaskConfig(name="efe", max_episode_steps=max_steps))


@register_robot("sawyer")
def _make_sawyer(**kwargs: Any) -> Any:
    """Sawyer arm from MetaWorld's MJCF (MIT), via the MJCF morph path."""
    kwargs.pop("type", None)
    base_position = kwargs.pop("base_position", (0.0, -0.3, 0.0))
    model_path = kwargs.pop("urdf_path", None) or kwargs.pop("model_path", None)
    if model_path is None:
        model_path = str(SAWYER_MJCF)
    elif model_path.startswith("mjcf:"):
        # resolve_robot_model infers the format from the file suffix; a
        # "mjcf:" scheme prefix would be treated as part of the path.
        model_path = model_path.removeprefix("mjcf:")
    return FrankaPanda(
        EmbodimentConfig(
            name="sawyer",
            base_position=base_position,
            urdf_path=model_path,
        )
    )


@register_robot("franka_panda")
def _make_franka_panda(**kwargs: Any) -> Any:
    """Franka Panda from VLABench's MJCF (MIT), the EFE single-arm robot."""
    kwargs.pop("type", None)
    base_position = kwargs.pop("base_position", (0.0, -0.4, 0.78))
    model_path = kwargs.pop("urdf_path", None) or kwargs.pop("model_path", None)
    if model_path is None:
        model_path = str(VLB_PANDA_MJCF)
    elif model_path.startswith("mjcf:"):
        model_path = model_path.removeprefix("mjcf:")
    return FrankaPanda(
        EmbodimentConfig(
            name="franka_panda",
            base_position=base_position,
            urdf_path=model_path,
        )
    )
