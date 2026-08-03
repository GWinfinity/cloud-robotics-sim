"""Replay task for RoboTwin demonstrations."""

from __future__ import annotations

import logging

import numpy as np

from cloud_robotics_sim.backend import SceneBackend
from cloud_robotics_sim.core.embodiment import RobotEmbodiment
from cloud_robotics_sim.core.scene import Scene
from cloud_robotics_sim.core.task import Task, TaskConfig
from cloud_robotics_sim.robotwin.bridge import RobotwinBridge, RobotwinFrame

logger = logging.getLogger(__name__)


class RobotwinReplayTask(Task):
    """Open-loop kinematic replay of a RoboTwin demonstration.

    At each step the task overwrites the robot and object states with the
    values recorded in the corresponding bridge frame. No closed-loop
    controller or reward shaping is performed; the task exists to verify
    that a Genesis scene can reproduce the recorded kinematics.
    """

    def __init__(
        self,
        bridge: RobotwinBridge,
        scene_backend: SceneBackend | None = None,
        config: TaskConfig | None = None,
    ) -> None:
        super().__init__(config or TaskConfig(name=bridge.task_name))
        self.bridge = bridge
        self.scene_backend = scene_backend
        self._frame_index: int = 0
        self._object_names: set[str] = set(bridge.object_assets.keys())
        self._step_dt = 1.0 / bridge.fps if bridge.fps > 0 else 0.02

    def reset(self, scene: Scene, robot: RobotEmbodiment, seed: int) -> dict:
        """Reset the replay to the first frame."""
        self.step_count = 0
        self.succeeded = False
        self._frame_index = 0

        if not self.bridge.frames:
            logger.warning("Bridge contains no frames; reset is a no-op")
            return {"frame_index": 0, "timestamp": 0.0}

        frame = self.bridge.frames[0]
        self._apply_frame(scene, robot, frame)

        return {
            "frame_index": self._frame_index,
            "timestamp": frame.timestamp,
            "num_frames": len(self.bridge.frames),
        }

    def step(
        self,
        scene: Scene,
        robot: RobotEmbodiment,
        action: np.ndarray,
    ) -> tuple[float, bool, bool, dict]:
        """Advance one replay frame and apply recorded states.

        The ``action`` argument is ignored because this is an open-loop replay.
        """
        self.step_count += 1
        if not self.bridge.frames:
            return (
                0.0,
                True,
                False,
                {
                    "frame_index": 0,
                    "timestamp": 0.0,
                    "step": self.step_count,
                    "success": False,
                },
            )

        self._frame_index = min(self._frame_index + 1, len(self.bridge.frames) - 1)

        frame = self.bridge.frames[self._frame_index]
        self._apply_frame(scene, robot, frame)

        terminated = self._frame_index >= len(self.bridge.frames) - 1
        truncated = False
        reward = 0.0

        info = {
            "frame_index": self._frame_index,
            "timestamp": frame.timestamp,
            "step": self.step_count,
            "success": False,
        }
        return reward, terminated, truncated, info

    def _apply_frame(
        self, scene: Scene, robot: RobotEmbodiment, frame: RobotwinFrame
    ) -> None:
        """Apply a single bridge frame to the scene."""
        # Update robot base pose and joint configuration.
        entity = robot.entity
        if entity is not None:
            try:
                entity.set_pos(frame.robot_base_pos)
                entity.set_quat(frame.robot_base_quat)
            except Exception as e:
                logger.warning(f"Failed to set robot base pose: {e}")

            try:
                robot.apply_action(frame.robot_command)
            except Exception as e:
                logger.warning(f"Failed to apply robot command: {e}")

        # Update object poses.
        for obj_name, state in frame.object_states.items():
            if obj_name not in scene.entities:
                continue
            obj_entity = scene.entities[obj_name]
            try:
                if "pos" in state:
                    obj_entity.set_pos(np.asarray(state["pos"], dtype=np.float64))
                if "quat" in state:
                    obj_entity.set_quat(np.asarray(state["quat"], dtype=np.float64))
                if "qpos" in state and hasattr(obj_entity, "set_qpos"):
                    obj_entity.set_qpos(np.asarray(state["qpos"], dtype=np.float64))
            except Exception as e:
                logger.warning(f"Failed to set object state for '{obj_name}': {e}")

    @property
    def current_frame(self) -> RobotwinFrame | None:
        """Return the currently active bridge frame."""
        if not self.bridge.frames:
            return None
        return self.bridge.frames[self._frame_index]

    @property
    def is_finished(self) -> bool:
        """Return True when the last frame has been reached."""
        return self._frame_index >= len(self.bridge.frames) - 1
