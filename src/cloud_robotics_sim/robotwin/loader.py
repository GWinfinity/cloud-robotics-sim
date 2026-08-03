"""Bridge loader and iterator for RoboTwin demonstrations."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

from cloud_robotics_sim.robotwin.bridge import RobotwinBridge, RobotwinFrame


class RobotwinBridgeLoader:
    """Loads a RobotwinBridge and provides frame-by-frame access.

    Example:
        >>> loader = RobotwinBridgeLoader("bridges/move_can_pot.pkl")
        >>> for frame in loader:
        ...     print(frame.timestamp, frame.robot_command)
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.bridge = RobotwinBridge.load(self.path)
        self._index = 0

    @property
    def task_name(self) -> str:
        """Return the task name stored in the bridge."""
        return self.bridge.task_name

    @property
    def seed(self) -> int:
        """Return the random seed stored in the bridge."""
        return self.bridge.seed

    @property
    def fps(self) -> float:
        """Return the playback frequency in frames per second."""
        return self.bridge.fps

    @property
    def num_frames(self) -> int:
        """Return the total number of frames."""
        return len(self.bridge.frames)

    @property
    def robot_urdf(self) -> str:
        """Return the robot URDF path stored in the bridge."""
        return self.bridge.robot_urdf

    @property
    def object_assets(self) -> dict[str, Any]:
        """Return the object asset descriptors."""
        return self.bridge.object_assets

    @property
    def cameras(self) -> list[Any]:
        """Return the camera configurations."""
        return self.bridge.cameras

    def summary(self) -> dict[str, Any]:
        """Return a summary of the loaded bridge."""
        return self.bridge.summary()

    def reset(self) -> None:
        """Reset the frame iterator to the first frame."""
        self._index = 0

    def __iter__(self) -> Iterator[RobotwinFrame]:
        """Iterate over all frames."""
        self.reset()
        return self

    def __next__(self) -> RobotwinFrame:
        """Return the next frame and advance the iterator."""
        if self._index >= self.num_frames:
            raise StopIteration
        frame = self.bridge.frames[self._index]
        self._index += 1
        return frame

    def __getitem__(self, index: int) -> RobotwinFrame:
        """Random access to a frame by index."""
        return self.bridge.frames[index]
