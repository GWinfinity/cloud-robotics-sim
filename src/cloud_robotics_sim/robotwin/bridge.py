"""Bridge data structures for RoboTwin trajectory replay.

The bridge format decouples RoboTwin's native SAPIEN demonstration data from
the current project's Genesis backend. It is intentionally simulator-agnostic
so the same recorded trajectory can be replayed on any backend.
"""

from __future__ import annotations

import builtins
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class CameraConfig:
    """Configuration for a single camera in the bridge.

    Attributes:
        name: Camera identifier, e.g. "head_camera".
        pos: World-space position (x, y, z).
        look_at: World-space look-at point (x, y, z).
        resolution: Image resolution in pixels (width, height).
        fov: Vertical field of view in degrees.
        near: Near clipping plane distance.
        far: Far clipping plane distance.
    """

    name: str
    pos: tuple[float, float, float]
    look_at: tuple[float, float, float]
    resolution: tuple[int, int]
    fov: float = 60.0
    near: float = 0.01
    far: float = 100.0


@dataclass
class ObjectAsset:
    """Asset reference for a manipulated object.

    Attributes:
        name: Object identifier used in the trajectory.
        asset_type: Either "mesh" for rigid GLB/OBJ or "urdf" for articulated objects.
        path: Filesystem path to the asset file.
        scale: Uniform or per-axis scale factor.
        is_articulation: Whether the object has movable joints.
    """

    name: str
    asset_type: str  # "mesh" | "urdf"
    path: str
    scale: tuple[float, float, float] | float = 1.0
    is_articulation: bool = False


@dataclass
class RobotwinFrame:
    """A single timestep of a recorded RoboTwin demonstration.

    Attributes:
        timestamp: Simulation time in seconds.
        robot_command: 14-D target joint position [L_arm(6), L_grip, R_arm(6), R_grip].
        robot_achieved_qpos: Actual generalized positions reported by the sim.
        robot_base_pos: World-space base position (x, y, z).
        robot_base_quat: World-space base orientation quaternion (w, x, y, z).
        object_states: Mapping from object name to its pose and optional joint state.
        camera_images: Optional mapping from camera name to RGB arrays.
    """

    timestamp: float
    robot_command: np.ndarray = field(
        default_factory=lambda: np.zeros(14, dtype=np.float64)
    )
    robot_achieved_qpos: np.ndarray = field(
        default_factory=lambda: np.zeros(14, dtype=np.float64)
    )
    robot_base_pos: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=np.float64)
    )
    robot_base_quat: np.ndarray = field(
        default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0])
    )
    object_states: dict[str, dict[str, Any]] = field(default_factory=dict)
    camera_images: dict[str, np.ndarray] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Normalize numpy arrays to float64."""
        self.robot_command = np.asarray(self.robot_command, dtype=np.float64)
        self.robot_achieved_qpos = np.asarray(
            self.robot_achieved_qpos, dtype=np.float64
        )
        self.robot_base_pos = np.asarray(self.robot_base_pos, dtype=np.float64)
        self.robot_base_quat = np.asarray(self.robot_base_quat, dtype=np.float64)


_BRIDGE_MODULE = "cloud_robotics_sim.robotwin.bridge"
_BRIDGE_CLASS_NAMES = frozenset(
    {"CameraConfig", "ObjectAsset", "RobotwinFrame", "RobotwinBridge"}
)
_BUILTIN_NAMES = frozenset(
    {
        "dict",
        "list",
        "tuple",
        "set",
        "frozenset",
        "str",
        "bytes",
        "int",
        "float",
        "bool",
        "complex",
    }
)


def _build_numpy_allowlist() -> dict[tuple[str, str], Any]:
    """Map allowed numpy pickle globals to their live callables.

    Both the ``numpy.core.*`` (numpy 1.x pickles) and ``numpy._core.*``
    (numpy 2.x pickles) spellings are accepted so bridges stay loadable across
    numpy versions. ``_frombuffer`` is the array-reconstruction entry point used
    by numpy >= 2.1 pickles; ``_reconstruct``/``scalar`` cover older pickles.
    """
    if hasattr(np, "_core"):  # numpy >= 2
        core = getattr(np, "_core")
    else:  # numpy 1.x
        core = getattr(np, "core")
    multiarray = getattr(core, "multiarray")
    numeric = getattr(core, "numeric")
    allowed: dict[tuple[str, str], Any] = {("numpy", "ndarray"): np.ndarray}
    allowed[("numpy", "dtype")] = np.dtype
    for module in ("numpy.core.multiarray", "numpy._core.multiarray"):
        for name in ("_reconstruct", "scalar"):
            fn = getattr(multiarray, name, None)
            if fn is not None:
                allowed[(module, name)] = fn
    for module in ("numpy.core.numeric", "numpy._core.numeric"):
        fn = getattr(numeric, "_frombuffer", None)
        if fn is not None:
            allowed[(module, "_frombuffer")] = fn
    return allowed


_NUMPY_ALLOWED_GLOBALS = _build_numpy_allowlist()


class _RestrictedUnpickler(pickle.Unpickler):
    """Unpickler restricted to the bridge dataclass graph.

    Only the globals needed to rebuild :class:`CameraConfig`,
    :class:`ObjectAsset`, :class:`RobotwinFrame` and :class:`RobotwinBridge`
    (plus their numpy arrays and builtin containers) can be resolved. Every
    other global — ``os``, ``sys``, ``subprocess``, ``builtins.eval``,
    ``__builtin__``, ``posix``, ``nt``, ... — raises
    :class:`pickle.UnpicklingError` before anything can execute, so a
    malicious ``__reduce__`` never runs.
    """

    def find_class(self, module: str, name: str) -> Any:
        if module == _BRIDGE_MODULE and name in _BRIDGE_CLASS_NAMES:
            return globals()[name]
        if module == "builtins" and name in _BUILTIN_NAMES:
            return getattr(builtins, name)
        try:
            return _NUMPY_ALLOWED_GLOBALS[(module, name)]
        except KeyError:
            raise pickle.UnpicklingError(
                f"Refusing to unpickle forbidden global {module!r}.{name!r}: "
                f"{_BRIDGE_MODULE}.load() only accepts bridge dataclass pickles"
            ) from None


@dataclass
class RobotwinBridge:
    """A complete RoboTwin demonstration episode.

    Attributes:
        task_name: RoboTwin task identifier, e.g. "move_can_pot".
        seed: Random seed used during collection.
        fps: Playback frequency in frames per second.
        robot_urdf: Path to the robot URDF used in the episode.
        robot_joint_names: Ordered list of actuated joint names.
        table_height: Table surface z-coordinate.
        object_assets: Mapping from object name to its asset descriptor.
        cameras: List of camera configurations.
        frames: List of per-frame states.
        metadata: Optional extra metadata (language instruction, success flag, etc.).
    """

    task_name: str
    seed: int
    fps: float
    robot_urdf: str
    robot_joint_names: list[str] = field(default_factory=list)
    table_height: float = 0.74
    object_assets: dict[str, ObjectAsset] = field(default_factory=dict)
    cameras: list[CameraConfig] = field(default_factory=list)
    frames: list[RobotwinFrame] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        """Return the number of frames in the bridge."""
        return len(self.frames)

    def save(self, path: str | Path) -> None:
        """Serialize the bridge to a pickle file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str | Path) -> "RobotwinBridge":
        """Load a bridge from a restricted pickle file.

        The stream is unpickled through :class:`_RestrictedUnpickler`, which
        only reconstructs the bridge dataclass graph (the four dataclasses,
        numpy arrays/scalars and builtin containers). Any other global — e.g.
        a malicious ``__reduce__`` pointing at ``os.system`` or
        ``builtins.eval`` — raises :class:`pickle.UnpicklingError` before it can
        execute. The ``isinstance`` check is kept as a second gate.
        """
        with open(path, "rb") as f:
            obj = _RestrictedUnpickler(f).load()
        if not isinstance(obj, cls):
            raise TypeError(f"Loaded object is not a {cls.__name__}")
        return obj

    def summary(self) -> dict[str, Any]:
        """Return a human-readable summary of the bridge."""
        return {
            "task_name": self.task_name,
            "seed": self.seed,
            "fps": self.fps,
            "num_frames": len(self.frames),
            "robot_urdf": self.robot_urdf,
            "num_objects": len(self.object_assets),
            "num_cameras": len(self.cameras),
        }
