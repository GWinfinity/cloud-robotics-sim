"""Hierarchical cuRobo planner integration (migration doc section 5.2).

Uses the local ``hierarchical_cuRobo_planner`` project (workspace GPU A* ->
batched cuRobo IK -> trajectory optimization) as the dual-arm / constrained
/high-dimensional planning backend, **replacing the raw cuRobo bridge**
originally sketched in the migration document.

Decision policy (doc section 5.2, unchanged):

- P0 default remains Genesis OMPL ``plan_path`` (zero extra deps,
  batch-friendly) for single-arm free-space reaching.
- :class:`HierarchicalCuRoboPlanner` is the fallback for dual-arm /
  constrained / long-horizon planning, and additionally brings workspace
  obstacle avoidance (A* voxel grid) that OMPL ``plan_path`` does not expose
  through the backend interface.

Installation (NOT a hard dependency of this project)::

    pip install -e D:/githbi/hierarchical_cuRobo_planner
    # plus a CUDA or MUSA PyTorch build and cuRobo for end-to-end planning;
    # without them the adapter degrades to the OMPL fallback.
"""

from __future__ import annotations

import importlib.util
import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from cloud_robotics_sim.backend.base import ArticulationBackend

logger = logging.getLogger(__name__)

__all__ = [
    "CuRoboPlannerConfig",
    "CuRoboPlannerUnavailableError",
    "HierarchicalCuRoboPlanner",
    "PlannerError",
    "is_curobo_planner_available",
    "plan_with_fallback",
]

_PACKAGE_NAME = "curobo_hierarchical_planner"
_INSTALL_HINT = (
    "hierarchical_cuRobo_planner is not installed. Install it with "
    "`pip install -e <path>/hierarchical_cuRobo_planner` plus a CUDA/MUSA "
    "PyTorch build and cuRobo for end-to-end planning."
)


class CuRoboPlannerUnavailableError(RuntimeError):
    """Raised when the external planner package cannot be imported/built."""


class PlannerError(RuntimeError):
    """Raised when a planning query fails (planner reported no success)."""


def is_curobo_planner_available() -> bool:
    """Return True if the external planner package is importable."""
    return importlib.util.find_spec(_PACKAGE_NAME) is not None


@dataclass
class CuRoboPlannerConfig:
    """Configuration for :class:`HierarchicalCuRoboPlanner`.

    Attributes:
        urdf_path: Robot URDF (use the converted mimic-free URDF).
        base_link: Robot base link name.
        ee_link: End-effector link name.
        workspace_bounds: ``(3, 2)`` workspace AABB in meters.
        voxel_size: A* voxel grid resolution in meters.
        inflation_radius: Safety margin added to obstacles in meters.
        obstacles: Workspace obstacles, e.g.
            ``{"type": "cuboid", "dims": [...], "pose": [...]}``.
        orientation_mode: EE orientation mode along the A* path
            (``identity`` / ``interpolate`` / ``tangent``).
        trajopt_dt: Trajectory optimization timestep in seconds.
        validate: Run FK/limit validation on the optimized trajectory.
        device: Torch device (``auto`` / ``cuda:0`` / ``musa:0`` / ``cpu``).
        backend_name: Compute backend (``auto`` / ``cuda`` / ``musa``).
    """

    urdf_path: str
    base_link: str
    ee_link: str
    workspace_bounds: np.ndarray = field(
        default_factory=lambda: np.array(
            [[-1.0, 1.0], [-1.0, 1.0], [0.0, 1.5]], dtype=np.float64
        )
    )
    voxel_size: float = 0.04
    inflation_radius: float = 0.05
    obstacles: list[dict[str, Any]] = field(default_factory=list)
    orientation_mode: str = "identity"
    trajopt_dt: float = 0.05
    validate: bool = True
    device: str = "auto"
    backend_name: str = "auto"

    def __post_init__(self) -> None:
        """Validate workspace bounds and voxel size."""
        bounds = np.asarray(self.workspace_bounds, dtype=np.float64)
        if bounds.shape != (3, 2):
            raise ValueError(
                f"workspace_bounds must have shape (3, 2), got {bounds.shape}"
            )
        self.workspace_bounds = bounds
        if self.voxel_size <= 0:
            raise ValueError("voxel_size must be positive")


class HierarchicalCuRoboPlanner:
    """Adapter wrapping ``curobo_hierarchical_planner.HierarchicalPlanner``.

    The heavy planner (cuRobo IK/TO adapters) is built lazily on first use
    so importing this module stays cheap and CUDA-free environments can
    still use the OMPL fallback.
    """

    def __init__(self, config: CuRoboPlannerConfig) -> None:
        self.config = config
        self._planner: Any | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _build(self) -> Any:
        """Lazily construct the external planner, or raise Unavailable."""
        if self._planner is not None:
            return self._planner
        if not is_curobo_planner_available():
            raise CuRoboPlannerUnavailableError(_INSTALL_HINT)
        try:
            from curobo_hierarchical_planner import HierarchicalPlanner
            from curobo_hierarchical_planner.types import (
                PlanningConfig,
                RobotModelConfig,
            )
        except ImportError as exc:  # pragma: no cover - defensive
            raise CuRoboPlannerUnavailableError(_INSTALL_HINT) from exc

        robot_cfg = RobotModelConfig(
            urdf_path=self.config.urdf_path,
            base_link=self.config.base_link,
            ee_link=self.config.ee_link,
        )
        planning_cfg = PlanningConfig(
            orientation_mode=self.config.orientation_mode,
            trajopt_dt=self.config.trajopt_dt,
            validate=self.config.validate,
        )
        try:
            self._planner = HierarchicalPlanner(
                robot_cfg,
                planning_cfg=planning_cfg,
                device=self.config.device,
                backend_name=self.config.backend_name,
            )
        except Exception as exc:
            raise CuRoboPlannerUnavailableError(
                f"Failed to build HierarchicalPlanner "
                f"(CUDA/MUSA + cuRobo required): {exc}"
            ) from exc
        return self._planner

    @property
    def is_built(self) -> bool:
        """Whether the underlying planner has been constructed."""
        return self._planner is not None

    # ------------------------------------------------------------------
    # Planning
    # ------------------------------------------------------------------

    def _make_request(
        self, start_qpos: np.ndarray, goal_pos: np.ndarray, goal_quat: np.ndarray | None
    ) -> Any:
        from curobo_hierarchical_planner.types import (
            PlanRequest,
            WorkspaceConfig,
        )

        world = WorkspaceConfig(
            bounds=self.config.workspace_bounds,
            voxel_size=self.config.voxel_size,
            inflation_radius=self.config.inflation_radius,
            obstacles=list(self.config.obstacles),
        )
        return PlanRequest(
            start_joint=np.asarray(start_qpos, dtype=np.float64),
            goal_ee_position=np.asarray(goal_pos, dtype=np.float64),
            goal_ee_quaternion=(
                None if goal_quat is None else np.asarray(goal_quat, dtype=np.float64)
            ),
            world=world,
            robot=self._build().robot_cfg,
        )

    def plan_to_ee_pose(
        self,
        start_qpos: np.ndarray,
        goal_pos: np.ndarray,
        goal_quat: np.ndarray | None = None,
    ) -> np.ndarray:
        """Plan a joint-space trajectory to an EE pose.

        Args:
            start_qpos: Start joint configuration, shape ``(n_dofs,)``.
            goal_pos: Goal EE world position, shape ``(3,)``.
            goal_quat: Optional goal EE quaternion ``(w, x, y, z)``.

        Returns:
            Trajectory of shape ``(T, n_dofs)`` (time-parameterized if the
            optimizer succeeded, otherwise the per-waypoint IK path).

        Raises:
            CuRoboPlannerUnavailableError: External planner not installed/built.
            PlannerError: The planner reported failure.
        """
        planner = self._build()
        request = self._make_request(start_qpos, goal_pos, goal_quat)
        result = planner.plan(request)
        if not result.success:
            raise PlannerError(
                f"Hierarchical planning failed: {result.message or 'no message'}"
            )
        trajectory = result.trajectory
        if trajectory is None:
            trajectory = result.joint_path
        if trajectory is None:
            raise PlannerError("Planner succeeded but returned no trajectory")
        trajectory = np.asarray(trajectory, dtype=np.float64)
        logger.info(
            "Hierarchical plan succeeded (%d waypoints, timings=%s)",
            trajectory.shape[0],
            getattr(result, "timings", {}),
        )
        return trajectory


def plan_with_fallback(
    robot: ArticulationBackend,
    goal_pos: np.ndarray,
    goal_quat: np.ndarray | None = None,
    *,
    ee_link: str | None = None,
    planner: HierarchicalCuRoboPlanner | None = None,
    num_waypoints: int = 50,
) -> tuple[np.ndarray, str]:
    """Plan to an EE pose, preferring the hierarchical cuRobo planner.

    Implements the doc section 5.2 routing: when a hierarchical planner is
    provided and usable, it handles the query (dual-arm / constrained /
    obstacle-rich cases); any unavailability or planning failure degrades
    to the Genesis OMPL path (``inverse_kinematics`` + ``plan_path``).

    Args:
        robot: Backend articulation used for the OMPL fallback.
        goal_pos: Goal EE world position, shape ``(3,)``.
        goal_quat: Optional goal EE quaternion ``(w, x, y, z)``.
        ee_link: EE link name for the OMPL IK fallback.
        planner: Optional hierarchical planner; ``None`` forces OMPL.
        num_waypoints: Waypoint count for the OMPL fallback.

    Returns:
        ``(trajectory, planner_name)`` where planner_name is
        ``"hierarchical_curobo"`` or ``"ompl"``.
    """
    if planner is not None:
        try:
            start = np.asarray(robot.get_qpos(), dtype=np.float64).reshape(-1)[
                : robot.n_dofs
            ]
            return (
                planner.plan_to_ee_pose(start, goal_pos, goal_quat),
                "hierarchical_curobo",
            )
        except (CuRoboPlannerUnavailableError, PlannerError) as exc:
            logger.warning("Falling back to OMPL plan_path: %s", exc)

    if ee_link is None:
        raise ValueError("ee_link is required for the OMPL fallback")
    q_goal = robot.inverse_kinematics(ee_link, pos=np.asarray(goal_pos), quat=goal_quat)
    trajectory = robot.plan_path(
        np.asarray(q_goal).reshape(-1), num_waypoints=num_waypoints
    )
    return trajectory, "ompl"
