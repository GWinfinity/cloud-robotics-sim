"""Genesis (Taichi/CUDA) backend implementation.

This backend preserves the existing Genesis-based behavior by adapting the
new Backend ABC to the legacy ``gs.*`` API surface.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from cloud_robotics_sim.backend.base import (
    ArticulationBackend,
    CameraBackend,
    DeformableEntityBackend,
    EntityBackend,
    RendererBackend,
    SceneBackend,
    SimulatorBackend,
)
from cloud_robotics_sim.backend.types import (
    ArticulationState,
    BackendName,
    DeformableConfig,
    DeformableMaterialType,
    DeformableState,
    LightDescription,
    PhysicsState,
    Pose,
    RenderOutput,
    ViewerOptions,
)
from cloud_robotics_sim.utils.genesis_compat import ensure_genesis_initialized
from cloud_robotics_sim.utils.robomat_compat import resolve_genesis_rigid_material

logger = logging.getLogger(__name__)

try:
    import genesis as gs

    HAS_GENESIS = True
except ImportError:
    HAS_GENESIS = False
    gs = None


def _require_genesis() -> None:
    if not HAS_GENESIS or gs is None:
        raise RuntimeError(
            "Genesis backend requested but 'genesis-world' is not installed."
        )


class GenesisEntityBackend(EntityBackend):
    """Genesis implementation of a generic rigid body entity."""

    def __init__(
        self,
        morph: Any,
        surface: Any | None = None,
        *,
        name: str | None = None,
    ) -> None:
        _require_genesis()
        self._morph = morph
        self._surface = surface
        self._name = name
        self._entity: Any = None

    @property
    def name(self) -> str | None:
        return self._name

    def _resolve_entity(self) -> Any:
        if self._entity is None:
            raise RuntimeError("Entity has not been added to a Genesis scene yet.")
        return self._entity

    def bind(self, entity: Any) -> None:
        """Called by GenesisSceneBackend after add_entity."""
        self._entity = entity

    def get_pos(self) -> np.ndarray:
        entity = self._resolve_entity()
        if hasattr(entity, "get_pos"):
            return np.asarray(entity.get_pos(), dtype=np.float64)
        # Deformable entities may not expose get_pos; fall back to the mean
        # particle position or the morph position.
        positions = getattr(entity, "get_positions", lambda: None)()
        if positions is not None:
            return np.asarray(np.mean(positions, axis=0), dtype=np.float64)
        morph_pos = getattr(self._morph, "pos", None)
        if morph_pos is not None:
            return np.asarray(morph_pos, dtype=np.float64)
        return np.zeros(3, dtype=np.float64)

    def set_pos(self, pos: np.ndarray) -> None:
        entity = self._resolve_entity()
        if hasattr(entity, "set_pos"):
            entity.set_pos(np.asarray(pos, dtype=np.float64))
        else:
            logger.debug("Genesis deformable entity does not support set_pos")

    def get_quat(self) -> np.ndarray:
        entity = self._resolve_entity()
        if hasattr(entity, "get_quat"):
            return np.asarray(entity.get_quat(), dtype=np.float64)
        morph_quat = getattr(self._morph, "quat", None)
        if morph_quat is not None:
            return np.asarray(morph_quat, dtype=np.float64)
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

    def set_quat(self, quat: np.ndarray) -> None:
        entity = self._resolve_entity()
        if hasattr(entity, "set_quat"):
            entity.set_quat(np.asarray(quat, dtype=np.float64))
        else:
            logger.debug("Genesis deformable entity does not support set_quat")

    def set_color(self, color: tuple[float, float, float, float]) -> None:
        entity = self._resolve_entity()
        if hasattr(entity, "set_color"):
            entity.set_color(color)
        else:
            logger.debug("Genesis entity does not support set_color")

    def apply_force(
        self,
        force: np.ndarray,
        pos: np.ndarray | None = None,
    ) -> None:
        entity = self._resolve_entity()
        if hasattr(entity, "apply_force"):
            entity.apply_force(
                np.asarray(force), pos=np.asarray(pos) if pos is not None else None
            )
        else:
            logger.debug("Genesis entity does not support apply_force")


class GenesisArticulationBackend(GenesisEntityBackend, ArticulationBackend):
    """Genesis implementation of an articulated robot."""

    @property
    def n_dofs(self) -> int:
        entity = self._resolve_entity()
        return int(getattr(entity, "n_dofs", 0) or getattr(entity, "n_qs", 0))

    @property
    def n_qs(self) -> int:
        entity = self._resolve_entity()
        return int(getattr(entity, "n_qs", 0) or getattr(entity, "n_dofs", 0))

    def get_qpos(self) -> np.ndarray:
        entity = self._resolve_entity()
        return np.asarray(entity.get_qpos(), dtype=np.float64)

    def set_qpos(
        self,
        qpos: np.ndarray,
        *,
        qs_idx_local: list[int] | None = None,
        **kwargs: Any,
    ) -> None:
        entity = self._resolve_entity()
        qpos_arr = np.asarray(qpos, dtype=np.float64)
        if qs_idx_local is not None and len(qs_idx_local) > 0:
            entity.set_qpos(qpos_arr, qs_idx_local=qs_idx_local)
        else:
            entity.set_qpos(qpos_arr)

    def get_qvel(self) -> np.ndarray:
        entity = self._resolve_entity()
        return np.asarray(entity.get_qvel(), dtype=np.float64)

    def set_qvel(self, qvel: np.ndarray) -> None:
        entity = self._resolve_entity()
        entity.set_qvel(np.asarray(qvel, dtype=np.float64))

    def control_dofs_position(
        self,
        targets: np.ndarray,
        stiffness: np.ndarray | None = None,
        damping: np.ndarray | None = None,
        dofs_idx_local: list[int] | None = None,
        **kwargs: Any,
    ) -> None:
        entity = self._resolve_entity()
        forward_kwargs: dict[str, Any] = {}
        if stiffness is not None:
            forward_kwargs["stiffness"] = np.asarray(stiffness)
        if damping is not None:
            forward_kwargs["damping"] = np.asarray(damping)
        if dofs_idx_local is not None:
            forward_kwargs["dofs_idx_local"] = dofs_idx_local
        entity.control_dofs_position(np.asarray(targets), **forward_kwargs)

    def control_dofs_velocity(self, targets: np.ndarray) -> None:
        entity = self._resolve_entity()
        entity.control_dofs_velocity(np.asarray(targets))

    def control_dofs_force(self, targets: np.ndarray) -> None:
        entity = self._resolve_entity()
        entity.control_dofs_force(np.asarray(targets))

    def get_state_batch(
        self,
        env_ids: list[int] | None = None,
    ) -> ArticulationState:
        # Genesis single-env path: env_ids is ignored.
        return ArticulationState(
            qpos=self.get_qpos(),
            qvel=self.get_qvel(),
            pos=self.get_pos(),
            quat=self.get_quat(),
        )

    def set_state_batch(
        self,
        state: ArticulationState,
        env_ids: list[int] | None = None,
    ) -> None:
        if state.pos is not None:
            self.set_pos(state.pos)
        if state.quat is not None:
            self.set_quat(state.quat)
        if state.qpos is not None:
            entity = self._resolve_entity()
            qpos_arr = np.asarray(state.qpos, dtype=np.float64)
            if hasattr(entity, "joints"):
                # For floating-base articulations, skip the 7-DOF root joint
                # by providing qs_idx_local of the actuated joints only.
                qs_idx_local: list[int] = []
                for joint in entity.joints:
                    joint_type = getattr(joint, "type", None)
                    if joint_type == gs.JOINT_TYPE.FREE:
                        continue
                    qs_local = getattr(joint, "qs_idx_local", None)
                    if qs_local is not None:
                        qs_idx_local.extend(list(qs_local))
                if qs_idx_local:
                    self.set_qpos(qpos_arr, qs_idx_local=qs_idx_local)
                    return
            self.set_qpos(qpos_arr)
        if state.qvel is not None:
            self.set_qvel(state.qvel)

    def control_position_batch(
        self,
        targets: np.ndarray,
        stiffness: np.ndarray | None = None,
        damping: np.ndarray | None = None,
    ) -> None:
        # Genesis single-env path: squeeze leading batch dim if present.
        targets_arr = np.asarray(targets)
        if targets_arr.ndim == 2 and targets_arr.shape[0] == 1:
            targets_arr = targets_arr[0]
        self.control_dofs_position(targets_arr, stiffness=stiffness, damping=damping)

    def get_end_effector_pose(self) -> Pose:
        entity = self._resolve_entity()
        if hasattr(entity, "get_end_effector_pose"):
            pose = entity.get_end_effector_pose()
            return Pose(pos=pose[:3], quat=pose[3:])
        # Fallback: return base pose.
        return Pose(pos=self.get_pos(), quat=self.get_quat())

    def get_joint_names(self) -> list[str]:
        """Return the names of all joints in the articulation."""
        entity = self._resolve_entity()
        if not hasattr(entity, "joints"):
            return []
        return [getattr(joint, "name", "") for joint in entity.joints]

    def get_joint_dofs_idx_local(self, joint_name: str) -> list[int]:
        """Return the local DoF indices for a given joint."""
        entity = self._resolve_entity()
        if not hasattr(entity, "joints"):
            return []
        for joint in entity.joints:
            if getattr(joint, "name", None) == joint_name:
                dofs = getattr(joint, "dofs_idx_local", None)
                return list(dofs) if dofs is not None else []
        return []

    def get_joint_qs_idx_local(self, joint_name: str) -> list[int]:
        """Return the local generalized-position indices for a given joint."""
        entity = self._resolve_entity()
        if not hasattr(entity, "joints"):
            return []
        for joint in entity.joints:
            if getattr(joint, "name", None) == joint_name:
                qs = getattr(joint, "qs_idx_local", None)
                return list(qs) if qs is not None else []
        return []

    def is_fixed_base(self) -> bool:
        """Return True if the articulation has a fixed (non-floating) base."""
        entity = self._resolve_entity()
        if not hasattr(entity, "joints"):
            return True
        for joint in entity.joints:
            if getattr(joint, "type", None) == gs.JOINT_TYPE.FREE:
                return False
        return True

    # ------------------------------------------------------------------
    # Extended interface (RoboTwin->Genesis migration skeleton)
    # ------------------------------------------------------------------

    def set_dofs_gains(
        self,
        kp: np.ndarray,
        kv: np.ndarray | None = None,
        force_range: tuple[np.ndarray, np.ndarray] | None = None,
        armature: np.ndarray | None = None,
        dofs_idx: list[int] | None = None,
        envs_idx: list[int] | None = None,
    ) -> None:
        """Set per-DoF PD gains from an embodiment ``config.yml`` mapping."""
        entity = self._resolve_entity()
        entity.set_dofs_kp(np.asarray(kp), dofs_idx_local=dofs_idx, envs_idx=envs_idx)
        if kv is not None:
            entity.set_dofs_kv(
                np.asarray(kv), dofs_idx_local=dofs_idx, envs_idx=envs_idx
            )
        if force_range is not None:
            lower, upper = force_range
            entity.set_dofs_force_range(
                np.asarray(lower),
                np.asarray(upper),
                dofs_idx_local=dofs_idx,
                envs_idx=envs_idx,
            )
        if armature is not None:
            entity.set_dofs_armature(
                np.asarray(armature), dofs_idx_local=dofs_idx, envs_idx=envs_idx
            )

    def inverse_kinematics(
        self,
        link_name: str,
        pos: np.ndarray,
        quat: np.ndarray | None = None,
        dofs_idx: list[int] | None = None,
        envs_idx: list[int] | None = None,
    ) -> np.ndarray:
        """Solve IK for a single link using Genesis' built-in solver."""
        entity = self._resolve_entity()
        link = entity.get_link(link_name)
        q = entity.inverse_kinematics(
            link,
            pos=np.asarray(pos, dtype=np.float64),
            quat=None if quat is None else np.asarray(quat, dtype=np.float64),
            dofs_idx_local=dofs_idx,
            envs_idx=envs_idx,
        )
        return np.asarray(q, dtype=np.float64)

    def inverse_kinematics_multilink(
        self,
        link_names: list[str],
        poss: list[np.ndarray] | np.ndarray,
        quats: list[np.ndarray] | np.ndarray | None = None,
        rot_mask: tuple[bool, bool, bool] | None = None,
        pos_mask: tuple[bool, bool, bool] | None = None,
        envs_idx: list[int] | None = None,
    ) -> np.ndarray:
        """Solve IK for multiple links (e.g. both arms of aloha) at once."""
        entity = self._resolve_entity()
        links = [entity.get_link(name) for name in link_names]
        kwargs: dict[str, Any] = {"envs_idx": envs_idx}
        if rot_mask is not None:
            kwargs["rot_mask"] = list(rot_mask)
        if pos_mask is not None:
            kwargs["pos_mask"] = list(pos_mask)
        q = entity.inverse_kinematics_multilink(
            links,
            poss=np.asarray(poss, dtype=np.float64),
            quats=None if quats is None else np.asarray(quats, dtype=np.float64),
            **kwargs,
        )
        return np.asarray(q, dtype=np.float64)

    def plan_path(
        self,
        qpos_goal: np.ndarray,
        qpos_start: np.ndarray | None = None,
        num_waypoints: int = 50,
        envs_idx: list[int] | None = None,
    ) -> np.ndarray:
        """Plan a collision-free joint-space path (OMPL RRTConnect)."""
        entity = self._resolve_entity()
        path = entity.plan_path(
            qpos_goal=np.asarray(qpos_goal, dtype=np.float64),
            qpos_start=(
                None if qpos_start is None else np.asarray(qpos_start, dtype=np.float64)
            ),
            num_waypoints=num_waypoints,
            envs_idx=envs_idx,
        )
        return np.asarray(path, dtype=np.float64)

    def get_link_pose(self, link_name: str) -> Pose:
        """Return the world-space pose of a link."""
        entity = self._resolve_entity()
        link = entity.get_link(link_name)
        return Pose(
            pos=np.asarray(link.get_pos(), dtype=np.float64),
            quat=np.asarray(link.get_quat(), dtype=np.float64),
        )

    def set_friction_ratio(
        self,
        ratios: np.ndarray,
        envs_idx: list[int] | None = None,
    ) -> None:
        """Per-env friction domain randomization."""
        entity = self._resolve_entity()
        entity.set_friction_ratio(np.asarray(ratios), envs_idx=envs_idx)

    def set_mass_shift(
        self,
        shifts: np.ndarray,
        envs_idx: list[int] | None = None,
    ) -> None:
        """Per-env link-mass domain randomization."""
        entity = self._resolve_entity()
        entity.set_mass_shift(np.asarray(shifts), envs_idx=envs_idx)

    def set_com_shift(
        self,
        shifts: np.ndarray,
        envs_idx: list[int] | None = None,
    ) -> None:
        """Per-env center-of-mass domain randomization."""
        entity = self._resolve_entity()
        entity.set_COM_shift(np.asarray(shifts), envs_idx=envs_idx)


class GenesisDeformableEntityBackend(DeformableEntityBackend):
    """Genesis implementation of a deformable (soft body / fluid) entity."""

    def __init__(
        self,
        morph: Any,
        material: Any,
        config: DeformableConfig,
        *,
        name: str | None = None,
    ) -> None:
        _require_genesis()
        self._morph = morph
        self._material = material
        self._config = config
        self._name = name
        self._entity: Any = None

    @property
    def name(self) -> str | None:
        return self._name

    def _resolve_entity(self) -> Any:
        if self._entity is None:
            raise RuntimeError(
                "Deformable entity has not been added to a Genesis scene yet."
            )
        return self._entity

    def bind(self, entity: Any) -> None:
        """Called by GenesisSceneBackend after add_entity."""
        self._entity = entity

    def get_pos(self) -> np.ndarray:
        entity = self._resolve_entity()
        if hasattr(entity, "get_pos"):
            return np.asarray(entity.get_pos(), dtype=np.float64)
        # Deformable entities may not expose get_pos; fall back to the mean
        # particle position or the morph position.
        positions = getattr(entity, "get_positions", lambda: None)()
        if positions is not None:
            return np.asarray(np.mean(positions, axis=0), dtype=np.float64)
        morph_pos = getattr(self._morph, "pos", None)
        if morph_pos is not None:
            return np.asarray(morph_pos, dtype=np.float64)
        return np.zeros(3, dtype=np.float64)

    def set_pos(self, pos: np.ndarray) -> None:
        entity = self._resolve_entity()
        if hasattr(entity, "set_pos"):
            entity.set_pos(np.asarray(pos, dtype=np.float64))
        else:
            logger.debug("Genesis deformable entity does not support set_pos")

    def get_quat(self) -> np.ndarray:
        entity = self._resolve_entity()
        if hasattr(entity, "get_quat"):
            return np.asarray(entity.get_quat(), dtype=np.float64)
        morph_quat = getattr(self._morph, "quat", None)
        if morph_quat is not None:
            return np.asarray(morph_quat, dtype=np.float64)
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

    def set_quat(self, quat: np.ndarray) -> None:
        entity = self._resolve_entity()
        if hasattr(entity, "set_quat"):
            entity.set_quat(np.asarray(quat, dtype=np.float64))
        else:
            logger.debug("Genesis deformable entity does not support set_quat")

    def set_color(self, color: tuple[float, float, float, float]) -> None:
        entity = self._resolve_entity()
        if hasattr(entity, "set_color"):
            entity.set_color(color)
        else:
            logger.debug("Genesis deformable entity does not support set_color")

    def apply_force(
        self,
        force: np.ndarray,
        pos: np.ndarray | None = None,
    ) -> None:
        entity = self._resolve_entity()
        if hasattr(entity, "apply_force"):
            entity.apply_force(
                np.asarray(force), pos=np.asarray(pos) if pos is not None else None
            )
        else:
            logger.debug("Genesis deformable entity does not support apply_force")

    def get_particle_positions(self) -> np.ndarray:
        entity = self._resolve_entity()
        positions = getattr(entity, "get_positions", lambda: None)()
        if positions is None:
            positions = getattr(entity, "init_positions", None)
        return np.asarray(positions, dtype=np.float64)

    def get_particle_velocities(self) -> np.ndarray | None:
        entity = self._resolve_entity()
        velocities = getattr(entity, "get_velocities", lambda: None)()
        if velocities is None:
            return None
        return np.asarray(velocities, dtype=np.float64)

    def set_vertex_constraints(
        self,
        indices: np.ndarray,
        positions: np.ndarray,
        *,
        soft: bool = False,
        stiffness: float = 1.0e4,
    ) -> None:
        entity = self._resolve_entity()
        if hasattr(entity, "set_vertex_constraints"):
            entity.set_vertex_constraints(
                np.asarray(indices, dtype=np.int32),
                np.asarray(positions, dtype=np.float64),
                is_soft_constraint=soft,
                stiffness=stiffness,
            )
        else:
            logger.debug(
                "Genesis deformable entity does not support vertex constraints"
            )

    def update_constraint_targets(
        self,
        indices: np.ndarray,
        positions: np.ndarray,
    ) -> None:
        entity = self._resolve_entity()
        if hasattr(entity, "update_constraint_targets"):
            entity.update_constraint_targets(
                np.asarray(indices, dtype=np.int32),
                np.asarray(positions, dtype=np.float64),
            )
        else:
            logger.debug(
                "Genesis deformable entity does not support updating constraint targets"
            )

    def get_deformable_state(self) -> DeformableState:
        return DeformableState(
            positions=self.get_particle_positions(),
            velocities=self.get_particle_velocities(),
        )

    def set_deformable_state(self, state: DeformableState) -> None:
        entity = self._resolve_entity()
        if state.positions is not None and hasattr(entity, "set_positions"):
            entity.set_positions(np.asarray(state.positions, dtype=np.float64))
        else:
            logger.debug("Genesis deformable entity does not support set_positions")


class GenesisCameraBackend(CameraBackend):
    """Genesis camera wrapper."""

    def __init__(self, name: str, camera: Any) -> None:
        self._name = name
        self._camera = camera

    @property
    def name(self) -> str:
        return self._name

    def render(
        self,
        *,
        rgb: bool = True,
        depth: bool = False,
        segmentation: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, ...]:
        result = self._camera.render(rgb=rgb, depth=depth, segmentation=segmentation)
        if isinstance(result, tuple):
            return tuple(np.asarray(r) for r in result)
        return np.asarray(result)

    def get_camera_params(self) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(intrinsic, extrinsic)`` for data-format alignment.

        The extrinsic is the camera-to-world transform (Genesis
        ``Camera.transform``), matching the RoboTwin HDF5 convention. For
        batched scenes (``n_envs > 1``) the transform is expressed in the
        global frame and includes the camera's env-grid offset.
        """
        cam = self._camera
        if getattr(cam, "is_built", False):
            intrinsic = np.asarray(cam.intrinsics, dtype=np.float64)
            extrinsic = np.asarray(cam.transform, dtype=np.float64)
        else:
            # Pre-build fallback: derive intrinsics from res/fov and compose
            # the extrinsic from the nominal pose.
            from cloud_robotics_sim.utils.camera import intrinsics_from_fov

            width, height = (int(v) for v in cam.res)
            intrinsic = intrinsics_from_fov(width, height, float(cam.fov))
            pos = np.asarray(cam.pos, dtype=np.float64).reshape(-1)[:3]
            lookat = np.asarray(cam.lookat, dtype=np.float64).reshape(-1)[:3]
            extrinsic = _lookat_to_matrix(pos, lookat)
        return intrinsic, extrinsic


def _lookat_to_matrix(
    pos: np.ndarray,
    lookat: np.ndarray,
    up: np.ndarray | None = None,
) -> np.ndarray:
    """Compose a 4x4 camera-to-world matrix (Genesis convention: forward=+x)."""
    if up is None:
        up = np.array([0.0, 0.0, 1.0])
    forward = lookat - pos
    norm = np.linalg.norm(forward)
    forward = forward / norm if norm > 1e-9 else np.array([1.0, 0.0, 0.0])
    left = np.cross(up, forward)
    left_norm = np.linalg.norm(left)
    if left_norm < 1e-9:
        # forward is parallel to up; pick an arbitrary perpendicular vector.
        alt = (
            np.array([1.0, 0.0, 0.0])
            if abs(forward[0]) < 0.9
            else np.array([0.0, 1.0, 0.0])
        )
        left = np.cross(forward, alt)
        left /= np.linalg.norm(left)
    else:
        left /= left_norm
    true_up = np.cross(forward, left)
    mat = np.eye(4, dtype=np.float64)
    mat[:3, 0] = forward
    mat[:3, 1] = -left
    mat[:3, 2] = true_up
    mat[:3, 3] = pos
    return mat


class GenesisRendererBackend(RendererBackend):
    """Genesis renderer adapter.

    Genesis ties rendering closely to the scene/viewer; this adapter provides
    a thin compatibility layer on top of the scene's camera entities.
    """

    def __init__(self, gs_scene: Any) -> None:
        self._gs_scene = gs_scene
        self._cameras: dict[str, GenesisCameraBackend] = {}

    def add_camera(
        self,
        name: str,
        pos: tuple[float, float, float],
        lookat: tuple[float, float, float],
        resolution: tuple[int, int],
        fov: float = 60.0,
    ) -> CameraBackend:
        cam = self._gs_scene.add_camera(
            pos=pos,
            lookat=lookat,
            res=resolution,
            fov=fov,
            GUI=False,
        )
        backend = GenesisCameraBackend(name, cam)
        self._cameras[name] = backend
        return backend

    def render(
        self,
        camera_name: str | None = None,
        *,
        rgb: bool = True,
        depth: bool = False,
        segmentation: bool = False,
    ) -> RenderOutput:
        if camera_name is None:
            camera_name = next(iter(self._cameras), None)
        if camera_name is None or camera_name not in self._cameras:
            raise ValueError(f"Camera '{camera_name}' not found")
        rgb_arr, depth_arr, seg_arr = None, None, None
        result = self._cameras[camera_name].render(
            rgb=rgb, depth=depth, segmentation=segmentation
        )
        if isinstance(result, tuple):
            parts = list(result)
            rgb_arr = parts.pop(0) if rgb else None
            depth_arr = parts.pop(0) if depth else None
            seg_arr = parts.pop(0) if segmentation else None
        else:
            rgb_arr = np.asarray(result) if rgb else None
        return RenderOutput(rgb=rgb_arr, depth=depth_arr, segmentation=seg_arr)

    def render_async(
        self,
        camera_names: list[str],
        *,
        rgb: bool = True,
        depth: bool = False,
    ) -> dict[str, Any]:
        # Genesis does not support async rendering; return synchronous results.
        return {
            name: self.render(name, rgb=rgb, depth=depth)
            for name in camera_names
            if name in self._cameras
        }


class GenesisSceneBackend(SceneBackend):
    """Genesis scene/world adapter."""

    def __init__(self, backend: "GenesisBackend", gs_scene: Any) -> None:
        _require_genesis()
        self._backend = backend
        self._gs_scene = gs_scene
        self._renderer = GenesisRendererBackend(gs_scene)
        self._built = False
        self._n_envs = 1
        self._entities: list[EntityBackend | ArticulationBackend] = []
        self._entity_names: dict[str, EntityBackend | ArticulationBackend] = {}

    @property
    def backend(self) -> "GenesisBackend":
        return self._backend

    @property
    def renderer(self) -> RendererBackend | None:
        return self._renderer

    def add_entity(self, entity: EntityBackend) -> None:
        if isinstance(entity, GenesisDeformableEntityBackend):
            gs_entity = self._gs_scene.add_entity(
                morph=entity._morph,
                material=entity._material,
            )
            entity.bind(gs_entity)
            self._entities.append(entity)
            if entity.name:
                self._entity_names[entity.name] = entity
            return

        if not isinstance(entity, GenesisEntityBackend):
            raise TypeError(
                "Genesis scene only accepts GenesisEntityBackend or "
                "GenesisDeformableEntityBackend instances"
            )
        kwargs: dict[str, Any] = {"morph": entity._morph}
        if entity._surface is not None:
            kwargs["surface"] = entity._surface
        gs_entity = self._gs_scene.add_entity(**kwargs)
        entity.bind(gs_entity)
        self._entities.append(entity)
        if entity.name:
            self._entity_names[entity.name] = entity

    def add_articulation(self, articulation: ArticulationBackend) -> None:
        # In Genesis, articulations are added the same way as entities.
        self.add_entity(articulation)

    def add_light(self, light: LightDescription) -> None:
        from cloud_robotics_sim.utils.genesis_compat import get_genesis_lights

        lights = get_genesis_lights()
        if lights is None:
            logger.debug("Genesis version does not expose gs.lights; skipping light")
            return
        if light.light_type.name == "AMBIENT":
            self._gs_scene.add_light(
                lights.Ambient(color=light.color, intensity=light.intensity)
            )
        elif light.light_type.name == "DIRECTIONAL":
            direction = light.direction or (0.0, 0.3, -1.0)
            self._gs_scene.add_light(
                lights.Directional(
                    pos=light.pos or (0.0, 0.0, 5.0),
                    direction=direction,
                    color=light.color,
                    intensity=light.intensity,
                    cast_shadow=light.cast_shadow,
                )
            )
        else:
            logger.warning("Unsupported Genesis light type: %s", light.light_type)

    def build(
        self, n_envs: int = 1, env_spacing: tuple[float, float] | None = None
    ) -> None:
        kwargs: dict[str, Any] = {}
        if n_envs > 1:
            kwargs["n_envs"] = n_envs
            if env_spacing is not None:
                kwargs["env_spacing"] = env_spacing
        self._gs_scene.build(**kwargs)
        self._built = True
        self._n_envs = n_envs

    def step(self) -> None:
        self._gs_scene.step()

    def reset(self) -> None:
        # Genesis resets via state restore; core logic handles recomposition.
        logger.debug("GenesisSceneBackend.reset is a no-op; rely on state rebuild")

    def get_physics_state(self) -> PhysicsState:
        """Return a snapshot of all tracked entities' poses and joint states."""
        time = 0.0
        if hasattr(self._gs_scene, "t"):
            time = float(self._gs_scene.t)

        entity_states: dict[str, Any] = {}
        for entity in self._entities:
            name = entity.name or f"__entity_{id(entity)}"
            state_entry: dict[str, Any] = {
                "pos": np.asarray(entity.get_pos(), dtype=np.float64),
                "quat": np.asarray(entity.get_quat(), dtype=np.float64),
            }
            if isinstance(entity, GenesisArticulationBackend):
                state_entry["qpos"] = np.asarray(entity.get_qpos(), dtype=np.float64)
                state_entry["qvel"] = np.asarray(entity.get_qvel(), dtype=np.float64)
            entity_states[name] = state_entry

        return PhysicsState(time=time, custom={"entities": entity_states})

    def set_physics_state(self, state: PhysicsState) -> None:
        """Restore tracked entities from a PhysicsState snapshot."""
        entity_states = (state.custom or {}).get("entities", {})
        if not entity_states:
            logger.debug("No entity states found in PhysicsState")
            return

        for entity in self._entities:
            name = entity.name or f"__entity_{id(entity)}"
            if name not in entity_states:
                continue
            entry = entity_states[name]
            if "pos" in entry:
                entity.set_pos(np.asarray(entry["pos"], dtype=np.float64))
            if "quat" in entry:
                entity.set_quat(np.asarray(entry["quat"], dtype=np.float64))
            if isinstance(entity, GenesisArticulationBackend):
                if "qpos" in entry:
                    entity.set_state_batch(
                        ArticulationState(
                            qpos=np.asarray(entry["qpos"], dtype=np.float64)
                        )
                    )
                if "qvel" in entry:
                    entity.set_qvel(np.asarray(entry["qvel"], dtype=np.float64))


class GenesisBackend(SimulatorBackend):
    """Genesis (Taichi/CUDA) simulator backend."""

    def __init__(self) -> None:
        self._initialized = False

    @property
    def name(self) -> BackendName:
        return BackendName.GENESIS

    def initialize(
        self,
        *,
        headless: bool = True,
        device: str = "cuda",
        **kwargs: Any,
    ) -> None:
        _require_genesis()
        use_cuda = device.lower() in {"cuda", "gpu", "auto"}
        ensure_genesis_initialized(headless=headless, use_cuda=use_cuda, **kwargs)
        self._initialized = True

    def create_scene(
        self,
        *,
        dt: float,
        substeps: int,
        headless: bool = True,
        viewer_options: ViewerOptions | None = None,
        fem_options: Any | None = None,
        pbd_options: Any | None = None,
        sph_options: Any | None = None,
        mpm_options: Any | None = None,
        sf_options: Any | None = None,
        renderer: Any | None = None,
    ) -> SceneBackend:
        _require_genesis()
        gs_viewer_options = None
        if not headless and viewer_options is not None:
            gs_viewer_options = gs.options.ViewerOptions(
                camera_pos=viewer_options.camera_pos,
                camera_lookat=viewer_options.camera_lookat,
                res=viewer_options.resolution,
                max_FPS=viewer_options.max_fps,
            )
        scene_kwargs: dict[str, Any] = {
            "viewer_options": gs_viewer_options,
            "sim_options": gs.options.SimOptions(dt=dt, substeps=substeps),
            "show_viewer": not headless,
        }
        if fem_options is not None:
            scene_kwargs["fem_options"] = fem_options
        if pbd_options is not None:
            scene_kwargs["pbd_options"] = pbd_options
        if sph_options is not None:
            scene_kwargs["sph_options"] = sph_options
        if mpm_options is not None:
            scene_kwargs["mpm_options"] = mpm_options
        if sf_options is not None:
            scene_kwargs["sf_options"] = sf_options
        if renderer is not None:
            scene_kwargs["renderer"] = renderer
        gs_scene = gs.Scene(**scene_kwargs)
        return GenesisSceneBackend(self, gs_scene)

    def _create_morph_kwargs(
        self,
        pos: tuple[float, float, float],
        quat: tuple[float, float, float, float] | None,
        color: tuple[float, float, float, float] | None,
        static: bool,
        friction: float,
        density: float | None,
        name: str | None,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"pos": pos, "fixed": static}
        if quat is not None:
            kwargs["quat"] = quat
        if density is not None:
            kwargs["density"] = density
        return kwargs

    def create_box(
        self,
        size: tuple[float, float, float],
        pos: tuple[float, float, float],
        quat: tuple[float, float, float, float] | None = None,
        color: tuple[float, float, float, float] | None = None,
        static: bool = True,
        friction: float = 0.5,
        density: float | None = None,
        name: str | None = None,
    ) -> EntityBackend:
        _require_genesis()
        kwargs = self._create_morph_kwargs(
            pos, quat, color, static, friction, density, name
        )
        morph = gs.morphs.Box(size=size, **kwargs)
        surface = gs.surfaces.Default(color=color, roughness=0.8) if color else None
        return GenesisEntityBackend(morph, surface, name=name)

    def create_sphere(
        self,
        radius: float,
        pos: tuple[float, float, float],
        quat: tuple[float, float, float, float] | None = None,
        color: tuple[float, float, float, float] | None = None,
        static: bool = True,
        friction: float = 0.5,
        density: float | None = None,
        name: str | None = None,
    ) -> EntityBackend:
        _require_genesis()
        kwargs = self._create_morph_kwargs(
            pos, quat, color, static, friction, density, name
        )
        morph = gs.morphs.Sphere(radius=radius, **kwargs)
        surface = gs.surfaces.Default(color=color, roughness=0.8) if color else None
        return GenesisEntityBackend(morph, surface, name=name)

    def create_cylinder(
        self,
        radius: float,
        height: float,
        pos: tuple[float, float, float],
        quat: tuple[float, float, float, float] | None = None,
        color: tuple[float, float, float, float] | None = None,
        static: bool = True,
        friction: float = 0.5,
        density: float | None = None,
        name: str | None = None,
    ) -> EntityBackend:
        _require_genesis()
        kwargs = self._create_morph_kwargs(
            pos, quat, color, static, friction, density, name
        )
        morph = gs.morphs.Cylinder(radius=radius, height=height, **kwargs)
        surface = gs.surfaces.Default(color=color, roughness=0.8) if color else None
        return GenesisEntityBackend(morph, surface, name=name)

    def create_mesh(
        self,
        file: str,
        pos: tuple[float, float, float],
        quat: tuple[float, float, float, float] | None = None,
        scale: tuple[float, float, float] | None = None,
        color: tuple[float, float, float, float] | None = None,
        static: bool = True,
        friction: float = 0.5,
        material: str | None = None,
        name: str | None = None,
    ) -> EntityBackend:
        _require_genesis()

        kwargs: dict[str, Any] = {"file": file, "pos": pos, "fixed": static}
        if quat is not None:
            kwargs["quat"] = quat
        if scale is not None:
            kwargs["scale"] = scale
        resolved = resolve_genesis_rigid_material(material)
        if resolved is not None:
            kwargs["material"] = resolved
        morph = gs.morphs.Mesh(**kwargs)
        surface = gs.surfaces.Default(color=color, roughness=0.8) if color else None
        return GenesisEntityBackend(morph, surface, name=name)

    def _resolve_deformable_morph(
        self,
        shape: str,
        pos: tuple[float, float, float],
        quat: tuple[float, float, float, float] | None,
        size: tuple[float, float, float] | None,
        radius: float | None,
        file: str | None,
        scale: tuple[float, float, float] | None,
        config: DeformableConfig,
    ) -> Any:
        """Build the Genesis morph for a deformable entity.

        For FEM solids, ``maxvolume`` is used to control tetrahedral resolution
        as a function of the requested multiscale ``resolution_level``.
        """
        kwargs: dict[str, Any] = {"pos": pos, "fixed": config.fixed}
        if quat is not None:
            kwargs["quat"] = quat

        # Map resolution level to a characteristic element size / volume.
        # Level 1 = coarse, 2 = medium, 3 = fine.
        # For a unit cube: maxvolume 0.01 (coarse) -> 0.001 (fine).
        level = max(1, min(3, config.resolution_level))
        scale_factor = 4.0 ** (3 - level)  # level 3 -> 1, level 1 -> 16

        if shape == "box":
            if size is None:
                raise ValueError("size is required for deformable box")
            volume = float(size[0] * size[1] * size[2])
            maxvolume = max(volume / (100.0 * scale_factor), volume / 5000.0)
            kwargs["size"] = size
            kwargs["maxvolume"] = maxvolume
            return gs.morphs.Box(**kwargs)

        if shape == "sphere":
            if radius is None:
                raise ValueError("radius is required for deformable sphere")
            volume = 4.0 / 3.0 * np.pi * radius**3
            maxvolume = max(volume / (100.0 * scale_factor), volume / 5000.0)
            kwargs["radius"] = radius
            kwargs["maxvolume"] = maxvolume
            return gs.morphs.Sphere(**kwargs)

        if shape == "mesh":
            if file is None:
                raise ValueError("file is required for deformable mesh")
            kwargs["file"] = file
            if scale is not None:
                kwargs["scale"] = scale
            # Mesh resolution is controlled by the input mesh; allow caller to
            # tune via maxvolume if a tetrahedralization is performed.
            kwargs.setdefault("maxvolume", -1.0)
            return gs.morphs.Mesh(**kwargs)

        raise ValueError(f"Unsupported deformable shape: {shape}")

    def _resolve_deformable_material(self, config: DeformableConfig) -> Any:
        """Build the Genesis material for a deformable entity."""
        material = config.material
        if material == DeformableMaterialType.FEM_ELASTIC:
            return gs.materials.FEM.Elastic(
                E=config.youngs_modulus,
                nu=config.poisson_ratio,
                rho=config.density,
                model="linear_corotated",
            )
        if material == DeformableMaterialType.FEM_CLOTH:
            return gs.materials.FEM.Cloth(
                E=config.youngs_modulus,
                nu=config.poisson_ratio,
                rho=config.density,
            )
        if material == DeformableMaterialType.PBD_ELASTIC:
            return gs.materials.PBD.Elastic(
                rho=config.density,
            )
        if material == DeformableMaterialType.PBD_CLOTH:
            return gs.materials.PBD.Cloth(
                rho=config.density,
            )
        if material == DeformableMaterialType.PBD_LIQUID:
            return gs.materials.PBD.Liquid(
                rho=config.density,
            )
        if material == DeformableMaterialType.SPH_LIQUID:
            return gs.materials.SPH.Liquid(
                rho=config.density,
            )
        raise NotImplementedError(
            f"Deformable material {material!r} is not implemented"
        )

    def create_deformable(
        self,
        config: DeformableConfig,
        shape: str,
        *,
        size: tuple[float, float, float] | None = None,
        radius: float | None = None,
        file: str | None = None,
        scale: tuple[float, float, float] | None = None,
        pos: tuple[float, float, float] = (0.0, 0.0, 0.0),
        quat: tuple[float, float, float, float] | None = None,
        color: tuple[float, float, float, float] | None = None,
        name: str | None = None,
    ) -> DeformableEntityBackend:
        _require_genesis()
        morph = self._resolve_deformable_morph(
            shape=shape,
            pos=pos,
            quat=quat,
            size=size,
            radius=radius,
            file=file,
            scale=scale,
            config=config,
        )
        material = self._resolve_deformable_material(config)
        backend = GenesisDeformableEntityBackend(
            morph=morph,
            material=material,
            config=config,
            name=name,
        )
        # Store color on the surface/material if possible; Genesis deformables
        # do not accept a surface, so we keep it on the backend for later use.
        backend._color = color  # type: ignore[attr-defined]
        return backend

    def load_mjcf(
        self,
        file: str,
        pos: tuple[float, float, float],
        **kwargs: Any,
    ) -> ArticulationBackend:
        _require_genesis()
        morph = gs.morphs.MJCF(file=file, pos=pos, **kwargs)
        return GenesisArticulationBackend(morph, name=file)

    def load_urdf(
        self,
        file: str,
        pos: tuple[float, float, float],
        **kwargs: Any,
    ) -> ArticulationBackend:
        _require_genesis()
        morph = gs.morphs.URDF(file=file, pos=pos, **kwargs)
        return GenesisArticulationBackend(morph, name=file)

    def create_light(self, description: LightDescription) -> LightDescription:
        return description
