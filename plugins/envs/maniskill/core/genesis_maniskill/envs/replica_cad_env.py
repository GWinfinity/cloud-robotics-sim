"""ReplicaCAD environment: ManiSkill's ReplicaCAD_SceneManipulation-style
scene running on the Genesis backend.

The scene geometry (apartment stage, rigid objects, articulated furniture)
comes from the ReplicaCAD scene dataset mirrored on ModelScope
(``jessy888/ManiSkill_replica_cad_dataset``); see
:mod:`genesis_maniskill.datasets.replicacad_assets` for download management.

Example::

    env = ReplicaCADEnv(scene_name="apt_0", robot_uid="franka",
                        task_type="pick_place", render_mode="human")
    obs, info = env.reset()
    for _ in range(50):
        obs, reward, terminated, truncated, info = env.step(
            env.action_space.sample())
    env.close()
"""

from __future__ import annotations

import genesis as gs
import numpy as np

from ..scenes.replica_cad_scene import ReplicaCADSceneBuilder
from .base_env import BaseEnv


class ReplicaCADEnv(BaseEnv):
    """ReplicaCAD apartment scene environment.

    Args:
        scene_name: ReplicaCAD scene configuration name (``apt_0``..``apt_5``
            or ``v3_sc*_staging_*``). See
            :meth:`ReplicaCADSceneBuilder.list_available_scenes`.
        dataset_root: Optional override for the dataset root (see
            ``CRS_REPLICACAD_ASSETS``).
        include_doors: Spawn articulated door templates (default False,
            matching ManiSkill's ReplicaCAD builder).
        **kwargs: Additional arguments passed to :class:`BaseEnv`.

    Note:
        GPU parallel simulation (``num_envs > 1``) is not supported yet.
    """

    def __init__(
        self,
        scene_name: str = "apt_0",
        dataset_root: str | None = None,
        include_doors: bool = False,
        **kwargs,
    ):
        if kwargs.get("num_envs", 1) != 1:
            raise NotImplementedError(
                "ReplicaCADEnv currently supports num_envs=1 only"
            )
        self.scene_name = scene_name
        self.dataset_root = dataset_root
        self.include_doors = include_doors
        self._scene_built = False

        scene_config = dict(kwargs.pop("scene_config", None) or {})
        scene_config.setdefault("include_doors", include_doors)

        super().__init__(
            scene_type="replicacad",
            scene_config=scene_config,
            **kwargs,
        )

    def _build_scene(self):
        """Build the ReplicaCAD scene (deferred ``scene.build()``)."""
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(
                dt=self.sim_dt,
                substeps=1,
            ),
            vis_options=gs.options.VisOptions(
                ambient_light=(0.3, 0.3, 0.3),
            ),
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(3.5, 3.5, 2.5),
                camera_lookat=(0.0, 0.0, 0.5),
            ),
            show_viewer=self.render_mode == "human",
        )

        self.scene_builder = ReplicaCADSceneBuilder(
            scene=self.scene,
            scene_name=self.scene_name,
            dataset_root=self.dataset_root,
            num_envs=self.num_envs,
            config=self.scene_config,
        )
        self.scene_builder.build()

        # Expose builder collections on the env (like KitchenEnv).
        self.objects = self.scene_builder.scene_objects
        self.movable_objects = self.scene_builder.movable_objects
        self.articulations = self.scene_builder.articulations
        self.background = self.scene_builder.background

        # NOTE: ``self.scene.build()`` is deferred to ``_setup_spaces`` so the
        # robot (``_build_agent``) and cameras (``_build_sensors``) are added
        # before the scene is compiled.

    def _setup_spaces(self):
        """Build the scene once all entities (robot, cameras) are added."""
        if not self._scene_built:
            self.scene.build()
            self.scene_builder.finalize()
            self._scene_built = True
        super()._setup_spaces()

    def _setup_cameras(self):
        """Setup ReplicaCAD cameras (added before ``scene.build()``)."""
        # Overhead entrance view covering the kitchen/dining area.
        self.cameras["base_camera"] = self.scene.add_camera(
            res=(128, 128),
            pos=(2.0, 2.0, 2.5),
            lookat=(0.0, 0.0, 0.5),
            fov=45,
        )

    def _get_rgb_obs(self) -> np.ndarray:
        """Render raw Genesis cameras (the plugin's Camera wrapper is unused)."""
        obs = {}
        for name, cam in self.cameras.items():
            rgb, *_ = cam.render(rgb=True)
            obs[name] = np.asarray(rgb)
        return obs

    def _get_rgbd_obs(self) -> np.ndarray:
        """Render RGB-D from raw Genesis cameras."""
        obs = {}
        for name, cam in self.cameras.items():
            rgb, depth, *_ = cam.render(rgb=True, depth=True)
            obs[name] = np.concatenate(
                [np.asarray(rgb), np.asarray(depth)[..., None]], axis=-1
            )
        return obs

    def _get_state_obs_dim(self) -> int:
        """Robot state + one 7-dof pose per movable object."""
        robot_dim = self.robot.state_dim
        object_dim = len(self.movable_objects) * 7  # pos(3) + quat(4)
        return robot_dim + object_dim

    def _get_state_obs(self) -> np.ndarray:
        """Robot state concatenated with movable object poses."""
        import torch

        robot_state = self.robot.get_state()
        object_states = []
        for obj in self.movable_objects.values():
            obj_state = torch.cat(
                [obj.get_pos().reshape(1, -1), obj.get_quat().reshape(1, -1)], dim=-1
            )
            object_states.append(obj_state)

        if object_states:
            object_states = torch.cat(object_states, dim=-1)
            obs = torch.cat([robot_state, object_states], dim=-1)
        else:
            obs = robot_state

        return obs.cpu().numpy()

    def _reset_scene(self):
        """Reset the scene and optionally randomize object placement."""
        super()._reset_scene()
        if self.scene_config.get("randomize_placement", False):
            self.scene_builder.randomize_object_placement()

    def list_scene_names(self) -> list[str]:
        """Return available ReplicaCAD scene configuration names."""
        return ReplicaCADSceneBuilder.list_available_scenes(root=self.dataset_root)

    def get_object(self, name: str):
        """Get a scene object (rigid or articulated) by name."""
        return self.scene_builder.scene_objects.get(name)

    def get_object_names(self) -> list:
        """Get list of all scene object names."""
        return list(self.scene_builder.scene_objects.keys())


__all__ = ["ReplicaCADEnv"]
