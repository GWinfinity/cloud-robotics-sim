"""MuJoCo-backed DynamicsBackend implementation."""
from __future__ import annotations

from typing import Any

import mujoco
import numpy as np

from dexterous_gnn_qp.core.backends.base import DynamicsBackend
from dexterous_gnn_qp.core.dynamics.mujoco_model import hand_jacobian
from dexterous_gnn_qp.core.env.loader import HandScene, build_scene
from dexterous_gnn_qp.core.env.sim import SimState, compute_sim_state


class MujocoDynamicsBackend(DynamicsBackend):
    """Wraps a MuJoCo HandScene and MjData into the backend-neutral interface."""

    def __init__(
        self,
        scene: HandScene,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        cfg: Any | None = None,
    ) -> None:
        self._scene = scene
        self._model = model
        self._data = data
        self._cfg = cfg

    @property
    def dt(self) -> float:
        return float(self._scene.dt)

    @property
    def scene(self) -> HandScene:
        return self._scene

    @property
    def model(self) -> mujoco.MjModel:
        return self._model

    @property
    def data(self) -> mujoco.MjData:
        return self._data

    @property
    def n_hand_dof(self) -> int:
        return int(self._scene.n_hand_dof)

    def get_hand_qpos(self) -> np.ndarray:
        return np.array(self._data.qpos[self._scene.hand_qpos_indices], dtype=float)

    def get_time(self) -> float:
        return float(self._data.time)

    def get_object_pose(self) -> tuple[np.ndarray, np.ndarray]:
        body = self._data.body(self._scene.object_body_id)
        return np.array(body.xpos, dtype=float), np.array(body.xquat, dtype=float)

    def compute_sim_state(
        self,
        mu: float | None = None,
        only_fingertips: bool = False,
    ) -> SimState:
        return compute_sim_state(
            self._model,
            self._data,
            self._scene,
            mu=mu,
            only_fingertips=only_fingertips,
        )

    def hand_jacobian(self, contact: Any) -> np.ndarray:
        return hand_jacobian(self._model, self._data, self._scene, contact)

    def set_initial_state(self, cfg: Any) -> None:
        """Set hand target configuration and object pose, then run mj_forward."""
        data = self._data
        model = self._model
        scene = self._scene

        if cfg.controller.hand_target is not None:
            data.qpos[scene.hand_qpos_indices] = cfg.controller.hand_target
        else:
            data.qpos[scene.hand_qpos_indices] = 0.0

        obj_quat = np.array(cfg.object.quaternion, dtype=float)
        obj_pos = np.array(cfg.object.position, dtype=float)
        data.qpos[scene.object_qpos_indices] = np.concatenate([obj_pos, obj_quat])
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)

    def apply_control(self, tau_hand: np.ndarray) -> None:
        """Apply hand torques plus PD feedback, and neutralize built-in actuators."""
        data = self._data
        model = self._model
        scene = self._scene

        q_target = (
            np.array(self._cfg.controller.hand_target, dtype=float)
            if self._cfg.controller.hand_target is not None
            else np.zeros(scene.n_hand_dof, dtype=float)
        )
        q = data.qpos[scene.hand_qpos_indices]
        v = data.qvel[scene.hand_dof_indices]
        tau_pd = float(self._cfg.controller.hand_kp) * (q_target - q) - float(
            self._cfg.controller.hand_kv
        ) * v
        data.qfrc_applied[scene.hand_dof_indices] = tau_hand + tau_pd

        # Neutralize position actuators by commanding the current joint position.
        if model.nu > 0:
            n = min(model.nu, len(scene.hand_qpos_indices))
            data.ctrl[:n] = data.qpos[scene.hand_qpos_indices[:n]]

    def step(self) -> None:
        mujoco.mj_step(self._model, self._data)

    @classmethod
    def from_config(cls, cfg: Any) -> "MujocoDynamicsBackend":
        """Build a MuJoCo backend from a DotDict config."""
        scene = build_scene(cfg)
        return cls(scene, scene.mj_model, scene.mj_data, cfg=cfg)
