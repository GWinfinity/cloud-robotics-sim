"""Genesis-backed DynamicsBackend implementation."""
from __future__ import annotations

from pathlib import Path
from typing import Any, List

import numpy as np

from dexterous_gnn_qp.core.backends.base import DynamicsBackend
from dexterous_gnn_qp.core.env.sim import Contact, SimState
from dexterous_gnn_qp.core.utils.config import DotDict

from cloud_robotics_sim.backends.genesis_backend import (
    GenesisArticulationBackend,
    GenesisEntityBackend,
)

import genesis as gs


class GenesisDynamicsBackend(DynamicsBackend):
    """Wraps a Genesis scene with a dexterous hand and a grasped object."""

    def __init__(
        self,
        scene: Any,
        hand_entity: Any,
        object_entity: Any,
        cfg: Any,
        fingertip_link_names: List[str] | None = None,
    ) -> None:
        self._scene = scene
        self._hand_entity = hand_entity
        self._object_entity = object_entity
        self._cfg = cfg

        self._hand_backend = GenesisArticulationBackend(
            hand_entity._morph if hasattr(hand_entity, "_morph") else None,
            name="hand",
        )
        self._hand_backend.bind(hand_entity)

        self._object_backend = GenesisArticulationBackend(
            object_entity._morph if hasattr(object_entity, "_morph") else None,
            name="object",
        )
        self._object_backend.bind(object_entity)

        self._hand_dof_indices = list(range(int(hand_entity.n_dofs)))
        self._object_dof_indices = list(range(int(object_entity.n_dofs)))
        self._step_count = 0

        self._fingertip_link_names = set(fingertip_link_names or [])
        self._fingertip_geom_indices: set[int] = set()
        if not self._fingertip_link_names:
            for link in hand_entity.links:
                name = getattr(link, "name", "")
                if name.endswith("_tip") or name.endswith("_ds"):
                    self._fingertip_link_names.add(name)
        for geom in hand_entity.geoms:
            link_name = getattr(getattr(geom, "link", None), "name", "")
            if link_name in self._fingertip_link_names:
                self._fingertip_geom_indices.add(int(geom.idx))

    @property
    def dt(self) -> float:
        return float(self._scene.sim.dt)

    @property
    def n_hand_dof(self) -> int:
        return len(self._hand_dof_indices)

    def get_hand_qpos(self) -> np.ndarray:
        return np.asarray(self._hand_entity.get_qpos(), dtype=float)

    def get_time(self) -> float:
        return float(self._step_count * self.dt)

    def get_object_pose(self) -> tuple[np.ndarray, np.ndarray]:
        qpos = np.asarray(self._object_entity.get_qpos(), dtype=float)
        return qpos[:3], qpos[3:7]

    def compute_sim_state(
        self,
        mu: float | None = None,
        only_fingertips: bool = False,
    ) -> SimState:
        M_hand = self._hand_backend.get_mass_matrix()
        C_hand = self._hand_backend.get_bias_force()
        M_obj = self._object_backend.get_mass_matrix()
        C_obj = self._object_backend.get_bias_force()

        q_hand = self.get_hand_qpos()
        v_hand = np.asarray(self._hand_entity.get_dofs_velocity(), dtype=float)

        obj_qpos = np.asarray(self._object_entity.get_qpos(), dtype=float)
        x_obj = obj_qpos[:3]
        quat_obj = obj_qpos[3:7]
        v_obj = np.asarray(self._object_entity.get_dofs_velocity(), dtype=float)

        contacts = self._extract_contacts(mu=mu, only_fingertips=only_fingertips)

        return SimState(
            contacts=contacts,
            M_full=None,  # not used by QP
            C_full=None,
            M_hand=M_hand,
            C_hand=C_hand,
            M_obj=M_obj,
            C_obj=C_obj,
            q_hand=q_hand,
            v_hand=v_hand,
            x_obj=x_obj,
            quat_obj=quat_obj,
            v_obj=v_obj,
        )

    def _in_hand_geom(self, geom_idx: int) -> bool:
        return int(self._hand_entity.geom_start) <= int(geom_idx) < int(
            self._hand_entity.geom_end
        )

    def _hand_link_name_from_geom(self, hand_geom_idx: int) -> str:
        local = int(hand_geom_idx) - int(self._hand_entity.geom_start)
        geoms = self._hand_entity.geoms
        if 0 <= local < len(geoms):
            link = getattr(geoms[local], "link", None)
            return str(getattr(link, "name", ""))
        return ""

    def _extract_contacts(
        self,
        mu: float | None = None,
        only_fingertips: bool = False,
    ) -> List[Contact]:
        """Build backend-neutral Contact list from Genesis hand-object contacts."""
        raw = self._hand_backend.get_contacts(with_entity=self._object_backend)
        if raw is None or not raw:
            return []

        n = int(raw["position"].shape[0])
        contacts: List[Contact] = []
        for i in range(n):
            geom_a = int(raw["geom_a"][i])
            geom_b = int(raw["geom_b"][i])

            hand_geom = geom_a if self._in_hand_geom(geom_a) else (
                geom_b if self._in_hand_geom(geom_b) else None
            )
            if hand_geom is None:
                continue

            object_geom = geom_b if hand_geom == geom_a else geom_a
            # Genesis raw normals point from B to A (observed empirically).  The
            # QP solvers expect the contact normal to point from the hand surface
            # into the object (outward from the object center), matching the
            # MuJoCo convention used by ``extract_contacts``.
            normal = np.array(raw["normal"][i], dtype=float)
            if self._in_hand_geom(geom_a):
                # A is hand, B is object => raw normal points object -> hand
                # (outward from the object); keep it.
                sign = 1.0
            else:
                # A is object, B is hand => raw normal points hand -> object
                # (inward); flip to point outward.
                sign = -1.0
            normal = sign * normal
            normal /= np.linalg.norm(normal) + 1e-12

            hand_link_name = self._hand_link_name_from_geom(hand_geom)

            if only_fingertips and int(hand_geom) not in self._fingertip_geom_indices:
                continue

            pos = np.array(raw["position"][i], dtype=float)
            t1, t2 = _tangent_basis(normal)
            mu_eff = float(mu) if mu is not None else 1.0

            contacts.append(
                Contact(
                    id=i,
                    pos=pos,
                    normal=normal,
                    tangent1=t1,
                    tangent2=t2,
                    hand_body_id=int(hand_geom),
                    hand_body_name=hand_link_name,
                    object_body_id=int(self._object_entity.link_start),
                    mu=mu_eff,
                    force_slice=slice(3 * len(contacts), 3 * (len(contacts) + 1)),
                )
            )
        return contacts

    def hand_jacobian(self, contact: Contact) -> np.ndarray:
        link = self._hand_entity.get_link(contact.hand_body_name)
        link_pos = np.asarray(link.get_pos(), dtype=float)
        link_quat = np.asarray(link.get_quat(), dtype=float)
        R = _quat_to_rot(link_quat)
        local_point = R.T @ (contact.pos - link_pos)
        J = self._hand_backend.get_jacobian(link, local_point=local_point)
        return J[:3, :].astype(float)

    def set_initial_state(self, cfg: Any) -> None:
        hand_target = (
            np.array(cfg.controller.hand_target, dtype=float)
            if cfg.controller.hand_target is not None
            else np.zeros(self.n_hand_dof, dtype=float)
        )
        obj_pos = np.array(cfg.object.position, dtype=float)
        obj_quat = np.array(cfg.object.quaternion, dtype=float)
        obj_qpos = np.concatenate([obj_pos, obj_quat])

        # Set the desired state before any integration so that subsequent
        # dynamics queries reflect the configuration we asked for.
        self._hand_entity.set_qpos(hand_target)
        self._hand_entity.set_dofs_velocity(np.zeros(self.n_hand_dof, dtype=float))
        self._object_entity.set_qpos(obj_qpos)
        self._object_entity.set_dofs_velocity(np.zeros(6, dtype=float))

        # Genesis populates contact structures only after the first simulation step.
        # Stepping from a penetrating configuration can give the object/hand large
        # velocities, so we step once to create the contacts, then restore the
        # desired zero-velocity configuration.
        self._scene.step()
        self._hand_entity.set_qpos(hand_target)
        self._hand_entity.set_dofs_velocity(np.zeros(self.n_hand_dof, dtype=float))
        self._object_entity.set_qpos(obj_qpos)
        self._object_entity.set_dofs_velocity(np.zeros(6, dtype=float))

    def apply_control(self, tau_hand: np.ndarray) -> None:
        cfg = self._cfg
        q_target = (
            np.array(cfg.controller.hand_target, dtype=float)
            if cfg.controller.hand_target is not None
            else np.zeros(self.n_hand_dof, dtype=float)
        )
        q = self.get_hand_qpos()
        v = np.asarray(self._hand_entity.get_dofs_velocity(), dtype=float)
        tau_pd = float(cfg.controller.hand_kp) * (q_target - q) - float(
            cfg.controller.hand_kv
        ) * v
        # The MJCF defines position actuators.  Zero their proportional gain so
        # they do not fight the QP torque; keep the derivative gain for joint
        # damping.  ``control_dofs_force`` then supplies the computed QP torque.
        self._hand_entity.set_dofs_kp(np.zeros(self.n_hand_dof, dtype=float))
        self._hand_entity.control_dofs_force(np.asarray(tau_hand + tau_pd, dtype=float))

    def step(self) -> None:
        self._scene.step()
        self._step_count += 1

    @classmethod
    def from_config(cls, cfg: Any) -> "GenesisDynamicsBackend":
        """Build a Genesis scene from a DotDict config."""
        if isinstance(cfg, dict):
            cfg = DotDict(cfg)

        dt = float(cfg.sim.dt)
        if not getattr(gs, "_initialized", False):
            gs.init(backend=gs.cpu)

        scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=dt),
            rigid_options=gs.options.RigidOptions(
                gravity=(0.0, 0.0, -9.81),
                enable_collision=True,
                enable_mujoco_compatibility=True,
                friction_cone=gs.friction_cone.elliptic,
            ),
            show_viewer=bool(cfg.sim.render),
        )

        # Load hand MJCF.
        hand_path = Path(cfg.robot.hand_mjcf_path)
        if not hand_path.is_absolute():
            from dexterous_gnn_qp.core.env.loader import _PLUGIN_ROOT

            hand_path = (_PLUGIN_ROOT / hand_path).resolve()

        hand_morph = gs.morphs.MJCF(
            file=str(hand_path),
            pos=tuple(cfg.robot.base_position),
            quat=tuple(cfg.robot.base_quaternion),
            requires_jac_and_IK=True,
        )
        # Use fingertip-level friction to match the MuJoCo menagerie defaults
        # (default geom friction 0.2, fingertip geoms 0.5).
        hand_material = gs.materials.Rigid(friction=1.0)
        hand_entity = scene.add_entity(hand_morph, material=hand_material)

        # Add grasp object.
        obj_cfg = cfg.object
        obj_type = str(obj_cfg.type).lower()
        size = float(obj_cfg.size)
        mass = float(obj_cfg.mass)
        friction = float(obj_cfg.friction[0])
        if obj_type == "sphere":
            volume = 4.0 / 3.0 * np.pi * size**3
            morph = gs.morphs.Sphere(
                radius=size,
                pos=tuple(obj_cfg.position),
                quat=tuple(obj_cfg.quaternion),
            )
        elif obj_type == "box":
            volume = (2.0 * size) ** 3
            morph = gs.morphs.Box(
                size=(size, size, size),
                pos=tuple(obj_cfg.position),
                quat=tuple(obj_cfg.quaternion),
            )
        elif obj_type == "cylinder":
            volume = np.pi * size**2 * (2.0 * size)
            morph = gs.morphs.Cylinder(
                radius=size,
                height=2.0 * size,
                pos=tuple(obj_cfg.position),
                quat=tuple(obj_cfg.quaternion),
            )
        else:
            raise ValueError(f"Unsupported object type: {obj_type}")

        density = mass / volume
        obj_material = gs.materials.Rigid(rho=density, friction=friction)
        object_entity = scene.add_entity(morph, material=obj_material)

        scene.build()
        return cls(scene, hand_entity, object_entity, cfg)


def _tangent_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return two orthonormal tangent vectors for a contact normal."""
    n = np.asarray(normal).flatten()
    norm = float(np.linalg.norm(n))
    if norm < 1e-8:
        return np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])
    n = n / norm
    if abs(n[2]) < 0.9:
        tmp = np.array([0.0, 0.0, 1.0])
    else:
        tmp = np.array([1.0, 0.0, 0.0])
    t1 = np.cross(n, tmp)
    t1 /= np.linalg.norm(t1) + 1e-12
    t2 = np.cross(n, t1)
    t2 /= np.linalg.norm(t2) + 1e-12
    return t1, t2


def _quat_to_rot(quat: np.ndarray) -> np.ndarray:
    """Convert quaternion [w,x,y,z] to rotation matrix."""
    w, x, y, z = quat
    return np.array(
        [
            [1 - 2 * (y**2 + z**2), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x**2 + z**2), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x**2 + y**2)],
        ],
        dtype=float,
    )
