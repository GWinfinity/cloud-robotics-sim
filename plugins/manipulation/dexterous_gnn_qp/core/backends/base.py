"""Backend-neutral interface for the QP grasping pipeline."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Tuple

import numpy as np

from dexterous_gnn_qp.core.env.sim import Contact, SimState


class DynamicsBackend(ABC):
    """Abstract interface hiding MuJoCo/Genesis specifics from QP and examples."""

    @property
    @abstractmethod
    def dt(self) -> float:
        """Simulation/integration timestep [s]."""
        ...

    @abstractmethod
    def compute_sim_state(
        self,
        mu: float | None = None,
        only_fingertips: bool = False,
    ) -> SimState:
        """Compute all dynamic quantities required by the QP solvers."""
        ...

    @abstractmethod
    def step(self) -> None:
        """Advance the simulation by one integration step."""
        ...

    @abstractmethod
    def set_initial_state(self, cfg: Any) -> None:
        """Place the hand and object at their target initial configuration."""
        ...

    @abstractmethod
    def apply_control(self, tau_hand: np.ndarray) -> None:
        """Apply hand joint torques for the upcoming step."""
        ...

    @abstractmethod
    def hand_jacobian(self, contact: Contact) -> np.ndarray:
        """Return the translational Jacobian (3 x n_hand_dof) of a contact point."""
        ...

    @abstractmethod
    def get_object_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return current object (position, quaternion[w,x,y,z])."""
        ...

    @abstractmethod
    def get_time(self) -> float:
        """Return current simulation time [s]."""
        ...

    @property
    @abstractmethod
    def n_hand_dof(self) -> int:
        """Number of actuated hand DOFs."""
        ...

    @abstractmethod
    def get_hand_qpos(self) -> np.ndarray:
        """Return current hand joint positions."""
        ...
