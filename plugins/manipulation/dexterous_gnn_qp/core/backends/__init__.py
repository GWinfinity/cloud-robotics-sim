"""Dynamics backend adapters for MuJoCo and Genesis."""

from __future__ import annotations

from typing import Any

from dexterous_gnn_qp.core.backends.base import DynamicsBackend
from dexterous_gnn_qp.core.backends.mujoco_backend import MujocoDynamicsBackend

__all__ = ["DynamicsBackend", "MujocoDynamicsBackend", "create_backend"]


def create_backend(cfg: Any) -> DynamicsBackend:
    """Build a DynamicsBackend from the ``sim.backend`` config key."""
    name = str(cfg.sim.get("backend", "mujoco")).lower()
    if name == "mujoco":
        return MujocoDynamicsBackend.from_config(cfg)
    if name == "genesis":
        from dexterous_gnn_qp.core.backends.genesis_backend import (
            GenesisDynamicsBackend,
        )

        return GenesisDynamicsBackend.from_config(cfg)
    raise ValueError(f"Unknown backend: {name}")
