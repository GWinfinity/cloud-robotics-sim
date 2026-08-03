"""Safety supervisor: workspace limits, speed limits, watchdog, estop.

This layer is what separates a teleoperation product from a demo. All
limits are independent of the mapping layer, so a misconfigured mapping
table can never command the robot outside its safe envelope.
"""

from __future__ import annotations

import enum
import math
from dataclasses import dataclass, field

import numpy as np

from .messages import PoseMsg


class SafetyState(enum.Enum):
    """Result of the freshness watchdog (plus the latched estop)."""

    OK = "ok"
    STALE = "stale"  # data older than freeze_timeout: freeze targets
    DISCONNECTED = "disconnected"  # data older than disconnect_timeout
    ESTOP = "estop"  # latched emergency stop


@dataclass
class SafetyConfig:
    """Workspace/speed limits and watchdog timeouts."""

    workspace_center: tuple[float, float, float] = (0.0, 0.0, 0.0)
    workspace_radius: float = math.inf  # metres
    max_ee_speed: float = math.inf  # metres per second
    freeze_timeout_ms: float = 100.0
    disconnect_timeout_ms: float = 1000.0

    @classmethod
    def from_dict(cls, data: dict) -> "SafetyConfig":
        return cls(
            workspace_center=tuple(data.get("workspace_center", (0.0, 0.0, 0.0))),
            workspace_radius=float(data.get("workspace_radius", math.inf)),
            max_ee_speed=float(data.get("max_ee_speed", math.inf)),
            freeze_timeout_ms=float(data.get("freeze_timeout_ms", 100.0)),
            disconnect_timeout_ms=float(data.get("disconnect_timeout_ms", 1000.0)),
        )


@dataclass
class SafetySupervisor:
    """Enforces the safe envelope on every commanded EE target."""

    config: SafetyConfig = field(default_factory=SafetyConfig)
    _estop_latched: bool = False

    # ------------------------------------------------------------------
    # emergency stop (latched)
    # ------------------------------------------------------------------

    def engage_estop(self) -> None:
        self._estop_latched = True

    def release_estop(self) -> None:
        self._estop_latched = False

    @property
    def estopped(self) -> bool:
        return self._estop_latched

    # ------------------------------------------------------------------
    # data freshness watchdog
    # ------------------------------------------------------------------

    def freshness(self, age_ms: float) -> SafetyState:
        if self._estop_latched:
            return SafetyState.ESTOP
        if age_ms > self.config.disconnect_timeout_ms:
            return SafetyState.DISCONNECTED
        if age_ms > self.config.freeze_timeout_ms:
            return SafetyState.STALE
        return SafetyState.OK

    # ------------------------------------------------------------------
    # target limiting
    # ------------------------------------------------------------------

    def clamp_target(
        self, target: PoseMsg, dt: float, reference: PoseMsg | None
    ) -> PoseMsg:
        """Clamp a target pose into the safe envelope.

        - Positions are projected back onto the workspace sphere (not
          dropped — dropping would feel like the controls "stick").
        - Per-tick translation is limited to ``max_ee_speed * dt`` relative
          to ``reference`` (the previously commanded target) to avoid IK
          jumps on network bursts.
        """
        pos = np.asarray(target.pos, dtype=np.float64).copy()

        centre = np.asarray(self.config.workspace_center, dtype=np.float64)
        radius = self.config.workspace_radius
        offset = pos - centre
        dist = float(np.linalg.norm(offset))
        if dist > radius and dist > 0.0:
            pos = centre + offset * (radius / dist)

        if reference is not None and math.isfinite(self.config.max_ee_speed):
            step = pos - np.asarray(reference.pos, dtype=np.float64)
            step_len = float(np.linalg.norm(step))
            max_step = self.config.max_ee_speed * max(dt, 1e-6)
            if step_len > max_step and step_len > 0.0:
                pos = np.asarray(reference.pos, dtype=np.float64) + step * (
                    max_step / step_len
                )

        return PoseMsg(pos=pos, quat=np.asarray(target.quat, dtype=np.float64))
