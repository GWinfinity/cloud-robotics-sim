"""L4 semantic layer: map raw controller signals to device-agnostic intents.

The mapping table is a YAML config, so switching robots (single Franka,
dual-arm, dexterous hand) is a config change, not a code change — and the
client never needs to know anything about the simulation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .messages import ControllerState, EventMsg, HandState, PoseMsg


@dataclass
class ArmMapping:
    """One arm entry: which controller drives which EE link and dofs."""

    name: str
    source: str  # "left" | "right"
    ee_link: str
    dofs: list[int]
    pos_scale: float = 1.0


@dataclass
class GripperMapping:
    """One gripper entry: trigger-driven dofs with open/close joint values."""

    name: str
    source: str  # controller side whose trigger drives the gripper
    dofs: list[int]
    open_value: float
    close_value: float

    def to_joint_value(self, close_fraction: float) -> float:
        """0.0 = fully open, 1.0 = fully closed."""
        f = min(max(close_fraction, 0.0), 1.0)
        return self.open_value + f * (self.close_value - self.open_value)


@dataclass
class ArmIntent:
    """Per-tick intent for one arm: filtered pose + clutch engagement."""

    mapping: ArmMapping
    pose: PoseMsg
    engaged: bool  # clutch (grip) held


@dataclass
class SemanticAction:
    """Device-agnostic output of the semantic layer for one control tick."""

    arm_intents: dict[str, ArmIntent] = field(default_factory=dict)
    gripper_cmds: dict[str, float] = field(default_factory=dict)  # 0..1 closed
    base_velocity: tuple[float, float] | None = None
    base_yaw: float | None = None
    events: list[str] = field(default_factory=list)  # semantic event names


class SemanticMapper:
    """YAML-driven mapping from ``ControllerState`` to ``SemanticAction``."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.grip_threshold = float(config.get("grip_threshold", 0.5))
        self.arms = [
            ArmMapping(
                name=a["name"],
                source=a["source"],
                ee_link=a["ee_link"],
                dofs=[int(d) for d in a["dofs"]],
                pos_scale=float(a.get("pos_scale", 1.0)),
            )
            for a in config.get("arms", [])
        ]
        self.grippers = [
            GripperMapping(
                name=g["name"],
                source=g["source"],
                dofs=[int(d) for d in g["dofs"]],
                open_value=float(g.get("open_value", 0.0)),
                close_value=float(g.get("close_value", 0.0)),
            )
            for g in config.get("grippers", [])
        ]
        base = config.get("base", {}) or {}
        self._base_vel_cfg = base.get("planar_velocity")
        self._base_yaw_cfg = base.get("yaw")
        self.buttons: dict[str, str] = dict(config.get("buttons", {}))

    @classmethod
    def from_yaml(cls, path: str | Path) -> "SemanticMapper":
        with open(path, encoding="utf-8") as f:
            return cls(yaml.safe_load(f))

    # ------------------------------------------------------------------

    def _hand(self, state: ControllerState, source: str) -> HandState:
        if source == "left":
            return state.left
        if source == "right":
            return state.right
        raise ValueError(f"unknown controller source {source!r}")

    def map(
        self,
        state: ControllerState,
        events: list[EventMsg] | None = None,
    ) -> SemanticAction:
        action = SemanticAction()

        for arm in self.arms:
            hand = self._hand(state, arm.source)
            action.arm_intents[arm.name] = ArmIntent(
                mapping=arm,
                pose=hand.pose,
                engaged=hand.grip >= self.grip_threshold,
            )

        for gripper in self.grippers:
            hand = self._hand(state, gripper.source)
            action.gripper_cmds[gripper.name] = hand.trigger

        if self._base_vel_cfg:
            src = self._hand(state, self._base_vel_cfg.get("source", "left"))
            scale = float(self._base_vel_cfg.get("scale", 1.0))
            action.base_velocity = (
                src.thumbstick[0] * scale,
                src.thumbstick[1] * scale,
            )
        if self._base_yaw_cfg:
            src = self._hand(state, self._base_yaw_cfg.get("source", "right"))
            scale = float(self._base_yaw_cfg.get("scale", 1.0))
            action.base_yaw = src.thumbstick[0] * scale

        for event in events or []:
            semantic = self.buttons.get(event.event)
            if semantic is not None:
                action.events.append(semantic)

        return action
