"""Clutch-based incremental retargeting: controller deltas -> EE targets.

VR controllers and robot end-effectors live in different spaces, so we do
not map absolute poses. The standard solution is a clutch: while the grip
is held, controller deltas (scaled) become EE deltas around the poses
anchored at the moment the clutch engaged. Releasing the grip freezes the
EE target and re-anchors on the next engagement.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .messages import PoseMsg
from .quat_utils import qfix_sign, qinv, qmul, qnormalize


@dataclass
class _ArmClutchState:
    engaged: bool = False
    ctrl_anchor: PoseMsg | None = None
    ee_anchor: PoseMsg | None = None
    last_target: PoseMsg | None = None


@dataclass
class ClutchRetargeter:
    """Stateful per-arm clutch retargeter (one instance per robot)."""

    _arms: dict[str, _ArmClutchState] = field(default_factory=dict)

    def update(
        self,
        arm: str,
        ctrl_pose: PoseMsg,
        ee_pose: PoseMsg,
        engaged: bool,
        pos_scale: float = 1.0,
    ) -> PoseMsg | None:
        """Return the new EE target pose, or None when the arm is disengaged.

        Args:
            arm: Arm identifier (one clutch state per arm).
            ctrl_pose: Current (filtered) controller pose, world frame.
            ee_pose: Current robot end-effector pose, world frame.
            engaged: Whether the clutch (grip) is currently held.
            pos_scale: Metres of EE motion per metre of controller motion.
        """
        state = self._arms.setdefault(arm, _ArmClutchState())

        if not engaged:
            state.engaged = False
            state.ctrl_anchor = None
            state.ee_anchor = None
            return None

        if not state.engaged:
            # (Re-)engage: anchor both poses; hold the current EE pose.
            state.engaged = True
            state.ctrl_anchor = PoseMsg(ctrl_pose.pos.copy(), ctrl_pose.quat.copy())
            state.ee_anchor = PoseMsg(ee_pose.pos.copy(), ee_pose.quat.copy())
            state.last_target = state.ee_anchor
            return state.last_target

        assert state.ctrl_anchor is not None and state.ee_anchor is not None
        ctrl_q = qfix_sign(ctrl_pose.quat, state.ctrl_anchor.quat)

        d_pos = (ctrl_pose.pos - state.ctrl_anchor.pos) * pos_scale
        d_quat = qmul(ctrl_q, qinv(state.ctrl_anchor.quat))

        target = PoseMsg(
            pos=state.ee_anchor.pos + d_pos,
            quat=qnormalize(qmul(d_quat, state.ee_anchor.quat)),
        )
        state.last_target = target
        return target

    def disengage(self, arm: str) -> None:
        """Explicitly release a clutch (e.g. on estop / session loss)."""
        self._arms.setdefault(arm, _ArmClutchState()).engaged = False

    def is_engaged(self, arm: str) -> bool:
        return self._arms.get(arm, _ArmClutchState()).engaged

    def last_target(self, arm: str) -> PoseMsg | None:
        return self._arms.get(arm, _ArmClutchState()).last_target

    def reset(self) -> None:
        self._arms.clear()
