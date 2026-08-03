"""L5 control layer: VRBridge wires all layers into one per-tick pipeline.

``step(dt)`` is called at the simulation rate (decoupled from the 90Hz
input stream). Each tick:

    drain events -> read mailbox -> watchdog check -> filter -> semantic
    map -> clutch retarget -> safety clamp -> IK -> position control
    -> optional recording

The robot is duck-typed against ``ArticulationBackend``: it needs
``get_qpos()``, ``control_dofs_position(arr)``, ``get_link_pose(name)``
and ``inverse_kinematics(link_name, pos, quat)``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import yaml

from .filters import PoseFilter
from .mapping import SemanticMapper
from .messages import PoseMsg
from .recorder_hook import TeleopRecorder
from .retargeting import ClutchRetargeter
from .safety import SafetyConfig, SafetyState, SafetySupervisor
from .session import SessionManager
from .sync_buffer import ClockSync, EventQueue, StateMailbox
from .transport import TransportServer

logger = logging.getLogger(__name__)


@dataclass
class BridgeConfig:
    """Network / filter / mapping / safety settings for one VRBridge."""

    host: str = "0.0.0.0"
    state_port: int = 5555
    control_port: int = 5556
    mapping_path: str = ""
    recording_output: str | None = None
    filter_min_cutoff: float = 1.0
    filter_beta: float = 0.02
    safety: SafetyConfig = field(default_factory=SafetyConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "BridgeConfig":
        path = Path(path)
        with open(path, encoding="utf-8") as f:
            data: dict[str, Any] = yaml.safe_load(f)
        network = data.get("network", {}) or {}
        filt = data.get("filter", {}) or {}
        mapping_path = str(data.get("mapping", ""))
        if mapping_path and not Path(mapping_path).is_absolute():
            mapping_path = str(path.parent / mapping_path)
        recording = data.get("recording", {}) or {}
        recording_output = recording.get("output")
        if recording_output:
            recording_output = str(recording_output)
            if not Path(recording_output).is_absolute():
                recording_output = str(path.parent / recording_output)
        return cls(
            host=str(network.get("host", "0.0.0.0")),
            state_port=int(network.get("state_port", 5555)),
            control_port=int(network.get("control_port", 5556)),
            mapping_path=mapping_path,
            recording_output=recording_output,
            filter_min_cutoff=float(filt.get("min_cutoff", 1.0)),
            filter_beta=float(filt.get("beta", 0.02)),
            safety=SafetyConfig.from_dict(data.get("safety", {}) or {}),
        )


class VRBridge:
    """Main entry point of the vr_bridge plugin."""

    def __init__(self, config: BridgeConfig, robot: Any = None) -> None:
        self.config = config
        self.robot = robot

        self.mailbox = StateMailbox(ClockSync())
        self.events = EventQueue()
        self.session = SessionManager()
        self.transport = TransportServer(
            host=config.host,
            state_port=config.state_port,
            control_port=config.control_port,
            mailbox=self.mailbox,
            events=self.events,
            session=self.session,
        )

        self.mapper = SemanticMapper.from_yaml(config.mapping_path)
        self.retargeter = ClutchRetargeter()
        self.safety = SafetySupervisor(config.safety)
        self.recorder = TeleopRecorder()
        self._filters = {
            side: PoseFilter(config.filter_min_cutoff, config.filter_beta)
            for side in ("left", "right")
        }

        # Optional callback fired on the `reset_episode` semantic event.
        self.on_reset: Callable[[], None] | None = None

        self._q_target: np.ndarray | None = None
        self._last_clamped: dict[str, PoseMsg] = {}
        self.ik_failures: int = 0

    @classmethod
    def from_yaml(cls, path: str | Path, robot: Any = None) -> "VRBridge":
        return cls(BridgeConfig.from_yaml(path), robot=robot)

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------

    def attach_robot(self, robot: Any) -> None:
        self.robot = robot
        self._q_target = None

    def start(self) -> None:
        self.transport.start()

    def stop(self) -> None:
        self.transport.stop()

    # ------------------------------------------------------------------
    # control tick
    # ------------------------------------------------------------------

    def step(self, dt: float, rgb: np.ndarray | None = None) -> dict[str, Any]:
        """Run one control tick at the simulation rate.

        Args:
            dt: Tick period in seconds (speed limiting / filter timing).
            rgb: Optional (H, W, 3) uint8 camera frame forwarded to the
                recorder (required for HDF5 / dreamdojo export).
        """
        events = self.events.drain()
        semantic_events = self._handle_events(events)

        sample = self.mailbox.get()
        age_ms = sample[1] if sample is not None else float("inf")
        safety_state = self.safety.freshness(age_ms)

        info: dict[str, Any] = {
            "safety_state": safety_state.value,
            "age_ms": age_ms,
            "events": semantic_events,
            "recording": self.recorder.recording,
            "session_active": self.session.is_active(),
        }

        if self.robot is None:
            return info
        if self._q_target is None:
            self._q_target = np.asarray(self.robot.get_qpos(), dtype=np.float64)

        if sample is None or safety_state is not SafetyState.OK:
            # Freeze: hold the last commanded target (or current pose).
            self.retargeter.reset()
            for filt in self._filters.values():
                filt.reset()
            self.robot.control_dofs_position(self._q_target)
            return info

        state, _ = sample
        action = self.mapper.map(state, events)
        self._apply_action(action, dt, info)

        self.recorder.capture(action, self._q_target, rgb=rgb)
        self.robot.control_dofs_position(self._q_target)
        return info

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _handle_events(self, events: list) -> list[str]:
        semantic: list[str] = []
        for event in events:
            name = self.mapper.buttons.get(event.event)
            if name is None:
                continue
            if name == "emergency_stop":
                # Toggle on press; release events are ignored. Real OpenXR
                # clients emit a press+release pair per physical tap, so
                # release-triggered reset would make the estop self-defeating.
                if event.pressed:
                    if self.safety.estopped:
                        self.safety.release_estop()
                    else:
                        self.safety.engage_estop()
            elif name == "record_toggle" and event.pressed:
                was_recording = self.recorder.recording
                self.recorder.toggle()
                if was_recording:
                    self._auto_save_recording()
            elif name == "reset_episode" and event.pressed:
                if self.on_reset is not None:
                    self.on_reset()
            semantic.append(name)
        return semantic

    def _auto_save_recording(self) -> None:
        """Persist + clear the finished episode when recording toggles off."""
        if not self.config.recording_output:
            return
        n_frames = self.recorder.n_frames
        try:
            path = self.recorder.save_hdf5(self.config.recording_output)
        except Exception as exc:  # noqa: BLE001 - keep teleop alive
            logger.warning("recording auto-save failed: %s", exc)
            return
        if path is not None:
            logger.info("teleop episode saved: %s (%d frames)", path, n_frames)
        self.recorder.clear()

    def _apply_action(self, action, dt: float, info: dict[str, Any]) -> None:
        assert self._q_target is not None
        for name, intent in action.arm_intents.items():
            mapping = intent.mapping
            filt = self._filters.setdefault(
                mapping.source,
                PoseFilter(self.config.filter_min_cutoff, self.config.filter_beta),
            )
            ctrl_pose = filt.apply(intent.pose, dt)
            ee_pose = self._ee_pose(mapping.ee_link)
            target = self.retargeter.update(
                name,
                ctrl_pose,
                ee_pose,
                engaged=intent.engaged,
                pos_scale=mapping.pos_scale,
            )
            if target is None:
                continue
            clamped = self.safety.clamp_target(target, dt, self._last_clamped.get(name))
            self._last_clamped[name] = clamped
            try:
                q_arm = self.robot.inverse_kinematics(
                    mapping.ee_link, clamped.pos, clamped.quat
                )
            except Exception as exc:  # noqa: BLE001 - keep teleop alive
                self.ik_failures += 1
                logger.warning("IK failed for %s: %s", name, exc)
                continue
            q_arm = np.asarray(q_arm, dtype=np.float64).ravel()
            for i, dof in enumerate(mapping.dofs):
                if i < q_arm.shape[0] and dof < self._q_target.shape[0]:
                    self._q_target[dof] = q_arm[i]

        for name, close_fraction in action.gripper_cmds.items():
            gripper = next(g for g in self.mapper.grippers if g.name == name)
            value = gripper.to_joint_value(close_fraction)
            for dof in gripper.dofs:
                if dof < self._q_target.shape[0]:
                    self._q_target[dof] = value

        if action.base_velocity is not None:
            info["base_velocity"] = action.base_velocity
        if action.base_yaw is not None:
            info["base_yaw"] = action.base_yaw

    def _ee_pose(self, link_name: str) -> PoseMsg:
        pose = self.robot.get_link_pose(link_name)
        return PoseMsg(
            pos=np.asarray(pose.pos, dtype=np.float64),
            quat=np.asarray(pose.quat, dtype=np.float64),
        )
