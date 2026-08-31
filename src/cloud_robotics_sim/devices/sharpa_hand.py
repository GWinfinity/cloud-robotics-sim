"""Sharpa Wave dexterous hand — Genesis-backed simulated device.

Wraps the Sharpa Wave URDF (vendored under
``plugins/do_as_i_do/assets/sharpa-urdf-usd-xml-main/``) behind the MHS-style
primitive interface. Unlike the lumped devices (furnace, UTM) this one owns a
Genesis scene and steps real rigid-body dynamics.

Safety behavior:

* joint targets are validated against URDF joint limits at the primitive
  layer;
* ``force_limit_n`` implements a power-and-force-limiting style guard
  (cf. GB/T 36008-2018): when any fingertip contact force exceeds the limit
  the fingers stop advancing and ``force_limit_tripped`` latches.

Test metrics in the reference file follow T/CIE 387-2026 (DOF count,
fingertip forces, grasp/precision-manipulation capability).
"""

from __future__ import annotations

import math
import re
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from .base import ReadPrimitive, SafetyViolationError, SimDevice, WritePrimitive

FINGERS = ("thumb", "index", "middle", "ring", "pinky")
_FLEXION_SUFFIXES = ("_FE", "_PIP", "_DIP", "_IP", "_CMC")


def _default_urdf(side: str) -> Path:
    root = Path(__file__).resolve().parents[3]
    return (
        root
        / "plugins"
        / "do_as_i_do"
        / "assets"
        / "sharpa-urdf-usd-xml-main"
        / "wave_01"
        / f"{side}_sharpa_wave"
        / f"{side}_sharpa_wave.urdf"
    )


class SharpaHandDevice(SimDevice):
    """Genesis-backed Sharpa Wave dexterous hand (22 actuated DOF)."""

    device_type = "sharpa_wave_hand"
    device_class = "dexterous_hand"
    compliance = ("T/CIE 387-2026", "GB/T 36008-2018", "GB/T 43200-2023")
    natural_language_notes = (
        "五指灵巧手（Sharpa Wave），拇指 5 自由度（含对掌 CMC_AA），"
        "其余四指各 4 自由度 + 小指 CMC。指尖有橡胶指垫（elastomer），"
        "捏取薄片物体时指尖力不超过 5N。长时间高负载抓握电机会升温降额。"
        "力限制动作后需将手指收回再复位 force_limit_tripped。"
    )

    def __init__(
        self,
        device_id: str,
        *,
        side: str = "right",
        headless: bool = True,
        with_grasp_object: bool = True,
        urdf_path: Any = None,
        mounted_on: str | None = None,
        force_limit_n: float = 140.0,
        sim_dt: float = 0.005,
        max_joint_speed_rad_s: float = 3.0,
    ) -> None:
        super().__init__(device_id)
        if side not in ("right", "left"):
            raise ValueError("side must be 'right' or 'left'")
        self.side = side
        self.headless = headless
        self.with_grasp_object = with_grasp_object
        self.urdf_path = Path(urdf_path) if urdf_path else _default_urdf(side)
        if not self.urdf_path.exists():
            raise FileNotFoundError(f"Sharpa URDF not found: {self.urdf_path}")
        self._mounted_on = mounted_on
        self.force_limit_n = force_limit_n
        self.sim_dt = sim_dt
        self.max_joint_speed_rad_s = max_joint_speed_rad_s

        self.speed_scale = 1.0
        self.force_limit_tripped = False
        self.motor_temp_c = 25.0
        self._built = False
        self._command: np.ndarray | None = None
        self._targets: np.ndarray | None = None

    # -- Genesis lifecycle -----------------------------------------------------

    def build(self) -> None:
        """Initialize Genesis, build the scene, and discover joints/limits."""
        if self._built:
            return
        import genesis as gs

        if not gs._initialized:
            gs.init(backend=gs.cpu, logging_level="warning")
        self._gs = gs
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.sim_dt, substeps=2),
            show_viewer=False,
        )
        # Genesis does not resolve ROS ``package://`` prefixes; rewrite mesh
        # paths to absolute filesystem paths into a cached resolved URDF.
        text = self.urdf_path.read_text(encoding="utf-8")
        pkg_dir = self.urdf_path.parent

        def _abs(match: re.Match) -> str:
            rel = match.group(1)
            if "/" in rel:
                rel = rel.split("/", 1)[1]  # drop the package name itself
            return f'filename="{(pkg_dir / rel).resolve().as_posix()}"'

        resolved = re.sub(r'filename="package://([^"]+)"', _abs, text)
        resolved_path = (
            Path(tempfile.gettempdir()) / f"{self.side}_sharpa_wave.resolved.urdf"
        )
        resolved_path.write_text(resolved, encoding="utf-8")
        self.hand = self.scene.add_entity(
            morph=gs.morphs.URDF(file=str(resolved_path), fixed=True),
        )
        self.grasp_object = None
        if self.with_grasp_object:
            # fingertip cluster when closed is near (0.06, 0.02, 0.095) m
            self.grasp_object = self.scene.add_entity(
                morph=gs.morphs.Sphere(
                    radius=0.025, pos=(0.06, 0.02, 0.10), fixed=True
                ),
                surface=gs.surfaces.Default(color=(0.8, 0.4, 0.2, 1.0)),
            )
        self.scene.build()

        # discover actuated joints in URDF order
        self.joint_names: list[str] = []
        self.dof_indices: list[int] = []
        for joint in self.hand.joints:
            idx = joint.dofs_idx_local
            idxs = idx if isinstance(idx, list) else [idx]
            if idxs and idxs[0] is not None:
                self.joint_names.append(joint.name)
                self.dof_indices.extend(int(i) for i in idxs)
        self.n_actuated = len(self.dof_indices)

        lower, upper = self.hand.get_dofs_limit()
        lower = np.asarray(lower.cpu().numpy()).flatten()
        upper = np.asarray(upper.cpu().numpy()).flatten()
        self._lower = lower[self.dof_indices]
        self._upper = upper[self.dof_indices]

        q0 = np.asarray(self.hand.get_dofs_position().cpu().numpy()).flatten()
        self._command = q0[self.dof_indices].copy()
        self._targets = self._command.copy()

        # fixed-joint fingertip links are merged into the distal phalanx (DP)
        self._fingertip_link_idx = {
            finger: self.hand.get_link(f"{self.side}_{finger}_DP").idx
            for finger in FINGERS
        }
        self._built = True

    def _require_built(self) -> None:
        if not self._built:
            raise RuntimeError(f"{self.device_id}: call build() before use")

    def close(self) -> None:
        """Release Genesis resources (best effort)."""
        if self._built:
            try:
                self.scene.destroy()
            except Exception:  # noqa: BLE001 - teardown must not raise
                pass
            self._built = False

    def mounted_on(self) -> str | None:
        return self._mounted_on

    # -- helpers -----------------------------------------------------------------

    def _name_to_dof(self) -> dict[str, int]:
        return {name: dof for name, dof in zip(self.joint_names, self.dof_indices)}

    def _is_flexion(self, joint_name: str) -> bool:
        return joint_name.endswith(_FLEXION_SUFFIXES)

    def _grasp_posture(self, primitive: str) -> np.ndarray:
        assert self._command is not None, "build() must be called first"
        targets: np.ndarray = self._command.copy()
        for i, name in enumerate(self.joint_names):
            lo, hi = self._lower[i], self._upper[i]
            span = hi - lo
            if not self._is_flexion(name):
                continue
            if primitive == "open":
                targets[i] = lo
            elif primitive == "power_grasp":
                targets[i] = lo + 0.8 * span
            elif primitive == "pinch":
                if f"{self.side}_thumb" in name or f"{self.side}_index" in name:
                    targets[i] = lo + 0.7 * span
                else:
                    targets[i] = lo
        return targets

    def _fingertip_forces(self) -> dict[str, float]:
        forces = {finger: 0.0 for finger in FINGERS}
        if self.grasp_object is None:
            return forces
        contacts = self.hand.get_contacts(with_entity=self.grasp_object)
        if not contacts or len(np.atleast_1d(np.asarray(contacts["link_a"]))) == 0:
            return forces
        link_to_finger = {v: k for k, v in self._fingertip_link_idx.items()}
        for side_key, force_key in (("link_a", "force_a"), ("link_b", "force_b")):
            links = np.atleast_1d(np.asarray(contacts[side_key]))
            side_forces = np.asarray(contacts[force_key])
            for link_idx, force_vec in zip(links, side_forces):
                finger = link_to_finger.get(int(link_idx))
                if finger is not None:
                    forces[finger] += float(np.linalg.norm(force_vec))
        return forces

    # -- primitives -----------------------------------------------------------

    @property
    def reads(self) -> dict[str, ReadPrimitive]:
        return {
            "joint_positions": ReadPrimitive(
                "joint_positions", "rad", "各关节角度（dict: 关节名 -> rad）"
            ),
            "joint_velocities": ReadPrimitive(
                "joint_velocities", "rad/s", "各关节角速度"
            ),
            "fingertip_forces_n": ReadPrimitive(
                "fingertip_forces_n", "N", "各指尖接触力幅值（dict: 手指 -> N）"
            ),
            "contact": ReadPrimitive("contact", "bool", "是否存在抓取接触"),
            "grasp_state": ReadPrimitive(
                "grasp_state", "str", "open/touching/grasping"
            ),
            "motor_temp_c": ReadPrimitive(
                "motor_temp_c", "°C", "驱动电机温度（代理模型）"
            ),
            "force_limit_tripped": ReadPrimitive(
                "force_limit_tripped", "bool", "力限制是否已动作"
            ),
            "speed_scale": ReadPrimitive("speed_scale", "1", "速度缩放系数"),
        }

    @property
    def writes(self) -> dict[str, WritePrimitive]:
        return {
            "joint_targets": WritePrimitive(
                "joint_targets",
                "rad",
                "关节目标角（dict: 关节名 -> rad，按 URDF 限位校验）",
            ),
            "grasp_primitive": WritePrimitive(
                "grasp_primitive",
                "str",
                "抓握原语",
                choices=("open", "power_grasp", "pinch"),
            ),
            "speed_scale": WritePrimitive(
                "speed_scale",
                "1",
                "速度缩放系数",
                minimum=0.0,
                maximum=1.0,
            ),
            "force_limit_n": WritePrimitive(
                "force_limit_n",
                "N",
                "指尖力限制（GB/T 36008-2018 功率与力限制模式）",
                minimum=1.0,
                maximum=400.0,
            ),
            "reset_force_limit": WritePrimitive(
                "reset_force_limit", "bool", "复位力限制（先收回手指）"
            ),
        }

    def _read(self, name: str) -> Any:
        self._require_built()
        if name == "joint_positions" or name == "joint_velocities":
            getter = (
                self.hand.get_dofs_position
                if name == "joint_positions"
                else self.hand.get_dofs_velocity
            )
            values = np.asarray(getter().cpu().numpy()).flatten()
            return {jn: float(values[d]) for jn, d in self._name_to_dof().items()}
        if name == "fingertip_forces_n":
            return self._fingertip_forces()
        if name == "contact":
            return any(f > 1e-6 for f in self._fingertip_forces().values())
        if name == "grasp_state":
            forces = self._fingertip_forces()
            if not any(f > 1e-6 for f in forces.values()):
                return "open"
            flexed = [
                float(np.asarray(self.hand.get_dofs_position().cpu().numpy())[d])
                for n, d in self._name_to_dof().items()
                if self._is_flexion(n)
            ]
            mean_flex = float(np.mean(flexed)) if flexed else 0.0
            return "grasping" if mean_flex > 0.3 else "touching"
        return {
            "motor_temp_c": self.motor_temp_c,
            "force_limit_tripped": self.force_limit_tripped,
            "speed_scale": self.speed_scale,
        }[name]

    def _write(self, name: str, value: Any) -> None:
        self._require_built()
        if name == "joint_targets":
            if not isinstance(value, dict):
                raise SafetyViolationError(
                    f"{self.device_id}: joint_targets must be a dict of "
                    "joint name -> angle (rad)"
                )
            name_to_idx = {n: i for i, n in enumerate(self.joint_names)}
            for joint_name, target in value.items():
                i = name_to_idx.get(joint_name)
                if i is None:
                    raise SafetyViolationError(
                        f"{self.device_id}: unknown joint {joint_name!r}"
                    )
                if not (self._lower[i] <= float(target) <= self._upper[i]):
                    raise SafetyViolationError(
                        f"{self.device_id}: {joint_name} target {target} outside "
                        f"[{self._lower[i]:.3f}, {self._upper[i]:.3f}] rad "
                        "(URDF joint limit)"
                    )
                self._targets[i] = float(target)  # type: ignore[index]
        elif name == "grasp_primitive":
            self._targets = self._grasp_posture(str(value))
        elif name == "speed_scale":
            self.speed_scale = float(value)
        elif name == "force_limit_n":
            self.force_limit_n = float(value)
        elif name == "reset_force_limit":
            if bool(value):
                self.force_limit_tripped = False

    # -- physics ------------------------------------------------------------------

    def step(self, dt: float) -> None:
        self._require_built()
        if dt <= 0:
            raise ValueError("dt must be positive")
        self.time_s += dt
        assert self._command is not None and self._targets is not None

        if self.force_limit_tripped:
            self._targets = self._command.copy()  # hold position

        n_sub = max(1, math.ceil(dt / self.sim_dt))
        max_delta = self.max_joint_speed_rad_s * self.speed_scale * self.sim_dt
        for _ in range(n_sub):
            delta = np.clip(self._targets - self._command, -max_delta, max_delta)
            self._command = self._command + delta
            full = np.asarray(self.hand.get_dofs_position().cpu().numpy()).flatten()
            full[self.dof_indices] = self._command
            self.hand.control_dofs_position(full.tolist())
            self.scene.step()

        # power-and-force limiting guard (GB/T 36008-2018 PFL mode)
        forces = self._fingertip_forces()
        if not self.force_limit_tripped and any(
            f > self.force_limit_n for f in forces.values()
        ):
            self.force_limit_tripped = True

        # motor temperature proxy: Joule heating ~ velocity^2, Newton cooling
        qvel = np.asarray(self.hand.get_dofs_velocity().cpu().numpy()).flatten()
        heat = 0.02 * float(np.sum(qvel[self.dof_indices] ** 2))
        self.motor_temp_c += (heat - 0.05 * (self.motor_temp_c - 25.0)) * dt

    # -- metadata ------------------------------------------------------------------

    def safety_limits(self) -> dict[str, Any]:
        return {
            "force_limit_n": self.force_limit_n,
            "pinch_force_reference_n": 140.0,
            "max_joint_speed_rad_s": self.max_joint_speed_rad_s,
            "standards_basis": ["GB/T 36008-2018", "T/CIE 387-2026"],
        }

    def test_metrics(self) -> dict[str, Any]:
        """T/CIE 387-2026 style capability metrics (filled after build)."""
        return {
            "dof": self.n_actuated if self._built else None,
            "side": self.side,
            "fingertip_links": sorted(self._fingertip_link_idx) if self._built else [],
        }
