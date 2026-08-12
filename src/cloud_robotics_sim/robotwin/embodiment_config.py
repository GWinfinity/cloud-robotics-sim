"""RoboTwin embodiment ``config.yml`` parsing and control-layer helpers.

Implements the config-side mappings of the migration document:

- section 2: ``config.yml`` joint stiffness/damping are parsed and applied
  through :meth:`ArticulationBackend.set_dofs_gains`;
- section 4.1: mimic joints expanded at asset-conversion time are handled in
  the control layer by replicating master joint targets onto slave joints
  with the recorded multiplier/offset (:class:`MimicJointMapper`).

Expected (tolerant) ``config.yml`` schema, matching the real RoboTwin 2.0
embodiment configs (``assets/robotwin/embodiments/*/config.yml``)::

    urdf_path: ./urdf/robot.urdf
    joint_stiffness: 1000          # scalar (broadcast) or per-DoF list
    joint_damping: 200             # scalar (broadcast) or per-DoF list
    gripper_stiffness: 1000        # optional gripper-specific gains
    gripper_damping: 200
    move_group: [fl_link6, fr_link6]   # ee LINK names (aliases: ee_links)
    ee_joints: [fl_joint6, fr_joint6]  # ee JOINT names
    gripper_name:
      - base: fl_joint7            # gripper base joint
        mimic: [[fl_joint8, 1., 0.]]   # [[slave, multiplier, offset]]
    dual_arm: true
    planner: curobo
    homestate: [[0, ...], [0, ...]]

Mimic mappings can also be merged from ``conversion_report.json`` produced
by ``tools/convert_assets.py`` (URDF-level ``<mimic>`` expansion). Note the
real RoboTwin 2.0 embodiment URDFs carry no active ``<mimic>`` elements -
the mimic relations live in ``gripper_name`` only.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from cloud_robotics_sim.backend.base import ArticulationBackend
from cloud_robotics_sim.robotwin.assets import ensure_for_path

logger = logging.getLogger(__name__)

__all__ = ["RobotwinEmbodimentConfig", "MimicJoint", "MimicJointMapper"]

_STIFFNESS_KEYS = ("joint_stiffness", "stiffness", "stiff")
_DAMPING_KEYS = ("joint_damping", "damping")
_GRIPPER_STIFFNESS_KEYS = ("gripper_stiffness", "gripper_stiffnes")
_GRIPPER_DAMPING_KEYS = ("gripper_damping",)
_EE_LINK_KEYS = ("ee_links", "move_group")
_EE_JOINT_KEYS = ("ee_joints",)
_GRIPPER_KEYS = ("gripper_joints", "grippers")


def _first_key(data: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in data:
            return data[key]
    return None


def _as_float_list(value: Any) -> list[float]:
    """Coerce a scalar or sequence of numbers to a list of floats."""
    if value is None:
        return []
    if isinstance(value, (int, float)):
        return [float(value)]
    return [float(v) for v in value]


@dataclass
class MimicJoint:
    """Mimic mapping for one slave joint (Genesis issue #678 workaround)."""

    master: str
    multiplier: float = 1.0
    offset: float = 0.0


@dataclass
class RobotwinEmbodimentConfig:
    """Parsed embodiment configuration.

    Attributes:
        name: Embodiment identifier (directory name or explicit field).
        urdf_path: Path to the (converted) URDF, as written in the config.
        joint_stiffness: Per-DoF PD stiffness from the config.
        joint_damping: Per-DoF PD damping from the config.
        ee_links: End-effector link names (``move_group`` in real configs).
        ee_joints: End-effector joint names.
        gripper_joints: Gripper base joint names.
        gripper_stiffness/gripper_damping: Gripper-specific PD gains.
        dual_arm: Whether the embodiment has two arms.
        planner: Planner name from the config (e.g. ``curobo``).
        homestate: Per-arm home joint configurations.
        mimic_map: Slave joint name -> :class:`MimicJoint` mapping, merged
            from ``gripper_name``/``mimic_joints`` and/or the conversion
            report.
    """

    name: str = ""
    urdf_path: str = ""
    joint_stiffness: list[float] = field(default_factory=list)
    joint_damping: list[float] = field(default_factory=list)
    gripper_stiffness: list[float] = field(default_factory=list)
    gripper_damping: list[float] = field(default_factory=list)
    ee_links: list[str] = field(default_factory=list)
    ee_joints: list[str] = field(default_factory=list)
    gripper_joints: list[str] = field(default_factory=list)
    mimic_map: dict[str, MimicJoint] = field(default_factory=dict)
    dual_arm: bool = False
    planner: str = ""
    homestate: list[list[float]] = field(default_factory=list)

    @classmethod
    def from_yaml(
        cls,
        path: str | Path,
        conversion_report: str | Path | None = None,
    ) -> "RobotwinEmbodimentConfig":
        """Load a config.yml, optionally merging a conversion report.

        Args:
            path: Path to the embodiment ``config.yml``.
            conversion_report: Optional path to ``conversion_report.json``
                produced by ``tools/convert_assets.py``. Its mimic mapping
                takes precedence over inline ``mimic_joints``.
        """
        path = Path(path)
        if not path.is_file() and ensure_for_path("embodiments", path):
            logger.info("RoboTwin embodiments auto-downloaded for %s", path)
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(data, dict):
            raise TypeError(f"Embodiment config must be a mapping: {path}")

        gripper_joints, gripper_mimic = _parse_gripper_name(data.get("gripper_name"))
        if not gripper_joints:
            gripper_joints = [str(v) for v in _first_key(data, _GRIPPER_KEYS) or []]

        mimic_map = _parse_mimic_map(data.get("mimic_joints"))
        mimic_map.update(gripper_mimic)

        config = cls(
            name=str(data.get("name") or path.parent.name),
            urdf_path=str(data.get("urdf_path") or data.get("urdf") or ""),
            joint_stiffness=_as_float_list(_first_key(data, _STIFFNESS_KEYS)),
            joint_damping=_as_float_list(_first_key(data, _DAMPING_KEYS)),
            gripper_stiffness=_as_float_list(_first_key(data, _GRIPPER_STIFFNESS_KEYS)),
            gripper_damping=_as_float_list(_first_key(data, _GRIPPER_DAMPING_KEYS)),
            ee_links=[str(v) for v in _first_key(data, _EE_LINK_KEYS) or []],
            ee_joints=[str(v) for v in _first_key(data, _EE_JOINT_KEYS) or []],
            gripper_joints=gripper_joints,
            mimic_map=mimic_map,
            dual_arm=bool(data.get("dual_arm", False)),
            planner=str(data.get("planner") or ""),
            homestate=[
                [float(x) for x in row]
                for row in (data.get("homestate") or [])
                if isinstance(row, (list, tuple))
            ],
        )
        if conversion_report is not None:
            config.mimic_map.update(_mimic_map_from_report(conversion_report, config))
        return config

    def apply_pd_gains(self, robot: ArticulationBackend) -> None:
        """Push stiffness/damping onto a backend articulation (doc section 2).

        Single-element lists (scalar configs, as in real RoboTwin 2.0
        ``config.yml`` files) are broadcast to all DoFs.
        """
        if not self.joint_stiffness:
            return
        kp = _broadcast(self.joint_stiffness, robot.n_dofs)
        kv = (
            _broadcast(self.joint_damping, robot.n_dofs) if self.joint_damping else None
        )
        robot.set_dofs_gains(np.asarray(kp), np.asarray(kv) if kv else None)


def _broadcast(values: list[float], n: int) -> list[float]:
    """Broadcast a length-1 list to length ``n``; otherwise pass through."""
    if len(values) == 1 and n > 1:
        return values * n
    return values


def _parse_gripper_name(raw: Any) -> tuple[list[str], dict[str, MimicJoint]]:
    """Parse the real RoboTwin ``gripper_name`` list.

    Format: ``[{base: fl_joint7, mimic: [[fl_joint8, 1., 0.]], ...}, ...]``.
    Returns ``(gripper base joint names, mimic map)``.
    """
    bases: list[str] = []
    mapping: dict[str, MimicJoint] = {}
    if not isinstance(raw, list):
        return bases, mapping
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        base = entry.get("base")
        if base:
            bases.append(str(base))
        for mimic in entry.get("mimic") or []:
            if isinstance(mimic, (list, tuple)) and len(mimic) >= 3:
                mapping[str(mimic[0])] = MimicJoint(
                    master=str(base or ""),
                    multiplier=float(mimic[1]),
                    offset=float(mimic[2]),
                )
    return bases, mapping


def _parse_mimic_map(raw: Any) -> dict[str, MimicJoint]:
    mapping: dict[str, MimicJoint] = {}
    if not isinstance(raw, dict):
        return mapping
    for slave, spec in raw.items():
        if isinstance(spec, dict):
            mapping[str(slave)] = MimicJoint(
                master=str(spec.get("master") or spec.get("joint") or ""),
                multiplier=float(spec.get("multiplier", 1.0)),
                offset=float(spec.get("offset", 0.0)),
            )
    return mapping


def _mimic_map_from_report(
    report_path: str | Path,
    config: RobotwinEmbodimentConfig,
) -> dict[str, MimicJoint]:
    """Extract the mimic mapping for this embodiment from a report."""
    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    urdf_name = Path(config.urdf_path).name
    for asset in report.get("assets", []):
        asset_urdf = asset.get("urdf", "")
        if urdf_name and Path(asset_urdf).name != urdf_name:
            continue
        return {
            str(slave): MimicJoint(
                master=str(spec.get("master", "")),
                multiplier=float(spec.get("multiplier", 1.0)),
                offset=float(spec.get("offset", 0.0)),
            )
            for slave, spec in (asset.get("mimic_map") or {}).items()
        }
    return {}


class MimicJointMapper:
    """Replicates master joint targets onto expanded mimic (slave) joints.

    After asset conversion, former mimic joints are plain actuated joints.
    Before issuing ``control_dofs_position``, slave targets must be computed
    from the master target as ``master * multiplier + offset`` (doc 4.1).
    """

    def __init__(
        self,
        mimic_map: dict[str, MimicJoint],
        joint_names: list[str],
    ) -> None:
        if not mimic_map:
            self._entries: list[tuple[int, int, float, float]] = []
            return
        index = {name: i for i, name in enumerate(joint_names)}
        entries: list[tuple[int, int, float, float]] = []
        for slave, spec in mimic_map.items():
            if slave not in index:
                raise KeyError(f"Mimic slave joint '{slave}' not in joint_names")
            if spec.master not in index:
                raise KeyError(f"Mimic master joint '{spec.master}' not in joint_names")
            entries.append(
                (index[slave], index[spec.master], spec.multiplier, spec.offset)
            )
        self._entries = entries

    @property
    def n_slaves(self) -> int:
        """Number of mapped slave joints."""
        return len(self._entries)

    def expand(self, targets: np.ndarray) -> np.ndarray:
        """Return a copy of ``targets`` with slave DoFs set from masters.

        Args:
            targets: Full joint-target vector ``(..., n_joints)``. The last
                dimension is indexed by joint order.

        Returns:
            New array with slave entries overwritten.
        """
        out = np.array(targets, dtype=np.float64, copy=True)
        for slave_idx, master_idx, multiplier, offset in self._entries:
            out[..., slave_idx] = out[..., master_idx] * multiplier + offset
        return out

    def apply(
        self,
        robot: ArticulationBackend,
        targets: np.ndarray,
        **kwargs: Any,
    ) -> None:
        """Expand mimic targets and issue ``control_dofs_position``."""
        robot.control_dofs_position(self.expand(targets), **kwargs)
