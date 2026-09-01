"""Tracking parameters export for deployment.

Ported from wbc_lab/export/tracking_params_yaml.py  -writes WBC
motion-tracking policy parameters to YAML for deploy runtimes.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import yaml

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "wbc_tracking_params_v1"

# Recognized reference observation term names
REFERENCE_OBS_TERM_NAMES = (
    "ref_joint_pos",
    "ref_joint_vel",
    "ref_base_height",
    "ref_base_lin_vel_b",
    "ref_base_ang_vel_b",
    "ref_gravity_b",
    "ref_anchor_pos_w",
    "ref_anchor_ori_6d",
)


class TrackingParamsExporter:
    """Exports WBC tracking parameters for deployment.

    Generates a YAML config file containing:
      - Joint PD gains and action scales
      - Actor observation layout
      - Tracking metadata (anchor body, motion bodies)
      - RSI bin statistics (if adaptive RSI enabled)
    """

    def __init__(
        self,
        robot_config: Any,
        joint_names: list[str] | None = None,
    ):
        self.robot_config = robot_config
        self.joint_names = joint_names or []

    def build_tracking_params(
        self,
        kp: list[float] | None = None,
        kd: list[float] | None = None,
        action_scale: float = 0.25,
        obs_layout: dict[str, int] | None = None,
        anchor_body: str = "torso_link",
        motion_bodies: list[str] | None = None,
        rsi_bin_stats: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Build the full tracking params dict.

        Returns:
            Dict conforming to wbc_tracking_params_v1 schema.
        """
        params: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "control": {
                "action_scale": action_scale,
                "dt": 0.005,
            },
            "tracking": {
                "anchor_body": anchor_body,
                "motion_bodies": motion_bodies or [],
                "num_joints": len(self.joint_names),
            },
        }

        if kp is not None and kd is not None:
            params["joint_pd"] = {
                "kp": kp,
                "kd": kd,
            }

        if obs_layout is not None:
            params["observation_layout"] = obs_layout

        if rsi_bin_stats is not None:
            params["rsi"] = {
                "num_bins": len(rsi_bin_stats),
                "failure_levels": rsi_bin_stats.tolist(),
            }

        return params

    def write_yaml(
        self,
        output_path: str | Path,
        **kwargs: Any,
    ) -> Path:
        """Build params and write to YAML file.

        Args:
            output_path: Path to write YAML file.
            **kwargs: Arguments passed to build_tracking_params().

        Returns:
            Path to written file.
        """
        params = self.build_tracking_params(**kwargs)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            yaml.dump(params, f, default_flow_style=False, sort_keys=False)

        logger.info("Wrote tracking params to %s", output_path)
        return output_path
