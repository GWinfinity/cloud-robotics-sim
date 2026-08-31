"""Metrics and logging for full vs hierarchical QP comparison."""
from __future__ import annotations

import csv
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


@dataclass
class StepMetrics:
    """Metrics recorded at a single control step."""

    time: float
    n_contacts: int
    n_skeleton: int
    full_solve_time_ms: float
    hier_solve_time_ms: float
    force_error_rel: float
    force_error_abs: float
    accel_error_rel: float
    accel_error_abs: float
    obj_pos_error_mm: float
    obj_rot_error_deg: float
    full_obj_accel: np.ndarray = field(repr=False)
    hier_obj_accel: np.ndarray = field(repr=False)


class BenchmarkLogger:
    """Collect step-level metrics and write them to CSV."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.rows: List[Dict[str, Any]] = []

    def log(
        self,
        sim_time: float,
        n_contacts: int,
        n_skeleton: int,
        full_res,
        hier_res,
        obj_pos_error: float,
        obj_rot_error: float,
    ) -> None:
        f_full = full_res.forces
        f_hier = hier_res.full_forces
        denom = max(1e-9, float(np.linalg.norm(f_full)))
        f_err_rel = float(np.linalg.norm(f_full - f_hier) / denom)
        f_err_abs = float(np.linalg.norm(f_full - f_hier))

        a_full = full_res.a_obj
        a_hier = hier_res.a_obj
        denom_a = max(1e-9, float(np.linalg.norm(a_full)))
        a_err_rel = float(np.linalg.norm(a_full - a_hier) / denom_a)
        a_err_abs = float(np.linalg.norm(a_full - a_hier))

        row = {
            "time": sim_time,
            "n_contacts": n_contacts,
            "n_skeleton": n_skeleton,
            "full_solve_time_ms": full_res.solve_time * 1000.0,
            "hier_solve_time_ms": hier_res.solve_time * 1000.0,
            "force_error_rel": f_err_rel,
            "force_error_abs": f_err_abs,
            "accel_error_rel": a_err_rel,
            "accel_error_abs": a_err_abs,
            "obj_pos_error_mm": obj_pos_error * 1000.0,
            "obj_rot_error_deg": np.degrees(obj_rot_error),
            # Alternating iteration diagnostics
            "hier_n_iterations": getattr(hier_res, "n_iterations", 1),
            "hier_residual_norm": getattr(hier_res, "residual_norm", 0.0),
            "hier_converged": getattr(hier_res, "converged", True),
        }
        self.rows.append(row)

    def summarize(self) -> Dict[str, float]:
        """Return median statistics over all logged steps."""
        if not self.rows:
            return {}
        keys = [k for k in self.rows[0] if k != "time"]
        return {k: float(np.median([r[k] for r in self.rows])) for k in keys}

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.rows[0].keys())
            writer.writeheader()
            writer.writerows(self.rows)


def rotation_angle_between(q1: np.ndarray, q2: np.ndarray) -> float:
    """Smallest rotation angle between two MuJoCo quaternions [w,x,y,z]."""
    dot = np.clip(np.dot(q1, q2), -1.0, 1.0)
    return 2.0 * np.arccos(abs(dot))
