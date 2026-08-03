"""Multiscale error indicator inspired by Deng Yu's domain-decomposition ideas.

This module provides heuristics to allocate computational resolution
non-uniformly across a deformable body.  Regions near contacts, large expected
deformation, or high curvature receive fine discretization; remote regions stay
coarse.  It is an engineering translation of the "local expensive computation"
principle, not a rigorous error bound.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    from cloud_robotics_sim.backend.types import DeformableConfig

ResolutionLevel = Literal[1, 2, 3]


@dataclass
class IndicatorConfig:
    """Tunable knobs for the multiscale indicator.

    Attributes:
        roi_radius: Distance from the region of interest beyond which resolution
            drops to the coarsest level.
        transition_width: Width of the smooth transition between resolution
            levels (meters).
        curvature_scale: Length scale used to normalize curvature contribution.
        contact_margin: Safety margin added around contact geometry when
            computing the distance field.
        fine_threshold: Normalized error indicator value above which level 3
            (fine) is recommended.
        medium_threshold: Normalized error indicator value above which level 2
            (medium) is recommended.
    """

    roi_radius: float = 0.15
    transition_width: float = 0.05
    curvature_scale: float = 0.05
    contact_margin: float = 0.02
    fine_threshold: float = 0.7
    medium_threshold: float = 0.3


class MultiscaleErrorIndicator:
    """Compute per-point resolution recommendations for a deformable body.

    The indicator blends three geometric cues:

    1. **Proximity to region of interest** (e.g. a gripper center).  This is the
       dominant term and directly reflects Deng Yu's idea of zooming expensive
       solvers only into non-equilibrium zones.
    2. **Local curvature / thinness**.  Thin or highly curved regions are more
       sensitive to discretization error, so they receive a resolution boost.
    3. **Contact risk** (optional).  Points closer to known colliders are marked
       for finer resolution.

    The final scalar is mapped to discrete resolution levels 1 (coarse), 2
    (medium), or 3 (fine).
    """

    def __init__(self, config: IndicatorConfig | None = None) -> None:
        self.config = config or IndicatorConfig()

    def evaluate(
        self,
        positions: np.ndarray,
        roi_center: tuple[float, float, float] = (0.0, 0.0, 0.0),
        colliders: list[tuple[float, float, float]] | None = None,
    ) -> np.ndarray:
        """Return a resolution level (1/2/3) for each input position.

        Args:
            positions: (N, 3) array of candidate points (e.g. particles or
                vertices of the deformable body).
            roi_center: Center of the high-interest region, typically a gripper
                or contact point.
            colliders: Optional list of points representing anticipated contact
                locations.

        Returns:
            (N,) integer array with values in {1, 2, 3}.
        """
        positions = np.asarray(positions, dtype=np.float64)
        if positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError("positions must be an (N, 3) array")

        roi_center_arr = np.asarray(roi_center, dtype=np.float64)
        roi_term = self._roi_term(positions, roi_center_arr)
        curvature_term = self._curvature_term(positions)
        contact_term = self._contact_term(positions, colliders)

        # Combine terms with weights that emphasize proximity.
        indicator = 0.6 * roi_term + 0.25 * curvature_term + 0.15 * contact_term
        return self._level_map(indicator)

    def _roi_term(self, positions: np.ndarray, roi_center: np.ndarray) -> np.ndarray:
        """Distance-from-ROI contribution, 0 (far) to 1 (inside ROI)."""
        cfg = self.config
        distances = np.linalg.norm(positions - roi_center, axis=1)
        # Smooth step from 1 at d=0 to 0 at d=roi_radius.
        x = (distances - (cfg.roi_radius - cfg.transition_width)) / max(
            cfg.transition_width, 1e-6
        )
        x = np.clip(x, 0.0, 1.0)
        return 1.0 - self._smoothstep(x)

    def _curvature_term(self, positions: np.ndarray) -> np.ndarray:
        """Estimate local geometric sensitivity using nearest-neighbor spread.

        For each point, a small neighborhood covariance is computed.  A high
        ratio of principal axes indicates a thin or elongated region, which
        benefits from finer resolution.
        """
        if len(positions) < 4:
            return np.zeros(len(positions))

        cfg = self.config
        k = min(8, len(positions) - 1)
        # Simple O(N^2) neighbor search; acceptable for the small point clouds
        # used in soft-body scene generation.
        deltas = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
        dists = np.linalg.norm(deltas, axis=2)
        nearest_idx = np.argsort(dists, axis=1)[:, 1 : k + 1]

        term = np.zeros(len(positions))
        for i, idx in enumerate(nearest_idx):
            neighbors = positions[idx]
            centered = neighbors - positions[i]
            cov = centered.T @ centered / max(len(neighbors), 1)
            eigvals = np.linalg.eigvalsh(cov)
            # Ratio of largest to smallest principal variance, clipped to avoid
            # division by zero and saturated for numerical stability.
            ratio = eigvals[-1] / max(eigvals[0], 1e-12)
            ratio = min(ratio, 100.0)
            # Normalize by scale; thin regions have larger ratios.
            term[i] = 1.0 - np.exp(
                -ratio * cfg.curvature_scale / max(cfg.curvature_scale, 1e-6)
            )
        return np.clip(term, 0.0, 1.0)

    def _contact_term(
        self,
        positions: np.ndarray,
        colliders: list[tuple[float, float, float]] | None,
    ) -> np.ndarray:
        """Contribution from anticipated contact points."""
        if colliders is None or len(colliders) == 0:
            return np.zeros(len(positions))

        collider_arr = np.asarray(colliders, dtype=np.float64)
        # Minimum distance from each position to any collider.
        deltas = positions[:, np.newaxis, :] - collider_arr[np.newaxis, :, :]
        distances = np.min(np.linalg.norm(deltas, axis=2), axis=1)
        effective = distances - self.config.contact_margin
        return np.asarray(
            np.clip(1.0 - effective / max(self.config.roi_radius, 1e-6), 0.0, 1.0)
        )

    def _level_map(self, indicator: np.ndarray) -> np.ndarray:
        """Map continuous indicator to discrete resolution levels."""
        levels = np.ones(len(indicator), dtype=np.int64)
        levels[indicator >= self.config.medium_threshold] = 2
        levels[indicator >= self.config.fine_threshold] = 3
        return levels

    @staticmethod
    def _smoothstep(x: np.ndarray) -> np.ndarray:
        """Cubic smoothstep: 0 at x=0, 1 at x=1."""
        x = np.clip(x, 0.0, 1.0)
        return np.asarray(x * x * (3.0 - 2.0 * x))


def build_multiscale_config(
    base_config: "DeformableConfig",
    indicator: MultiscaleErrorIndicator,
    positions: np.ndarray,
    roi_center: tuple[float, float, float] = (0.0, 0.0, 0.0),
    colliders: list[tuple[float, float, float]] | None = None,
) -> list["DeformableConfig"]:
    """Split a deformable body into zone-specific configs.

    This is the main bridge between the Deng-Yu-style indicator and the backend
    abstraction.  It is most useful when the backend supports spatially varying
    resolution (e.g. distinct FEM bodies with different ``maxvolume``).

    Returns:
        A list of DeformableConfig objects, one per distinct resolution level
        recommended by the indicator.  Each config has its ``resolution_level``
        adjusted to the indicator's recommendation for that zone.  Callers can
        use the levels to create multiple deformable primitives or to choose a
        single conservative resolution.
    """
    from cloud_robotics_sim.backend.types import DeformableConfig

    levels = indicator.evaluate(positions, roi_center=roi_center, colliders=colliders)
    configs: list[DeformableConfig] = []
    for level in sorted(set(levels.tolist())):
        zone_config = DeformableConfig(
            material=base_config.material,
            youngs_modulus=base_config.youngs_modulus,
            poisson_ratio=base_config.poisson_ratio,
            density=base_config.density,
            resolution_level=level,
            solver_iterations=base_config.solver_iterations,
            region_of_interest=base_config.region_of_interest,
            fixed=base_config.fixed,
        )
        configs.append(zone_config)
    return configs
