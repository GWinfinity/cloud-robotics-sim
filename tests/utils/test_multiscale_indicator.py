"""Unit tests for the multiscale error indicator."""

from __future__ import annotations

import numpy as np
import pytest

from cloud_robotics_sim.utils.multiscale_indicator import (
    IndicatorConfig,
    MultiscaleErrorIndicator,
)


@pytest.fixture
def indicator() -> MultiscaleErrorIndicator:
    """Return a multiscale error indicator with tight test thresholds."""
    return MultiscaleErrorIndicator(
        IndicatorConfig(
            roi_radius=0.1,
            transition_width=0.02,
            fine_threshold=0.55,
            medium_threshold=0.25,
        )
    )


def test_center_points_get_fine_resolution(
    indicator: MultiscaleErrorIndicator,
) -> None:
    """Points at the ROI center should receive the finest resolution."""
    positions = np.array([[0.0, 0.0, 0.0], [0.01, 0.0, 0.0]])
    levels = indicator.evaluate(positions, roi_center=(0.0, 0.0, 0.0))
    assert np.all(levels == 3)


def test_far_points_get_coarse_resolution(
    indicator: MultiscaleErrorIndicator,
) -> None:
    """Points far from the ROI should receive the coarsest resolution."""
    positions = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    levels = indicator.evaluate(positions, roi_center=(0.0, 0.0, 0.0))
    assert np.all(levels == 1)


def test_invalid_positions_shape(indicator: MultiscaleErrorIndicator) -> None:
    """A non-(N, 3) positions array should raise ValueError."""
    with pytest.raises(ValueError):
        indicator.evaluate(np.array([0.0, 0.0, 0.0]))  # type: ignore[arg-type]


def test_contact_term_boosts_resolution(
    indicator: MultiscaleErrorIndicator,
) -> None:
    """Adding a collider at a point should not reduce its resolution level."""
    positions = np.array([[0.08, 0.0, 0.0]])
    levels_without_contact = indicator.evaluate(positions, roi_center=(0.0, 0.0, 0.0))
    levels_with_contact = indicator.evaluate(
        positions,
        roi_center=(0.0, 0.0, 0.0),
        colliders=[(0.08, 0.0, 0.0)],
    )
    assert levels_with_contact[0] >= levels_without_contact[0]


def test_smoothstep_bounds() -> None:
    """Smoothstep should map (-inf, 0) -> 0, (1, inf) -> 1, and 0.5 -> ~0.5."""
    x = np.array([-0.1, 0.0, 0.5, 1.0, 1.1])
    y = MultiscaleErrorIndicator._smoothstep(x)
    assert y[0] == pytest.approx(0.0)
    assert y[1] == pytest.approx(0.0)
    assert y[2] == pytest.approx(0.5, abs=0.1)
    assert y[3] == pytest.approx(1.0)
    assert y[4] == pytest.approx(1.0)


def test_curvature_term_handles_small_clouds() -> None:
    """The curvature term should not crash on single-point clouds."""
    indicator = MultiscaleErrorIndicator(
        IndicatorConfig(
            curvature_scale=0.05,
            fine_threshold=0.55,
            medium_threshold=0.25,
        )
    )
    positions = np.array([[0.0, 0.0, 0.0]])
    levels = indicator.evaluate(positions, roi_center=(0.0, 0.0, 0.0))
    assert levels[0] == 3  # Only the ROI term contributes.
