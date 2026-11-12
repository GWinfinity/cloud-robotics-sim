"""Unit tests for impulse-triggered household fracture."""

from __future__ import annotations

import numpy as np
import pytest

from cloud_robotics_sim.robotwin.fracture import (
    FractureTrigger,
    contact_impulse,
    fragment_velocities,
)


def _contacts(force: float) -> dict:
    return {
        "geom_a": np.array([10]),
        "geom_b": np.array([0]),
        "force_a": np.array([[0.0, 0.0, force]]),
        "force_b": np.array([[0.0, 0.0, -force]]),
    }


def test_contact_impulse_uses_entity_side_and_valid_mask() -> None:
    """Mixed A/B contacts resolve to forces acting on the requested entity."""
    contacts = {
        "geom_a": np.array([10, 0]),
        "geom_b": np.array([0, 10]),
        "force_a": np.array([[1.0, 0.0, 0.0], [9.0, 9.0, 9.0]]),
        "force_b": np.array([[0.0, 0.0, 0.0], [0.0, 3.0, 0.0]]),
        "valid_mask": np.array([True, True]),
    }
    impulse = contact_impulse(contacts, 0.1, geom_start=10, geom_end=11)
    assert impulse == pytest.approx(np.linalg.norm([1.0, 3.0, 0.0]) * 0.1)
    contacts["valid_mask"] = np.array([True, False])
    assert contact_impulse(contacts, 0.1, 10, 11) == pytest.approx(0.1)


def test_trigger_is_thresholded_and_single_shot() -> None:
    """Subthreshold contact is safe and the fracture transition fires once."""
    trigger = FractureTrigger(impulse_threshold=0.1)
    assert not trigger.update(_contacts(1.0), 0.05, 10, 11)
    assert trigger.peak_impulse == pytest.approx(0.05)
    assert trigger.update(_contacts(2.0), 0.05, 10, 11)
    assert trigger.broken
    assert trigger.fracture_step == 2
    assert not trigger.update(_contacts(100.0), 0.05, 10, 11)
    assert trigger.fracture_step == 2


def test_fragment_velocities_inherit_linear_and_angular_twist() -> None:
    """Fragment center velocities preserve v + omega x r."""
    velocities = fragment_velocities(
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 2.0],
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
    )
    assert np.allclose(velocities, [[1.0, 2.0, 0.0], [-1.0, 0.0, 0.0]])


def test_invalid_fracture_parameters() -> None:
    """Invalid timestep and threshold values fail explicitly."""
    with pytest.raises(ValueError, match="dt"):
        contact_impulse({}, 0.0, 0, 1)
    with pytest.raises(ValueError, match="threshold"):
        FractureTrigger(impulse_threshold=0.0)
