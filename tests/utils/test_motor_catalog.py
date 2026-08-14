"""Tests for the robot motor catalog utilities."""

from __future__ import annotations

import pytest

from cloud_robotics_sim.utils.motor_catalog import (
    estimate_joule_power,
    estimate_resistance_from_motor,
    get_catalog,
    get_category,
    get_motor,
    get_thermal_defaults,
    list_categories,
    list_motors,
)


class TestMotorCatalog:
    """Basic tests for motor catalog loading and queries."""

    def test_get_catalog_structure(self):
        catalog = get_catalog()
        assert "metadata" in catalog
        assert "categories" in catalog
        assert "thermal_defaults" in catalog
        assert "humanoid_joint" in catalog["categories"]

    def test_list_categories(self):
        categories = list_categories()
        expected = {
            "humanoid_joint",
            "cobot_joint",
            "industrial_servo",
            "quadruped",
            "drone",
            "agv",
            "dexterous_hand",
        }
        assert expected.issubset(set(categories))

    def test_list_motors_all(self):
        motors = list_motors()
        assert "hip/knee_large" in motors
        assert "yaskawa_sgm7g_55apk" in motors
        assert "unitree_b2_joint" in motors

    def test_list_motors_by_category(self):
        motors = list_motors("drone")
        assert "racing_fpv" in motors
        assert "consumer_aerial" in motors

    def test_list_motors_unknown_category(self):
        with pytest.raises(ValueError):
            list_motors("not_a_category")

    def test_get_motor(self):
        motor = get_motor("hip/knee_large")
        assert motor["name"] == "hip/knee_large"
        assert "voltage_v" in motor
        assert "current_a" in motor
        assert "power_w" in motor

    def test_get_motor_unknown(self):
        with pytest.raises(ValueError):
            get_motor("not_a_motor")

    def test_get_category(self):
        category = get_category("cobot_joint")
        assert "description" in category
        assert "entries" in category
        assert len(category["entries"]) > 0

    def test_get_thermal_defaults(self):
        defaults = get_thermal_defaults()
        assert "copper" in defaults
        assert defaults["copper"]["conductivity_s_per_m"] == pytest.approx(5.8e7)

    def test_estimate_joule_power(self):
        assert estimate_joule_power(10.0, 0.1) == pytest.approx(10.0)

    def test_estimate_resistance_from_motor(self):
        resistance = estimate_resistance_from_motor("yaskawa_sgm7g_13apk")
        assert resistance > 0
