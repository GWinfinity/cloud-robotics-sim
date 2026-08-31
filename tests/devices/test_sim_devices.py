"""Tests for the MHS-style device abstraction layer and the muffle furnace."""

from __future__ import annotations

import pytest
import yaml

from cloud_robotics_sim.devices import (
    InterlockBypass,
    MuffleFurnace,
    RelayStuckClosed,
    SafetyViolationError,
    SensorDrift,
    SensorStuck,
    UnknownPrimitiveError,
    create_device,
    load_standards_index,
)


def fast_furnace(device_id: str = "furnace_test") -> MuffleFurnace:
    """Small, fast furnace instance so tests converge in seconds."""
    return MuffleFurnace(
        device_id,
        thermal_mass_j_k=2_000.0,
        heater_power_w=3_000.0,
        wall_loss_w_k=1.0,
        door_loss_w_k=20.0,
        max_setpoint_c=1200.0,
        overtemp_cutout_c=1250.0,
    )


def run_until(device: MuffleFurnace, cond, max_steps: int = 200_000) -> bool:
    """Step the device until ``cond`` holds; return False on timeout."""
    for _ in range(max_steps):
        device.step(1.0)
        if cond():
            return True
    return False


# --------------------------------------------------------------------------
# Primitive layer
# --------------------------------------------------------------------------


def test_unknown_primitives_raise() -> None:
    """Unknown primitives raise."""
    dev = fast_furnace()
    with pytest.raises(UnknownPrimitiveError):
        dev.read("nope")
    with pytest.raises(UnknownPrimitiveError):
        dev.write("nope", 1)


def test_setpoint_above_max_rejected() -> None:
    """Setpoint above max rejected."""
    dev = fast_furnace()
    with pytest.raises(SafetyViolationError):
        dev.write("setpoint_c", dev.max_setpoint_c + 1.0)


def test_ramp_rate_bounds_rejected() -> None:
    """Ramp rate bounds rejected."""
    dev = fast_furnace()
    with pytest.raises(SafetyViolationError):
        dev.write("ramp_rate_c_min", 0.0)


def test_create_device_registry() -> None:
    """Create device registry."""
    dev = create_device("muffle_furnace", "f1")
    assert isinstance(dev, MuffleFurnace)
    with pytest.raises(KeyError):
        create_device("warp_drive", "x")


# --------------------------------------------------------------------------
# Thermal behavior
# --------------------------------------------------------------------------


def test_heats_and_settles_near_setpoint() -> None:
    """Heats and settles near setpoint."""
    dev = fast_furnace()
    dev.write("setpoint_c", 400.0)
    dev.write("start", True)
    assert run_until(dev, lambda: abs(dev.read("chamber_temp_c") - 400.0) < 2.0)
    # settles inside the hysteresis band around the program target
    for _ in range(600):
        dev.step(1.0)
    assert abs(dev.read("chamber_temp_c") - 400.0) < 5.0


def test_idle_furnace_cools_to_ambient() -> None:
    """Idle furnace cools to ambient."""
    dev = fast_furnace()
    dev.temp_c = 300.0
    for _ in range(5_000):
        dev.step(1.0)
    assert dev.read("chamber_temp_c") < 300.0


def test_door_interlock_cuts_heater() -> None:
    """Door interlock cuts heater."""
    dev = fast_furnace()
    dev.write("setpoint_c", 500.0)
    dev.write("start", True)
    assert run_until(dev, lambda: dev.read("heater_on"))
    dev.write("door_open", True)
    assert dev.read("heater_on") is False
    temp_before = dev.read("chamber_temp_c")
    for _ in range(300):
        dev.step(1.0)
    assert dev.read("heater_on") is False  # stays cut while door open
    assert dev.read("chamber_temp_c") < temp_before


def test_door_interlock_bypass_fault() -> None:
    """Door interlock bypass fault."""
    dev = fast_furnace()
    dev.write("setpoint_c", 500.0)
    dev.write("start", True)
    dev.write("door_open", True)
    dev.inject_fault(InterlockBypass(name="bypass", interlock_name="door"))
    assert run_until(dev, lambda: dev.read("heater_on"), max_steps=10)


def test_overtemp_cutout_trips_and_latches() -> None:
    """Overtemp cutout trips and latches."""
    dev = fast_furnace()
    dev.write("setpoint_c", 1200.0)
    dev.write("start", True)
    dev.inject_fault(RelayStuckClosed(name="welded_relay"))
    assert run_until(dev, lambda: dev.read("overtemp_tripped"))
    assert dev.read("heater_on") is False
    # cannot restart while tripped
    with pytest.raises(SafetyViolationError):
        dev.write("start", True)
    # reset clears the latch
    dev.write("reset_overtemp", True)
    assert dev.read("overtemp_tripped") is False


# --------------------------------------------------------------------------
# Sensor faults
# --------------------------------------------------------------------------


def test_sensor_drift_distorts_read_only() -> None:
    """Sensor drift distorts read only."""
    dev = fast_furnace()
    dev.temp_c = 500.0
    dev.inject_fault(SensorDrift(name="drift", channel="chamber_temp_c", offset=-80.0))
    assert dev.read("chamber_temp_c") == pytest.approx(420.0)
    assert dev.temp_c == pytest.approx(500.0)  # true state untouched


def test_sensor_stuck_freezes_channel() -> None:
    """Sensor stuck freezes channel."""
    dev = fast_furnace()
    dev.write("setpoint_c", 300.0)
    dev.write("start", True)
    dev.inject_fault(SensorStuck(name="stuck_tc", channel="chamber_temp_c"))
    frozen = dev.read("chamber_temp_c")
    for _ in range(300):
        dev.step(1.0)
    assert dev.read("chamber_temp_c") == frozen
    assert dev.temp_c > frozen


# --------------------------------------------------------------------------
# Reference file
# --------------------------------------------------------------------------


def test_standards_index_loads_both_catalogs() -> None:
    """Standards index loads both catalogs."""
    index = load_standards_index()
    assert "GB/T 36008-2018" in index  # robot catalog
    assert "GB 5959.4-2008" in index  # lab equipment catalog


def test_reference_file_contents() -> None:
    """Reference file contents."""
    dev = fast_furnace("muffle_lab_01")
    ref = dev.reference_file()
    assert ref["device_id"] == "muffle_lab_01"
    assert ref["device_type"] == "muffle_furnace"
    read_names = {r["name"] for r in ref["readable"]}
    assert "chamber_temp_c" in read_names
    sp = next(w for w in ref["writable"] if w["name"] == "setpoint_c")
    assert sp["maximum"] == dev.max_setpoint_c
    comp_ids = {c["id"] for c in ref["compliance"]}
    assert comp_ids == set(MuffleFurnace.compliance)
    names = {c["name"] for c in ref["compliance"]}
    assert any("电热装置的安全" in n for n in names)
    assert ref["safety_limits"]["door_interlock"] is True
    assert ref["natural_language_notes"]


def test_reference_file_yaml_roundtrip() -> None:
    """Reference file yaml roundtrip."""
    dev = fast_furnace("muffle_lab_01")
    parsed = yaml.safe_load(dev.reference_file_yaml())
    assert parsed["device_id"] == "muffle_lab_01"
