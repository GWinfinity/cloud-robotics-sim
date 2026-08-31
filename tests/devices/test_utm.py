"""Tests for the electronic universal testing machine device."""

from __future__ import annotations

import pytest

from cloud_robotics_sim.devices import (
    SafetyViolationError,
    SensorDrift,
    UniversalTestingMachine,
    create_device,
)


def make_utm(**kwargs) -> UniversalTestingMachine:
    """UTM with a default Q235 specimen loaded."""
    dev = UniversalTestingMachine("utm_test", **kwargs)
    dev.write("load_specimen", True)
    return dev


def run_until(dev: UniversalTestingMachine, cond, max_steps: int = 500_000) -> bool:
    """Step the machine until ``cond`` holds; return False on timeout."""
    for _ in range(max_steps):
        dev.step(0.1)
        if cond():
            return True
    return False


def test_create_device_registry() -> None:
    """Utm is registered under its device type."""
    dev = create_device("electronic_utm", "u1")
    assert isinstance(dev, UniversalTestingMachine)


def test_start_without_specimen_rejected() -> None:
    """Start without specimen rejected."""
    dev = UniversalTestingMachine("utm_empty")
    with pytest.raises(SafetyViolationError):
        dev.write("start", True)


def test_rate_and_mode_validation() -> None:
    """Rate and mode validation."""
    dev = make_utm()
    with pytest.raises(SafetyViolationError):
        dev.write("loading_rate_mm_min", dev.max_speed_mm_min + 1.0)
    with pytest.raises(SafetyViolationError):
        dev.write("control_mode", "telekinesis")


def test_elastic_slope_matches_specimen_modulus() -> None:
    """Elastic slope matches specimen modulus."""
    dev = make_utm()
    dev.write("control_mode", "displacement")
    dev.write("loading_rate_mm_min", 30.0)
    dev.write("start", True)
    assert run_until(dev, lambda: dev.read("force_n") > 5000.0)
    dev.write("stop", True)
    # stiffness = 1 / (L0/(E*A0) + 1/C_frame)
    spec = dev.specimen
    k_spec = spec.elastic_modulus_gpa * 1000.0 * spec.area_mm2 / spec.gauge_length_mm
    k_expected = 1.0 / (1.0 / k_spec + 1.0 / dev.frame_stiffness_n_mm)
    k_meas = dev.read("force_n") / dev.read("crosshead_disp_mm")
    assert k_meas == pytest.approx(k_expected, rel=0.02)


def test_yield_and_tensile_strength_match_curve() -> None:
    """Yield and tensile strength match curve."""
    dev = make_utm()
    dev.write("control_mode", "displacement")
    dev.write("loading_rate_mm_min", 60.0)
    dev.write("start", True)
    spec = dev.specimen
    f_max_expected = spec.tensile_strength_mpa * spec.area_mm2
    peak = 0.0
    for _ in range(500_000):
        dev.step(0.1)
        peak = max(peak, dev.read("force_n"))
        if dev.state == "fractured":
            break
    assert dev.state == "fractured"
    assert peak == pytest.approx(f_max_expected, rel=0.05)


def test_fracture_auto_stop_and_extensometer_damage() -> None:
    """Fracture auto stop and extensometer damage."""
    dev = make_utm()
    dev.write("attach_extensometer", True)
    dev.write("control_mode", "displacement")
    dev.write("loading_rate_mm_min", 120.0)
    dev.write("start", True)
    assert run_until(dev, lambda: dev.state == "fractured")
    assert dev.read("force_n") == 0.0
    # extensometer left on through fracture is damaged (GB/T 228.1 practice)
    assert dev.read("extensometer_damaged") is True
    with pytest.raises(SafetyViolationError):
        dev.write("attach_extensometer", True)


def test_extensometer_none_when_detached() -> None:
    """Extensometer none when detached."""
    dev = make_utm()
    assert dev.read("extensometer_strain") is None
    dev.write("attach_extensometer", True)
    assert dev.read("extensometer_strain") == pytest.approx(0.0)


def test_overload_protection_trips_and_latches() -> None:
    """Overload protection trips and latches."""
    # FS far below the yield force -> overload before yield
    dev = make_utm(max_force_n=10_000.0)
    dev.write("control_mode", "displacement")
    dev.write("loading_rate_mm_min", 60.0)
    dev.write("start", True)
    assert run_until(dev, lambda: dev.read("overload_tripped"))
    assert dev.state == "overload"
    with pytest.raises(SafetyViolationError):
        dev.write("start", True)
    dev.write("reset_overload", True)
    assert dev.read("overload_tripped") is False


def test_force_control_runaway_past_yield() -> None:
    """Force control runaway past yield."""
    dev = make_utm()
    dev.write("control_mode", "force")
    dev.write("force_rate_n_s", 5000.0)
    dev.write("start", True)
    hit_max_speed = False
    for _ in range(500_000):
        dev.step(0.05)
        if dev.read("crosshead_speed_mm_min") >= dev.max_speed_mm_min - 1e-6:
            hit_max_speed = True
        if dev.state == "fractured":
            break
    assert hit_max_speed  # force saturates at yield -> integrator windup
    assert dev.state == "fractured"


def test_load_cell_drift_fault() -> None:
    """Load cell drift fault."""
    dev = make_utm()
    dev.write("control_mode", "displacement")
    dev.write("start", True)
    assert run_until(dev, lambda: dev.read("force_n") > 3000.0)
    dev.inject_fault(SensorDrift(name="cell_drift", channel="force_n", offset=500.0))
    assert dev.read("force_n") == pytest.approx(dev.force_n + 500.0)


def test_reference_file_contents() -> None:
    """Reference file contents."""
    dev = make_utm()
    ref = dev.reference_file()
    assert ref["device_class"] == "utm"
    names = {c["name"] for c in ref["compliance"]}
    assert any("电子式万能试验机" in n for n in names)
    assert any("拉伸试验" in n for n in names)
    assert ref["safety_limits"]["force_accuracy_class"] == 1.0
    assert ref["safety_limits"]["max_force_n"] == dev.max_force_n
