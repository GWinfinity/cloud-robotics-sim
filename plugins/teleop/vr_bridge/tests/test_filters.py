"""One Euro filter tests: convergence, smoothing vs. latency trade-off."""

from __future__ import annotations

import numpy as np
import pytest
from vr_bridge.core.filters import OneEuroFilter, PoseFilter, QuatOneEuro, Vec3OneEuro
from vr_bridge.core.messages import PoseMsg


def test_first_sample_passes_through():
    """The very first sample is returned unfiltered (filter priming)."""
    filt = OneEuroFilter(min_cutoff=1.0, beta=0.02)
    assert filt(3.14, dt=0.01) == pytest.approx(3.14)


def test_static_input_converges():
    """A constant signal stays (numerically) on the constant value."""
    filt = OneEuroFilter(min_cutoff=1.0, beta=0.02)
    filt(1.0, dt=0.01)
    out = [filt(1.0, dt=0.01) for _ in range(200)]
    assert out[-1] == pytest.approx(1.0, abs=1e-9)


def test_noisy_static_signal_is_smoothed():
    """Around a static signal the output jitter is much smaller than input."""
    rng = np.random.default_rng(0)
    noise = rng.normal(0.0, 0.01, size=500)
    filt = OneEuroFilter(min_cutoff=1.0, beta=0.02)
    filt(0.5, dt=0.01)
    out = np.array([filt(0.5 + n, dt=0.01) for n in noise])
    assert out.std() < noise.std() * 0.5
    assert abs(out.mean() - 0.5) < 0.005


def test_high_speed_motion_has_lower_lag_than_fixed_lowpass():
    """The beta term opens the cutoff at speed, reducing tracking lag."""
    dt = 0.01
    ramp = np.linspace(0.0, 2.0, 100)  # 2 m/s, clearly "fast"

    adaptive = OneEuroFilter(min_cutoff=1.0, beta=0.5)
    fixed = OneEuroFilter(min_cutoff=1.0, beta=0.0)
    for filt in (adaptive, fixed):
        filt(ramp[0], dt)
    out_adaptive = np.array([adaptive(x, dt) for x in ramp[1:]])
    out_fixed = np.array([fixed(x, dt) for x in ramp[1:]])

    lag_adaptive = np.abs(out_adaptive - ramp[1:]).mean()
    lag_fixed = np.abs(out_fixed - ramp[1:]).mean()
    assert lag_adaptive < lag_fixed


def test_step_response_is_bounded_and_converges():
    """A unit step is tracked without overshoot and converges to 1."""
    filt = OneEuroFilter(min_cutoff=1.0, beta=0.02)
    filt(0.0, dt=0.01)
    out = np.array([filt(1.0, dt=0.01) for _ in range(500)])
    assert np.all(out >= -1e-9)
    assert np.all(out <= 1.0 + 1e-9)
    assert out[-1] == pytest.approx(1.0, abs=1e-3)


def test_zero_dt_does_not_explode():
    """dt=0 is clamped internally instead of dividing by zero."""
    filt = OneEuroFilter()
    filt(0.0, dt=0.0)
    out = filt(1.0, dt=0.0)
    assert np.isfinite(out)


def test_reset_restores_passthrough():
    """After reset the next sample passes through unfiltered again."""
    filt = OneEuroFilter()
    filt(0.0, dt=0.01)
    filt(0.1, dt=0.01)
    filt.reset()
    assert filt(5.0, dt=0.01) == pytest.approx(5.0)


def test_vec3_filter_shape_and_values():
    """Vec3OneEuro filters each axis independently."""
    filt = Vec3OneEuro(min_cutoff=1.0, beta=0.02)
    v = np.array([1.0, 2.0, 3.0])
    out = filt(v, dt=0.01)
    assert out.shape == (3,)
    assert out == pytest.approx(v)


def test_quat_filter_returns_unit_quaternion():
    """QuatOneEuro output is always a unit quaternion."""
    filt = QuatOneEuro(min_cutoff=1.0, beta=0.02)
    q = np.array([1.0, 0.0, 0.0, 0.0])
    filt(q, dt=0.01)
    for angle in np.linspace(0.0, np.pi, 20):
        sample = np.array([np.cos(angle / 2), 0.0, 0.0, np.sin(angle / 2)])
        out = filt(sample, dt=0.01)
        assert np.linalg.norm(out) == pytest.approx(1.0)


def test_quat_filter_hemisphere_continuity():
    """Sign-flipped equivalent quaternions must not cause output jumps."""
    filt = QuatOneEuro(min_cutoff=1.0, beta=0.02)
    q = np.array([np.cos(0.1), 0.0, 0.0, np.sin(0.1)])
    filt(q, dt=0.01)
    out_pos = filt(q, dt=0.01)
    out_neg = filt(-q, dt=0.01)  # same rotation, opposite hemisphere
    assert float(np.dot(out_pos, out_neg)) > 0.999


def test_pose_filter_roundtrip_and_reset():
    """PoseFilter filters pos+quat jointly; reset re-primes both."""
    filt = PoseFilter(min_cutoff=1.0, beta=0.02)
    pose = PoseMsg(
        pos=np.array([0.1, 0.2, 0.3]),
        quat=np.array([1.0, 0.0, 0.0, 0.0]),
    )
    out = filt.apply(pose, dt=0.01)
    assert out.pos == pytest.approx(pose.pos)
    assert out.quat == pytest.approx(pose.quat)
    filt.reset()
    new_pose = PoseMsg(
        pos=np.array([1.0, 1.0, 1.0]),
        quat=np.array([0.0, 1.0, 0.0, 0.0]),
    )
    out = filt.apply(new_pose, dt=0.01)
    assert out.pos == pytest.approx(new_pose.pos)
