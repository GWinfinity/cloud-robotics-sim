"""L3 signal layer: One Euro filters for poses and analog channels.

One Euro filter (Casiez et al., CHI 2012) adapts its cutoff to signal
speed: smooth at low speed, low-latency at high speed — a better fit for
hand-tracking jitter than a fixed low-pass.
"""

from __future__ import annotations

import math

import numpy as np

from .messages import PoseMsg
from .quat_utils import nlerp, qfix_sign, qnormalize


class OneEuroFilter:
    """Scalar One Euro filter."""

    def __init__(
        self,
        min_cutoff: float = 1.0,
        beta: float = 0.02,
        d_cutoff: float = 1.0,
    ) -> None:
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self._x_prev: float | None = None
        self._dx_prev: float = 0.0

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / max(dt, 1e-6))

    def __call__(self, x: float, dt: float) -> float:
        if self._x_prev is None:
            self._x_prev = x
            return x
        dx = (x - self._x_prev) / max(dt, 1e-6)
        a_d = self._alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1.0 - a_d) * self._dx_prev
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._alpha(cutoff, dt)
        x_hat = a * x + (1.0 - a) * self._x_prev
        self._x_prev, self._dx_prev = x_hat, dx_hat
        return x_hat

    def reset(self) -> None:
        self._x_prev = None
        self._dx_prev = 0.0


class Vec3OneEuro:
    """Axis-independent One Euro filtering for 3D vectors."""

    def __init__(self, **kwargs: float) -> None:
        self._axes = [OneEuroFilter(**kwargs) for _ in range(3)]

    def __call__(self, v: np.ndarray, dt: float) -> np.ndarray:
        return np.array(
            [f(float(v[i]), dt) for i, f in enumerate(self._axes)],
            dtype=np.float64,
        )

    def reset(self) -> None:
        for f in self._axes:
            f.reset()


class QuatOneEuro:
    """One Euro filtering on quaternion components with hemisphere fix."""

    def __init__(self, **kwargs: float) -> None:
        self._filters = [OneEuroFilter(**kwargs) for _ in range(4)]
        self._prev: np.ndarray | None = None

    def __call__(self, q: np.ndarray, dt: float) -> np.ndarray:
        q = qnormalize(q)
        if self._prev is None:
            self._prev = q
            return q
        q = qfix_sign(q, self._prev)
        out = np.array(
            [f(float(q[i]), dt) for i, f in enumerate(self._filters)],
            dtype=np.float64,
        )
        out = nlerp(self._prev, qnormalize(out), 1.0)
        self._prev = out
        return out

    def reset(self) -> None:
        self._prev = None
        for f in self._filters:
            f.reset()


class PoseFilter:
    """Combined position + orientation filter for one controller."""

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.02) -> None:
        self.pos_filter = Vec3OneEuro(min_cutoff=min_cutoff, beta=beta)
        self.quat_filter = QuatOneEuro(min_cutoff=min_cutoff, beta=beta)
        self.scalar_filter = OneEuroFilter(min_cutoff=min_cutoff, beta=beta)

    def apply(self, pose: PoseMsg, dt: float) -> PoseMsg:
        return PoseMsg(
            pos=self.pos_filter(pose.pos, dt),
            quat=self.quat_filter(pose.quat, dt),
        )

    def reset(self) -> None:
        self.pos_filter.reset()
        self.quat_filter.reset()
        self.scalar_filter.reset()
