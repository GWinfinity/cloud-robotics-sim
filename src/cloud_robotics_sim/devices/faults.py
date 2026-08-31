"""Fault-injection models for simulated devices.

Faults are attached to a :class:`~cloud_robotics_sim.devices.base.SimDevice`
via ``inject_fault`` and serve two purposes:

* distort read channels (e.g. sensor drift), so agents must detect
  discrepancies between reported and true state;
* perturb device internals during :meth:`SimDevice.step` (e.g. a stuck
  heater relay), so agents must recognize physical failures from data.

They are the building blocks of the safety-evaluation scenarios catalogued in
``data/failures/``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Fault:
    """Base fault model.

    Attributes:
        name: Human-readable fault identifier.
        active: Whether the fault currently affects the device.
    """

    name: str
    active: bool = True

    def distort_read(self, device: Any, channel: str, value: Any) -> Any:
        """Return the (possibly distorted) value for a read channel."""
        return value

    def on_step(self, device: Any, dt: float) -> None:
        """Hook invoked each physics step, after control logic."""


@dataclass
class SensorDrift(Fault):
    """Linear drift on one read channel: ``reported = scale * true + offset``."""

    channel: str = ""
    offset: float = 0.0
    scale: float = 1.0

    def distort_read(self, device: Any, channel: str, value: Any) -> Any:
        if self.active and channel == self.channel and isinstance(value, (int, float)):
            return self.scale * value + self.offset
        return value


@dataclass
class SensorStuck(Fault):
    """Freeze one read channel at the value seen when the fault activates."""

    channel: str = ""
    _frozen: Any = field(default=None, repr=False)
    _captured: bool = field(default=False, repr=False)

    def distort_read(self, device: Any, channel: str, value: Any) -> Any:
        if not self.active or channel != self.channel:
            return value
        if not self._captured:
            self._frozen = value
            self._captured = True
        return self._frozen


@dataclass
class RelayStuckClosed(Fault):
    """Force a boolean actuator flag on the device to stay closed (on).

    Classic furnace scenario: the heater control relay welds shut, so the
    controller's commands no longer remove power. Independent safety hardware
    (e.g. an over-temperature cutout in series) can still interrupt power —
    whether it does is up to the device's step logic.
    """

    flag_attr: str = "heater_on"

    def on_step(self, device: Any, dt: float) -> None:
        if self.active and hasattr(device, self.flag_attr):
            setattr(device, self.flag_attr, True)


@dataclass
class InterlockBypass(Fault):
    """Bypass a safety interlock on the device.

    The device checks :attr:`interlock_name` against its active faults where
    it would normally enforce the interlock.
    """

    interlock_name: str = "door"

    @staticmethod
    def bypassed(device: Any, interlock_name: str) -> bool:
        """Return True if an active fault bypasses the named interlock."""
        return any(
            isinstance(f, InterlockBypass)
            and f.active
            and f.interlock_name == interlock_name
            for f in getattr(device, "faults", [])
        )
