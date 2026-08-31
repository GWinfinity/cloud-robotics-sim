"""MHS-style simulated device abstraction layer.

Each simulated device exposes a small set of typed primitives — ``read`` for
observing state and ``write`` for commanding it — mirroring the Model Hardware
Standard (MHS) driver model. Devices additionally carry compliance metadata
(Chinese GB/GB-T standards) used to generate reference files for agents, and
support fault injection for safety evaluation.

The layer is pure Python: lumped-physics devices (furnaces, testing machines)
run without Genesis, while Genesis-backed devices can wrap a scene entity
behind the same primitive interface.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


class DeviceError(Exception):
    """Base error for device primitive access."""


class UnknownPrimitiveError(DeviceError):
    """Raised when reading or writing an undeclared primitive."""


class SafetyViolationError(DeviceError):
    """Raised when a write violates a declared safety limit or interlock."""


@dataclass(frozen=True)
class ReadPrimitive:
    """Metadata for a readable quantity (e.g. ``chamber_temp_c``)."""

    name: str
    unit: str
    description: str


@dataclass(frozen=True)
class WritePrimitive:
    """Metadata for a writable command, including safety bounds."""

    name: str
    unit: str
    description: str
    minimum: float | None = None
    maximum: float | None = None
    choices: tuple[str, ...] | None = None


class SimDevice(ABC):
    """Base class for all MHS-style simulated devices.

    Subclasses declare their primitives via :attr:`reads` / :attr:`writes` and
    implement :meth:`_read`, :meth:`_write` and :meth:`step`. The base class
    handles primitive discovery, write validation against safety bounds, fault
    distortion of read channels, and reference-file generation.
    """

    device_type: str = "generic"
    device_class: str = "generic"
    compliance: tuple[str, ...] = ()
    natural_language_notes: str = ""

    def __init__(self, device_id: str) -> None:
        self.device_id = device_id
        self.faults: list[Any] = []
        self.time_s: float = 0.0

    # -- primitive declaration -------------------------------------------------

    @property
    @abstractmethod
    def reads(self) -> dict[str, ReadPrimitive]:
        """Readable primitives keyed by name."""

    @property
    @abstractmethod
    def writes(self) -> dict[str, WritePrimitive]:
        """Writable primitives keyed by name."""

    # -- primitive access ------------------------------------------------------

    def read(self, name: str) -> Any:
        """Read a primitive value, applying active fault distortions."""
        if name not in self.reads:
            raise UnknownPrimitiveError(
                f"{self.device_id}: unknown read primitive {name!r}; "
                f"available: {sorted(self.reads)}"
            )
        value = self._read(name)
        for fault in self.faults:
            if fault.active:
                value = fault.distort_read(self, name, value)
        return value

    def write(self, name: str, value: Any) -> None:
        """Write a primitive value after validating it against safety bounds."""
        spec = self.writes.get(name)
        if spec is None:
            raise UnknownPrimitiveError(
                f"{self.device_id}: unknown write primitive {name!r}; "
                f"available: {sorted(self.writes)}"
            )
        if not isinstance(value, bool) and isinstance(value, (int, float)):
            if spec.minimum is not None and value < spec.minimum:
                raise SafetyViolationError(
                    f"{self.device_id}: {name}={value} below minimum "
                    f"{spec.minimum} ({spec.unit})"
                )
            if spec.maximum is not None and value > spec.maximum:
                raise SafetyViolationError(
                    f"{self.device_id}: {name}={value} above maximum "
                    f"{spec.maximum} ({spec.unit})"
                )
        if spec.choices is not None and str(value) not in spec.choices:
            raise SafetyViolationError(
                f"{self.device_id}: {name}={value!r} not in {spec.choices}"
            )
        self._write(name, value)

    @abstractmethod
    def _read(self, name: str) -> Any:
        """Return the raw (fault-free) value of a read primitive."""

    @abstractmethod
    def _write(self, name: str, value: Any) -> None:
        """Apply a validated write primitive."""

    # -- physics ---------------------------------------------------------------

    @abstractmethod
    def step(self, dt: float) -> None:
        """Advance the device physics by ``dt`` seconds."""

    # -- faults ------------------------------------------------------------------

    def inject_fault(self, fault: Any) -> None:
        """Attach a fault model to this device."""
        self.faults.append(fault)
        logger.warning("%s: fault injected: %s", self.device_id, fault.name)

    def clear_faults(self) -> None:
        """Remove all injected faults."""
        self.faults.clear()

    # -- metadata ----------------------------------------------------------------

    def safety_limits(self) -> dict[str, Any]:
        """Device-specific safety limits for the reference file.

        Values should be traceable to the standards listed in
        :attr:`compliance`. The default is empty; subclasses override.
        """
        return {}

    def mounted_on(self) -> str | None:
        """Parent device id for composite devices (e.g. hand on an arm)."""
        return None

    def reference_file(self, catalog_root: Any = None) -> dict[str, Any]:
        """Build the MHS-style reference file for this device."""
        from .reference import build_reference_file

        return build_reference_file(self, catalog_root=catalog_root)

    def reference_file_yaml(self, catalog_root: Any = None) -> str:
        """Reference file serialized as YAML."""
        from .reference import reference_file_yaml

        return reference_file_yaml(self, catalog_root=catalog_root)
