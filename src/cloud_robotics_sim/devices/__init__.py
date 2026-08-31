"""MHS-style simulated device library.

Exposes the device abstraction layer (:class:`SimDevice`), fault models, and
concrete devices. Each device ships read/write primitives plus a generated
reference file backed by the standards catalogs in ``data/standards/``.
"""

from .base import (
    DeviceError,
    ReadPrimitive,
    SafetyViolationError,
    SimDevice,
    UnknownPrimitiveError,
    WritePrimitive,
)
from .faults import Fault, InterlockBypass, RelayStuckClosed, SensorDrift, SensorStuck
from .mcp_adapter import DeviceHub, call_tool, list_tools
from .muffle_furnace import MuffleFurnace
from .reference import build_reference_file, load_standards_index, reference_file_yaml
from .sharpa_hand import SharpaHandDevice
from .universal_testing_machine import TensileSpecimen, UniversalTestingMachine

DEVICE_TYPES: dict[str, type[SimDevice]] = {
    MuffleFurnace.device_type: MuffleFurnace,
    UniversalTestingMachine.device_type: UniversalTestingMachine,
    SharpaHandDevice.device_type: SharpaHandDevice,
}


def create_device(device_type: str, device_id: str, **kwargs: object) -> SimDevice:
    """Create a device by registered type name."""
    try:
        cls = DEVICE_TYPES[device_type]
    except KeyError:
        raise KeyError(
            f"unknown device type {device_type!r}; available: {sorted(DEVICE_TYPES)}"
        ) from None
    return cls(device_id, **kwargs)


__all__ = [
    "DEVICE_TYPES",
    "DeviceError",
    "DeviceHub",
    "Fault",
    "InterlockBypass",
    "MuffleFurnace",
    "ReadPrimitive",
    "RelayStuckClosed",
    "SafetyViolationError",
    "SensorDrift",
    "SensorStuck",
    "SharpaHandDevice",
    "SimDevice",
    "TensileSpecimen",
    "UniversalTestingMachine",
    "UnknownPrimitiveError",
    "WritePrimitive",
    "build_reference_file",
    "call_tool",
    "create_device",
    "list_tools",
    "load_standards_index",
    "reference_file_yaml",
]
