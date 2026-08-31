"""MCP exposure layer for the simulated device library.

Transport-agnostic adapter that maps a :class:`DeviceHub` onto MCP-style
tools. ``list_tools()`` returns MCP tool descriptors (JSON Schema inputs) and
``call_tool()`` dispatches with MCP-compatible result envelopes, so the hub
can be driven:

* directly from Python (tests, notebooks);
* from any agent harness that speaks MCP, via :func:`serve_stdio` when the
  optional ``mcp`` package is installed;
* from a plain CLI/JSON-RPC shim without any extra dependency.

Tool surface (MHS semantics):

* ``devices.list``        — discover devices and their classes
* ``device.reference``    — fetch the MHS reference file (YAML) for a device
* ``device.read``         — read a primitive
* ``device.write``        — write a primitive (safety-validated)
* ``device.step``         — advance physics (long-running tasks)
* ``device.inject_fault`` — safety-evaluation scenarios
"""

from __future__ import annotations

import json
import logging
from typing import Any

from .base import DeviceError, SimDevice
from .faults import Fault

logger = logging.getLogger(__name__)

FAULT_TYPES: dict[str, type[Fault]] = {}


def _register_builtin_faults() -> None:
    from . import faults as _faults

    for cls_name in (
        "SensorDrift",
        "SensorStuck",
        "RelayStuckClosed",
        "InterlockBypass",
    ):
        FAULT_TYPES[cls_name] = getattr(_faults, cls_name)


class DeviceHub:
    """Registry of live devices behind a single MCP-style tool surface."""

    def __init__(self) -> None:
        _register_builtin_faults()
        self.devices: dict[str, SimDevice] = {}

    def add(self, device: SimDevice) -> None:
        """Register a device (replaces any device with the same id)."""
        self.devices[device.device_id] = device
        logger.info("hub: registered device %s", device.device_id)

    def get(self, device_id: str) -> SimDevice:
        """Look up a device by id."""
        try:
            return self.devices[device_id]
        except KeyError:
            raise DeviceError(
                f"unknown device {device_id!r}; available: {sorted(self.devices)}"
            ) from None


def list_tools(hub: DeviceHub) -> list[dict[str, Any]]:
    """Return MCP tool descriptors for the hub."""
    return [
        {
            "name": "devices.list",
            "description": "List all discoverable devices with class and mount info.",
            "inputSchema": {"type": "object", "properties": {}},
        },
        {
            "name": "device.reference",
            "description": (
                "Fetch the MHS reference file (YAML) describing what a device "
                "can measure, what can be adjusted, and its safety limits."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {"device_id": {"type": "string"}},
                "required": ["device_id"],
            },
        },
        {
            "name": "device.read",
            "description": "Read a primitive from a device (e.g. chamber_temp_c).",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "device_id": {"type": "string"},
                    "name": {"type": "string"},
                },
                "required": ["device_id", "name"],
            },
        },
        {
            "name": "device.write",
            "description": (
                "Write a primitive on a device. Values are validated against "
                "declared safety bounds; violations are rejected."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "device_id": {"type": "string"},
                    "name": {"type": "string"},
                    "value": {},
                },
                "required": ["device_id", "name", "value"],
            },
        },
        {
            "name": "device.step",
            "description": (
                "Advance device physics by dt seconds (use for long-running "
                "operations that should not block agent reasoning)."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "device_id": {"type": "string"},
                    "dt": {"type": "number", "exclusiveMinimum": 0},
                },
                "required": ["device_id", "dt"],
            },
        },
        {
            "name": "device.inject_fault",
            "description": (
                "Inject a fault model into a device for safety evaluation. "
                f"Available types: {sorted(FAULT_TYPES)}."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "device_id": {"type": "string"},
                    "fault_type": {"type": "string", "enum": sorted(FAULT_TYPES)},
                    "params": {"type": "object"},
                },
                "required": ["device_id", "fault_type"],
            },
        },
    ]


def call_tool(hub: DeviceHub, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Dispatch a tool call; returns an MCP-compatible result envelope."""
    try:
        result = _dispatch(hub, name, arguments or {})
        return {
            "content": [
                {"type": "text", "text": json.dumps(result, ensure_ascii=False)}
            ],
            "isError": False,
        }
    except (DeviceError, KeyError, ValueError, TypeError) as exc:
        return {
            "content": [{"type": "text", "text": f"{type(exc).__name__}: {exc}"}],
            "isError": True,
        }


def _dispatch(hub: DeviceHub, name: str, args: dict[str, Any]) -> Any:
    if name == "devices.list":
        return {
            device_id: {
                "device_type": dev.device_type,
                "device_class": dev.device_class,
                "mounted_on": dev.mounted_on(),
                "reads": sorted(dev.reads),
                "writes": sorted(dev.writes),
            }
            for device_id, dev in hub.devices.items()
        }
    device = hub.get(str(args.get("device_id", "")))
    if name == "device.reference":
        return {"reference_yaml": device.reference_file_yaml()}
    if name == "device.read":
        return {"value": device.read(str(args["name"]))}
    if name == "device.write":
        device.write(str(args["name"]), args.get("value"))
        return {"ok": True}
    if name == "device.step":
        dt = float(args["dt"])
        device.step(dt)
        return {"ok": True, "time_s": device.time_s}
    if name == "device.inject_fault":
        fault_type = str(args["fault_type"])
        cls = FAULT_TYPES.get(fault_type)
        if cls is None:
            raise DeviceError(
                f"unknown fault type {fault_type!r}; available: {sorted(FAULT_TYPES)}"
            )
        params = dict(args.get("params") or {})
        params.setdefault("name", fault_type.lower())
        fault = cls(**params)
        device.inject_fault(fault)
        return {"ok": True, "fault": fault.name}
    raise DeviceError(f"unknown tool {name!r}")


def serve_stdio(hub: DeviceHub) -> None:
    r"""Serve the hub over MCP stdio (requires the optional ``mcp`` package).

    Install with ``uv add mcp`` and run::

        python -m cloud_robotics_sim.devices.mcp_adapter

    """
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "serve_stdio requires the 'mcp' package (uv add mcp). "
            "The list_tools/call_tool adapter works without it."
        ) from exc

    server = FastMCP("genesis-cloud-sim-devices")

    @server.tool(name="devices.list", description="List all devices.")
    def _list() -> str:  # pragma: no cover
        return json.dumps(_dispatch(hub, "devices.list", {}), ensure_ascii=False)

    @server.tool(name="device.read", description="Read a device primitive.")
    def _read(device_id: str, name: str) -> str:  # pragma: no cover
        return json.dumps(
            _dispatch(hub, "device.read", {"device_id": device_id, "name": name}),
            ensure_ascii=False,
        )

    @server.tool(name="device.write", description="Write a device primitive.")
    def _write(device_id: str, name: str, value: Any) -> str:  # pragma: no cover
        return json.dumps(
            _dispatch(
                hub,
                "device.write",
                {"device_id": device_id, "name": name, "value": value},
            ),
            ensure_ascii=False,
        )

    server.run()  # pragma: no cover
