"""Tests for the MCP exposure layer (DeviceHub + tool dispatch)."""

from __future__ import annotations

import json

import yaml

from cloud_robotics_sim.devices import MuffleFurnace
from cloud_robotics_sim.devices.mcp_adapter import DeviceHub, call_tool, list_tools


def make_hub() -> DeviceHub:
    """Hub with one fast furnace."""
    hub = DeviceHub()
    hub.add(
        MuffleFurnace(
            "furnace_hub",
            thermal_mass_j_k=2_000.0,
            heater_power_w=3_000.0,
            wall_loss_w_k=1.0,
        )
    )
    return hub


def payload(result: dict) -> dict:
    """Decode the JSON payload of a successful tool result."""
    assert result["isError"] is False
    return json.loads(result["content"][0]["text"])


def test_list_tools_schema() -> None:
    """List tools schema."""
    tools = list_tools(make_hub())
    names = {t["name"] for t in tools}
    assert names == {
        "devices.list",
        "device.reference",
        "device.read",
        "device.write",
        "device.step",
        "device.inject_fault",
    }
    for tool in tools:
        assert "description" in tool
        assert tool["inputSchema"]["type"] == "object"


def test_devices_list_discovers_primitives() -> None:
    """Devices list discovers primitives."""
    result = payload(call_tool(make_hub(), "devices.list", {}))
    assert "furnace_hub" in result
    assert result["furnace_hub"]["device_class"] == "furnace"
    assert "chamber_temp_c" in result["furnace_hub"]["reads"]


def test_read_write_roundtrip() -> None:
    """Read write roundtrip."""
    hub = make_hub()
    call_tool(
        hub,
        "device.write",
        {"device_id": "furnace_hub", "name": "setpoint_c", "value": 300.0},
    )
    out = payload(
        call_tool(
            hub, "device.read", {"device_id": "furnace_hub", "name": "setpoint_c"}
        )
    )
    assert out["value"] == 300.0


def test_write_safety_violation_returns_error_envelope() -> None:
    """Write safety violation returns error envelope."""
    hub = make_hub()
    result = call_tool(
        hub,
        "device.write",
        {"device_id": "furnace_hub", "name": "setpoint_c", "value": 99999.0},
    )
    assert result["isError"] is True
    assert "SafetyViolationError" in result["content"][0]["text"]


def test_unknown_device_returns_error() -> None:
    """Unknown device returns error."""
    result = call_tool(make_hub(), "device.read", {"device_id": "ghost", "name": "x"})
    assert result["isError"] is True


def test_unknown_tool_returns_error() -> None:
    """Unknown tool returns error."""
    result = call_tool(make_hub(), "device.destroy", {})
    assert result["isError"] is True


def test_step_advances_time() -> None:
    """Step advances time."""
    hub = make_hub()
    call_tool(hub, "device.step", {"device_id": "furnace_hub", "dt": 5.0})
    dev = hub.get("furnace_hub")
    assert dev.time_s == 5.0


def test_inject_fault_via_tool() -> None:
    """Inject fault via tool."""
    hub = make_hub()
    out = payload(
        call_tool(
            hub,
            "device.inject_fault",
            {
                "device_id": "furnace_hub",
                "fault_type": "SensorDrift",
                "params": {"channel": "chamber_temp_c", "offset": -50.0},
            },
        )
    )
    assert out["ok"] is True
    dev = hub.get("furnace_hub")
    assert dev.read("chamber_temp_c") == dev.temp_c - 50.0
    bad = call_tool(
        hub,
        "device.inject_fault",
        {"device_id": "furnace_hub", "fault_type": "EmpathyFault"},
    )
    assert bad["isError"] is True


def test_reference_tool_returns_yaml() -> None:
    """Reference tool returns yaml."""
    out = payload(
        call_tool(make_hub(), "device.reference", {"device_id": "furnace_hub"})
    )
    parsed = yaml.safe_load(out["reference_yaml"])
    assert parsed["device_id"] == "furnace_hub"
    assert parsed["compliance"]


def test_full_agent_workflow() -> None:
    """Full agent workflow: discover, read reference, operate, observe."""
    hub = make_hub()
    # 1. discover
    devices = payload(call_tool(hub, "devices.list", {}))
    device_id = next(iter(devices))
    # 2. learn the device from its reference file
    ref = yaml.safe_load(
        payload(call_tool(hub, "device.reference", {"device_id": device_id}))[
            "reference_yaml"
        ]
    )
    sp = next(w for w in ref["writable"] if w["name"] == "setpoint_c")
    target = 0.5 * sp["maximum"]
    # 3. operate within limits
    call_tool(
        hub,
        "device.write",
        {"device_id": device_id, "name": "setpoint_c", "value": target},
    )
    call_tool(
        hub,
        "device.write",
        {"device_id": device_id, "name": "ramp_rate_c_min", "value": 50.0},
    )
    call_tool(
        hub, "device.write", {"device_id": device_id, "name": "start", "value": True}
    )
    # 4. long-running heat-up without agent reasoning at each step
    for _ in range(240):
        call_tool(hub, "device.step", {"device_id": device_id, "dt": 5.0})
    temp = payload(
        call_tool(
            hub, "device.read", {"device_id": device_id, "name": "chamber_temp_c"}
        )
    )["value"]
    assert abs(temp - target) < 10.0
