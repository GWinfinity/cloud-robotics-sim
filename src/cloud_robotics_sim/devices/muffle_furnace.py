"""Muffle furnace (box-type resistance furnace) simulated device.

Lumped thermal model, no Genesis dependency::

    C_th * dT/dt = P_heater - h_wall * (T - T_amb) - [door open] h_door * (T - T_amb)

Control is bang-bang around a ramp-limited program target with hysteresis,
mirroring a real furnace controller. Safety behavior follows
GB 5959.4-2008 / GB/T 5959.1-2019 / GB 4793.6-2008:

* a door interlock removes heater power while the door is open;
* an independent over-temperature cutout (series contactor) trips above
  ``overtemp_cutout_c`` and latches until ``reset_overtemp``;
* setpoints above ``max_setpoint_c`` are rejected at the primitive layer.

Fault-injection scenarios: thermocouple drift/stuck (SensorDrift /
SensorStuck on ``chamber_temp_c``), welded heater relay
(RelayStuckClosed), bypassed door interlock (InterlockBypass).
"""

from __future__ import annotations

from typing import Any

from .base import ReadPrimitive, SafetyViolationError, SimDevice, WritePrimitive
from .faults import InterlockBypass


class MuffleFurnace(SimDevice):
    """Lumped box-type resistance furnace."""

    device_type = "muffle_furnace"
    device_class = "furnace"
    compliance = ("GB/T 5959.1-2019", "GB 5959.4-2008", "GB 4793.6-2008")
    natural_language_notes = (
        "箱式电阻炉（马弗炉）。炉膛热惯性大：升温数分钟到数十分钟，"
        "开门取放样品后需等待热平衡恢复再读数。炉门联锁：开门时切断加热；"
        "独立超温保护动作后必须人工复位。取放高温样品使用坩埚钳，"
        "炉口正前方为高温辐射区。"
    )

    def __init__(
        self,
        device_id: str,
        *,
        ambient_c: float = 25.0,
        thermal_mass_j_k: float = 20_000.0,
        heater_power_w: float = 5_000.0,
        wall_loss_w_k: float = 2.5,
        door_loss_w_k: float = 40.0,
        max_setpoint_c: float = 1200.0,
        overtemp_cutout_c: float = 1250.0,
        hysteresis_c: float = 1.0,
        max_ramp_c_min: float = 50.0,
    ) -> None:
        super().__init__(device_id)
        self.ambient_c = ambient_c
        self.thermal_mass_j_k = thermal_mass_j_k
        self.heater_power_w = heater_power_w
        self.wall_loss_w_k = wall_loss_w_k
        self.door_loss_w_k = door_loss_w_k
        self.max_setpoint_c = max_setpoint_c
        self.overtemp_cutout_c = overtemp_cutout_c
        self.hysteresis_c = hysteresis_c
        self.max_ramp_c_min = max_ramp_c_min

        # state
        self.temp_c = ambient_c
        self.setpoint_c = ambient_c
        self.ramp_rate_c_min = 20.0
        self._program_target_c = ambient_c  # ramp-limited target
        self.running = False
        self.heater_on = False
        self.door_open = False
        self.overtemp_tripped = False

    # -- primitives -----------------------------------------------------------

    @property
    def reads(self) -> dict[str, ReadPrimitive]:
        return {
            "chamber_temp_c": ReadPrimitive(
                "chamber_temp_c", "°C", "热电偶测得的炉膛温度"
            ),
            "setpoint_c": ReadPrimitive("setpoint_c", "°C", "当前设定温度"),
            "program_target_c": ReadPrimitive(
                "program_target_c", "°C", "经升温速率限制后的程序目标温度"
            ),
            "door_open": ReadPrimitive("door_open", "bool", "炉门是否打开"),
            "heater_on": ReadPrimitive("heater_on", "bool", "加热器是否通电"),
            "running": ReadPrimitive("running", "bool", "程序是否在运行"),
            "overtemp_tripped": ReadPrimitive(
                "overtemp_tripped", "bool", "超温保护是否已动作"
            ),
        }

    @property
    def writes(self) -> dict[str, WritePrimitive]:
        return {
            "setpoint_c": WritePrimitive(
                "setpoint_c",
                "°C",
                "设定目标温度",
                minimum=self.ambient_c,
                maximum=self.max_setpoint_c,
            ),
            "ramp_rate_c_min": WritePrimitive(
                "ramp_rate_c_min",
                "°C/min",
                "升温速率上限",
                minimum=0.1,
                maximum=self.max_ramp_c_min,
            ),
            "start": WritePrimitive("start", "bool", "启动温度程序"),
            "stop": WritePrimitive("stop", "bool", "停止程序并切断加热"),
            "door_open": WritePrimitive(
                "door_open", "bool", "打开/关闭炉门（联锁：开门切断加热）"
            ),
            "reset_overtemp": WritePrimitive(
                "reset_overtemp", "bool", "复位超温保护（需先排除故障）"
            ),
        }

    def _read(self, name: str) -> Any:
        return {
            "chamber_temp_c": self.temp_c,
            "setpoint_c": self.setpoint_c,
            "program_target_c": self._program_target_c,
            "door_open": self.door_open,
            "heater_on": self.heater_on,
            "running": self.running,
            "overtemp_tripped": self.overtemp_tripped,
        }[name]

    def _write(self, name: str, value: Any) -> None:
        if name == "setpoint_c":
            self.setpoint_c = float(value)
        elif name == "ramp_rate_c_min":
            self.ramp_rate_c_min = float(value)
        elif name == "start":
            if bool(value):
                if self.overtemp_tripped:
                    raise SafetyViolationError(
                        f"{self.device_id}: cannot start, over-temperature "
                        "cutout is tripped; call reset_overtemp first"
                    )
                self.running = True
            else:
                self.running = False
        elif name == "stop":
            self.running = False
            self.heater_on = False
        elif name == "door_open":
            self.door_open = bool(value)
            if self.door_open:
                self.heater_on = False  # door interlock acts immediately
        elif name == "reset_overtemp":
            if bool(value):
                self.overtemp_tripped = False
                self.heater_on = False
                self.running = False

    # -- physics ----------------------------------------------------------------

    def step(self, dt: float) -> None:
        if dt <= 0:
            raise ValueError("dt must be positive")
        self.time_s += dt

        # ramp-limited program target
        if self.running:
            step_c = self.ramp_rate_c_min * dt / 60.0
            delta = self.setpoint_c - self._program_target_c
            self._program_target_c += max(-step_c, min(step_c, delta))

        # control logic (bang-bang with hysteresis)
        door_cut = self.door_open and not InterlockBypass.bypassed(self, "door")
        if self.running and not self.overtemp_tripped and not door_cut:
            if self.temp_c < self._program_target_c - self.hysteresis_c:
                self.heater_on = True
            elif self.temp_c > self._program_target_c + self.hysteresis_c:
                self.heater_on = False
        else:
            self.heater_on = False

        # fault perturbation (e.g. welded relay forces heater on)
        for fault in self.faults:
            if fault.active:
                fault.on_step(self, dt)

        # the independent cutout contactor wins over any fault
        if self.overtemp_tripped:
            self.heater_on = False

        # thermal integration
        power_w = self.heater_power_w if self.heater_on else 0.0
        loss_w = self.wall_loss_w_k * (self.temp_c - self.ambient_c)
        if self.door_open:
            loss_w += self.door_loss_w_k * (self.temp_c - self.ambient_c)
        self.temp_c += (power_w - loss_w) / self.thermal_mass_j_k * dt
        if self.temp_c < self.ambient_c:
            self.temp_c = self.ambient_c

        # over-temperature cutout (latching)
        if self.temp_c >= self.overtemp_cutout_c and not self.overtemp_tripped:
            self.overtemp_tripped = True
            self.heater_on = False
            self.running = False

    # -- metadata ----------------------------------------------------------------

    def safety_limits(self) -> dict[str, Any]:
        return {
            "max_setpoint_c": self.max_setpoint_c,
            "overtemp_cutout_c": self.overtemp_cutout_c,
            "door_interlock": True,
            "max_ramp_c_min": self.max_ramp_c_min,
            "standards_basis": ["GB 5959.4-2008", "GB/T 5959.1-2019"],
        }
