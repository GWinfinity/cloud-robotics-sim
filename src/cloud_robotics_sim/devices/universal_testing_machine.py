"""Electronic universal testing machine (UTM) simulated device.

Lumped model: the specimen is represented by a piecewise engineering
stress-strain curve (elastic -> yield plateau -> strain hardening -> necking
-> fracture) parameterized like a material data sheet, and the load train by
a linear compliance. Given a crosshead displacement ``x`` the machine solves

    x = strain * L0 + F(strain) / C_frame

by fixed-point iteration (converges fast because ``F/C_frame << L0``).

Behavior follows the cited standards:

* GB/T 228.1-2021 — stress/strain-rate control intent, results such as
  ReH/Rm/A come out of the curve; the extensometer must be removed after
  yield (leaving it on until fracture damages it);
* GB/T 16491-2022 — force accuracy class, overload alarm + automatic stop at
  95% of full scale, fracture auto-stop;
* GB/T 2611-2022 — travel limits and general protections.

Fault scenarios: load-cell drift (SensorDrift on ``force_n``), extensometer
left attached through fracture (damages it), force-control runaway past
yield (classic lab accident: force saturates at yield, the integrator winds
up and the crosshead slams to max speed).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .base import ReadPrimitive, SafetyViolationError, SimDevice, WritePrimitive


@dataclass(frozen=True)
class TensileSpecimen:
    """Round tensile specimen material/geometry (room temperature)."""

    material: str = "Q235"
    elastic_modulus_gpa: float = 200.0
    yield_strength_mpa: float = 235.0  # ReL
    tensile_strength_mpa: float = 400.0  # Rm
    uniform_elongation_pct: float = 10.0  # Agt
    elongation_pct: float = 25.0  # A (fracture strain)
    gauge_length_mm: float = 50.0  # L0
    diameter_mm: float = 10.0

    @property
    def area_mm2(self) -> float:
        d = self.diameter_mm
        return 3.141592653589793 * d * d / 4.0

    def stress_mpa(self, strain: float) -> float:
        """Piecewise engineering stress-strain curve."""
        e = self.elastic_modulus_gpa * 1000.0  # MPa
        re_ = self.yield_strength_mpa
        rm = self.tensile_strength_mpa
        eps_y = re_ / e
        eps_plateau_end = eps_y + 0.002
        eps_u = self.uniform_elongation_pct / 100.0
        eps_f = self.elongation_pct / 100.0
        if strain <= 0.0:
            return 0.0
        if strain <= eps_y:
            return e * strain
        if strain <= eps_plateau_end:
            return re_
        if strain <= eps_u:
            t = (strain - eps_plateau_end) / (eps_u - eps_plateau_end)
            return float(re_ + (rm - re_) * t**0.5)
        if strain <= eps_f:
            t = (strain - eps_u) / (eps_f - eps_u)
            return rm * (1.0 - 0.3 * t)
        return 0.0  # broken


class UniversalTestingMachine(SimDevice):
    """Electronic UTM with displacement/force control modes."""

    device_type = "electronic_utm"
    device_class = "utm"
    compliance = ("GB/T 16491-2022", "GB/T 2611-2022", "GB/T 228.1-2021")
    natural_language_notes = (
        "门式双空间电子万能试验机，上空间拉伸、下空间压缩/弯曲。"
        "屈服后必须摘除引伸计，否则试样断裂时损坏引伸计。"
        "力控制模式在屈服阶段会饱和飞车，塑性段务必用位移/应变速率控制。"
        "试样断裂瞬间可能有碎屑飞溅，确认防护罩关闭后再加载。"
        "试验前检查力传感器清零；试样夹偏会在曲线上留下非物理波动。"
    )

    def __init__(
        self,
        device_id: str,
        *,
        specimen: TensileSpecimen | None = None,
        max_force_n: float = 50_000.0,
        frame_stiffness_n_mm: float = 200_000.0,
        max_travel_mm: float = 800.0,
        max_speed_mm_min: float = 500.0,
        force_class: float = 1.0,
        overload_ratio: float = 0.95,
        force_control_gain: float = 2.0,
    ) -> None:
        super().__init__(device_id)
        self.specimen = specimen or TensileSpecimen()
        self.max_force_n = max_force_n
        self.frame_stiffness_n_mm = frame_stiffness_n_mm
        self.max_travel_mm = max_travel_mm
        self.max_speed_mm_min = max_speed_mm_min
        self.force_class = force_class
        self.overload_ratio = overload_ratio
        self.force_control_gain = force_control_gain  # (mm/min) per N of error

        # state
        self.specimen_loaded = False
        self.state = "idle"  # idle | running | stopped | fractured | overload
        self.control_mode = "displacement"
        self.loading_rate_mm_min = 10.0
        self.force_rate_n_s = 100.0
        self.crosshead_mm = 0.0
        self.force_n = 0.0
        self.force_target_n = 0.0
        self.extensometer_attached = False
        self.extensometer_damaged = False
        self.overload_tripped = False
        self._speed_mm_min = 0.0
        self._returning = False

    # -- specimen --------------------------------------------------------------

    @property
    def strain(self) -> float:
        """Specimen engineering strain from crosshead position and force."""
        elastic = self.force_n / self.frame_stiffness_n_mm
        elong = max(self.crosshead_mm - elastic, 0.0)
        return elong / self.specimen.gauge_length_mm

    def _specimen_force_n(self, strain: float) -> float:
        return self.specimen.stress_mpa(strain) * self.specimen.area_mm2

    # -- primitives -------------------------------------------------------------

    @property
    def reads(self) -> dict[str, ReadPrimitive]:
        return {
            "force_n": ReadPrimitive("force_n", "N", "载荷传感器测得的试验力"),
            "crosshead_disp_mm": ReadPrimitive("crosshead_disp_mm", "mm", "横梁位移"),
            "extensometer_strain": ReadPrimitive(
                "extensometer_strain", "1", "引伸计测得的标距内应变（未安装时为 None）"
            ),
            "crosshead_speed_mm_min": ReadPrimitive(
                "crosshead_speed_mm_min", "mm/min", "横梁当前速度"
            ),
            "state": ReadPrimitive(
                "state", "str", "idle/running/stopped/fractured/overload"
            ),
            "control_mode": ReadPrimitive(
                "control_mode", "str", "当前控制模式 displacement/force"
            ),
            "extensometer_attached": ReadPrimitive(
                "extensometer_attached", "bool", "引伸计是否安装"
            ),
            "extensometer_damaged": ReadPrimitive(
                "extensometer_damaged", "bool", "引伸计是否已损坏"
            ),
            "overload_tripped": ReadPrimitive(
                "overload_tripped", "bool", "过载保护是否已动作"
            ),
        }

    @property
    def writes(self) -> dict[str, WritePrimitive]:
        return {
            "load_specimen": WritePrimitive(
                "load_specimen", "bool", "装夹试样（装好后清零位移与力）"
            ),
            "control_mode": WritePrimitive(
                "control_mode",
                "str",
                "控制模式",
                choices=("displacement", "force"),
            ),
            "loading_rate_mm_min": WritePrimitive(
                "loading_rate_mm_min",
                "mm/min",
                "位移控制速率",
                minimum=0.05,
                maximum=self.max_speed_mm_min,
            ),
            "force_rate_n_s": WritePrimitive(
                "force_rate_n_s",
                "N/s",
                "力控制加载速率",
                minimum=1.0,
                maximum=self.max_force_n / 10.0,
            ),
            "start": WritePrimitive("start", "bool", "开始试验"),
            "stop": WritePrimitive("stop", "bool", "停止试验"),
            "return_crosshead": WritePrimitive(
                "return_crosshead", "bool", "横梁以最大速率返回零点"
            ),
            "attach_extensometer": WritePrimitive(
                "attach_extensometer", "bool", "安装/摘除引伸计"
            ),
            "reset_overload": WritePrimitive("reset_overload", "bool", "复位过载保护"),
        }

    def _read(self, name: str) -> Any:
        if name == "extensometer_strain":
            return self.strain if self.extensometer_attached else None
        return {
            "force_n": self.force_n,
            "crosshead_disp_mm": self.crosshead_mm,
            "crosshead_speed_mm_min": self._speed_mm_min,
            "state": self.state,
            "control_mode": self.control_mode,
            "extensometer_attached": self.extensometer_attached,
            "extensometer_damaged": self.extensometer_damaged,
            "overload_tripped": self.overload_tripped,
        }[name]

    def _write(self, name: str, value: Any) -> None:
        if name == "load_specimen":
            if bool(value):
                self.specimen_loaded = True
                self.crosshead_mm = 0.0
                self.force_n = 0.0
                self.state = "idle"
            else:
                if self.state == "running":
                    raise SafetyViolationError(
                        f"{self.device_id}: cannot unload specimen while running"
                    )
                self.specimen_loaded = False
        elif name == "control_mode":
            self.control_mode = str(value)
        elif name == "loading_rate_mm_min":
            self.loading_rate_mm_min = float(value)
        elif name == "force_rate_n_s":
            self.force_rate_n_s = float(value)
        elif name == "start":
            if bool(value):
                if not self.specimen_loaded:
                    raise SafetyViolationError(f"{self.device_id}: no specimen loaded")
                if self.overload_tripped:
                    raise SafetyViolationError(
                        f"{self.device_id}: overload tripped; reset_overload first"
                    )
                if self.state == "fractured":
                    raise SafetyViolationError(
                        f"{self.device_id}: specimen fractured; load a new one"
                    )
                self.state = "running"
                self.force_target_n = self.force_n
            else:
                self.state = "stopped"
        elif name == "stop":
            if self.state == "running":
                self.state = "stopped"
        elif name == "return_crosshead":
            if bool(value) and self.state not in ("running",):
                self._returning = True
        elif name == "attach_extensometer":
            if bool(value) and self.extensometer_damaged:
                raise SafetyViolationError(
                    f"{self.device_id}: extensometer damaged; replace it first"
                )
            self.extensometer_attached = bool(value)
        elif name == "reset_overload":
            if bool(value):
                self.overload_tripped = False
                if self.state == "overload":
                    self.state = "stopped"

    # -- physics ------------------------------------------------------------------

    def step(self, dt: float) -> None:
        if dt <= 0:
            raise ValueError("dt must be positive")
        self.time_s += dt
        self._speed_mm_min = 0.0

        if self._returning and self.state != "running":
            self.crosshead_mm = max(
                0.0, self.crosshead_mm - self.max_speed_mm_min * dt / 60.0
            )
            self._speed_mm_min = -self.max_speed_mm_min
            if self.crosshead_mm == 0.0:
                self._returning = False
            self._update_force()
            return

        if self.state != "running":
            return

        if self.control_mode == "displacement":
            speed = self.loading_rate_mm_min
        else:  # force control: chase a ramping force target
            self.force_target_n += self.force_rate_n_s * dt
            error = self.force_target_n - self.force_n
            speed = self.force_control_gain * error
            speed = max(-self.max_speed_mm_min, min(self.max_speed_mm_min, speed))
            speed = max(speed, 0.0)  # tensile direction only

        self.crosshead_mm += speed * dt / 60.0
        self._speed_mm_min = speed

        # travel limit protection (GB/T 2611-2022)
        if self.crosshead_mm >= self.max_travel_mm:
            self.crosshead_mm = self.max_travel_mm
            self.state = "stopped"

        self._update_force()

        # fracture detection -> automatic stop (GB/T 16491-2022)
        if self.strain >= self.specimen.elongation_pct / 100.0:
            self.state = "fractured"
            self.force_n = 0.0
            if self.extensometer_attached:
                self.extensometer_damaged = True
                self.extensometer_attached = False
            return

        # overload protection: alarm + automatic stop (GB/T 16491-2022)
        if self.force_n >= self.overload_ratio * self.max_force_n:
            self.overload_tripped = True
            self.state = "overload"

    def _update_force(self) -> None:
        if not self.specimen_loaded or self.state == "fractured":
            self.force_n = 0.0
            return
        # Solve x = strain*L0 + F(strain)/C for strain. The left side is
        # monotonic in strain for realistic frame stiffnesses (the necking
        # slope never exceeds L0*C), so bisection is robust everywhere.
        l0 = self.specimen.gauge_length_mm
        x = self.crosshead_mm
        lo, hi = 0.0, x / l0
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if mid * l0 + self._specimen_force_n(mid) / self.frame_stiffness_n_mm <= x:
                lo = mid
            else:
                hi = mid
        self.force_n = self._specimen_force_n(lo)

    # -- metadata ------------------------------------------------------------------

    def safety_limits(self) -> dict[str, Any]:
        return {
            "max_force_n": self.max_force_n,
            "force_accuracy_class": self.force_class,
            "overload_trip_ratio": self.overload_ratio,
            "max_travel_mm": self.max_travel_mm,
            "max_speed_mm_min": self.max_speed_mm_min,
            "standards_basis": ["GB/T 16491-2022", "GB/T 2611-2022"],
        }

    def specimen_info(self) -> dict[str, Any]:
        """Material/geometry of the loaded specimen (for reports)."""
        return asdict(self.specimen)
