"""Interactive Genesis simulation of the Television System (US 1,773,980).

Philo Farnsworth's 1930 patent replaced mechanical scanning disks with a
fully electronic system: an electron gun, magnetic deflection coils, and a
raster scanned across a fluorescent screen, with the beam current modulated
by the image signal.

Physics model (lumped, plain Python; Genesis renders the tube)
--------------------------------------------------------------
- Electron gun: the cathode-ray beam is accelerated through ``V_acc``,
  reaching ``v = sqrt(2 e V_acc / m_e)`` (1-10 kV gives 0.06c-0.20c). The
  non-relativistic kinetic energy is used throughout; at 10 kV the exact
  relativistic speed is only ~0.5% lower, well inside demonstrator
  tolerances.
- Magnetic deflection: in the deflection-coil field ``B`` the beam follows
  an arc of radius ``r = m_e v / (e B)``; for a short coil of length
  ``L_coil`` the small-angle deflection is ``theta ~ L_coil / r`` and the
  screen displacement is ``theta * D`` with ``D`` the coil-to-screen
  distance. Because displacement scales as ``1/v ~ 1/sqrt(V_acc)``, the
  high-voltage beam is "magnetically stiff": quadrupling the accelerating
  voltage halves the deflection for the same coil drive.
- Raster scan: a horizontal sawtooth at ``f_h`` and a vertical sawtooth at
  ``f_v = f_h / N_lines`` sweep the spot line by line over the screen. A
  test pattern (crosshair plus grid) modulates the beam current, so the
  spot brightness reports the image function at the current spot position.

Interactive parameters
----------------------
- ``accel_voltage_kv``: electron-gun accelerating voltage, 1-10 kV.
- ``deflection_drive``: deflection-coil drive amplitude, 0-1 (1 = full
  screen scan at the reference voltage of 5 kV).
- ``num_lines``: number of scan lines per frame, 30-441.
"""

from __future__ import annotations

import logging

import numpy as np

from cloud_robotics_sim.patents.base import PatentSimConfig, PatentSimulation, SimState
from cloud_robotics_sim.patents.registry import register_patent

logger = logging.getLogger(__name__)

E_CHARGE = 1.602176634e-19  # C, elementary charge
M_ELECTRON = 9.1093837015e-31  # kg, electron mass
C_LIGHT = 299792458.0  # m/s, speed of light


@register_patent(
    "US1773980",
    {
        "title": "Television System",
        "inventors": ["Philo T. Farnsworth"],
        "grant_date": "1930-08-26",
        "breakthrough": "All-electronic image dissector and magnetic raster",
    },
)
class TelevisionSystemSimulation(PatentSimulation):
    """Genesis simulation of Farnsworth's all-electronic television."""

    # Tube geometry (meters, demonstrator scale).
    TUBE_LENGTH: float = 0.4  # coil-to-screen distance D
    COIL_LENGTH: float = 0.05  # axial length of the deflection coils
    SCREEN_WIDTH: float = 0.2  # m
    SCREEN_HEIGHT: float = 0.15  # m
    TUBE_RADIUS: float = 0.13  # m, glass envelope

    # Scan timing.
    FRAME_RATE: float = 30.0  # Hz, vertical sawtooth frequency f_v

    # Reference voltage at which drive = 1 scans the full screen.
    REFERENCE_VOLTAGE_KV: float = 5.0

    def __init__(self, config: PatentSimConfig | None = None) -> None:
        super().__init__(config)
        self.config.patent_id = self.config.patent_id or "US1773980"
        self._init_parameters(
            {
                "accel_voltage_kv": 5.0,  # kV, 1 .. 10
                "deflection_drive": 1.0,  # 0 .. 1
                "num_lines": 30.0,  # 30 .. 441
            }
        )
        # Scan state (phases are pure functions of _scan_time).
        self._scan_time: float = 0.0

    @property
    def patent_title(self) -> str:
        return "Television System (Farnsworth image dissector)"

    # ------------------------------------------------------------------
    # Genesis scene
    # ------------------------------------------------------------------

    def build(self) -> None:
        """Create and build the Genesis scene with the CRT."""
        from cloud_robotics_sim.utils.genesis_compat import ensure_genesis_initialized

        try:
            import genesis as gs
        except ImportError as exc:
            raise RuntimeError("genesis-world is not installed") from exc

        ensure_genesis_initialized(
            headless=self.config.headless, device=self.config.device
        )
        self._scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.config.dt),
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(0.6, -0.6, 0.45),
                camera_lookat=(0.0, 0.0, 0.22),
            ),
            show_viewer=not self.config.headless,
        )
        self._scene.add_entity(gs.morphs.Plane())

        rigid = gs.materials.Rigid(rho=1000.0, friction=0.5)
        # Glass envelope (cylinder) standing upright, gun at the bottom.
        self._scene.add_entity(
            gs.morphs.Cylinder(
                radius=self.TUBE_RADIUS,
                height=self.TUBE_LENGTH,
                pos=(0.0, 0.0, self.TUBE_LENGTH / 2),
                fixed=True,
            ),
            material=gs.materials.Rigid(rho=2500.0, friction=0.3),
            surface=gs.surfaces.Default(color=(0.75, 0.85, 0.9, 0.35)),
        )
        # Electron gun assembly at the base of the tube.
        self._scene.add_entity(
            gs.morphs.Cylinder(
                radius=0.03, height=0.1, pos=(0.0, 0.0, 0.05), fixed=True
            ),
            material=rigid,
            surface=gs.surfaces.Default(color=(0.2, 0.2, 0.22, 1.0)),
        )
        # Deflection-coil torus hinted by a short sleeve around the tube.
        self._scene.add_entity(
            gs.morphs.Cylinder(
                radius=self.TUBE_RADIUS + 0.02,
                height=self.COIL_LENGTH,
                pos=(0.0, 0.0, 0.18),
                fixed=True,
            ),
            material=rigid,
            surface=gs.surfaces.Default(color=(0.72, 0.45, 0.2, 1.0)),
        )
        # Fluorescent screen: a fixed plate closing the top of the tube.
        self._scene.add_entity(
            gs.morphs.Box(
                size=(self.SCREEN_WIDTH, self.SCREEN_HEIGHT, 0.01),
                pos=(0.0, 0.0, self.TUBE_LENGTH),
                fixed=True,
            ),
            material=rigid,
            surface=gs.surfaces.Default(color=(0.05, 0.12, 0.05, 1.0)),
        )
        # Luminescent spot, posed from the scan model every step.
        self._entities["spot"] = self._scene.add_entity(
            gs.morphs.Sphere(radius=0.008, pos=(0.0, 0.0, self.TUBE_LENGTH - 0.01)),
            material=gs.materials.Rigid(rho=100.0, friction=0.5),
            surface=gs.surfaces.Default(
                color=(0.4, 1.0, 0.4, 1.0), emissive=(0.2, 0.9, 0.2)
            ),
        )

        # Camera must be added before the scene is built.
        self._camera = self._scene.add_camera(
            pos=(0.6, -0.6, 0.45),
            lookat=(0.0, 0.0, 0.22),
            res=self.config.resolution,
            fov=45,
            GUI=False,
        )
        self._scene.build()
        self._built = True

    # ------------------------------------------------------------------
    # Simulation lifecycle
    # ------------------------------------------------------------------

    def reset(self) -> SimState:
        """Reset the raster scan to the start of the first frame."""
        if not self._built:
            raise RuntimeError("Simulation has not been built. Call build() first.")
        self._time = 0.0
        self._scan_time = 0.0
        self._pose_spot()
        return self.get_state()

    def step(self) -> SimState:
        """Advance the raster scan and the scene."""
        if not self._built:
            raise RuntimeError("Simulation has not been built. Call build() first.")
        for _ in range(self.config.substeps):
            self._advance_scan(self.config.dt)
            self._pose_spot()
            self._scene.step()
        self._time += self.config.dt * self.config.substeps
        return self.get_state()

    def get_state(self) -> SimState:
        """Return the simulation state including television metrics."""
        state = super().get_state()
        state.metrics.update(self._compute_metrics())
        return state

    # ------------------------------------------------------------------
    # Physics
    # ------------------------------------------------------------------

    def accel_voltage(self) -> float:
        """Accelerating voltage in volts (parameter clipped to 1-10 kV)."""
        kv = float(np.clip(self._parameters["accel_voltage_kv"], 1.0, 10.0))
        return kv * 1000.0

    def beam_velocity(self, voltage_kv: float | None = None) -> float:
        """Beam speed in m/s, ``v = sqrt(2 e V / m_e)`` (non-relativistic).

        Args:
            voltage_kv: Optional accelerating voltage override in kV;
                defaults to the current ``accel_voltage_kv`` parameter.
        """
        volts = self.accel_voltage() if voltage_kv is None else voltage_kv * 1000.0
        return float(np.sqrt(2.0 * E_CHARGE * volts / M_ELECTRON))

    def num_lines(self) -> int:
        """Number of scan lines per frame (30-441)."""
        return int(np.clip(round(self._parameters["num_lines"]), 30, 441))

    def frame_frequency(self) -> float:
        """Vertical sawtooth frequency ``f_v`` in Hz."""
        return self.FRAME_RATE

    def line_frequency(self) -> float:
        """Horizontal sawtooth frequency ``f_h = f_v * N_lines`` in Hz."""
        return float(self.FRAME_RATE * self.num_lines())

    def _reference_field(self, half_extent: float) -> float:
        """Coil flux density (tesla) that scans ``half_extent`` at 5 kV."""
        v_ref = self.beam_velocity(self.REFERENCE_VOLTAGE_KV)
        return (
            half_extent
            * M_ELECTRON
            * v_ref
            / (E_CHARGE * self.COIL_LENGTH * self.TUBE_LENGTH)
        )

    def max_deflection(
        self,
        half_extent: float,
        voltage_kv: float | None = None,
        drive: float | None = None,
    ) -> float:
        """Peak screen displacement in meters for the given beam settings.

        The spot displacement is ``theta * D`` with
        ``theta = L_coil * e * B / (m_e * v)``; the coil drive scales ``B``
        linearly, so displacement scales as ``drive / v ~ drive / sqrt(V)``.
        """
        volts_kv = (
            float(np.clip(self._parameters["accel_voltage_kv"], 1.0, 10.0))
            if voltage_kv is None
            else voltage_kv
        )
        drive_value = (
            float(np.clip(self._parameters["deflection_drive"], 0.0, 1.0))
            if drive is None
            else drive
        )
        field = drive_value * self._reference_field(half_extent)
        if field <= 0.0:
            return 0.0
        radius = M_ELECTRON * self.beam_velocity(volts_kv) / (E_CHARGE * field)
        theta = self.COIL_LENGTH / radius
        return float(theta * self.TUBE_LENGTH)

    def spot_position(self, scan_time: float | None = None) -> tuple[float, float]:
        """Screen coordinates ``(x, y)`` of the spot in meters.

        ``x`` sweeps left to right at the line rate; ``y`` steps top to
        bottom at the frame rate. The origin is the screen center, so a
        completed frame returns the spot to the top-left corner.

        Args:
            scan_time: Absolute scan time in seconds; defaults to the
                current internal scan time.
        """
        t = self._scan_time if scan_time is None else scan_time
        phase_h = (t * self.line_frequency()) % 1.0
        phase_v = (t * self.frame_frequency()) % 1.0
        half_x = self.max_deflection(self.SCREEN_WIDTH / 2)
        half_y = self.max_deflection(self.SCREEN_HEIGHT / 2)
        spot_x = (2.0 * phase_h - 1.0) * half_x
        spot_y = (1.0 - 2.0 * phase_v) * half_y
        return float(spot_x), float(spot_y)

    def frames_drawn(self) -> int:
        """Number of complete frames scanned since reset."""
        return int(self._scan_time * self.frame_frequency())

    def line_number(self) -> int:
        """Current scan line within the frame (0-indexed)."""
        phase_v = (self._scan_time * self.frame_frequency()) % 1.0
        return min(int(phase_v * self.num_lines()), self.num_lines() - 1)

    def test_pattern(self, x: float, y: float) -> float:
        """Test-pattern brightness (0-1) at screen coordinates in meters.

        A crosshair through the origin plus a grid every quarter of the
        half-extent: bright lines on a dim background, like the monoscope
        test cards used with early electronic cameras.
        """
        half_x = self.SCREEN_WIDTH / 2
        half_y = self.SCREEN_HEIGHT / 2
        u = x / half_x
        v = y / half_y
        on_cross = abs(u) < 0.03 or abs(v) < 0.03
        grid_u = abs((u * 4.0 + 0.5) % 1.0 - 0.5) < 0.02
        grid_v = abs((v * 4.0 + 0.5) % 1.0 - 0.5) < 0.02
        return 1.0 if (on_cross or grid_u or grid_v) else 0.1

    def _advance_scan(self, dt: float) -> None:
        """Advance the raster scan clock by ``dt`` seconds."""
        self._scan_time += dt

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def _pose_spot(self) -> None:
        """Write the current spot position into the Genesis scene."""
        spot = self._entities.get("spot")
        if spot is None:
            return
        x, y = self.spot_position()
        # The tube axis is vertical: screen x/y map to world x/y at the
        # screen plane on top of the tube.
        pos = np.array([x, y, self.TUBE_LENGTH - 0.008])
        try:
            spot.set_pos(pos)
            spot.set_quat(np.array([1.0, 0.0, 0.0, 0.0]))
            if hasattr(spot, "set_dofs_velocity"):
                spot.set_dofs_velocity(np.zeros(6))
        except Exception as exc:  # noqa: BLE001 - pose is best-effort
            logger.debug("Could not pose spot: %s", exc)

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def _compute_metrics(self) -> dict[str, float]:
        """Return current television metrics."""
        spot_x, spot_y = self.spot_position()
        return {
            "spot_x": float(spot_x),
            "spot_y": float(spot_y),
            "beam_velocity_ms": float(self.beam_velocity()),
            "beam_velocity_c": float(self.beam_velocity() / C_LIGHT),
            "frames_drawn": float(self.frames_drawn()),
            "line_number": float(self.line_number()),
            "line_frequency_hz": float(self.line_frequency()),
            "frame_frequency_hz": float(self.frame_frequency()),
            "max_deflection_x_m": float(self.max_deflection(self.SCREEN_WIDTH / 2)),
            "beam_brightness": float(self.test_pattern(spot_x, spot_y)),
        }
