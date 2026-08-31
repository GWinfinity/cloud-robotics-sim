"""Interactive Genesis simulation of Improvement in Telegraphy (US 174,465).

Alexander Graham Bell's 1876 patent transmitted speech electrically for
the first time: a vibrating diaphragm modulates an electric current whose
*undulations* mirror the sound wave, and an identical instrument at the
far end reproduces it. This demonstrator models the magneto transmitter /
receiver pair that Bell described (permanent-magnet biased armatures).

Physics model (lumped, plain Python; Genesis renders the instruments)
---------------------------------------------------------------------
- Transmitter diaphragm: a driven damped oscillator,
  ``m x'' + c x' + k x = A p(t)`` with voice pressure
  ``p(t) = pressure_pa * sin(2 pi f t)``. The diaphragm is tuned to a
  ~1 kHz resonance with Q ~ 20, giving the instrument its characteristic
  frequency response.
- Magneto induction: a permanent magnet biases flux across the air gap
  ``g = g0 - x``; the reluctance is ``R(g) = g / (mu0 A_pole)``, so the
  bias flux is ``Phi = MMF / R`` and the induced EMF follows the chain
  rule, ``e = -N dPhi/dt = N MMF mu0 A_pole / g^2 * x'``.
- Line: a lumped RL loop, ``L di/dt + (R_line + 2 R_coil) i =
  e_tx - e_rx`` with ``R_line = 8 ohm/km * line_length_km``.
- Receiver: the line current drives an identical biased armature with
  force ``F = i N dPhi/dg``; its diaphragm velocity is the acoustic
  output. The transmitter feels the Lenz reaction ``-i N dPhi/dg``, so
  energy is conserved through the coupled electro-mechanical system.
- Integration: classical RK4 at a fixed inner step of 20 us (the 1 kHz
  diaphragm resonance demands it).

Interactive parameters
----------------------
- ``voice_freq_hz``: voice tone frequency, 100-4000 Hz.
- ``voice_pressure_pa``: voice pressure amplitude at the mouthpiece,
  0.1-10 Pa.
- ``line_length_km``: transmission line length, 0-100 km.
"""

from __future__ import annotations

import logging
import math

import numpy as np

from cloud_robotics_sim.patents.base import PatentSimConfig, PatentSimulation, SimState
from cloud_robotics_sim.patents.registry import register_patent

logger = logging.getLogger(__name__)

MU0 = 4.0e-7 * math.pi  # H/m, vacuum permeability


@register_patent(
    "US174465",
    {
        "title": "Improvement in Telegraphy (Telephone)",
        "inventors": ["Alexander Graham Bell"],
        "grant_date": "1876-03-07",
        "breakthrough": "Variable resistance undulating acoustic speech transmission",
    },
)
class ImprovementInTelegraphyTelephoneSimulation(PatentSimulation):
    """Genesis simulation of Bell's magneto telephone transmitter/receiver."""

    # Diaphragm constants (SI units, demonstrator scale).
    DIAPHRAGM_MASS: float = 5.0e-5  # kg
    RESONANCE_HZ: float = 1000.0  # diaphragm tuning
    QUALITY_FACTOR: float = 20.0
    DIAPHRAGM_AREA: float = 1.0e-3  # m^2, acoustic pickup area

    # Magnetic circuit constants.
    GAP_ZERO: float = 5.0e-4  # m, resting air gap
    N_TURNS: int = 500  # coil turns on each instrument
    MMF_BIAS: float = 2.0  # A-turns, permanent-magnet bias
    POLE_AREA: float = 1.0e-4  # m^2

    # Electrical loop constants.
    R_COIL: float = 10.0  # ohm, per instrument
    L_LOOP: float = 5.0e-3  # H, total loop inductance
    LINE_OHM_PER_KM: float = 8.0

    INNER_DT: float = 1.0e-5  # s, inner RK4 step (1 kHz resonance)
    RMS_TAU: float = 2.0e-3  # s, sliding-RMS time constant for metrics

    # Visual layout constants (meters).
    VISUAL_GAIN: float = 200.0  # diaphragm displacement exaggeration
    PHONE_X: float = 1.2  # half-distance between the two instruments

    def __init__(self, config: PatentSimConfig | None = None) -> None:
        super().__init__(config)
        self.config.patent_id = self.config.patent_id or "US174465"
        self._init_parameters(
            {
                "voice_freq_hz": 1000.0,  # 100 .. 4000 Hz
                "voice_pressure_pa": 2.0,  # 0.1 .. 10 Pa
                "line_length_km": 10.0,  # 0 .. 100 km
            }
        )
        # Physics state: diaphragm positions/velocities and line current.
        self._x_tx: float = 0.0
        self._v_tx: float = 0.0
        self._x_rx: float = 0.0
        self._v_rx: float = 0.0
        self._current: float = 0.0
        self._drive_phase: float = 0.0  # rad, voice drive phase
        # Sliding-RMS accumulators (exponential moving mean of squares).
        self._rms_tx: float = 0.0
        self._rms_rx: float = 0.0
        self._rms_current: float = 0.0
        self._rms_emf: float = 0.0

    @property
    def patent_title(self) -> str:
        return "Improvement in Telegraphy (Bell magneto telephone)"

    # ------------------------------------------------------------------
    # Genesis scene
    # ------------------------------------------------------------------

    def build(self) -> None:
        """Create and build the Genesis scene with the two telephones."""
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
                camera_pos=(0.0, -3.2, 1.6),
                camera_lookat=(0.0, 0.0, 0.6),
            ),
            show_viewer=not self.config.headless,
        )
        self._scene.add_entity(gs.morphs.Plane())

        rigid = gs.materials.Rigid(rho=1000.0, friction=0.5)
        # Central telegraph pole with crossarm (the "line" between phones).
        self._scene.add_entity(
            gs.morphs.Cylinder(
                radius=0.06, height=2.0, pos=(0.0, 0.4, 1.0), fixed=True
            ),
            material=gs.materials.Rigid(rho=600.0, friction=0.8),
            surface=gs.surfaces.Default(color=(0.4, 0.28, 0.15, 1.0)),
        )
        self._scene.add_entity(
            gs.morphs.Box(size=(2.4, 0.08, 0.08), pos=(0.0, 0.4, 1.9), fixed=True),
            material=rigid,
            surface=gs.surfaces.Default(color=(0.35, 0.25, 0.13, 1.0)),
        )
        # Two "butter-can" instruments: a fixed cylindrical shell plus a
        # thin diaphragm disc whose pose follows the lumped model.
        for name, x in (("tx", -self.PHONE_X), ("rx", self.PHONE_X)):
            self._scene.add_entity(
                gs.morphs.Cylinder(
                    radius=0.18, height=0.12, pos=(x, 0.0, 0.56), fixed=True
                ),
                material=rigid,
                surface=gs.surfaces.Default(color=(0.15, 0.15, 0.17, 1.0)),
            )
            self._entities[f"{name}_diaphragm"] = self._scene.add_entity(
                gs.morphs.Cylinder(radius=0.15, height=0.01, pos=(x, 0.0, 0.63)),
                material=gs.materials.Rigid(rho=7800.0, friction=0.3),
                surface=gs.surfaces.Default(color=(0.8, 0.75, 0.6, 1.0)),
            )

        # Camera must be added before the scene is built.
        self._camera = self._scene.add_camera(
            pos=(0.0, -3.2, 1.6),
            lookat=(0.0, 0.0, 0.6),
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
        """Reset both diaphragms and the line current to rest."""
        if not self._built:
            raise RuntimeError("Simulation has not been built. Call build() first.")
        self._time = 0.0
        self._x_tx = 0.0
        self._v_tx = 0.0
        self._x_rx = 0.0
        self._v_rx = 0.0
        self._current = 0.0
        self._drive_phase = 0.0
        self._rms_tx = 0.0
        self._rms_rx = 0.0
        self._rms_current = 0.0
        self._rms_emf = 0.0
        self._pose_diaphragms()
        return self.get_state()

    def step(self) -> SimState:
        """Advance the electro-mechanical dynamics and the scene."""
        if not self._built:
            raise RuntimeError("Simulation has not been built. Call build() first.")
        for _ in range(self.config.substeps):
            self._integrate_circuit(self.config.dt)
            self._pose_diaphragms()
            self._scene.step()
        self._time += self.config.dt * self.config.substeps
        return self.get_state()

    def get_state(self) -> SimState:
        """Return the simulation state including telephone metrics."""
        state = super().get_state()
        state.metrics.update(self._compute_metrics())
        return state

    # ------------------------------------------------------------------
    # Physics
    # ------------------------------------------------------------------

    @property
    def stiffness(self) -> float:
        """Diaphragm stiffness tuned to ``RESONANCE_HZ`` in N/m."""
        omega = 2.0 * math.pi * self.RESONANCE_HZ
        return self.DIAPHRAGM_MASS * omega * omega

    @property
    def damping(self) -> float:
        """Diaphragm viscous damping for ``QUALITY_FACTOR`` in N s/m."""
        omega = 2.0 * math.pi * self.RESONANCE_HZ
        return self.DIAPHRAGM_MASS * omega / self.QUALITY_FACTOR

    def coupling(self, displacement: float) -> float:
        """Electro-mechanical coupling ``N dPhi/dg`` in V s/m (= N/A).

        The bias flux gradient grows as the air gap closes; the gap is
        clamped above a tenth of its rest value to avoid the singularity.
        """
        gap = max(self.GAP_ZERO - displacement, 0.1 * self.GAP_ZERO)
        return self.N_TURNS * self.MMF_BIAS * MU0 * self.POLE_AREA / (gap * gap)

    def line_resistance(self) -> float:
        """Total loop resistance in ohm (line plus both coils)."""
        length = float(np.clip(self._parameters["line_length_km"], 0.0, 100.0))
        return self.LINE_OHM_PER_KM * length + 2.0 * self.R_COIL

    def _derivatives(self, y: tuple[float, ...]) -> tuple[float, ...]:
        """Right-hand side of the coupled electro-mechanical ODE system."""
        x_tx, v_tx, x_rx, v_rx, current = y
        pressure = float(np.clip(self._parameters["voice_pressure_pa"], 0.0, 10.0))
        drive = self.DIAPHRAGM_AREA * pressure * math.sin(self._drive_phase)

        k_tx = self.coupling(x_tx)
        k_rx = self.coupling(x_rx)
        emf_tx = k_tx * v_tx
        emf_rx = k_rx * v_rx
        di = (emf_tx - emf_rx - self.line_resistance() * current) / self.L_LOOP

        k_spring = self.stiffness
        c_damp = self.damping
        a_tx = (
            drive - c_damp * v_tx - k_spring * x_tx - k_tx * current
        ) / self.DIAPHRAGM_MASS
        a_rx = (k_rx * current - c_damp * v_rx - k_spring * x_rx) / self.DIAPHRAGM_MASS
        return (v_tx, a_tx, v_rx, a_rx, di)

    def _integrate_circuit(self, dt: float) -> None:
        """Advance the transmitter/line/receiver state over ``dt`` (RK4)."""
        inner_dt = min(dt, self.INNER_DT)
        steps = max(1, int(round(dt / inner_dt)))
        inner_dt = dt / steps
        freq = float(np.clip(self._parameters["voice_freq_hz"], 100.0, 4000.0))
        alpha = min(1.0, inner_dt / self.RMS_TAU)
        y: tuple[float, ...] = (
            self._x_tx,
            self._v_tx,
            self._x_rx,
            self._v_rx,
            self._current,
        )
        for _ in range(steps):
            k1 = self._derivatives(y)
            k2 = self._derivatives(tuple(a + 0.5 * inner_dt * b for a, b in zip(y, k1)))
            k3 = self._derivatives(tuple(a + 0.5 * inner_dt * b for a, b in zip(y, k2)))
            k4 = self._derivatives(tuple(a + inner_dt * b for a, b in zip(y, k3)))
            y = tuple(
                a + (inner_dt / 6.0) * (b1 + 2.0 * b2 + 2.0 * b3 + b4)
                for a, b1, b2, b3, b4 in zip(y, k1, k2, k3, k4)
            )
            self._drive_phase += inner_dt * 2.0 * math.pi * freq
            # Sliding-RMS bookkeeping for the metrics.
            emf = self.coupling(y[0]) * y[1]
            self._rms_tx += alpha * (y[0] * y[0] - self._rms_tx)
            self._rms_rx += alpha * (y[2] * y[2] - self._rms_rx)
            self._rms_current += alpha * (y[4] * y[4] - self._rms_current)
            self._rms_emf += alpha * (emf * emf - self._rms_emf)
        self._x_tx, self._v_tx, self._x_rx, self._v_rx, self._current = y
        self._drive_phase = math.fmod(self._drive_phase, 2.0 * math.pi)

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def _pose_diaphragms(self) -> None:
        """Write the diaphragm displacements into the Genesis scene."""
        identity = np.array([1.0, 0.0, 0.0, 0.0])
        poses = {
            "tx_diaphragm": (-self.PHONE_X, self._x_tx),
            "rx_diaphragm": (self.PHONE_X, self._x_rx),
        }
        for name, (x, disp) in poses.items():
            entity = self._entities.get(name)
            if entity is None:
                continue
            z = 0.63 + disp * self.VISUAL_GAIN
            try:
                entity.set_pos(np.array([x, 0.0, z]))
                entity.set_quat(identity)
                if hasattr(entity, "set_dofs_velocity"):
                    entity.set_dofs_velocity(np.zeros(6))
            except Exception as exc:  # noqa: BLE001 - pose is best-effort
                logger.debug("Could not pose %s: %s", name, exc)

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def _compute_metrics(self) -> dict[str, float]:
        """Return current telephone metrics."""
        tx_rms = math.sqrt(max(self._rms_tx, 0.0))
        rx_rms = math.sqrt(max(self._rms_rx, 0.0))
        attenuation = rx_rms / tx_rms if tx_rms > 1e-15 else 0.0
        return {
            "tx_disp_um": float(tx_rms * 1.0e6),
            "rx_disp_um": float(rx_rms * 1.0e6),
            "line_current_ma": float(math.sqrt(max(self._rms_current, 0.0)) * 1.0e3),
            "tx_emf_rms_v": float(math.sqrt(max(self._rms_emf, 0.0))),
            "attenuation": float(attenuation),
            "line_resistance_ohm": float(self.line_resistance()),
        }
