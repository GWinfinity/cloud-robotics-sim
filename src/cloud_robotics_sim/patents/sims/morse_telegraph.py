"""Interactive Genesis simulation of the Electro-Magnetic Telegraph (US 1,647).

Samuel Morse's telegraph: an operator key closes a battery circuit; the
current, delayed by the line's resistance and inductance, energizes an
electromagnet at the receiver, whose armature clicks down and embosses
dots and dashes onto the paper tape.

Physics model (lumped, plain Python; Genesis renders the receiver)
------------------------------------------------------------------
- Circuit: ``L di/dt = V - i R`` with ``R = R_coil + line_km * R_LINE``
  and ``L = L_coil + line_km * L_LINE``. Long lines both delay the
  current rise and — historically the reason for relay stations —
  attenuate it below the armature's pull-in current.
- Electromagnet: Maxwell's force law, ``F = K_m i^2 / (gap + x0)^2``,
  grows steeply as the air gap closes.
- Armature: spring-return lever with preload, ``m x'' = F_mag - P0 -
  k x - c x'`` clamped to its travel stops. Because the holding force at
  zero gap far exceeds the force at rest, the armature shows pull-in /
  drop-out hysteresis (the "click" of the sounder).
- Telegraphy: key closures are timed and decoded into Morse characters
  (dot < dash, letter and word gaps) so scripted or interactive keying
  prints real text.

Interactive parameters
----------------------
- ``key``: operator key, 1 = closed, 0 = open.
- ``battery_voltage``: supply in volts.
- ``line_length_km``: transmission-line length (adds R and L).
"""

from __future__ import annotations

import logging

import numpy as np

from cloud_robotics_sim.patents.base import PatentSimConfig, PatentSimulation, SimState
from cloud_robotics_sim.patents.registry import register_patent

logger = logging.getLogger(__name__)

# ITU Morse table for the letters and digits used by the decoder.
MORSE_TABLE = {
    ".-": "A",
    "-...": "B",
    "-.-.": "C",
    "-..": "D",
    ".": "E",
    "..-.": "F",
    "--.": "G",
    "....": "H",
    "..": "I",
    ".---": "J",
    "-.-": "K",
    ".-..": "L",
    "--": "M",
    "-.": "N",
    "---": "O",
    ".--.": "P",
    "--.-": "Q",
    ".-.": "R",
    "...": "S",
    "-": "T",
    "..-": "U",
    "...-": "V",
    ".--": "W",
    "-..-": "X",
    "-.--": "Y",
    "--..": "Z",
    "-----": "0",
    ".----": "1",
    "..---": "2",
    "...--": "3",
    "....-": "4",
    ".....": "5",
    "-....": "6",
    "--...": "7",
    "---..": "8",
    "----.": "9",
}


@register_patent(
    "US1647",
    {
        "title": "Electro-Magnetic Telegraph",
        "inventors": ["Samuel F. B. Morse"],
        "grant_date": "1840-06-20",
        "breakthrough": "Regenerative relay amplifiers and binary code",
    },
)
class ElectromagneticTelegraphSimulation(PatentSimulation):
    """Genesis simulation of Morse's electromagnetic telegraph receiver."""

    # Circuit constants (SI units, demonstrator scale).
    R_COIL: float = 5.0  # ohm, electromagnet winding
    L_COIL: float = 0.5  # H, iron-core coil inductance
    R_LINE: float = 10.0  # ohm / km, iron telegraph wire
    L_LINE: float = 2.0e-3  # H / km

    # Armature constants. The preload and geometry set a pull-in current
    # of ~0.2 A; the drop-out current is far lower (hysteresis).
    ARMATURE_MASS: float = 0.01  # kg
    ARMATURE_GAP: float = 2.0e-3  # m, open air gap
    GAP_OFFSET: float = 0.5e-3  # m, residual gap at full closure
    SPRING_PRELOAD: float = 0.05  # N, holds the armature open
    SPRING_RATE: float = 25.0  # N / m
    DAMPING: float = 0.3  # N s / m

    # Morse timing (seconds): dot length threshold, letter/word gaps.
    DASH_MIN: float = 0.24  # press longer than this is a dash
    LETTER_GAP: float = 0.3  # release longer than this ends a letter
    WORD_GAP: float = 0.84  # release longer than this ends a word

    INNER_DT: float = 2.0e-4  # s, inner integrator resolution

    def __init__(self, config: PatentSimConfig | None = None) -> None:
        super().__init__(config)
        self.config.patent_id = self.config.patent_id or "US1647"
        self._init_parameters(
            {
                "key": 0.0,  # 0/1, operator key
                "battery_voltage": 12.0,  # V
                "line_length_km": 5.0,  # km
            }
        )
        self._current: float = 0.0  # A
        self._travel: float = 0.0  # m, armature deflection (0=open)
        self._travel_rate: float = 0.0  # m/s
        self._sounder_down: bool = False
        self._clicks: int = 0
        # Morse timing state.
        self._key_since: float | None = None
        self._release_since: float | None = 0.0
        self._symbols: str = ""
        self.decoded_text: str = ""

    @property
    def patent_title(self) -> str:
        return "Electro-Magnetic Telegraph (Morse receiver)"

    # ------------------------------------------------------------------
    # Genesis scene
    # ------------------------------------------------------------------

    def build(self) -> None:
        """Create and build the Genesis scene with the receiver."""
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
                camera_pos=(0.35, -0.35, 0.3),
                camera_lookat=(0.0, 0.0, 0.08),
            ),
            show_viewer=not self.config.headless,
        )
        self._scene.add_entity(gs.morphs.Plane())

        rigid = gs.materials.Rigid(rho=1000.0, friction=0.6)
        wood = gs.surfaces.Default(color=(0.5, 0.35, 0.18, 1.0))
        brass = gs.surfaces.Default(color=(0.72, 0.53, 0.2, 1.0))
        iron = gs.surfaces.Default(color=(0.25, 0.25, 0.28, 1.0))

        # Base board.
        self._scene.add_entity(
            gs.morphs.Box(size=(0.3, 0.2, 0.015), pos=(0.0, 0.0, 0.008), fixed=True),
            material=rigid,
            surface=wood,
        )
        # Electromagnet: horizontal coil (iron core in dark, winding hinted
        # by a slightly larger copper sleeve).
        self._entities["coil"] = self._scene.add_entity(
            gs.morphs.Cylinder(
                radius=0.02, height=0.07, pos=(0.0, 0.0, 0.05), fixed=True
            ),
            material=rigid,
            surface=gs.surfaces.Default(color=(0.75, 0.45, 0.2, 1.0)),
        )
        # Armature lever above the coil pole; posed from the lumped model.
        self._entities["armature"] = self._scene.add_entity(
            gs.morphs.Box(size=(0.12, 0.015, 0.006), pos=(0.0, 0.0, 0.085)),
            material=gs.materials.Rigid(rho=7800.0, friction=0.6),
            surface=iron,
        )
        # Pivot post under the armature's left end.
        self._scene.add_entity(
            gs.morphs.Box(
                size=(0.01, 0.02, 0.05), pos=(-0.055, 0.0, 0.045), fixed=True
            ),
            material=rigid,
            surface=brass,
        )
        # Battery box and the operator's key lever.
        self._scene.add_entity(
            gs.morphs.Box(size=(0.07, 0.05, 0.05), pos=(0.1, 0.06, 0.04), fixed=True),
            material=rigid,
            surface=iron,
        )
        self._entities["key"] = self._scene.add_entity(
            gs.morphs.Box(size=(0.09, 0.012, 0.005), pos=(-0.1, 0.05, 0.045)),
            material=rigid,
            surface=brass,
        )

        # Camera must be added before the scene is built.
        self._camera = self._scene.add_camera(
            pos=(0.35, -0.35, 0.3),
            lookat=(0.0, 0.0, 0.08),
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
        """Reset the circuit, armature, and Morse decoder."""
        if not self._built:
            raise RuntimeError("Simulation has not been built. Call build() first.")
        self._time = 0.0
        self._current = 0.0
        self._travel = 0.0
        self._travel_rate = 0.0
        self._sounder_down = False
        self._clicks = 0
        self._key_since = None
        self._release_since = 0.0
        self._symbols = ""
        self.decoded_text = ""
        self._pose_armature()
        return self.get_state()

    def step(self) -> SimState:
        """Advance the circuit/armature dynamics and the scene."""
        if not self._built:
            raise RuntimeError("Simulation has not been built. Call build() first.")
        for _ in range(self.config.substeps):
            self._integrate(self.config.dt)
            self._pose_armature()
            self._scene.step()
        self._time += self.config.dt * self.config.substeps
        return self.get_state()

    def get_state(self) -> SimState:
        """Return the simulation state including telegraph metrics."""
        state = super().get_state()
        state.metrics.update(self._compute_metrics())
        return state

    # ------------------------------------------------------------------
    # Physics
    # ------------------------------------------------------------------

    def _circuit_constants(self) -> tuple[float, float]:
        """Return total ``(R, L)`` including the transmission line."""
        km = float(np.clip(self._parameters["line_length_km"], 0.0, 500.0))
        return (
            self.R_COIL + km * self.R_LINE,
            self.L_COIL + km * self.L_LINE,
        )

    def steady_current(self) -> float:
        """Steady-state coil current in amps with the key closed."""
        voltage = float(np.clip(self._parameters["battery_voltage"], 0.0, 100.0))
        r_total, _ = self._circuit_constants()
        return voltage / r_total

    def magnetic_force(self, current: float, travel: float) -> float:
        """Maxwell attraction on the armature in newtons.

        ``F = K_m i^2 / (g - x + x0)^2`` grows as the air gap closes.
        ``K_m`` is chosen so the pull-in current at rest is ~0.2 A.
        """
        gap = self.ARMATURE_GAP - travel + self.GAP_OFFSET
        k_m = self.SPRING_PRELOAD * (self.ARMATURE_GAP + self.GAP_OFFSET) ** 2 / 0.2**2
        return float(k_m * current * current / gap**2)

    def pull_in_current(self) -> float:
        """Coil current at which the armature starts to move (amps)."""
        return 0.2

    def drop_out_current(self) -> float:
        """Coil current below which a closed armature releases (amps)."""
        holding = self.SPRING_PRELOAD + self.SPRING_RATE * self.ARMATURE_GAP
        k_m = self.SPRING_PRELOAD * (self.ARMATURE_GAP + self.GAP_OFFSET) ** 2 / 0.2**2
        return float(np.sqrt(holding * self.GAP_OFFSET**2 / k_m))

    def _integrate(self, dt: float) -> None:
        """Integrate circuit, armature, and Morse timing over ``dt``."""
        key = float(self._parameters["key"]) >= 0.5
        voltage = float(np.clip(self._parameters["battery_voltage"], 0.0, 100.0))
        r_total, l_total = self._circuit_constants()
        inner_dt = min(dt, self.INNER_DT)
        steps = max(1, int(round(dt / inner_dt)))
        inner_dt = dt / steps
        for _ in range(steps):
            # Circuit: L di/dt = V - i R (key open -> V = 0).
            drive = voltage if key else 0.0
            self._current += inner_dt * (drive - self._current * r_total) / l_total
            self._current = max(self._current, 0.0)

            # Armature lever with travel stops at 0 (open) and gap (closed).
            force = (
                self.magnetic_force(self._current, self._travel)
                - self.SPRING_PRELOAD
                - self.SPRING_RATE * self._travel
                - self.DAMPING * self._travel_rate
            )
            self._travel_rate += inner_dt * force / self.ARMATURE_MASS
            self._travel += inner_dt * self._travel_rate
            if self._travel < 0.0:
                self._travel, self._travel_rate = 0.0, max(self._travel_rate, 0.0)
            if self._travel > self.ARMATURE_GAP:
                self._travel = self.ARMATURE_GAP
                self._travel_rate = min(self._travel_rate, 0.0)

            # Sounder clicks: transitions register a click.
            down = self._travel > 0.9 * self.ARMATURE_GAP
            if down != self._sounder_down:
                self._clicks += 1
                self._sounder_down = down

            self._integrate_morse(key, inner_dt)

    # ------------------------------------------------------------------
    # Morse decoding
    # ------------------------------------------------------------------

    def _integrate_morse(self, key: bool, dt: float) -> None:
        """Time key closures and decode dots/dashes into text."""
        if key:
            if self._key_since is None:
                self._key_since = 0.0  # key just pressed
            else:
                self._key_since += dt
            if self._release_since is not None:
                gap = self._release_since
                self._release_since = None
                self._end_letter(gap)
        else:
            if self._key_since is not None:
                press = self._key_since
                self._key_since = None
                self._symbols += "-" if press >= self.DASH_MIN else "."
            if self._release_since is None:
                self._release_since = 0.0
            else:
                self._release_since += dt

    def _end_letter(self, gap: float) -> None:
        """Flush pending symbols once a letter gap has elapsed.

        Args:
            gap: Duration of the key release that just ended. Only gaps
                longer than ``LETTER_GAP`` end a letter; gaps beyond
                ``WORD_GAP`` also end a word.
        """
        if gap < self.LETTER_GAP:
            return
        if self._symbols:
            self.decoded_text += MORSE_TABLE.get(self._symbols, "?")
            self._symbols = ""
        if gap >= self.WORD_GAP:
            self.decoded_text += " "

    def flush_decoder(self) -> None:
        """End any pending letter (call at the end of a message)."""
        self._end_letter(self.LETTER_GAP)

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def _pose_armature(self) -> None:
        """Tilt the armature lever about its pivot from the travel."""
        armature = self._entities.get("armature")
        if armature is None:
            return
        lever = 0.11  # m, pivot-to-pole-face horizontal distance
        angle = self._travel / lever  # small-angle rotation about Y
        half = angle / 2.0
        quat = np.array([np.cos(half), 0.0, np.sin(half), 0.0])
        pivot = np.array([-0.055, 0.0, 0.072])
        rest_center = np.array([0.005, 0.0, 0.085])  # relative to pivot
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        offset = rest_center - pivot
        rotated = pivot + np.array(
            [
                cos_a * offset[0] + sin_a * offset[2],
                offset[1],
                -sin_a * offset[0] + cos_a * offset[2],
            ]
        )
        try:
            armature.set_pos(rotated)
            armature.set_quat(quat)
            if hasattr(armature, "set_dofs_velocity"):
                armature.set_dofs_velocity(np.zeros(6))
        except Exception as exc:  # noqa: BLE001 - pose is best-effort
            logger.debug("Could not pose armature: %s", exc)

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def _compute_metrics(self) -> dict[str, float]:
        """Return current telegraph metrics."""
        return {
            "coil_current_a": float(self._current),
            "steady_current_a": float(self.steady_current()),
            "magnetic_force_n": float(self.magnetic_force(self._current, self._travel)),
            "armature_travel_mm": float(self._travel * 1000.0),
            "sounder_down": float(self._sounder_down),
            "clicks": float(self._clicks),
            "pull_in_current_a": float(self.pull_in_current()),
            "drop_out_current_a": float(self.drop_out_current()),
            "symbols_pending": float(len(self._symbols)),
            "letters_decoded": float(len(self.decoded_text.replace(" ", ""))),
        }
