"""Interactive Genesis simulation of the Electrical Transformer (US 593,138).

Tesla's 1897 high-potential transformer (the "Tesla coil"): a capacitor
charged by the supply discharges through a spark gap into a few-turn
primary; the primary is magnetically coupled to a long secondary whose
self-capacitance forms a resonant circuit tuned to the same frequency.
Energy sloshes between the two circuits at the beat (coupling) frequency,
building the secondary to enormous voltages until the top load discharges.

Physics model (lumped coupled oscillators, plain Python + SciPy)
----------------------------------------------------------------
State ``x = [v1, i1, v2, i2]`` obeys the coupled equations::

    C1 v1' = -i1            (gap closed; plus supply recharge when open)
    C2 v2' =  i2
    L1 i1' + M i2' = v1 - r1 i1
    M i1' + L2 i2' = -v2 - r2 i2

with mutual inductance ``M = k sqrt(L1 L2)``. Because the system is
linear between spark-gap events, each inner step is integrated *exactly*
with the matrix exponential ``expm(A dt)`` — one matrix per gap state —
so the stiff RF oscillation needs no tiny explicit timestep.

- Spark gap: open while the supply recharges ``C1``; fires when
  ``|v1|`` reaches ``gap_threshold``; quenches when the primary current
  crosses zero.
- Tuning: primary and secondary are tuned to the same free resonance
  ``f0 = 1 / (2 pi sqrt(L C))``. The ``detune`` parameter scales the
  secondary capacitance; off resonance the voltage gain collapses.
- The secondary peak approaches ``V1 sqrt(C1 / C2)`` on tune — the
  high-potential transformation of the patent.

Interactive parameters
----------------------
- ``supply_voltage``: charging supply in volts (sets the gap fire rate).
- ``coupling``: coupling coefficient ``k`` (0.05-0.6).
- ``gap_threshold``: spark-gap breakdown voltage in volts.
- ``detune``: secondary capacitance multiplier (1.0 = on tune).
"""

from __future__ import annotations

import logging

import numpy as np

from cloud_robotics_sim.patents.base import PatentSimConfig, PatentSimulation, SimState
from cloud_robotics_sim.patents.registry import register_patent

logger = logging.getLogger(__name__)


@register_patent(
    "US593138",
    {
        "title": "Electrical Transformer",
        "inventors": ["Nikola Tesla"],
        "grant_date": "1897-11-02",
        "breakthrough": "Graded spiral windings for high-potential transformation",
    },
)
class ElectricalTransformerSimulation(PatentSimulation):
    """Genesis simulation of Tesla's coupled-resonance transformer."""

    # Circuit constants (demonstrator scale: 2 kHz resonance keeps the
    # exact integrator cheap while preserving the coupled-LC physics).
    L_PRIMARY: float = 30.0e-3  # H
    C_PRIMARY: float = 211.0e-9  # F
    L_SECONDARY: float = 1.2  # H
    C_SECONDARY: float = 5.28e-9  # F
    R_PRIMARY: float = 2.0  # ohm (includes spark-gap arc resistance)
    R_SECONDARY: float = 40.0  # ohm
    CHARGE_TAU: float = 2.0e-3  # s, supply recharge time constant
    QUENCH_TAU: float = 10.0e-6  # s, primary current decay with gap open
    QUENCH_CURRENT: float = 0.05  # A, arc extinction threshold

    MAX_INNER_DT: float = 5.0e-6  # s (~40 steps per RF cycle at 2 kHz)

    def __init__(self, config: PatentSimConfig | None = None) -> None:
        super().__init__(config)
        self.config.patent_id = self.config.patent_id or "US593138"
        self._init_parameters(
            {
                "supply_voltage": 8000.0,  # V
                "coupling": 0.2,  # 0.05 .. 0.6
                "gap_threshold": 8000.0,  # V breakdown
                "detune": 1.0,  # secondary C multiplier (1 = on tune)
            }
        )
        self._v1: float = 0.0  # primary capacitor voltage
        self._i1: float = 0.0  # primary current
        self._v2: float = 0.0  # secondary (top-load) voltage
        self._i2: float = 0.0  # secondary current
        self._gap_conducting: bool = False
        self._gap_firings: int = 0
        self._secondary_peak: float = 0.0

    @property
    def patent_title(self) -> str:
        return "Electrical Transformer (Tesla coil)"

    # ------------------------------------------------------------------
    # Genesis scene
    # ------------------------------------------------------------------

    def build(self) -> None:
        """Create and build the Genesis scene with the transformer."""
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
                camera_pos=(0.6, -0.6, 0.5),
                camera_lookat=(0.0, 0.0, 0.3),
            ),
            show_viewer=not self.config.headless,
        )
        self._scene.add_entity(gs.morphs.Plane())

        rigid = gs.materials.Rigid(rho=1000.0, friction=0.5)
        # Base table.
        self._scene.add_entity(
            gs.morphs.Box(size=(0.5, 0.4, 0.02), pos=(0.0, 0.0, 0.01), fixed=True),
            material=rigid,
            surface=gs.surfaces.Default(color=(0.4, 0.28, 0.15, 1.0)),
        )
        # Primary: flat spiral approximated by a short wide cylinder.
        self._entities["primary"] = self._scene.add_entity(
            gs.morphs.Cylinder(
                radius=0.16, height=0.03, pos=(0.0, 0.0, 0.045), fixed=True
            ),
            material=rigid,
            surface=gs.surfaces.Default(color=(0.75, 0.45, 0.2, 1.0)),
        )
        # Secondary: tall helical coil (slim cylinder).
        self._entities["secondary"] = self._scene.add_entity(
            gs.morphs.Cylinder(
                radius=0.05, height=0.45, pos=(0.0, 0.0, 0.28), fixed=True
            ),
            material=rigid,
            surface=gs.surfaces.Default(color=(0.7, 0.5, 0.25, 1.0)),
        )
        # Top-load toroid approximated by a sphere.
        self._entities["topload"] = self._scene.add_entity(
            gs.morphs.Sphere(radius=0.09, pos=(0.0, 0.0, 0.55), fixed=True),
            material=rigid,
            surface=gs.surfaces.Default(color=(0.8, 0.8, 0.85, 1.0)),
        )
        # Spark gap: two brass balls on a bar.
        self._scene.add_entity(
            gs.morphs.Box(size=(0.1, 0.02, 0.02), pos=(0.25, 0.1, 0.05), fixed=True),
            material=rigid,
            surface=gs.surfaces.Default(color=(0.72, 0.53, 0.2, 1.0)),
        )
        for x in (0.21, 0.29):
            self._scene.add_entity(
                gs.morphs.Sphere(radius=0.015, pos=(x, 0.1, 0.07), fixed=True),
                material=rigid,
                surface=gs.surfaces.Default(color=(0.8, 0.65, 0.3, 1.0)),
            )
        # Capacitor tank (Leyden-jar box).
        self._scene.add_entity(
            gs.morphs.Box(size=(0.08, 0.08, 0.1), pos=(-0.2, 0.1, 0.07), fixed=True),
            material=rigid,
            surface=gs.surfaces.Default(color=(0.3, 0.3, 0.32, 1.0)),
        )

        # Camera must be added before the scene is built.
        self._camera = self._scene.add_camera(
            pos=(0.6, -0.6, 0.5),
            lookat=(0.0, 0.0, 0.3),
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
        """Reset both circuits to zero energy."""
        if not self._built:
            raise RuntimeError("Simulation has not been built. Call build() first.")
        self._time = 0.0
        self._v1 = 0.0
        self._i1 = 0.0
        self._v2 = 0.0
        self._i2 = 0.0
        self._gap_conducting = False
        self._gap_firings = 0
        self._secondary_peak = 0.0
        return self.get_state()

    def step(self) -> SimState:
        """Advance the coupled oscillators and the scene."""
        if not self._built:
            raise RuntimeError("Simulation has not been built. Call build() first.")
        for _ in range(self.config.substeps):
            self._integrate_circuits(self.config.dt)
            self._scene.step()
        self._time += self.config.dt * self.config.substeps
        return self.get_state()

    def get_state(self) -> SimState:
        """Return the simulation state including transformer metrics."""
        state = super().get_state()
        state.metrics.update(self._compute_metrics())
        return state

    # ------------------------------------------------------------------
    # Physics
    # ------------------------------------------------------------------

    def resonance_hz(self) -> float:
        """Free resonance of the (tuned) primary circuit in Hz."""
        return float(1.0 / (2.0 * np.pi * np.sqrt(self.L_PRIMARY * self.C_PRIMARY)))

    def mutual_inductance(self) -> float:
        """Mutual inductance ``M = k sqrt(L1 L2)`` in H."""
        k = float(np.clip(self._parameters["coupling"], 0.05, 0.6))
        return float(k * np.sqrt(self.L_PRIMARY * self.L_SECONDARY))

    def ideal_gain(self) -> float:
        """Ideal secondary voltage gain ``sqrt(C1 / C2)`` on tune."""
        detune = float(np.clip(self._parameters["detune"], 0.5, 2.0))
        return float(np.sqrt(self.C_PRIMARY / (self.C_SECONDARY * detune)))

    def _system_matrix(self, gap_closed: bool) -> np.ndarray:
        """Build the 4x4 state matrix for one gap state.

        State ``x = [v1, i1, v2, i2]``. With the gap closed the two LC
        circuits are coupled through M; with the gap open the primary
        capacitor recharges from the supply and the secondary rings down
        freely.
        """
        detune = float(np.clip(self._parameters["detune"], 0.5, 2.0))
        c2 = self.C_SECONDARY * detune
        a = np.zeros((4, 4))
        if gap_closed:
            m = self.mutual_inductance()
            det = self.L_PRIMARY * self.L_SECONDARY - m * m
            # v1' = -i1 / C1
            a[0, 1] = -1.0 / self.C_PRIMARY
            # i1' = (L2 (v1 - r1 i1) + M (v2 + r2 i2)) / det
            a[1, 0] = self.L_SECONDARY / det
            a[1, 1] = -self.L_SECONDARY * self.R_PRIMARY / det
            a[1, 2] = m / det
            a[1, 3] = m * self.R_SECONDARY / det
            # v2' = i2 / C2
            a[2, 3] = 1.0 / c2
            # i2' = (-L1 (v2 + r2 i2) - M (v1 - r1 i1)) / det
            a[3, 0] = -m / det
            a[3, 1] = m * self.R_PRIMARY / det
            a[3, 2] = -self.L_PRIMARY / det
            a[3, 3] = -self.L_PRIMARY * self.R_SECONDARY / det
        else:
            # Primary capacitor recharges toward the supply; any residual
            # primary current quenches quickly.
            a[0, 0] = -1.0 / self.CHARGE_TAU
            a[1, 1] = -1.0 / self.QUENCH_TAU
            # Secondary rings down freely (primary loop open: M i1' ~ 0).
            a[2, 3] = 1.0 / c2
            a[3, 2] = -1.0 / self.L_SECONDARY
            a[3, 3] = -self.R_SECONDARY / self.L_SECONDARY
        return a

    def _integrate_circuits(self, dt: float) -> None:
        """Integrate the coupled oscillators over ``dt`` seconds.

        Each gap state is linear time-invariant, so the inner step uses
        the exact matrix exponential; only the state-dependent gap events
        (fire / quench) are sampled at the inner resolution.
        """
        from scipy.linalg import expm

        supply = float(np.clip(self._parameters["supply_voltage"], 0.0, 20000.0))
        gap_v = float(np.clip(self._parameters["gap_threshold"], 100.0, 20000.0))
        inner_dt = min(dt, self.MAX_INNER_DT)
        steps = max(1, int(round(dt / inner_dt)))
        inner_dt = dt / steps

        expm_closed = expm(self._system_matrix(True) * inner_dt)
        expm_open = expm(self._system_matrix(False) * inner_dt)
        # Constant supply input with the gap open: v1' += supply / tau.
        recharge = supply * inner_dt / self.CHARGE_TAU

        state = np.array([self._v1, self._i1, self._v2, self._i2])
        for _ in range(steps):
            if self._gap_conducting:
                state = expm_closed @ state
                # Quench at primary current zero-crossing.
                if abs(state[1]) < self.QUENCH_CURRENT:
                    self._gap_conducting = False
                    state[1] = 0.0
            else:
                state = expm_open @ state
                state[0] += recharge
                if state[0] >= gap_v:
                    self._gap_conducting = True
                    self._gap_firings += 1
            self._secondary_peak = max(self._secondary_peak, abs(state[2]))
        self._v1, self._i1, self._v2, self._i2 = (float(s) for s in state)

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def _compute_metrics(self) -> dict[str, float]:
        """Return current transformer metrics."""
        return {
            "primary_voltage_v": float(self._v1),
            "primary_current_a": float(self._i1),
            "secondary_voltage_v": float(self._v2),
            "secondary_peak_v": float(self._secondary_peak),
            "secondary_peak_kv": float(self._secondary_peak / 1000.0),
            "gap_firings": float(self._gap_firings),
            "gap_conducting": float(self._gap_conducting),
            "resonance_hz": float(self.resonance_hz()),
            "ideal_gain": float(self.ideal_gain()),
            "energy_primary_mj": float(0.5 * self.C_PRIMARY * self._v1**2 * 1e3),
            "energy_secondary_mj": float(0.5 * self.C_SECONDARY * self._v2**2 * 1e3),
        }
