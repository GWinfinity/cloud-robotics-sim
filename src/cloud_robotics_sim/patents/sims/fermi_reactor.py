"""Interactive Genesis simulation of the Neutronic Reactor (US 2,708,656).

Enrico Fermi and Leo Szilard's patent covers the first self-sustaining
nuclear chain-reacting pile: a heterogeneous uranium/graphite lattice with
neutron-absorbing (cadmium) control rods to regulate the reaction.

Physics model (lumped, plain Python; Genesis renders the pile)
--------------------------------------------------------------
- Point kinetics with six delayed-neutron precursor groups (U-235 thermal
  fission data): ``dn/dt = ((rho - beta) / Lambda) n + sum(lam_i C_i) + S``
  and ``dC_i/dt = (beta_i / Lambda) n - lam_i C_i``, where ``S`` is an
  external start-up source (~1e4 n/s). The prompt neutron generation time
  ``Lambda = 1e-3 s`` is enlarged for demonstration purposes; a real
  thermal reactor has ``Lambda ~ 1e-4 s``.
- Reactivity: ``rho = rho_rod + alpha_T (T_fuel - T_ref)``. The control
  rods contribute ``rho_rod = (rod_position * 3.5 - 3.0) * beta``
  (rod_position 0 = fully inserted, 1 = fully withdrawn; full withdrawal
  leaves only ``+0.5 beta``, below prompt critical). The negative fuel
  temperature coefficient ``alpha_T = -3e-5 /K`` makes the pile
  self-regulating: any power excursion heats the fuel and shuts itself
  down.
- Thermal model: fission power ``P = k_p n`` heats a lumped fuel mass,
  ``C_th dT/dt = P - h * cooling * (T - T_amb)``.
- Integration: the coupled stiff ODE system is advanced with classical
  RK4 at a fixed inner step of 1 ms.

Interactive parameters
----------------------
- ``rod_position``: control-rod withdrawal, 0 (fully inserted) to 1
  (fully withdrawn).
- ``cooling``: coolant flow multiplier, 0.5 to 2.0.
- ``source_strength``: external neutron source in n/s used for start-up.
"""

from __future__ import annotations

import logging

import numpy as np

from cloud_robotics_sim.patents.base import PatentSimConfig, PatentSimulation, SimState
from cloud_robotics_sim.patents.registry import register_patent

logger = logging.getLogger(__name__)


@register_patent(
    "US2708656",
    {
        "title": "Neutronic Reactor",
        "inventors": ["Enrico Fermi", "Leo Szilard"],
        "grant_date": "1955-05-17",
        "breakthrough": "Heterogeneous graphite lattice and cadmium criticality",
    },
)
class NeutronicReactorSimulation(PatentSimulation):
    """Genesis simulation of Fermi and Szilard's neutronic reactor."""

    # Delayed-neutron data for U-235 thermal fission (six groups).
    BETA_I: tuple[float, ...] = (
        2.15e-4,
        1.424e-3,
        1.274e-3,
        2.568e-3,
        7.48e-4,
        2.73e-4,
    )
    LAMBDA_I: tuple[float, ...] = (0.0124, 0.0305, 0.111, 0.301, 1.14, 3.01)  # 1/s
    BETA_TOTAL: float = 6.5e-3  # ~sum(BETA_I)
    GEN_TIME: float = 1.0e-3  # s, prompt generation time (demo-enlarged)

    ALPHA_T: float = -3.0e-5  # 1/K, negative fuel temperature coefficient
    POWER_COEFF: float = 1.0e3  # W per unit neutron population
    HEAT_CAPACITY: float = 5.0e6  # J/K, lumped fuel heat capacity
    COOLING_COEFF: float = 1.0e4  # W/K at cooling = 1
    T_AMBIENT: float = 300.0  # K
    T_REFERENCE: float = 300.0  # K, feedback reference temperature

    INNER_DT: float = 1.0e-3  # s, inner RK4 step for the stiff kinetics

    # Visual layout constants (meters).
    ROD_TRAVEL: float = 1.0  # rod stroke between inserted and withdrawn
    ROD_INSERTED_Z: float = 0.55  # rod center height when fully inserted

    def __init__(self, config: PatentSimConfig | None = None) -> None:
        super().__init__(config)
        self.config.patent_id = self.config.patent_id or "US2708656"
        self._init_parameters(
            {
                "rod_position": 0.3,  # 0 .. 1 (0 = inserted, 1 = withdrawn)
                "cooling": 1.0,  # 0.5 .. 2 coolant flow multiplier
                "source_strength": 1.0e4,  # n/s external start-up source
            }
        )
        # Physics state: neutron population, six precursor concentrations,
        # and the lumped fuel temperature.
        self._neutrons: float = 0.0
        self._precursors: np.ndarray = np.zeros(6)
        self._fuel_temp: float = self.T_AMBIENT
        self._last_dndt: float = 0.0

    @property
    def patent_title(self) -> str:
        return "Neutronic Reactor (Fermi-Szilard pile)"

    # ------------------------------------------------------------------
    # Genesis scene
    # ------------------------------------------------------------------

    def build(self) -> None:
        """Create and build the Genesis scene with the reactor pile."""
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
                camera_pos=(3.5, -3.5, 2.5),
                camera_lookat=(0.0, 0.0, 0.8),
            ),
            show_viewer=not self.config.headless,
        )
        self._scene.add_entity(gs.morphs.Plane())

        rigid = gs.materials.Rigid(rho=1000.0, friction=0.5)
        # Concrete pad.
        self._scene.add_entity(
            gs.morphs.Box(size=(3.0, 3.0, 0.1), pos=(0.0, 0.0, 0.05), fixed=True),
            material=rigid,
            surface=gs.surfaces.Default(color=(0.55, 0.55, 0.55, 1.0)),
        )
        # Graphite moderator core (fixed block).
        self._entities["core"] = self._scene.add_entity(
            gs.morphs.Box(size=(1.2, 1.2, 1.0), pos=(0.0, 0.0, 0.6), fixed=True),
            material=gs.materials.Rigid(rho=1700.0, friction=0.8),
            surface=gs.surfaces.Default(color=(0.25, 0.25, 0.28, 1.0)),
        )
        # Three cadmium control rods; their height follows ``rod_position``.
        for k, x in enumerate((-0.3, 0.0, 0.3)):
            self._entities[f"rod{k}"] = self._scene.add_entity(
                gs.morphs.Cylinder(
                    radius=0.05,
                    height=1.2,
                    pos=(x, 0.0, self.ROD_INSERTED_Z),
                ),
                material=gs.materials.Rigid(rho=8600.0, friction=0.3),
                surface=gs.surfaces.Default(color=(0.75, 0.65, 0.2, 1.0)),
            )

        # Camera must be added before the scene is built.
        self._camera = self._scene.add_camera(
            pos=(3.5, -3.5, 2.5),
            lookat=(0.0, 0.0, 0.8),
            res=self.config.resolution,
            fov=50,
            GUI=False,
        )
        self._scene.build()
        self._built = True

    # ------------------------------------------------------------------
    # Simulation lifecycle
    # ------------------------------------------------------------------

    def reset(self) -> SimState:
        """Reset the pile to a shut-down, ambient-temperature state."""
        if not self._built:
            raise RuntimeError("Simulation has not been built. Call build() first.")
        self._time = 0.0
        self._neutrons = 0.0
        self._precursors = np.zeros(6)
        self._fuel_temp = self.T_AMBIENT
        self._last_dndt = 0.0
        self._pose_rods()
        return self.get_state()

    def step(self) -> SimState:
        """Advance the reactor kinetics and the scene."""
        if not self._built:
            raise RuntimeError("Simulation has not been built. Call build() first.")
        for _ in range(self.config.substeps):
            self._integrate_kinetics(self.config.dt)
            self._pose_rods()
            self._scene.step()
        self._time += self.config.dt * self.config.substeps
        return self.get_state()

    def get_state(self) -> SimState:
        """Return the simulation state including reactor metrics."""
        state = super().get_state()
        state.metrics.update(self._compute_metrics())
        return state

    # ------------------------------------------------------------------
    # Physics
    # ------------------------------------------------------------------

    def reactivity(self, fuel_temp: float | None = None) -> float:
        """Total reactivity in absolute units (delta-k/k).

        Combines the control-rod worth with the negative fuel-temperature
        feedback. Full withdrawal leaves ``+0.5 beta`` — supercritical but
        well below prompt critical.
        """
        rod = float(np.clip(self._parameters["rod_position"], 0.0, 1.0))
        temp = self._fuel_temp if fuel_temp is None else fuel_temp
        rho_rod = (rod * 3.5 - 3.0) * self.BETA_TOTAL
        return float(rho_rod + self.ALPHA_T * (temp - self.T_REFERENCE))

    def _derivatives(self, y: np.ndarray) -> np.ndarray:
        """Right-hand side of the coupled kinetics/thermal ODE system."""
        beta_i = np.asarray(self.BETA_I)
        lam_i = np.asarray(self.LAMBDA_I)
        n = float(y[0])
        precursors = y[1:7]
        temp = float(y[7])
        source = float(max(self._parameters["source_strength"], 0.0))
        cooling = float(np.clip(self._parameters["cooling"], 0.5, 2.0))

        rho = self.reactivity(fuel_temp=temp)
        dndt = ((rho - self.BETA_TOTAL) / self.GEN_TIME) * n
        dndt += float(np.dot(lam_i, precursors)) + source
        dcdt = (beta_i / self.GEN_TIME) * n - lam_i * precursors
        power = self.POWER_COEFF * n
        dtdt = (
            power - self.COOLING_COEFF * cooling * (temp - self.T_AMBIENT)
        ) / self.HEAT_CAPACITY

        out = np.empty(8)
        out[0] = dndt
        out[1:7] = dcdt
        out[7] = dtdt
        return out

    def _integrate_kinetics(self, dt: float) -> None:
        """Advance the point-kinetics and thermal state over ``dt`` (RK4)."""
        inner_dt = min(dt, self.INNER_DT)
        steps = max(1, int(round(dt / inner_dt)))
        inner_dt = dt / steps
        y = np.empty(8)
        y[0] = self._neutrons
        y[1:7] = self._precursors
        y[7] = self._fuel_temp
        for _ in range(steps):
            k1 = self._derivatives(y)
            k2 = self._derivatives(y + 0.5 * inner_dt * k1)
            k3 = self._derivatives(y + 0.5 * inner_dt * k2)
            k4 = self._derivatives(y + inner_dt * k3)
            y = y + (inner_dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            # Neutron populations and temperature stay non-negative.
            y[0:7] = np.maximum(y[0:7], 0.0)
        self._neutrons = float(y[0])
        self._precursors = y[1:7]
        self._fuel_temp = float(y[7])
        self._last_dndt = float(self._derivatives(y)[0])

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def _pose_rods(self) -> None:
        """Write the control-rod withdrawal into the Genesis scene."""
        rod = float(np.clip(self._parameters["rod_position"], 0.0, 1.0))
        z = self.ROD_INSERTED_Z + rod * self.ROD_TRAVEL
        identity = np.array([1.0, 0.0, 0.0, 0.0])
        for k, x in enumerate((-0.3, 0.0, 0.3)):
            entity = self._entities.get(f"rod{k}")
            if entity is None:
                continue
            try:
                entity.set_pos(np.array([x, 0.0, z]))
                entity.set_quat(identity)
                if hasattr(entity, "set_dofs_velocity"):
                    entity.set_dofs_velocity(np.zeros(6))
            except Exception as exc:  # noqa: BLE001 - pose is best-effort
                logger.debug("Could not pose rod%d: %s", k, exc)

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def _compute_metrics(self) -> dict[str, float]:
        """Return current reactor metrics."""
        rho = self.reactivity()
        power = self.POWER_COEFF * self._neutrons
        if self._last_dndt > 1e-30 and self._neutrons > 0.0:
            period = self._neutrons / self._last_dndt
        else:
            period = float("inf")
        return {
            "power_w": float(power),
            "neutron_level": float(self._neutrons),
            "period_s": float(period),
            "reactivity_pcm": float(rho * 1.0e5),
            "reactivity_dollars": float(rho / self.BETA_TOTAL),
            "fuel_temp_k": float(self._fuel_temp),
            "precursor_total": float(np.sum(self._precursors)),
            "rod_position": float(self._parameters["rod_position"]),
        }
