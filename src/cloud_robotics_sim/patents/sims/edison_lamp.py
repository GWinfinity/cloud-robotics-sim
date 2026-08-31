"""Interactive Genesis simulation of the Electric-Lamp (US 223,898).

Thomas Edison's breakthrough was a *high-resistance* carbon filament in a
high vacuum: thin enough to reach incandescence at practical voltages,
and resistive enough that many lamps could run in parallel on one
distribution network.

Physics model (lumped, plain Python; Genesis provides the scene)
----------------------------------------------------------------
- Electrical: Joule heating ``P = V^2 / R(T)``. Carbon has a *negative*
  temperature coefficient of resistance, ``R(T) = R0 * (1 + alpha *
  (T - T0))`` with ``alpha < 0`` — watch the resistance fall as the
  filament heats, the opposite of modern tungsten.
- Thermal balance: ``C dT/dt = P_elec - P_rad - P_cond``. Radiation
  follows Stefan-Boltzmann, ``eps * sigma * A * (T^4 - Tamb^4)``.
  Conduction through the leads and convection by residual gas are
  combined into ``(k_lead + (1 - vacuum) * k_gas) * (T - Tamb)``, so a
  leaky bulb runs dimmer at the same voltage.
- Lifetime: carbon sublimation follows an Arrhenius law,
  ``w(T) = w_ref * exp(Ea/R * (1/T_ref - 1/T))`` — a few hundred kelvin
  hotter trades weeks of life for brightness. Oxidation in a poor vacuum
  multiplies the rate. At ``wear = 1`` the filament burns out and the
  circuit opens.
- Light: luminous flux is computed by integrating the Planck spectrum
  against the photopic response (380-780 nm), so luminous efficacy rises
  steeply with temperature — the reason Edison ran the filament as hot
  as his vacuum pumps allowed.

Interactive parameters
----------------------
- ``voltage``: supply voltage in volts (0-120).
- ``vacuum``: bulb vacuum quality, 1 = high vacuum, 0 = full air leak.
"""

from __future__ import annotations

import logging

import numpy as np

from cloud_robotics_sim.patents.base import PatentSimConfig, PatentSimulation, SimState
from cloud_robotics_sim.patents.registry import register_patent

logger = logging.getLogger(__name__)

STEFAN_BOLTZMANN = 5.670374419e-8  # W / (m^2 K^4)
PLANCK_H = 6.62607015e-34
BOLTZMANN_K = 1.380649e-23
LIGHT_SPEED = 2.99792458e8
LUMINOUS_EFFICACY_MAX = 683.0  # lm / W at 555 nm


@register_patent(
    "US223898",
    {
        "title": "Electric-Lamp",
        "inventors": ["Thomas A. Edison"],
        "grant_date": "1880-01-27",
        "breakthrough": "High-resistance carbon filament in high vacuum",
    },
)
class ElectriclampSimulation(PatentSimulation):
    """Genesis simulation of Edison's carbon-filament lamp."""

    # Electrical constants (SI units, roughly 1880 Edison lamp scale).
    V_RATED: float = 100.0  # V, dynamo supply
    R_COLD: float = 450.0  # ohm at T_AMBIENT (the high-resistance filament)
    TCR_ALPHA: float = -3.0e-4  # 1/K, negative for carbon

    # Thermal constants.
    T_AMBIENT: float = 300.0  # K
    FILAMENT_AREA: float = 4.0e-5  # m^2 radiating surface
    EMISSIVITY: float = 0.8  # carbonized bamboo
    # Effective heat capacity of filament + supports, chosen so the
    # thermal time constant lands at ~10 ms for demonstration stepping.
    HEAT_CAPACITY: float = 9.4e-4  # J/K
    K_LEAD: float = 3.0e-3  # W/K, conduction through the platinum leads
    K_GAS: float = 0.05  # W/K, convection by residual gas at zero vacuum

    # Lifetime constants: wear rate w_ref at T_REF, Arrhenius slope.
    # w(2400 K) = 2.78e-6 /s corresponds to a ~100 h rated lifetime.
    WEAR_REF: float = 2.78e-6  # 1/s at T_REF
    T_REF: float = 2400.0  # K
    EA_OVER_R: float = 84_000.0  # K (carbon sublimation ~700 kJ/mol)
    OXIDATION_MULT: float = 50.0  # extra wear factor at zero vacuum

    # Inner integrator resolution for the stiff thermal ODE.
    INNER_DT: float = 5.0e-4  # s

    _efficacy_table: tuple[np.ndarray, np.ndarray] | None = None

    def __init__(self, config: PatentSimConfig | None = None) -> None:
        super().__init__(config)
        self.config.patent_id = self.config.patent_id or "US223898"
        self._init_parameters(
            {
                "voltage": self.V_RATED,  # V, dynamo supply voltage
                "vacuum": 1.0,  # 0 .. 1, bulb vacuum quality
            }
        )
        self._temperature: float = self.T_AMBIENT
        self._wear: float = 0.0
        self._burned_out: bool = False

    @property
    def patent_title(self) -> str:
        return "Electric-Lamp (Edison carbon filament)"

    # ------------------------------------------------------------------
    # Genesis scene
    # ------------------------------------------------------------------

    def build(self) -> None:
        """Create and build the Genesis scene with the lamp."""
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
                camera_pos=(0.35, -0.35, 0.25),
                camera_lookat=(0.0, 0.0, 0.12),
            ),
            show_viewer=not self.config.headless,
        )
        self._scene.add_entity(gs.morphs.Plane())

        rigid = gs.materials.Rigid(rho=1000.0, friction=0.5)
        # Screw base (brass).
        self._entities["base"] = self._scene.add_entity(
            gs.morphs.Cylinder(
                radius=0.025, height=0.05, pos=(0.0, 0.0, 0.045), fixed=True
            ),
            material=rigid,
            surface=gs.surfaces.Default(color=(0.72, 0.53, 0.2, 1.0)),
        )
        # Glass neck above the base.
        self._scene.add_entity(
            gs.morphs.Cylinder(
                radius=0.02, height=0.04, pos=(0.0, 0.0, 0.09), fixed=True
            ),
            material=rigid,
            surface=gs.surfaces.Default(color=(0.85, 0.92, 1.0, 0.4)),
        )
        # Bulb envelope (transparent glass).
        self._entities["bulb"] = self._scene.add_entity(
            gs.morphs.Sphere(radius=0.06, pos=(0.0, 0.0, 0.17), fixed=True),
            material=rigid,
            surface=gs.surfaces.Default(color=(0.85, 0.92, 1.0, 0.25)),
        )
        # Platinum lead wires.
        for x in (-0.008, 0.008):
            self._scene.add_entity(
                gs.morphs.Cylinder(
                    radius=0.001, height=0.05, pos=(x, 0.0, 0.135), fixed=True
                ),
                material=rigid,
                surface=gs.surfaces.Default(color=(0.8, 0.8, 0.85, 1.0)),
            )
        # Carbon filament: hairpin approximated by a slim horizontal bar
        # joining the lead tips.
        self._entities["filament"] = self._scene.add_entity(
            gs.morphs.Box(
                size=(0.018, 0.0016, 0.0016), pos=(0.0, 0.0, 0.16), fixed=True
            ),
            material=rigid,
            surface=gs.surfaces.Default(color=(0.15, 0.1, 0.08, 1.0)),
        )

        # Camera must be added before the scene is built.
        self._camera = self._scene.add_camera(
            pos=(0.35, -0.35, 0.25),
            lookat=(0.0, 0.0, 0.14),
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
        """Reset the filament to ambient temperature with no wear."""
        if not self._built:
            raise RuntimeError("Simulation has not been built. Call build() first.")
        self._time = 0.0
        self._temperature = self.T_AMBIENT
        self._wear = 0.0
        self._burned_out = False
        return self.get_state()

    def step(self) -> SimState:
        """Advance the thermal dynamics and the Genesis scene."""
        if not self._built:
            raise RuntimeError("Simulation has not been built. Call build() first.")
        for _ in range(self.config.substeps):
            self._integrate_thermal(self.config.dt)
            self._scene.step()
        self._time += self.config.dt * self.config.substeps
        return self.get_state()

    def get_state(self) -> SimState:
        """Return the simulation state including lamp metrics."""
        state = super().get_state()
        state.metrics.update(self._compute_metrics())
        return state

    # ------------------------------------------------------------------
    # Physics
    # ------------------------------------------------------------------

    def resistance(self, temperature: float | None = None) -> float:
        """Filament resistance in ohms (negative carbon TCR)."""
        t = self._temperature if temperature is None else temperature
        return self.R_COLD * (1.0 + self.TCR_ALPHA * (t - self.T_AMBIENT))

    def input_power(self, voltage: float, temperature: float) -> float:
        """Joule heating ``V^2 / R(T)`` in watts; zero once burned out."""
        if self._burned_out or voltage <= 0.0:
            return 0.0
        return voltage * voltage / self.resistance(temperature)

    def radiated_power(self, temperature: float) -> float:
        """Stefan-Boltzmann radiative loss in watts."""
        return (
            self.EMISSIVITY
            * STEFAN_BOLTZMANN
            * self.FILAMENT_AREA
            * (temperature**4 - self.T_AMBIENT**4)
        )

    def conducted_power(self, temperature: float, vacuum: float) -> float:
        """Lead conduction plus residual-gas convection loss in watts."""
        k = self.K_LEAD + (1.0 - vacuum) * self.K_GAS
        return k * (temperature - self.T_AMBIENT)

    def wear_rate(self, temperature: float, vacuum: float) -> float:
        """Fractional carbon wear per second (Arrhenius + oxidation)."""
        if temperature <= 0.0:
            return 0.0
        arrhenius = np.exp(self.EA_OVER_R * (1.0 / self.T_REF - 1.0 / temperature))
        oxidation = 1.0 + self.OXIDATION_MULT * (1.0 - vacuum)
        return float(self.WEAR_REF * arrhenius * oxidation)

    def _integrate_thermal(self, dt: float) -> None:
        """Integrate the lumped thermal ODE over ``dt`` seconds.

        Uses a fixed-step semi-implicit inner loop sized to the thermal
        time constant so the stiff ``T^4`` radiation term stays stable at
        the outer ``config.dt`` resolution.
        """
        voltage = float(np.clip(self._parameters["voltage"], 0.0, 120.0))
        vacuum = float(np.clip(self._parameters["vacuum"], 0.0, 1.0))
        t = self._temperature
        inner_dt = min(dt, self.INNER_DT)
        steps = max(1, int(round(dt / inner_dt)))
        inner_dt = dt / steps
        for _ in range(steps):
            net = (
                self.input_power(voltage, t)
                - self.radiated_power(t)
                - self.conducted_power(t, vacuum)
            )
            t += inner_dt * net / self.HEAT_CAPACITY
            t = max(t, self.T_AMBIENT)
            if not self._burned_out:
                self._wear += self.wear_rate(t, vacuum) * inner_dt
                if self._wear >= 1.0:
                    self._wear = 1.0
                    self._burned_out = True
                    logger.info("Filament burned out after %.2f h", self._time / 3600)
        self._temperature = t

    # ------------------------------------------------------------------
    # Blackbody light output
    # ------------------------------------------------------------------

    @classmethod
    def _luminous_efficacy_table(cls) -> tuple[np.ndarray, np.ndarray]:
        """Precompute blackbody luminous efficacy (lm/W radiated) vs T.

        Integrates the Planck spectrum against a Gaussian approximation
        of the photopic response V(lambda) centered at 555 nm.
        """
        if cls._efficacy_table is not None:
            return cls._efficacy_table
        wavelengths = np.linspace(1.0e-7, 3.0e-6, 400)  # m
        photopic = np.exp(-0.5 * ((wavelengths - 555e-9) / 42e-9) ** 2)
        photopic[wavelengths < 380e-9] = 0.0
        photopic[wavelengths > 780e-9] = 0.0
        temps = np.arange(600.0, 4001.0, 50.0)
        c2 = PLANCK_H * LIGHT_SPEED / BOLTZMANN_K
        prefactor = 2.0 * PLANCK_H * LIGHT_SPEED**2 / wavelengths**5
        efficacy = np.zeros_like(temps)
        for i, t in enumerate(temps):
            b = prefactor / np.expm1(c2 / (wavelengths * t))
            total = np.trapezoid(b, wavelengths)
            visible = np.trapezoid(b * photopic, wavelengths)
            efficacy[i] = LUMINOUS_EFFICACY_MAX * visible / total
        cls._efficacy_table = (temps, efficacy)
        return cls._efficacy_table

    def luminous_flux(self, temperature: float | None = None) -> float:
        """Luminous flux in lumens from the radiated blackbody power."""
        t = self._temperature if temperature is None else temperature
        temps, efficacy = self._luminous_efficacy_table()
        eff = float(np.interp(t, temps, efficacy))
        return eff * max(self.radiated_power(t), 0.0)

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def _compute_metrics(self) -> dict[str, float]:
        """Return current lamp metrics."""
        voltage = float(np.clip(self._parameters["voltage"], 0.0, 120.0))
        vacuum = float(np.clip(self._parameters["vacuum"], 0.0, 1.0))
        p_in = self.input_power(voltage, self._temperature)
        p_rad = max(self.radiated_power(self._temperature), 0.0)
        lumens = self.luminous_flux()
        rate = self.wear_rate(self._temperature, vacuum)
        remaining_h = (1.0 - self._wear) / rate / 3600.0 if rate > 0 else float("inf")
        return {
            "filament_temp_k": float(self._temperature),
            "resistance_ohm": float(self.resistance()),
            "input_power_w": float(p_in),
            "radiated_power_w": float(p_rad),
            "conducted_power_w": float(self.conducted_power(self._temperature, vacuum)),
            "luminous_flux_lm": float(lumens),
            "efficacy_lm_per_w": float(lumens / p_in) if p_in > 0 else 0.0,
            "wear": float(self._wear),
            "lifetime_hours_remaining": float(remaining_h),
            "burned_out": float(self._burned_out),
        }
