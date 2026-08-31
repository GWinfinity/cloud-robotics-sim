"""Interactive Genesis simulation of Method of Treating Foodstuffs (US 2,495,429).

Percy Spencer's Raytheon patent turned the cavity magnetron — a World War II
radar tube — into the microwave oven: microwaves at 2.45 GHz are fed into a
sealed metal cavity, where they form standing waves and heat food
dielectrically from within. The patent already notes the two practical
problems this creates: hot/cold spots at the standing-wave antinodes/nodes,
and the boiling plateau while free water evaporates.

Physics model (lumped, plain Python; Genesis provides the scene)
----------------------------------------------------------------
- Magnetron: input power ``P_in = power_w`` watts (duty cycle simplified to
  1.0; the 2.45 GHz carrier itself is not resolved). A fixed fraction
  ``ETA`` of the input power is absorbed by the food.
- Cavity standing wave: the food is divided into a 3x3 grid of lumped
  cells. Cell ``i`` absorbs ``P_i = ETA * P_in * w_i / sum(w)`` with the
  ``sin^2`` mode pattern ``w_i = 1 + 2 * sin^2(pi*x) * sin^2(pi*y)``, so the
  central antinode absorbs about twice the power of the corner nodes.
- Cell heat balance: ``m_i * c * dT_i/dt = P_i - h * (T_i - T_amb)`` with a
  heat capacity that follows the remaining water content,
  ``C_i = m_dry * c_dry + m_water * c_water``.
- Evaporation: once a cell reaches 100 C, excess power evaporates water
  (``L_v = 2260 kJ/kg``) instead of raising the temperature — the cell is
  pinned at 100 C until dry, then resumes heating (and eventually chars;
  chemistry beyond drying is out of scope).
- Turntable: with ``turntable_on`` the cells cycle through the standing-wave
  pattern, so every cell sees the *mean* weight ``w_mean`` and the absorbed
  power equalizes; with the turntable off each cell keeps its fixed ``w_i``.

Interactive parameters
----------------------
- ``power_w``: magnetron input power in watts (0-1200).
- ``turntable_on``: 1 = rotate the food tray, 0 = fixed position.
- ``food_mass_g``: total food mass in grams (50-500), 60% water by mass.
"""

from __future__ import annotations

import logging

import numpy as np

from cloud_robotics_sim.patents.base import PatentSimConfig, PatentSimulation, SimState
from cloud_robotics_sim.patents.registry import register_patent

logger = logging.getLogger(__name__)

GRID = 3  # food is split into a GRID x GRID lattice of lumped cells
N_CELLS = GRID * GRID


@register_patent(
    "US2495429",
    {
        "title": "Method of Treating Foodstuffs",
        "inventors": ["Percy L. Spencer"],
        "grant_date": "1950-01-24",
        "breakthrough": "Cavity magnetron dielectric microwave heating",
    },
)
class MethodOfTreatingFoodstuffsSimulation(PatentSimulation):
    """Genesis simulation of Spencer's microwave oven."""

    # Magnetron constants (demonstrator scale).
    ETA: float = 0.65  # fraction of input power absorbed by the food

    # Thermal constants (SI units).
    T_AMBIENT_C: float = 25.0  # deg C
    BOILING_C: float = 100.0  # deg C, water evaporation plateau
    H_CELL: float = 0.2  # W/K, convective loss per cell to the cavity air
    C_WATER: float = 4186.0  # J/(kg K)
    C_DRY: float = 1500.0  # J/(kg K), dry food matter
    WATER_FRACTION: float = 0.6  # initial water mass fraction of the food
    LATENT_HEAT: float = 2.26e6  # J/kg, vaporization of water

    # Turntable rotation rate for the visual tray (~6 rpm, like a real oven).
    TURNTABLE_RATE: float = 0.6  # rad/s

    # Inner integrator resolution for the thermal ODE.
    INNER_DT: float = 0.5  # s

    def __init__(self, config: PatentSimConfig | None = None) -> None:
        super().__init__(config)
        self.config.patent_id = self.config.patent_id or "US2495429"
        self._init_parameters(
            {
                "power_w": 800.0,  # W, magnetron input power
                "turntable_on": 1.0,  # 0/1
                "food_mass_g": 300.0,  # g, total food mass
            }
        )
        # Physics state.
        self._temperatures: np.ndarray = np.full(N_CELLS, self.T_AMBIENT_C)
        self._water_g: np.ndarray = np.full(N_CELLS, self._initial_water_per_cell_g())
        self._energy_j: float = 0.0
        self._turntable_angle: float = 0.0

    @property
    def patent_title(self) -> str:
        return "Method of Treating Foodstuffs (Spencer microwave oven)"

    # ------------------------------------------------------------------
    # Genesis scene
    # ------------------------------------------------------------------

    def build(self) -> None:
        """Create and build the Genesis scene with the microwave oven."""
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
                camera_pos=(0.55, -0.55, 0.4),
                camera_lookat=(0.0, 0.0, 0.2),
            ),
            show_viewer=not self.config.headless,
        )
        self._scene.add_entity(gs.morphs.Plane())

        rigid = gs.materials.Rigid(rho=1000.0, friction=0.5)
        steel = gs.surfaces.Default(color=(0.55, 0.56, 0.6, 1.0))
        # Oven cavity: floor, back wall, side walls and ceiling form the
        # metal box that confines the standing wave. All fixed.
        self._scene.add_entity(
            gs.morphs.Box(size=(0.5, 0.5, 0.02), pos=(0.0, 0.0, 0.09), fixed=True),
            material=rigid,
            surface=steel,
        )
        self._scene.add_entity(
            gs.morphs.Box(size=(0.5, 0.02, 0.32), pos=(0.0, 0.24, 0.25), fixed=True),
            material=rigid,
            surface=steel,
        )
        for x in (-0.24, 0.24):
            self._scene.add_entity(
                gs.morphs.Box(size=(0.02, 0.5, 0.32), pos=(x, 0.0, 0.25), fixed=True),
                material=rigid,
                surface=steel,
            )
        self._scene.add_entity(
            gs.morphs.Box(size=(0.5, 0.5, 0.02), pos=(0.0, 0.0, 0.4), fixed=True),
            material=rigid,
            surface=steel,
        )
        # Magnetron waveguide stub on the ceiling.
        self._scene.add_entity(
            gs.morphs.Box(size=(0.08, 0.08, 0.06), pos=(0.15, 0.15, 0.43), fixed=True),
            material=rigid,
            surface=gs.surfaces.Default(color=(0.35, 0.35, 0.38, 1.0)),
        )

        # Turntable tray and the food block are posed from the lumped
        # thermal model every step (the tray spins when enabled).
        self._entities["turntable"] = self._scene.add_entity(
            gs.morphs.Cylinder(radius=0.16, height=0.015, pos=(0.0, 0.0, 0.11)),
            material=rigid,
            surface=gs.surfaces.Default(color=(0.9, 0.9, 0.92, 1.0)),
        )
        self._entities["food"] = self._scene.add_entity(
            gs.morphs.Box(size=(0.18, 0.18, 0.08), pos=(0.0, 0.0, 0.16)),
            material=rigid,
            surface=gs.surfaces.Default(color=(0.85, 0.55, 0.3, 1.0)),
        )

        # Camera must be added before the scene is built.
        self._camera = self._scene.add_camera(
            pos=(0.55, -0.55, 0.4),
            lookat=(0.0, 0.0, 0.2),
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
        """Reset the food to ambient temperature with full water content."""
        if not self._built:
            raise RuntimeError("Simulation has not been built. Call build() first.")
        self._time = 0.0
        self._temperatures = np.full(N_CELLS, self.T_AMBIENT_C)
        self._water_g = np.full(N_CELLS, self._initial_water_per_cell_g())
        self._energy_j = 0.0
        self._turntable_angle = 0.0
        self._pose_scene()
        return self.get_state()

    def step(self) -> SimState:
        """Advance the thermal dynamics and the Genesis scene."""
        if not self._built:
            raise RuntimeError("Simulation has not been built. Call build() first.")
        for _ in range(self.config.substeps):
            self._integrate_heating(self.config.dt)
            self._pose_scene()
            self._scene.step()
        self._time += self.config.dt * self.config.substeps
        return self.get_state()

    def get_state(self) -> SimState:
        """Return the simulation state including oven metrics."""
        state = super().get_state()
        state.metrics.update(self._compute_metrics())
        return state

    # ------------------------------------------------------------------
    # Physics
    # ------------------------------------------------------------------

    @staticmethod
    def standing_wave_weights() -> np.ndarray:
        """Standing-wave absorption pattern of the cavity (3x3 grid).

        Returns:
            Raw weights ``1 + 2 * sin^2(pi*x) * sin^2(pi*y)``; the central
            antinode (weight 3) absorbs twice the power of the corner nodes
            (weight 1.5), with edge midpoints in between (weight 2).
        """
        axis = np.sin(np.pi * (np.arange(GRID) + 1.0) / (GRID + 1.0)) ** 2
        return (1.0 + 2.0 * np.outer(axis, axis)).ravel()

    def effective_weights(self) -> np.ndarray:
        """Per-cell absorption weights after turntable averaging.

        With the turntable on, every cell cycles through the standing-wave
        pattern and sees the long-term mean weight; with it off, each cell
        keeps its fixed position weight.
        """
        pattern = self.standing_wave_weights()
        turntable = float(self._parameters["turntable_on"]) >= 0.5
        if turntable:
            return np.full(N_CELLS, float(np.mean(pattern)))
        return pattern

    def absorbed_powers(self) -> np.ndarray:
        """Per-cell absorbed microwave power in watts."""
        power_w = float(np.clip(self._parameters["power_w"], 0.0, 1200.0))
        weights = self.effective_weights()
        return self.ETA * power_w * weights / float(np.sum(weights))

    def _food_mass_kg(self) -> float:
        """Total food mass in kg from the ``food_mass_g`` parameter."""
        return float(np.clip(self._parameters["food_mass_g"], 50.0, 500.0)) / 1000.0

    def _initial_water_per_cell_g(self) -> float:
        """Initial water mass per cell in grams."""
        return self._food_mass_kg() * self.WATER_FRACTION / N_CELLS * 1000.0

    def _integrate_heating(self, dt: float) -> None:
        """Integrate the cell heat/evaporation balance over ``dt`` seconds.

        Uses a fixed-step inner loop so the evaporative boiling plateau is
        resolved cleanly at any outer ``config.dt``.
        """
        powers = self.absorbed_powers()
        cell_mass_kg = self._food_mass_kg() / N_CELLS
        inner_dt = min(dt, self.INNER_DT)
        steps = max(1, int(round(dt / inner_dt)))
        inner_dt = dt / steps
        for _ in range(steps):
            self._energy_j += float(np.sum(powers)) * inner_dt
            if float(self._parameters["turntable_on"]) >= 0.5:
                self._turntable_angle += self.TURNTABLE_RATE * inner_dt
            for i in range(N_CELLS):
                water_kg = self._water_g[i] / 1000.0
                dry_kg = max(
                    cell_mass_kg - self._initial_water_per_cell_g() / 1000.0, 0.0
                )
                heat_cap = dry_kg * self.C_DRY + water_kg * self.C_WATER
                temp = float(self._temperatures[i])
                net = float(powers[i]) - self.H_CELL * (temp - self.T_AMBIENT_C)
                energy = net * inner_dt
                new_temp = temp + energy / heat_cap
                if water_kg > 0.0 and new_temp > self.BOILING_C:
                    # Reaching the boiling plateau: energy beyond what is
                    # needed to hit 100 C evaporates water instead.
                    e_boil = max((self.BOILING_C - temp), 0.0) * heat_cap
                    e_evap = max(energy - e_boil, 0.0)
                    evaporated = e_evap / self.LATENT_HEAT
                    if evaporated >= water_kg:
                        # The cell dries out mid-step; the leftover energy
                        # heats the remaining dry matter past 100 C.
                        leftover = (evaporated - water_kg) * self.LATENT_HEAT
                        self._water_g[i] = 0.0
                        new_temp = self.BOILING_C + leftover / (dry_kg * self.C_DRY)
                    else:
                        self._water_g[i] -= evaporated * 1000.0
                        new_temp = self.BOILING_C
                self._temperatures[i] = max(new_temp, self.T_AMBIENT_C)

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def _pose_scene(self) -> None:
        """Write the turntable rotation into the Genesis scene."""
        quat = np.array(
            [
                np.cos(self._turntable_angle / 2.0),
                0.0,
                0.0,
                np.sin(self._turntable_angle / 2.0),
            ]
        )
        poses = {
            "turntable": np.array([0.0, 0.0, 0.11]),
            "food": np.array([0.0, 0.0, 0.16]),
        }
        for name, pos in poses.items():
            entity = self._entities.get(name)
            if entity is None:
                continue
            try:
                entity.set_pos(pos)
                entity.set_quat(quat)
                if hasattr(entity, "set_dofs_velocity"):
                    entity.set_dofs_velocity(np.zeros(6))
            except Exception as exc:  # noqa: BLE001 - pose is best-effort
                logger.debug("Could not pose %s: %s", name, exc)

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def _compute_metrics(self) -> dict[str, float]:
        """Return current oven metrics."""
        temps = self._temperatures
        t_max = float(np.max(temps))
        t_min = float(np.min(temps))
        uniformity = 1.0 - (t_max - t_min) / max(t_max, 1.0)
        return {
            "t_mean_c": float(np.mean(temps)),
            "t_max_c": t_max,
            "t_min_c": t_min,
            "uniformity": float(uniformity),
            "water_remaining_g": float(np.sum(self._water_g)),
            "energy_kj": float(self._energy_j / 1000.0),
            "magnetron_power_w": float(
                np.clip(self._parameters["power_w"], 0.0, 1200.0)
            ),
            "absorbed_power_w": float(np.sum(self.absorbed_powers())),
            "turntable_on": float(float(self._parameters["turntable_on"]) >= 0.5),
        }
