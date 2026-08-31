"""Interactive Genesis simulation of the Rocket Apparatus (US 1,155,986).

Robert Goddard's 1914/1915 patents introduced the bipropellant combustion
chamber fed by pumps, the de Laval (convergent-divergent) nozzle, and
multi-staging: drop the empty lower stage and keep accelerating.

Physics model (lumped, plain Python; Genesis renders the vehicle)
-----------------------------------------------------------------
- Propulsion: thrust ``T = mdot * v_e * throttle`` with constant exhaust
  velocity ``v_e`` (the de Laval nozzle expands the combustion gases to
  supersonic speed); mass decreases as propellant burns, so acceleration
  grows over the burn — the variable-mass rocket equation.
- Aerodynamics: drag ``0.5 * rho(h) * v^2 * C_d * A`` with an
  exponential atmosphere; gravity is held constant (low altitudes).
- Staging: when the lower stage runs dry and ``auto_stage`` is set, the
  empty stage separates and falls away on a ballistic arc while the
  upper stage ignites its own motor. Tsiolkovsky's ideal delta-v,
  ``v_e * ln(m0/m1)``, is reported per stage for comparison with the
  achieved (gravity- and drag-reduced) velocity.
- Gimbal: the nozzle can tilt a few degrees, producing a horizontal
  thrust component and downrange drift (attitude dynamics are out of
  scope for this demonstrator).

Interactive parameters
----------------------
- ``throttle``: engine throttle, 0-1.
- ``gimbal``: nozzle deflection, -1..1 (mapped to +/-8 degrees).
- ``payload_kg``: payload mass on top of stage 2.
- ``auto_stage``: 1 = separate and ignite stage 2 automatically at
  stage-1 burnout; 0 = never stage (single-stage flight).
"""

from __future__ import annotations

import logging

import numpy as np

from cloud_robotics_sim.patents.base import PatentSimConfig, PatentSimulation, SimState
from cloud_robotics_sim.patents.registry import register_patent

logger = logging.getLogger(__name__)

GRAVITY = 9.80665  # m/s^2
RHO_SEA_LEVEL = 1.225  # kg/m^3
SCALE_HEIGHT = 8500.0  # m, exponential atmosphere


@register_patent(
    "US1155986",
    {
        "title": "Rocket Apparatus",
        "inventors": ["Robert H. Goddard"],
        "grant_date": "1915-10-05",
        "breakthrough": "Bipropellant combustion chamber and de Laval nozzle",
    },
)
class RocketApparatusSimulation(PatentSimulation):
    """Genesis simulation of Goddard's two-stage rocket."""

    # Vehicle constants (SI units, demonstrator scale).
    STAGE1_DRY: float = 40.0  # kg
    STAGE1_PROP: float = 120.0  # kg
    STAGE2_DRY: float = 15.0  # kg
    STAGE2_PROP: float = 45.0  # kg
    MDOT_MAX: float = 4.0  # kg/s propellant flow at full throttle
    EXHAUST_VELOCITY: float = 2000.0  # m/s (de Laval nozzle)
    DRAG_COEFF: float = 0.5
    REFERENCE_AREA: float = 0.03  # m^2
    GIMBAL_MAX_DEG: float = 8.0  # degrees of nozzle deflection
    STAGE1_LENGTH: float = 1.6  # m (visual staging offset)

    INNER_DT: float = 5.0e-3  # s, inner integrator resolution

    def __init__(self, config: PatentSimConfig | None = None) -> None:
        super().__init__(config)
        self.config.patent_id = self.config.patent_id or "US1155986"
        self._init_parameters(
            {
                "throttle": 1.0,  # 0 .. 1
                "gimbal": 0.0,  # -1 .. 1
                "payload_kg": 10.0,  # kg
                "auto_stage": 1.0,  # 0/1
            }
        )
        # Flight state.
        self._altitude: float = 0.0
        self._downrange: float = 0.0
        self._v_vertical: float = 0.0
        self._v_horizontal: float = 0.0
        self._prop1: float = self.STAGE1_PROP
        self._prop2: float = self.STAGE2_PROP
        self._separated: bool = False
        self._on_pad: bool = True
        # Separated stage-1 ballistic state.
        self._booster_altitude: float = 0.0
        self._booster_downrange: float = 0.0
        self._booster_vv: float = 0.0
        self._booster_vh: float = 0.0
        self._last_thrust: float = 0.0
        self._last_drag: float = 0.0

    @property
    def patent_title(self) -> str:
        return "Rocket Apparatus (Goddard two-stage)"

    # ------------------------------------------------------------------
    # Genesis scene
    # ------------------------------------------------------------------

    def build(self) -> None:
        """Create and build the Genesis scene with the rocket."""
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
                camera_pos=(8.0, -8.0, 5.0),
                camera_lookat=(0.0, 0.0, 2.0),
            ),
            show_viewer=not self.config.headless,
        )
        self._scene.add_entity(gs.morphs.Plane())

        rigid = gs.materials.Rigid(rho=1000.0, friction=0.5)
        # Launch pad.
        self._scene.add_entity(
            gs.morphs.Box(size=(1.0, 1.0, 0.1), pos=(0.0, 0.0, 0.05), fixed=True),
            material=rigid,
            surface=gs.surfaces.Default(color=(0.35, 0.35, 0.38, 1.0)),
        )
        # Stage 1 (lower, larger) and stage 2 (upper) bodies; both are
        # posed from the lumped flight model every step.
        self._entities["stage1"] = self._scene.add_entity(
            gs.morphs.Cylinder(
                radius=0.15, height=self.STAGE1_LENGTH, pos=(0.0, 0.0, 0.9)
            ),
            material=rigid,
            surface=gs.surfaces.Default(color=(0.85, 0.85, 0.85, 1.0)),
        )
        self._entities["stage2"] = self._scene.add_entity(
            gs.morphs.Cylinder(
                radius=0.11,
                height=0.8,
                pos=(0.0, 0.0, 0.9 + self.STAGE1_LENGTH / 2 + 0.4),
            ),
            material=rigid,
            surface=gs.surfaces.Default(color=(0.75, 0.3, 0.2, 1.0)),
        )
        self._entities["nose"] = self._scene.add_entity(
            gs.morphs.Sphere(
                radius=0.11, pos=(0.0, 0.0, 0.9 + self.STAGE1_LENGTH / 2 + 0.8)
            ),
            material=rigid,
            surface=gs.surfaces.Default(color=(0.9, 0.9, 0.9, 1.0)),
        )

        # Camera must be added before the scene is built.
        self._camera = self._scene.add_camera(
            pos=(8.0, -8.0, 5.0),
            lookat=(0.0, 0.0, 2.0),
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
        """Reset the vehicle to the pad, fully fuelled."""
        if not self._built:
            raise RuntimeError("Simulation has not been built. Call build() first.")
        self._time = 0.0
        self._altitude = 0.0
        self._downrange = 0.0
        self._v_vertical = 0.0
        self._v_horizontal = 0.0
        self._prop1 = self.STAGE1_PROP
        self._prop2 = self.STAGE2_PROP
        self._separated = False
        self._on_pad = True
        self._booster_altitude = 0.0
        self._booster_downrange = 0.0
        self._booster_vv = 0.0
        self._booster_vh = 0.0
        self._last_thrust = 0.0
        self._last_drag = 0.0
        self._pose_vehicles()
        return self.get_state()

    def step(self) -> SimState:
        """Advance the flight dynamics and the scene."""
        if not self._built:
            raise RuntimeError("Simulation has not been built. Call build() first.")
        for _ in range(self.config.substeps):
            self._integrate_flight(self.config.dt)
            self._pose_vehicles()
            self._scene.step()
        self._time += self.config.dt * self.config.substeps
        return self.get_state()

    def get_state(self) -> SimState:
        """Return the simulation state including flight metrics."""
        state = super().get_state()
        state.metrics.update(self._compute_metrics())
        return state

    # ------------------------------------------------------------------
    # Physics
    # ------------------------------------------------------------------

    def total_mass(self) -> float:
        """Current total mass of the flying vehicle in kg."""
        payload = float(np.clip(self._parameters["payload_kg"], 0.0, 100.0))
        mass = self.STAGE2_DRY + self._prop2 + payload
        if not self._separated:
            mass += self.STAGE1_DRY + self._prop1
        return mass

    def active_propellant(self) -> float:
        """Propellant remaining in the currently burning stage."""
        return self._prop1 if not self._separated else self._prop2

    def thrust(self) -> float:
        """Current engine thrust in newtons."""
        if self._on_pad and self.active_propellant() <= 0.0:
            return 0.0
        if self.active_propellant() <= 0.0:
            return 0.0
        throttle = float(np.clip(self._parameters["throttle"], 0.0, 1.0))
        return self.MDOT_MAX * throttle * self.EXHAUST_VELOCITY

    def drag(self, altitude: float, velocity: float) -> float:
        """Aerodynamic drag in newtons (opposing the velocity)."""
        rho = float(RHO_SEA_LEVEL * np.exp(-max(altitude, 0.0) / SCALE_HEIGHT))
        speed = abs(velocity)
        return 0.5 * rho * speed * speed * self.DRAG_COEFF * self.REFERENCE_AREA

    def ideal_delta_v(self, stage: int) -> float:
        """Tsiolkovsky ideal delta-v of the given stage in m/s."""
        payload = float(np.clip(self._parameters["payload_kg"], 0.0, 100.0))
        if stage == 1:
            m0 = (
                self.STAGE1_DRY
                + self.STAGE1_PROP
                + self.STAGE2_DRY
                + self.STAGE2_PROP
                + payload
            )
            m1 = m0 - self.STAGE1_PROP
        else:
            m0 = self.STAGE2_DRY + self.STAGE2_PROP + payload
            m1 = m0 - self.STAGE2_PROP
        return float(self.EXHAUST_VELOCITY * np.log(m0 / m1))

    def _integrate_flight(self, dt: float) -> None:
        """Integrate the variable-mass flight dynamics over ``dt``."""
        inner_dt = min(dt, self.INNER_DT)
        steps = max(1, int(round(dt / inner_dt)))
        inner_dt = dt / steps
        for _ in range(steps):
            thrust = self.thrust()
            gimbal = float(np.clip(self._parameters["gimbal"], -1.0, 1.0))
            tilt = np.deg2rad(self.GIMBAL_MAX_DEG) * gimbal
            mass = self.total_mass()

            # Thrust components (vertical axis plus gimbal tilt).
            t_vert = thrust * np.cos(tilt)
            t_horiz = thrust * np.sin(tilt)
            drag_v = self.drag(self._altitude, self._v_vertical)
            drag_h = self.drag(self._altitude, self._v_horizontal)

            if self._on_pad:
                # Hold-down clamps: lift off only when thrust beats weight.
                if t_vert <= mass * GRAVITY:
                    self._last_thrust = thrust
                    self._last_drag = 0.0
                    self._burn_propellant(inner_dt)
                    continue
                self._on_pad = False

            acc_v = (
                t_vert - mass * GRAVITY - np.sign(self._v_vertical) * drag_v
            ) / mass
            acc_h = (t_horiz - np.sign(self._v_horizontal) * drag_h) / mass
            self._v_vertical += inner_dt * acc_v
            self._v_horizontal += inner_dt * acc_h
            self._altitude += inner_dt * self._v_vertical
            self._downrange += inner_dt * self._v_horizontal
            self._last_thrust = thrust
            self._last_drag = drag_v
            self._burn_propellant(inner_dt)

            # Staging at stage-1 burnout.
            auto = float(self._parameters.get("auto_stage", 1.0)) >= 0.5
            if not self._separated and self._prop1 <= 0.0 and auto:
                self._separate()

            # The separated booster follows a free ballistic arc.
            if self._separated:
                self._integrate_booster(inner_dt)

            # Ground impact ends the flight (and the booster's).
            if self._altitude <= 0.0 and not self._on_pad:
                self._altitude = 0.0
                self._v_vertical = min(self._v_vertical, 0.0)
                self._v_horizontal = 0.0

    def _burn_propellant(self, dt: float) -> None:
        """Consume propellant at the throttled flow rate."""
        throttle = float(np.clip(self._parameters["throttle"], 0.0, 1.0))
        mdot = self.MDOT_MAX * throttle * dt
        if not self._separated:
            self._prop1 = max(self._prop1 - mdot, 0.0)
        else:
            self._prop2 = max(self._prop2 - mdot, 0.0)

    def _separate(self) -> None:
        """Separate the empty first stage and ignite the second."""
        self._separated = True
        self._booster_altitude = self._altitude
        self._booster_downrange = self._downrange
        self._booster_vv = self._v_vertical
        self._booster_vh = self._v_horizontal
        logger.info("Stage separation at %.0f m, %.1f s", self._altitude, self._time)

    def _integrate_booster(self, dt: float) -> None:
        """Ballistic fall of the separated first stage."""
        if self._booster_altitude <= 0.0:
            return
        drag = self.drag(self._booster_altitude, self._booster_vv)
        self._booster_vv += dt * (
            -GRAVITY - np.sign(self._booster_vv) * drag / self.STAGE1_DRY
        )
        self._booster_altitude += dt * self._booster_vv
        self._booster_downrange += dt * self._booster_vh
        if self._booster_altitude < 0.0:
            self._booster_altitude = 0.0
            self._booster_vv = 0.0

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def _pose_vehicles(self) -> None:
        """Write the flight state into the Genesis scene."""
        identity = np.array([1.0, 0.0, 0.0, 0.0])

        def pose(name: str, x: float, z: float) -> None:
            entity = self._entities.get(name)
            if entity is None:
                return
            try:
                entity.set_pos(np.array([x, 0.0, z]))
                entity.set_quat(identity)
                if hasattr(entity, "set_dofs_velocity"):
                    entity.set_dofs_velocity(np.zeros(6))
            except Exception as exc:  # noqa: BLE001 - pose is best-effort
                logger.debug("Could not pose %s: %s", name, exc)

        if self._separated:
            pose("stage1", self._booster_downrange, self._booster_altitude + 0.9)
            upper_alt = self._altitude
            pose("stage2", self._downrange, upper_alt + 0.5)
            pose("nose", self._downrange, upper_alt + 0.95)
        else:
            base = self._altitude
            pose("stage1", self._downrange, base + 0.9)
            pose("stage2", self._downrange, base + 0.9 + self.STAGE1_LENGTH / 2 + 0.4)
            pose("nose", self._downrange, base + 0.9 + self.STAGE1_LENGTH / 2 + 0.85)

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def _compute_metrics(self) -> dict[str, float]:
        """Return current flight metrics."""
        mass = self.total_mass()
        accel = self._last_thrust / mass if mass > 0 else 0.0
        return {
            "altitude_m": float(self._altitude),
            "downrange_m": float(self._downrange),
            "velocity_ms": float(self._v_vertical),
            "velocity_horizontal_ms": float(self._v_horizontal),
            "mass_total_kg": float(mass),
            "thrust_n": float(self._last_thrust),
            "drag_n": float(self._last_drag),
            "acceleration_g": float(accel / GRAVITY),
            "propellant_stage1_kg": float(self._prop1),
            "propellant_stage2_kg": float(self._prop2),
            "thrust_to_weight": float(self._last_thrust / (mass * GRAVITY)),
            "ideal_delta_v_stage1_ms": float(self.ideal_delta_v(1)),
            "ideal_delta_v_stage2_ms": float(self.ideal_delta_v(2)),
            "separated": float(self._separated),
            "on_pad": float(self._on_pad),
        }
