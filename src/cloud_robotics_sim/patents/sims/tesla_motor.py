"""Interactive Genesis simulation of the Electro-Magnetic Motor (US 381,968).

Nikola Tesla's breakthrough: two (or more) alternating currents displaced
in phase produce a *rotating* magnetic field without any commutator or
brushes — the field itself drags the induced rotor around.

Physics model (lumped, plain Python; Genesis renders the machine)
-----------------------------------------------------------------
- Stator: two phase windings in space quadrature fed by
  ``i_a = I sin(w t)`` and ``i_b = I sin(w t - phi)``. Following the
  classical double-revolving-field theory, the air-gap MMF decomposes
  into forward and backward rotating fields whose amplitudes depend on
  the phase offset ``phi``: at ``phi = 90 deg`` the field is purely
  forward-rotating (the patent's polyphase case); at ``phi = 0`` the
  pulsating field splits into equal counter-rotating components whose
  torques cancel at standstill — a single-phase motor cannot self-start.
- Rotor: each revolving field induces currents and produces torque
  following the Kloss curve ``T ~ B^2 * s / (s^2 + s_m^2)`` where ``s``
  is the slip relative to that field (the backward field sees slip
  ``2 - s``).
- Mechanics: ``J dw/dt = T_net - T_load - b * w``.

Interactive parameters
----------------------
- ``frequency``: supply frequency in Hz (sets the synchronous speed).
- ``current``: stator current amplitude, 0-1 per unit (field strength).
- ``phase_offset``: electrical phase displacement between the two
  windings in degrees; 90 is the patent's quarter-phase arrangement, 0
  collapses the field to a pure pulsation (no starting torque), 270
  reverses the rotation.
- ``load``: brake torque, 0-1 per unit.
"""

from __future__ import annotations

import logging

import numpy as np

from cloud_robotics_sim.patents.base import PatentSimConfig, PatentSimulation, SimState
from cloud_robotics_sim.patents.registry import register_patent

logger = logging.getLogger(__name__)


@register_patent(
    "US381968",
    {
        "title": "Electro-Magnetic Motor",
        "inventors": ["Nikola Tesla"],
        "grant_date": "1888-05-01",
        "breakthrough": "Brushless polyphase AC rotating magnetic stator field",
    },
)
class ElectromagneticMotorSimulation(PatentSimulation):
    """Genesis simulation of Tesla's two-phase induction motor."""

    # Machine constants (SI units, demonstrator scale).
    POLE_PAIRS: int = 2  # four-pole stator
    RATED_FREQUENCY: float = 60.0  # Hz
    ROTOR_INERTIA: float = 0.01  # kg m^2
    FRICTION: float = 1.0e-4  # N m s, viscous bearing friction
    TORQUE_CONST: float = 0.5  # N m per (p.u. field)^2 at the Kloss peak
    PULLOUT_SLIP: float = 0.2  # slip at peak torque (s_m)
    LOAD_MAX: float = 0.08  # N m brake torque at load=1

    # Inner integrator resolution for the mechanical ODE.
    INNER_DT: float = 2.0e-3  # s

    def __init__(self, config: PatentSimConfig | None = None) -> None:
        super().__init__(config)
        self.config.patent_id = self.config.patent_id or "US381968"
        self._init_parameters(
            {
                "frequency": self.RATED_FREQUENCY,  # Hz
                "current": 1.0,  # 0 .. 1 per unit
                "phase_offset": 90.0,  # degrees between phases
                "load": 0.2,  # 0 .. 1 per unit brake torque
            }
        )
        self._rotor_speed: float = 0.0  # rad/s mechanical
        self._rotor_angle: float = 0.0  # rad
        self._field_angle: float = 0.0  # rad, forward field position
        self._last_torque: float = 0.0

    @property
    def patent_title(self) -> str:
        return "Electro-Magnetic Motor (Tesla two-phase induction)"

    # ------------------------------------------------------------------
    # Genesis scene
    # ------------------------------------------------------------------

    def build(self) -> None:
        """Create and build the Genesis scene with the motor."""
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
                camera_pos=(0.4, -0.4, 0.35),
                camera_lookat=(0.0, 0.0, 0.12),
            ),
            show_viewer=not self.config.headless,
        )
        self._scene.add_entity(gs.morphs.Plane())

        rigid = gs.materials.Rigid(rho=1000.0, friction=0.5)
        # Mounting plate.
        self._scene.add_entity(
            gs.morphs.Box(size=(0.3, 0.3, 0.02), pos=(0.0, 0.0, 0.01), fixed=True),
            material=rigid,
            surface=gs.surfaces.Default(color=(0.3, 0.3, 0.32, 1.0)),
        )
        # Stator frame: cylinder shell with four pole pieces (two phases in
        # space quadrature). Alternate colors mark the two phases.
        self._entities["stator"] = self._scene.add_entity(
            gs.morphs.Cylinder(
                radius=0.11, height=0.09, pos=(0.0, 0.0, 0.13), fixed=True
            ),
            material=rigid,
            surface=gs.surfaces.Default(color=(0.45, 0.45, 0.5, 1.0)),
        )
        pole_colors = [(0.75, 0.25, 0.2, 1.0), (0.2, 0.4, 0.8, 1.0)]
        for k in range(4):
            angle = k * np.pi / 2
            px, py = 0.095 * np.cos(angle), 0.095 * np.sin(angle)
            self._scene.add_entity(
                gs.morphs.Box(
                    size=(0.035, 0.035, 0.1),
                    pos=(px, py, 0.13),
                    fixed=True,
                ),
                material=rigid,
                surface=gs.surfaces.Default(color=pole_colors[k % 2]),
            )
        # Rotor: conductive drum plus an asymmetric marker bar so rotation
        # is visible. The pose is driven from the lumped mechanical model.
        self._entities["rotor"] = self._scene.add_entity(
            gs.morphs.Cylinder(radius=0.055, height=0.11, pos=(0.0, 0.0, 0.13)),
            material=gs.materials.Rigid(rho=2700.0, friction=0.1),
            surface=gs.surfaces.Default(color=(0.8, 0.65, 0.35, 1.0)),
        )
        self._entities["rotor_marker"] = self._scene.add_entity(
            gs.morphs.Box(size=(0.1, 0.012, 0.012), pos=(0.0, 0.0, 0.19)),
            material=rigid,
            surface=gs.surfaces.Default(color=(0.1, 0.1, 0.1, 1.0)),
        )

        # Camera must be added before the scene is built.
        self._camera = self._scene.add_camera(
            pos=(0.4, -0.4, 0.35),
            lookat=(0.0, 0.0, 0.12),
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
        """Reset the rotor to standstill with the field at angle zero."""
        if not self._built:
            raise RuntimeError("Simulation has not been built. Call build() first.")
        self._time = 0.0
        self._rotor_speed = 0.0
        self._rotor_angle = 0.0
        self._field_angle = 0.0
        self._last_torque = 0.0
        self._pose_rotor()
        return self.get_state()

    def step(self) -> SimState:
        """Advance the electro-mechanical dynamics and the scene."""
        if not self._built:
            raise RuntimeError("Simulation has not been built. Call build() first.")
        for _ in range(self.config.substeps):
            self._integrate_mechanics(self.config.dt)
            self._pose_rotor()
            self._scene.step()
        self._time += self.config.dt * self.config.substeps
        return self.get_state()

    def get_state(self) -> SimState:
        """Return the simulation state including motor metrics."""
        state = super().get_state()
        state.metrics.update(self._compute_metrics())
        return state

    # ------------------------------------------------------------------
    # Physics
    # ------------------------------------------------------------------

    def synchronous_speed(self, frequency: float) -> float:
        """Synchronous field speed ``2 pi f / p`` in rad/s (mechanical)."""
        return 2.0 * np.pi * frequency / self.POLE_PAIRS

    def field_amplitudes(
        self, current: float, phase_offset_deg: float
    ) -> tuple[float, float]:
        """Forward/backward revolving-field amplitudes (double revolving
        field theory): for equal phase currents displaced by ``phi``, the
        forward component is ``I (1 + sin phi) / 2`` and the backward one
        takes the remainder.
        """
        phi = np.deg2rad(phase_offset_deg)
        forward = current * (1.0 + np.sin(phi)) / 2.0
        backward = current - forward
        return float(forward), float(backward)

    def kloss_torque(self, field: float, slip: float) -> float:
        """Induction torque of one revolving field (Kloss curve).

        ``T = K B^2 s / (s^2 + s_m^2)`` — zero at synchronism, peaked at
        the pull-out slip ``s_m``, roughly linear for small slip.
        """
        s_m = self.PULLOUT_SLIP
        return float(
            self.TORQUE_CONST * field * field * slip / (slip * slip + s_m * s_m)
        )

    def electromagnetic_torque(self, rotor_speed: float) -> float:
        """Net electromagnetic torque at the given mechanical speed.

        The forward field sees slip ``s``; the backward field rotates
        against the rotor and sees slip ``2 - s``, braking it.
        """
        current = float(np.clip(self._parameters["current"], 0.0, 1.0))
        phase = float(self._parameters["phase_offset"])
        frequency = float(np.clip(self._parameters["frequency"], 0.0, 200.0))
        w_sync = self.synchronous_speed(frequency)
        if w_sync <= 0.0 or current <= 0.0:
            return 0.0
        fwd, bwd = self.field_amplitudes(current, phase)
        slip = float(np.clip(1.0 - rotor_speed / w_sync, -2.0, 2.0))
        t_fwd = self.kloss_torque(fwd, slip)
        t_bwd = self.kloss_torque(bwd, 2.0 - slip)
        return t_fwd - t_bwd

    def _integrate_mechanics(self, dt: float) -> None:
        """Integrate ``J dw/dt = T - T_load - b w`` over ``dt`` seconds."""
        load = float(np.clip(self._parameters["load"], 0.0, 1.0)) * self.LOAD_MAX
        frequency = float(np.clip(self._parameters["frequency"], 0.0, 200.0))
        inner_dt = min(dt, self.INNER_DT)
        steps = max(1, int(round(dt / inner_dt)))
        inner_dt = dt / steps
        for _ in range(steps):
            torque = self.electromagnetic_torque(self._rotor_speed)
            accel = (torque - self.FRICTION * self._rotor_speed) / self.ROTOR_INERTIA
            speed = self._rotor_speed + inner_dt * accel
            # Coulomb brake: opposes motion but can never reverse it; if
            # the brake would cross zero speed within a step, the rotor
            # sticks (static friction holds it).
            if abs(speed) > 1e-12:
                braked = speed - np.sign(speed) * load * inner_dt / self.ROTOR_INERTIA
                speed = float(braked if braked * speed > 0 else 0.0)
            self._rotor_speed = speed
            self._rotor_angle += inner_dt * self._rotor_speed
            # Track the forward field angle for the metrics/overlay.
            self._field_angle += inner_dt * self.synchronous_speed(frequency)
            self._last_torque = torque

    def _pose_rotor(self) -> None:
        """Write the lumped rotor angle into the Genesis scene."""
        half = self._rotor_angle / 2.0
        quat = np.array([np.cos(half), 0.0, 0.0, np.sin(half)])
        centers = {"rotor": (0.0, 0.0, 0.13), "rotor_marker": (0.0, 0.0, 0.19)}
        for name, center in centers.items():
            entity = self._entities.get(name)
            if entity is None:
                continue
            try:
                entity.set_pos(np.array(center))
                entity.set_quat(quat)
                if hasattr(entity, "set_dofs_velocity"):
                    entity.set_dofs_velocity(np.zeros(6))
            except Exception as exc:  # noqa: BLE001 - pose is best-effort
                logger.debug("Could not pose %s: %s", name, exc)

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def _compute_metrics(self) -> dict[str, float]:
        """Return current motor metrics."""
        current = float(np.clip(self._parameters["current"], 0.0, 1.0))
        phase = float(self._parameters["phase_offset"])
        frequency = float(np.clip(self._parameters["frequency"], 0.0, 200.0))
        w_sync = self.synchronous_speed(frequency)
        fwd, bwd = self.field_amplitudes(current, phase)
        slip = 1.0 - self._rotor_speed / w_sync if w_sync > 0 else 0.0
        rpm = self._rotor_speed * 60.0 / (2.0 * np.pi)
        return {
            "rotor_speed_rpm": float(rpm),
            "synchronous_rpm": float(w_sync * 60.0 / (2.0 * np.pi)),
            "slip": float(slip),
            "torque_nm": float(self._last_torque),
            "mech_power_w": float(self._last_torque * self._rotor_speed),
            "forward_field_pu": fwd,
            "backward_field_pu": bwd,
            "field_angle_deg": float(np.degrees(self._field_angle) % 360.0),
            "rotor_angle_deg": float(np.degrees(self._rotor_angle) % 360.0),
        }
