"""Interactive Genesis simulation of the Wright Flyer (US 821,393).

This simulation models the 1903 Wright Flyer as a rigid-body aircraft with
simplified lumped aerodynamics. The pilot can adjust wing warp, rudder,
elevator, thrust, and wind speed to observe pitch, roll, yaw, and adverse yaw
behavior.

Aerodynamic model
-----------------
- Body-frame velocity is computed from the world velocity and orientation.
- Angle of attack ``alpha`` and sideslip ``beta`` drive lift, drag, and side
  force coefficients.
- Wing warp creates differential lift between the left and right wings,
  generating a roll moment and (through differential induced drag) a yaw
  moment.
- The elevator produces a pitch moment; the rudder produces a yaw moment.
  Following Claim 18 of the patent, the rudder is by default chained to the
  wing-warping cradle (``coupled=1``), so a warp input automatically commands
  a proportional rudder deflection; set ``coupled=0`` for independent rudder.
- Thrust is applied at the propeller line.
- Gravity and a configurable ambient wind complete the force balance.

The model is intentionally simple: it demonstrates the 3-axis flight-control
principle from the patent without requiring a full CFD solver.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from cloud_robotics_sim.patents.base import PatentSimConfig, PatentSimulation, SimState
from cloud_robotics_sim.patents.registry import register_patent

logger = logging.getLogger(__name__)

# Smithsonian NASM scan of the 1903 Wright Flyer (CC0), reused from the
# classic-patents.com public models. Falls back to a box when the asset is
# not present locally.
WRIGHT_FLYER_MESH = (
    Path(__file__).resolve().parents[4]
    / "assets"
    / "patents"
    / "wright_flyer"
    / "smithsonian-nasm-1903-flyer.stl"
)


@register_patent(
    "US821393",
    {
        "title": "Flying-Machine",
        "inventors": ["Orville Wright", "Wilbur Wright"],
        "grant_date": "1906-05-22",
        "breakthrough": "3-axis aerodynamic flight control via wing warping",
    },
)
class WrightFlyerSimulation(PatentSimulation):
    """Genesis simulation of the Wright Flyer with interactive 3-axis controls."""

    # Aircraft reference parameters (SI units, roughly Wright Flyer scale).
    MASS: float = 340.0  # kg (Flyer + pilot)
    WING_SPAN: float = 12.3  # m
    WING_CHORD: float = 2.0  # m
    WING_AREA: float = 47.0  # m^2
    RHO: float = 1.225  # kg/m^3, sea-level air density

    # Uniform scale factor applied to the Smithsonian mesh so that its span
    # matches WING_SPAN. The STL spans 141.38 raw units along Y (Blender
    # export, Z-up, nose along +X), so 12.3 / 141.38 = 0.087.
    MESH_SCALE: float = 0.087

    # Aerodynamic coefficients (simplified, dimensionless).
    CL0: float = 0.9  # low-speed high-lift airfoil, trimmed cruise
    CL_ALPHA: float = 4.8  # per radian
    CD0: float = 0.045
    K_INDUCED: float = 0.045
    CY_BETA: float = -0.6  # per radian
    CL_WARP: float = 0.8  # lift coefficient change per unit warp
    CD_WARP: float = 0.15  # induced drag change per unit warp
    ELEVATOR_LIFT: float = 2.5  # lift coefficient multiplier for elevator
    RUDDER_FORCE: float = 1.8  # side-force coefficient multiplier for rudder
    THRUST_MAX: float = 600.0  # N (two small piston engines)

    # Claim 18 linkage: the rudder is chained to the wing-warping cradle.
    # The original site maps rudder_deg = round(warp_deg * 0.45) over a
    # +/-15 deg warp and +/-25 deg rudder range, i.e. 0.27 in the normalized
    # units used here.
    WARP_RUDDER_COUPLING: float = 0.45 * 15.0 / 25.0  # = 0.27

    def __init__(self, config: PatentSimConfig | None = None) -> None:
        super().__init__(config)
        self.config.patent_id = self.config.patent_id or "US821393"
        self._init_parameters(
            {
                "wing_warp": 0.0,  # -1 .. 1, differential twist
                "rudder": 0.0,  # -1 .. 1, yaw control (used when coupled=0)
                "elevator": 0.0,  # -1 .. 1, pitch control
                "thrust": 0.0,  # 0 .. 1, throttle
                "wind_speed": 0.0,  # m/s, ambient headwind
                "coupled": 1.0,  # 0/1, Claim 18 warp-rudder linkage
            }
        )

        # Cached Genesis entities
        self._aircraft: Any = None
        self._wind_text: Any = None

    @property
    def patent_title(self) -> str:
        return "Flying-Machine (Wright Flyer)"

    def build(self) -> None:
        """Create and build the Genesis scene with the Flyer."""
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
                camera_pos=(15.0, -15.0, 8.0),
                camera_lookat=(0.0, 0.0, 3.0),
            ),
            show_viewer=not self.config.headless,
        )

        # Ground plane at Kitty Hawk sea level.
        self._scene.add_entity(gs.morphs.Plane())

        aircraft = self._spawn_aircraft(gs)
        self._aircraft = aircraft
        self._entities["aircraft"] = aircraft

        # Camera must be added before the scene is built.
        self._camera = self._scene.add_camera(
            pos=(-8.0, -8.0, 6.0),
            lookat=(0.0, 0.0, 3.0),
            res=self.config.resolution,
            fov=60,
            GUI=False,
        )

        # Build the scene.
        self._scene.build()

        # Mesh volume derived from convex decomposition is far larger than the
        # real airframe's, so the density-based mass comes out wrong (e.g.
        # 8400 kg instead of 340 kg). Override mass post-build; Genesis scales
        # the inertia tensor by the same ratio, preserving the shape
        # distribution of the airframe.
        try:
            measured = float(aircraft.get_mass())
            if abs(measured - self.MASS) > 1e-3:
                logger.info(
                    "Wright Flyer mass override: %.1f kg -> %.1f kg",
                    measured,
                    self.MASS,
                )
                aircraft.set_mass(self.MASS)
        except (AttributeError, RuntimeError) as exc:
            logger.warning("Could not override aircraft mass: %s", exc)

        self._built = True

    def _spawn_aircraft(self, gs: Any) -> Any:
        """Spawn the aircraft rigid body.

        Prefers the Smithsonian NASM mesh when the asset is available;
        otherwise falls back to a flat box approximation. The exact mass is
        enforced post-build via ``set_mass`` in both cases.
        """
        spawn_pos = (0.0, 0.0, 3.0)
        if WRIGHT_FLYER_MESH.exists():
            return self._scene.add_entity(
                gs.morphs.Mesh(
                    file=str(WRIGHT_FLYER_MESH),
                    scale=self.MESH_SCALE,
                    pos=spawn_pos,
                ),
                material=gs.materials.Rigid(
                    # Exact mass is enforced post-build via set_mass().
                    rho=100.0,
                    friction=0.5,
                ),
                surface=gs.surfaces.Default(
                    color=(0.8, 0.7, 0.5, 1.0),
                ),
            )

        logger.info(
            "Wright Flyer mesh not found at %s; using box fallback", WRIGHT_FLYER_MESH
        )
        # Volume = 3.2 * 12.3 * 0.1 ~= 3.94 m^3.
        box_volume = 3.2 * self.WING_SPAN * 0.1
        return self._scene.add_entity(
            gs.morphs.Box(
                size=(3.2, self.WING_SPAN, 0.1),
                pos=spawn_pos,
            ),
            material=gs.materials.Rigid(
                rho=self.MASS / box_volume,
                friction=0.5,
            ),
            surface=gs.surfaces.Default(
                color=(0.8, 0.7, 0.5, 1.0),
            ),
        )

    def reset(self) -> SimState:
        """Reset the aircraft to a stable flight condition."""
        if not self._built:
            raise RuntimeError("Simulation has not been built. Call build() first.")

        self._time = 0.0
        self._aircraft.set_pos(np.array([0.0, 0.0, 10.0]))
        self._aircraft.set_quat(np.array([1.0, 0.0, 0.0, 0.0]))
        if hasattr(self._aircraft, "set_dofs_velocity"):
            # Cruise-velocity initial condition (~13 m/s historical cruise).
            self._aircraft.set_dofs_velocity(np.array([13.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
        elif hasattr(self._aircraft, "set_vel"):
            self._aircraft.set_vel(np.array([13.0, 0.0, 0.0]))

        # Stabilize for a few steps.
        for _ in range(10):
            self._scene.step()

        return self.get_state()

    def step(self) -> SimState:
        """Advance physics and apply aerodynamic forces for one control step."""
        if not self._built:
            raise RuntimeError("Simulation has not been built. Call build() first.")

        for _ in range(self.config.substeps):
            self._apply_aerodynamics()
            self._scene.step()

        self._time += self.config.dt * self.config.substeps
        return self.get_state()

    def get_state(self) -> SimState:
        """Return the full simulation state including aerodynamic metrics."""
        state = super().get_state()
        metrics = self._compute_metrics()
        state.metrics.update(metrics)
        return state

    def _apply_aerodynamics(self) -> None:
        """Compute and apply lift, drag, thrust, and control forces."""
        aircraft = self._aircraft

        pos = np.asarray(aircraft.get_pos(), dtype=float)
        quat = np.asarray(aircraft.get_quat(), dtype=float)
        vel_world = np.asarray(aircraft.get_vel(), dtype=float)
        # Angular velocity (world frame). Genesis 1.3.2 exposes get_ang();
        # the older hasattr(get_angular_velocity) guard silently returned
        # zeros, which disabled all angular damping.
        if hasattr(aircraft, "get_ang"):
            ang_vel = np.asarray(aircraft.get_ang(), dtype=float)
        elif hasattr(aircraft, "get_angular_velocity"):
            ang_vel = np.asarray(aircraft.get_angular_velocity(), dtype=float)
        else:
            ang_vel = np.zeros(3)

        # Wind is along -X (headwind) in world frame.
        wind_world = np.array([-self._parameters["wind_speed"], 0.0, 0.0], dtype=float)
        air_vel = vel_world - wind_world
        airspeed = float(np.linalg.norm(air_vel))

        if airspeed < 0.1:
            return

        # Rotation matrix from body frame to world frame.
        rot = self._quat_to_matrix(quat)
        # Air velocity in body frame.
        v_body = rot.T @ air_vel

        # Angle of attack and sideslip.
        alpha = float(np.arctan2(-v_body[2], v_body[0]))
        beta = float(np.arctan2(v_body[1], v_body[0]))

        q = 0.5 * self.RHO * airspeed**2

        # Base lift and drag coefficients. The linear CL law is clamped to a
        # stalled range so that extreme attitudes (during rolls/spins) cannot
        # produce absurd forces; sin^2(alpha) adds bluff-body drag post-stall.
        cl = float(np.clip(self.CL0 + self.CL_ALPHA * alpha, -1.2, 1.6))
        cd = self.CD0 + self.K_INDUCED * cl**2 + 0.8 * float(np.sin(alpha)) ** 2

        # Body-frame unit vectors. e_drag points opposite the airflow; lift is
        # perpendicular in the body x-z plane and must point +Z (up) in level
        # flight, hence the cross-product order (side_axis x e_drag).
        e_drag = -v_body / airspeed
        e_lift = np.cross(np.array([0.0, 1.0, 0.0]), e_drag)
        e_lift_norm = np.linalg.norm(e_lift)
        if e_lift_norm > 1e-9:
            e_lift /= e_lift_norm
        else:
            e_lift = np.array([0.0, 0.0, 1.0])
        e_side = np.cross(e_drag, e_lift)
        e_side /= max(np.linalg.norm(e_side), 1e-9)

        # Convert unit vectors to world frame.
        e_drag_world = rot @ e_drag
        e_lift_world = rot @ e_lift
        e_side_world = rot @ e_side

        # Wing warp: differential lift and induced drag.
        warp = float(np.clip(self._parameters["wing_warp"], -1.0, 1.0))
        cl_left = cl + self.CL_WARP * warp
        cl_right = cl - self.CL_WARP * warp

        # Roll damping (the Clp derivative): a body roll rate p changes each
        # wing's local angle of attack by +/- p*y_wing/V, producing a
        # differential lift that opposes the roll. Without this term a held
        # warp input rolls the aircraft indefinitely.
        y_wing = self.WING_SPAN / 4.0
        p_body = float((rot.T @ ang_vel)[0])
        cl_damp = self.CL_ALPHA * p_body * y_wing / max(airspeed, 1.0)
        cl_left += cl_damp
        cl_right -= cl_damp

        cd_left = cd + self.CD_WARP * abs(warp)
        cd_right = cd + self.CD_WARP * abs(warp)

        wing_area_half = self.WING_AREA / 2.0
        lift_left = q * wing_area_half * cl_left
        lift_right = q * wing_area_half * cl_right
        drag_left = q * wing_area_half * cd_left
        drag_right = q * wing_area_half * cd_right

        # Aggregate all aerodynamic and control forces/torques about the CG.
        total_force = np.zeros(3, dtype=float)
        total_torque = np.zeros(3, dtype=float)

        # Left/right wing forces at quarter-chord positions.
        for side, sign, lift, drag in [
            ("left", -1.0, lift_left, drag_left),
            ("right", 1.0, lift_right, drag_right),
        ]:
            force = lift * e_lift_world + drag * e_drag_world
            point = pos + rot @ np.array([0.0, sign * y_wing, 0.0])
            total_force += force
            total_torque += np.cross(point - pos, force)

        # Side force from sideslip (yaw stability).
        cy = self.CY_BETA * beta
        side_force = q * self.WING_AREA * cy * e_side_world
        total_force += side_force

        # Elevator: pitch control force applied at the canard.
        elevator = float(np.clip(self._parameters["elevator"], -1.0, 1.0))
        elevator_force = (
            q * self.WING_AREA * 0.15 * self.ELEVATOR_LIFT * elevator * e_lift_world
        )
        elevator_point = pos + rot @ np.array([1.8, 0.0, 0.0])
        total_force += elevator_force
        total_torque += np.cross(elevator_point - pos, elevator_force)

        # Rudder: yaw control side force at the tail. With the Claim 18
        # linkage engaged (the historical default), the rudder follows the
        # wing warp instead of the pilot's separate rudder input.
        rudder = self._effective_rudder(warp)
        rudder_force = (
            q * self.WING_AREA * 0.08 * self.RUDDER_FORCE * rudder * e_side_world
        )
        rudder_point = pos + rot @ np.array([-1.6, 0.0, 0.0])
        total_force += rudder_force
        total_torque += np.cross(rudder_point - pos, rudder_force)

        # Thrust: applied along body +X at the propeller line.
        thrust = float(np.clip(self._parameters["thrust"], 0.0, 1.0))
        thrust_force = self.THRUST_MAX * thrust * rot[:, 0]
        thrust_point = pos + rot @ np.array([-0.6, 0.0, 0.0])
        total_force += thrust_force
        total_torque += np.cross(thrust_point - pos, thrust_force)

        # Damping to tame high-frequency oscillations. Angular damping is sized
        # relative to the pitch inertia (~290 kg m^2) so that control inputs
        # do not spin the aircraft end-over-end.
        total_force += -0.15 * airspeed * air_vel
        total_torque += -50.0 * ang_vel

        # Clamp the generalized force: during spins or ground contact the
        # lumped model can otherwise produce spikes that NaN the solver.
        np.clip(total_force, -2.0e4, 2.0e4, out=total_force)
        np.clip(total_torque, -5.0e3, 5.0e3, out=total_torque)

        # Apply generalized force: [Fx, Fy, Fz, Tx, Ty, Tz] in world frame.
        aircraft.control_dofs_force(
            np.concatenate([total_force, total_torque]).astype(float)
        )

        # Cache metrics for state reporting.
        self._last_metrics = {
            "airspeed": airspeed,
            "altitude": float(pos[2]),
            "alpha_deg": float(np.degrees(alpha)),
            "beta_deg": float(np.degrees(beta)),
            "cl": cl,
            "cd": cd,
            "lift_total": float(lift_left + lift_right),
            "drag_total": float(drag_left + drag_right),
            "thrust_n": float(self.THRUST_MAX * thrust),
            "rudder_effective": rudder,
        }

    def _effective_rudder(self, warp: float) -> float:
        """Return the rudder command, honoring the Claim 18 warp linkage.

        When ``coupled`` is enabled (the historical default), the rudder is
        chained to the wing-warping cradle and follows the warp input;
        otherwise the pilot's independent ``rudder`` parameter is used.
        """
        if float(self._parameters.get("coupled", 1.0)) >= 0.5:
            return float(np.clip(self.WARP_RUDDER_COUPLING * warp, -1.0, 1.0))
        return float(np.clip(self._parameters["rudder"], -1.0, 1.0))

    def _compute_metrics(self) -> dict[str, float]:
        """Return current flight metrics."""
        metrics: dict[str, float] = {}
        if hasattr(self, "_last_metrics"):
            metrics.update(self._last_metrics)
        if self._aircraft is not None:
            pos = np.asarray(self._aircraft.get_pos(), dtype=float)
            vel = np.asarray(self._aircraft.get_vel(), dtype=float)
            quat = np.asarray(self._aircraft.get_quat(), dtype=float)
            metrics["altitude"] = float(pos[2])
            metrics["airspeed"] = float(np.linalg.norm(vel))
            roll, pitch, yaw = self._quat_to_euler(quat)
            metrics["roll_deg"] = float(np.degrees(roll))
            metrics["pitch_deg"] = float(np.degrees(pitch))
            metrics["yaw_deg"] = float(np.degrees(yaw))
        return metrics

    @staticmethod
    def _quat_to_matrix(quat: np.ndarray) -> np.ndarray:
        """Convert quaternion (w, x, y, z) to a 3x3 rotation matrix."""
        w, x, y, z = (float(v) for v in quat[:4])
        return np.array(
            [
                [
                    1 - 2 * (y * y + z * z),
                    2 * (x * y - z * w),
                    2 * (x * z + y * w),
                ],
                [
                    2 * (x * y + z * w),
                    1 - 2 * (x * x + z * z),
                    2 * (y * z - x * w),
                ],
                [
                    2 * (x * z - y * w),
                    2 * (y * z + x * w),
                    1 - 2 * (x * x + y * y),
                ],
            ],
            dtype=float,
        )

    @staticmethod
    def _quat_to_euler(quat: np.ndarray) -> tuple[float, float, float]:
        """Convert quaternion (w, x, y, z) to roll, pitch, yaw in radians."""
        w, x, y, z = (float(v) for v in quat[:4])
        sinr_cosp = 2.0 * (w * x + y * z)
        cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        sinp = 2.0 * (w * y - z * x)
        pitch = np.arcsin(np.clip(sinp, -1.0, 1.0))

        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)

        return float(roll), float(pitch), float(yaw)
