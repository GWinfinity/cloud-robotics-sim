"""Interactive Genesis simulation of the Sewing Machine (US 4,750).

Elias Howe's 1846 lockstitch machine: an eye-pointed needle carries the
thread down through the cloth; as the needle rises from bottom dead center
the thread forms a loop, and a reciprocating shuttle carrying a second
thread passes through that loop, locking the stitch. A feed advances the
cloth one stitch length per needle revolution.

Physics model (lumped kinematics, plain Python; Genesis renders it)
-------------------------------------------------------------------
- Needle bar: slider-crank drive from the hand wheel,
  ``x = r (1 - cos theta) + (r^2 / 2L) sin^2 theta`` with
  ``theta = omega t`` and ``omega = wheel_rpm * 2 pi / 60``. The crank
  term dominates; the finite connecting rod ``L`` adds the second-order
  harmonic that makes the needle dwell longer near the top of the stroke.
  At bottom dead center (``theta = pi``) the needle tip is through the
  cloth; the thread loop opens on the rise past BDC.
- Shuttle: reciprocates along a straight race, approximated by a sine at
  wheel frequency with a phase lag so the shuttle point enters the loop
  just after the needle starts rising from BDC.
- Stitch formation: each completed wheel revolution is one stitch cycle.
  The thread-tension window decides the outcome — below ``TENSION_MIN``
  the loop is too small for the shuttle point to catch (missed stitch),
  above ``TENSION_MAX`` the thread snaps and the machine halts.
- Feed: on every successful stitch the feed dog advances the cloth by
  ``stitch_length_mm``, so cloth position equals stitches times length.

Interactive parameters
----------------------
- ``wheel_rpm``: hand-wheel speed, 10-600 rpm (one stitch per revolution).
- ``stitch_length_mm``: cloth feed per stitch, 0.5-4 mm.
- ``thread_tension``: thread tension, 0-1; 0.3-0.7 is the working window.
"""

from __future__ import annotations

import logging

import numpy as np

from cloud_robotics_sim.patents.base import PatentSimConfig, PatentSimulation, SimState
from cloud_robotics_sim.patents.registry import register_patent

logger = logging.getLogger(__name__)


@register_patent(
    "US4750",
    {
        "title": "Sewing Machine",
        "inventors": ["Elias Howe Jr."],
        "grant_date": "1846-09-10",
        "breakthrough": "Eye-pointed needle and reciprocating shuttle lockstitch",
    },
)
class SewingMachineSimulation(PatentSimulation):
    """Genesis simulation of Howe's lockstitch sewing machine."""

    # Slider-crank geometry (meters).
    CRANK_RADIUS: float = 0.015  # r, hand-wheel crank throw
    ROD_LENGTH: float = 0.06  # L, connecting rod length
    NEEDLE_TOP_Z: float = 0.09  # needle-bar center height at TDC

    # Shuttle kinematics: sine of wheel angle with a phase lag that puts
    # the shuttle point at the needle just after bottom dead center.
    SHUTTLE_AMPLITUDE: float = 0.02  # m
    SHUTTLE_PHASE: float = 0.6 * np.pi  # rad, lag behind the needle

    # Thread-tension window (dimensionless 0-1).
    TENSION_MIN: float = 0.3  # below: loop too small, missed stitch
    TENSION_MAX: float = 0.7  # above: thread breaks, machine halts

    def __init__(self, config: PatentSimConfig | None = None) -> None:
        super().__init__(config)
        self.config.patent_id = self.config.patent_id or "US4750"
        self._init_parameters(
            {
                "wheel_rpm": 300.0,  # 10 .. 600
                "stitch_length_mm": 2.0,  # 0.5 .. 4
                "thread_tension": 0.5,  # 0 .. 1, working window 0.3 .. 0.7
            }
        )
        # Mechanism state.
        self._wheel_angle: float = 0.0  # rad, accumulated hand-wheel angle
        self._stitches: int = 0
        self._missed_stitches: int = 0
        self._cloth_position_mm: float = 0.0
        self._thread_broken: bool = False

    @property
    def patent_title(self) -> str:
        return "Sewing Machine (Howe lockstitch)"

    # ------------------------------------------------------------------
    # Genesis scene
    # ------------------------------------------------------------------

    def build(self) -> None:
        """Create and build the Genesis scene with the machine."""
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
                camera_lookat=(0.0, 0.0, 0.06),
            ),
            show_viewer=not self.config.headless,
        )
        self._scene.add_entity(gs.morphs.Plane())

        rigid = gs.materials.Rigid(rho=7800.0, friction=0.6)
        iron = gs.surfaces.Default(color=(0.15, 0.15, 0.17, 1.0))
        # Bed plate and the overhanging arm of the machine (both fixed).
        self._scene.add_entity(
            gs.morphs.Box(size=(0.4, 0.16, 0.02), pos=(0.0, 0.0, 0.01), fixed=True),
            material=rigid,
            surface=iron,
        )
        self._scene.add_entity(
            gs.morphs.Box(size=(0.05, 0.06, 0.14), pos=(-0.15, 0.0, 0.09), fixed=True),
            material=rigid,
            surface=iron,
        )
        self._scene.add_entity(
            gs.morphs.Box(size=(0.3, 0.05, 0.04), pos=(0.0, 0.0, 0.15), fixed=True),
            material=rigid,
            surface=iron,
        )
        # Needle bar (vertical cylinder), posed from the slider-crank.
        self._entities["needle"] = self._scene.add_entity(
            gs.morphs.Cylinder(
                radius=0.003, height=0.07, pos=(0.05, 0.0, self.NEEDLE_TOP_Z)
            ),
            material=rigid,
            surface=gs.surfaces.Default(color=(0.8, 0.8, 0.85, 1.0)),
        )
        # Shuttle on its race below the cloth, posed from the phase-lagged
        # sine approximation.
        self._entities["shuttle"] = self._scene.add_entity(
            gs.morphs.Sphere(radius=0.008, pos=(0.05, 0.0, 0.025)),
            material=gs.materials.Rigid(rho=7800.0, friction=0.4),
            surface=gs.surfaces.Default(color=(0.72, 0.53, 0.2, 1.0)),
        )
        # Cloth strip, fed along +x one stitch length per stitch.
        self._entities["cloth"] = self._scene.add_entity(
            gs.morphs.Box(size=(0.2, 0.06, 0.004), pos=(0.05, 0.0, 0.032)),
            material=gs.materials.Rigid(rho=300.0, friction=0.8),
            surface=gs.surfaces.Default(color=(0.85, 0.8, 0.7, 1.0)),
        )

        # Camera must be added before the scene is built.
        self._camera = self._scene.add_camera(
            pos=(0.35, -0.35, 0.25),
            lookat=(0.0, 0.0, 0.06),
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
        """Reset the wheel, stitch counters, and cloth feed."""
        if not self._built:
            raise RuntimeError("Simulation has not been built. Call build() first.")
        self._time = 0.0
        self._wheel_angle = 0.0
        self._stitches = 0
        self._missed_stitches = 0
        self._cloth_position_mm = 0.0
        self._thread_broken = False
        self._pose_mechanism()
        return self.get_state()

    def step(self) -> SimState:
        """Advance the machine kinematics and the scene."""
        if not self._built:
            raise RuntimeError("Simulation has not been built. Call build() first.")
        for _ in range(self.config.substeps):
            self._advance(self.config.dt)
            self._pose_mechanism()
            self._scene.step()
        self._time += self.config.dt * self.config.substeps
        return self.get_state()

    def get_state(self) -> SimState:
        """Return the simulation state including sewing metrics."""
        state = super().get_state()
        state.metrics.update(self._compute_metrics())
        return state

    # ------------------------------------------------------------------
    # Physics
    # ------------------------------------------------------------------

    def wheel_rpm(self) -> float:
        """Hand-wheel speed in rpm (parameter clipped to 10-600)."""
        return float(np.clip(self._parameters["wheel_rpm"], 10.0, 600.0))

    def angular_velocity(self) -> float:
        """Hand-wheel angular velocity ``omega`` in rad/s."""
        return float(self.wheel_rpm() * 2.0 * np.pi / 60.0)

    def stitch_length_mm(self) -> float:
        """Cloth feed per stitch in mm (parameter clipped to 0.5-4)."""
        return float(np.clip(self._parameters["stitch_length_mm"], 0.5, 4.0))

    def thread_tension(self) -> float:
        """Thread tension, clipped to 0-1."""
        return float(np.clip(self._parameters["thread_tension"], 0.0, 1.0))

    def needle_position(self, theta: float) -> float:
        """Needle displacement below top dead center in meters.

        Slider-crank kinematics ``x = r(1 - cos t) + (r^2/2L) sin^2 t``;
        zero at TDC, ``2r`` at BDC where the eye of the needle passes
        through the cloth.
        """
        r, rod = self.CRANK_RADIUS, self.ROD_LENGTH
        x = r * (1.0 - np.cos(theta)) + (r * r / (2.0 * rod)) * np.sin(theta) ** 2
        return float(x)

    def shuttle_position(self, theta: float) -> float:
        """Shuttle displacement along its race in meters.

        A sine at wheel frequency lagging the needle so the shuttle point
        threads the loop while the needle rises from bottom dead center.
        """
        return float(self.SHUTTLE_AMPLITUDE * np.sin(theta - self.SHUTTLE_PHASE))

    def needle_phase(self) -> float:
        """Current wheel angle modulo one revolution, in radians."""
        return float(self._wheel_angle % (2.0 * np.pi))

    def _advance(self, dt: float) -> None:
        """Advance the wheel by ``dt`` and complete any finished stitches.

        A broken thread halts the machine: the wheel freezes and no further
        stitches or feed occur until ``reset()``.
        """
        if self._thread_broken:
            return
        previous = int(self._wheel_angle / (2.0 * np.pi))
        self._wheel_angle += self.angular_velocity() * dt
        completed = int(self._wheel_angle / (2.0 * np.pi)) - previous
        for _ in range(completed):
            self._complete_stitch()

    def _complete_stitch(self) -> None:
        """Resolve one stitch cycle against the thread-tension window."""
        tension = self.thread_tension()
        if tension > self.TENSION_MAX:
            # Thread snaps: the machine halts without forming the stitch.
            self._thread_broken = True
            logger.info(
                "Thread broke at %.1f rpm after %d stitches",
                self.wheel_rpm(),
                self._stitches,
            )
            return
        if tension < self.TENSION_MIN:
            # Loop too small: the shuttle point misses it (skipped stitch,
            # and an unseamed gap means the feed has nothing to pull).
            self._missed_stitches += 1
            return
        self._stitches += 1
        self._cloth_position_mm += self.stitch_length_mm()

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def _pose_mechanism(self) -> None:
        """Write needle, shuttle, and cloth poses into the scene."""
        identity = np.array([1.0, 0.0, 0.0, 0.0])
        theta = self._wheel_angle

        def pose(name: str, pos: np.ndarray) -> None:
            entity = self._entities.get(name)
            if entity is None:
                return
            try:
                entity.set_pos(pos)
                entity.set_quat(identity)
                if hasattr(entity, "set_dofs_velocity"):
                    entity.set_dofs_velocity(np.zeros(6))
            except Exception as exc:  # noqa: BLE001 - pose is best-effort
                logger.debug("Could not pose %s: %s", name, exc)

        needle_z = self.NEEDLE_TOP_Z - self.needle_position(theta)
        pose("needle", np.array([0.05, 0.0, needle_z]))
        shuttle_x = 0.05 + self.shuttle_position(theta)
        pose("shuttle", np.array([shuttle_x, 0.0, 0.025]))
        cloth_x = 0.05 + self._cloth_position_mm / 1000.0
        pose("cloth", np.array([cloth_x, 0.0, 0.032]))

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def _compute_metrics(self) -> dict[str, float]:
        """Return current sewing-machine metrics."""
        theta = self._wheel_angle
        tension = self.thread_tension()
        return {
            "stitches": float(self._stitches),
            "missed_stitches": float(self._missed_stitches),
            "cloth_position_mm": float(self._cloth_position_mm),
            "needle_phase": float(self.needle_phase()),
            "needle_position_mm": float(self.needle_position(theta) * 1000.0),
            "shuttle_position_mm": float(self.shuttle_position(theta) * 1000.0),
            "rpm": float(self.wheel_rpm()),
            "thread_broken": float(self._thread_broken),
            "tension_ok": float(self.TENSION_MIN <= tension <= self.TENSION_MAX),
        }
