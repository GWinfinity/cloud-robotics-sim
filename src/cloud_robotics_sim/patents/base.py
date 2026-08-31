"""Base classes for classic patent simulations.

Patent simulations are interactive demonstrators built on Genesis physics.
They are intentionally simpler than the original site's bespoke WASM physics
kernels: Genesis provides rigid/soft-body dynamics, while patent-specific
analytical models (lift/drag, thermal radiation, dipole fields, etc.) are
implemented in plain Python and applied as forces or overlays.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PatentSimConfig:
    """Configuration for a patent simulation.

    Attributes:
        patent_id: Canonical patent identifier, e.g. ``US821393``.
        headless: Whether to run without the interactive viewer.
        dt: Physics timestep in seconds.
        substeps: Substeps per ``step()`` call.
        resolution: Camera resolution as (width, height).
        device: Genesis compute device (``cuda`` or ``cpu``).
        seed: Random seed for reproducibility.
        parameters: Interactive parameter overrides.
    """

    patent_id: str = ""
    headless: bool = True
    dt: float = 0.01
    substeps: int = 10
    resolution: tuple[int, int] = (640, 480)
    device: str = "cuda"
    seed: int = 0
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass
class SimState:
    """Serializable snapshot of a patent simulation.

    Attributes:
        time: Simulation time in seconds.
        parameters: Current interactive parameter values.
        metrics: Derived metrics (lift, drag, voltage, etc.).
        bodies: Dictionary mapping body names to pose/velocity dicts.
    """

    time: float = 0.0
    parameters: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    bodies: dict[str, dict[str, Any]] = field(default_factory=dict)


class PatentSimulation(ABC):
    """Abstract base class for a classic-patents.com interactive simulation.

    Subclasses implement ``build()``, ``reset()``, ``step()``, and ``render()``
    to recreate a historical invention in Genesis. The class exposes a uniform
    API so a CLI runner or future web service can drive any patent without
    knowing its internals.

    Example:
        >>> from cloud_robotics_sim.patents import PatentSimConfig, PatentSimulation
        >>> config = PatentSimConfig(patent_id="US821393", headless=True)
        >>> sim = WrightFlyerSimulation(config)
        >>> sim.build()
        >>> sim.reset()
        >>> for _ in range(100):
        ...     sim.step()
        >>> frame = sim.render()
        >>> sim.close()
    """

    def __init__(self, config: PatentSimConfig | None = None) -> None:
        self.config = config or PatentSimConfig()
        self._built: bool = False
        self._closed: bool = False
        self._time: float = 0.0
        self._parameters: dict[str, Any] = {}
        self._default_parameters: dict[str, Any] = {}
        self._scene: Any = None
        self._camera: Any = None
        self._entities: dict[str, Any] = {}

    @property
    def patent_id(self) -> str:
        """Return the canonical patent identifier."""
        return self.config.patent_id

    @property
    def patent_title(self) -> str:
        """Return a human-readable title.

        Subclasses should override this.
        """
        return self.__class__.__name__

    @property
    def is_built(self) -> bool:
        """Return True if the Genesis scene has been built."""
        return self._built

    @property
    def camera(self) -> Any:
        """Return the simulation camera (a Genesis ``Camera``), if any."""
        return self._camera

    def get_entity(self, name: str) -> Any:
        """Return a named Genesis entity, or None if it does not exist."""
        return self._entities.get(name)

    @abstractmethod
    def build(self) -> None:
        """Create and build the Genesis scene.

        This method should:
          1. Initialize Genesis if needed.
          2. Create a ``gs.Scene``.
          3. Add bodies, lights, and a camera.
          4. Call ``scene.build()``.
          5. Set ``self._built = True``.
        """
        pass

    @abstractmethod
    def reset(self) -> SimState:
        """Reset the simulation to its initial state.

        Returns:
            Initial simulation state.
        """
        pass

    @abstractmethod
    def step(self) -> SimState:
        """Advance the simulation by ``config.dt`` seconds.

        Returns:
            Updated simulation state.
        """
        pass

    def set_parameter(self, name: str, value: Any) -> None:
        """Update an interactive parameter.

        Args:
            name: Parameter name.
            value: New value.

        Raises:
            KeyError: If the parameter is not recognized.
        """
        if name not in self._default_parameters:
            raise KeyError(
                f"Unknown parameter '{name}' for {self.patent_id}. "
                f"Available: {list(self._default_parameters.keys())}"
            )
        self._parameters[name] = value

    def get_parameter(self, name: str) -> Any:
        """Return the current value of an interactive parameter."""
        return self._parameters[name]

    def list_parameters(self) -> list[str]:
        """Return the list of interactive parameter names."""
        return list(self._default_parameters.keys())

    def get_state(self) -> SimState:
        """Return a serializable snapshot of the simulation.

        Subclasses may override this to add patent-specific metrics.
        """
        bodies: dict[str, dict[str, Any]] = {}
        for name, entity in self._entities.items():
            bodies[name] = self._entity_pose_state(entity)
        return SimState(
            time=self._time,
            parameters=dict(self._parameters),
            metrics={},
            bodies=bodies,
        )

    def render(self) -> np.ndarray | None:
        """Render the current frame from the simulation camera.

        Returns:
            RGB image as a numpy array, or None if no camera exists.
        """
        if self._camera is None:
            return None
        try:
            result = self._camera.render(rgb=True)
            if isinstance(result, tuple):
                return np.asarray(result[0])
            return np.asarray(result)
        except Exception as exc:
            logger.debug("Render failed: %s", exc)
            return None

    def close(self) -> None:
        """Clean up Genesis resources."""
        if self._closed:
            return
        self._closed = True
        self._built = False
        try:
            from cloud_robotics_sim.utils.cache_cleanup import cleanup_after_simulation

            cleanup_after_simulation()
        except Exception as exc:
            logger.debug("Cache cleanup failed: %s", exc)

    def _entity_pose_state(self, entity: Any) -> dict[str, Any]:
        """Extract a serializable pose/velocity dict from a Genesis entity."""
        state: dict[str, Any] = {
            "pos": None,
            "quat": None,
            "vel": None,
            "ang_vel": None,
        }
        try:
            if hasattr(entity, "get_pos"):
                state["pos"] = np.asarray(entity.get_pos(), dtype=float).tolist()
            if hasattr(entity, "get_quat"):
                state["quat"] = np.asarray(entity.get_quat(), dtype=float).tolist()
            if hasattr(entity, "get_vel"):
                state["vel"] = np.asarray(entity.get_vel(), dtype=float).tolist()
            elif hasattr(entity, "get_linear_velocity"):
                state["vel"] = np.asarray(
                    entity.get_linear_velocity(), dtype=float
                ).tolist()
            if hasattr(entity, "get_ang_vel"):
                state["ang_vel"] = np.asarray(
                    entity.get_ang_vel(), dtype=float
                ).tolist()
            elif hasattr(entity, "get_angular_velocity"):
                state["ang_vel"] = np.asarray(
                    entity.get_angular_velocity(), dtype=float
                ).tolist()
        except Exception as exc:
            logger.debug("Failed to extract entity state: %s", exc)
        return state

    def _init_parameters(self, defaults: dict[str, Any]) -> None:
        """Initialize interactive parameters from defaults and config overrides."""
        self._default_parameters = dict(defaults)
        self._parameters = dict(defaults)
        for name, value in self.config.parameters.items():
            if name in self._parameters:
                self._parameters[name] = value
            else:
                logger.warning(
                    "Ignoring unknown parameter override '%s' for %s",
                    name,
                    self.patent_id,
                )


class StubPatentSimulation(PatentSimulation):
    """Base class for patent simulations that are not yet fully implemented.

    A stub builds a minimal Genesis scene and prints a warning when stepped,
    ensuring the full catalog is registered and runnable.
    """

    def build(self) -> None:
        """Build a minimal placeholder scene."""
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
                camera_pos=(2.0, 2.0, 2.0),
                camera_lookat=(0.0, 0.0, 0.0),
            ),
            show_viewer=not self.config.headless,
        )
        self._scene.add_entity(gs.morphs.Plane())
        self._camera = self._scene.add_camera(
            pos=(2.0, 2.0, 2.0),
            lookat=(0.0, 0.0, 0.0),
            res=self.config.resolution,
            fov=60,
            GUI=False,
        )
        self._scene.build()
        self._built = True

    def reset(self) -> SimState:
        """Reset the stub scene."""
        self._time = 0.0
        logger.warning(
            "%s (%s) is a stub simulation; physics model is not yet implemented.",
            self.patent_title,
            self.patent_id,
        )
        return self.get_state()

    def step(self) -> SimState:
        """Step the stub scene."""
        if not self._built:
            raise RuntimeError("Simulation has not been built. Call build() first.")
        for _ in range(self.config.substeps):
            self._scene.step()
        self._time += self.config.dt * self.config.substeps
        return self.get_state()
