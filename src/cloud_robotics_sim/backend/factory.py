"""Backend factory and registry.

Provides runtime discovery and construction of SimulatorBackend instances.
Backends are registered lazily to avoid importing heavy physics engines
until they are actually requested.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Callable

from cloud_robotics_sim.backend.types import BackendName

if TYPE_CHECKING:
    from cloud_robotics_sim.backend.base import SimulatorBackend

logger = logging.getLogger(__name__)


BackendFactory = Callable[[], "SimulatorBackend"]


class BackendRegistry:
    """Registry of backend factories keyed by BackendName."""

    def __init__(self) -> None:
        self._factories: dict[BackendName, BackendFactory] = {}

    def register(
        self,
        name: BackendName,
        factory: BackendFactory,
    ) -> None:
        """Register a backend factory."""
        self._factories[name] = factory
        logger.debug("Registered backend factory: %s", name)

    def get(self, name: BackendName | str) -> "SimulatorBackend":
        """Create and return a backend instance by name."""
        if isinstance(name, str):
            name = BackendName(name)

        if name not in self._factories:
            self._auto_register(name)

        factory = self._factories.get(name)
        if factory is None:
            raise ValueError(
                f"Unknown backend '{name}'. "
                f"Available: {list(self._factories.keys())}"
            )
        return factory()

    def _auto_register(self, name: BackendName) -> None:
        """Lazily import and register well-known backends."""
        if name == BackendName.GENESIS:
            from cloud_robotics_sim.backends.genesis_backend import GenesisBackend

            self.register(name, GenesisBackend)
        elif name == BackendName.MT_LAMBDA:
            from cloud_robotics_sim.backends.mt_lambda_backend import MTLambdaBackend

            self.register(name, MTLambdaBackend)
        elif name == BackendName.MUJOCO:
            # Standard MuJoCo CPU backend is reserved for debugging/fallback.
            # It is not implemented in the initial migration phase.
            logger.warning(
                "Standard MuJoCo backend is not yet implemented; "
                "use 'genesis' or 'mt_lambda'."
            )

    def available(self) -> list[BackendName]:
        """Return names of all registered backend factories."""
        return list(self._factories.keys())


# Global registry singleton
_global_registry = BackendRegistry()


def register_backend(name: BackendName, factory: BackendFactory) -> None:
    """Register a custom backend factory globally."""
    _global_registry.register(name, factory)


def get_backend(name: BackendName | str) -> "SimulatorBackend":
    """Return a backend instance by name.

    Args:
        name: One of 'genesis', 'mt_lambda', or 'mujoco'.

    Returns:
        An initialized SimulatorBackend instance (caller must call
        ``backend.initialize()`` before use).
    """
    return _global_registry.get(name)


def available_backends() -> list[BackendName]:
    """Return currently registered backend names."""
    return _global_registry.available()
