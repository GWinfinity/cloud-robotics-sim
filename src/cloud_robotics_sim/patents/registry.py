"""Registry for classic patent simulations.

The registry maps canonical patent identifiers (e.g. ``US821393``) to factory
functions that create ``PatentSimulation`` instances. This allows runners,
examples, and future web services to instantiate simulations by ID without
hardcoding imports.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, TypeVar

from cloud_robotics_sim.patents.base import PatentSimConfig, PatentSimulation

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=PatentSimulation)


class PatentRegistry:
    """Registry for patent simulation factories.

    Example:
        >>> registry = PatentRegistry()
        >>> @registry.register("US821393")
        ... class WrightFlyerSimulation(PatentSimulation):
        ...     pass
        >>> sim = registry.create("US821393", headless=True)
    """

    def __init__(self) -> None:
        self._factories: dict[str, Callable[..., PatentSimulation]] = {}
        self._metadata: dict[str, dict[str, Any]] = {}

    def register(
        self,
        patent_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> Callable[[type[T]], type[T]]:
        """Decorator for registering a patent simulation class.

        Args:
            patent_id: Canonical patent identifier.
            metadata: Optional metadata (title, inventors, grant date, etc.).

        Returns:
            The decorated class unchanged.
        """

        def decorator(cls: type[T]) -> type[T]:
            if patent_id in self._factories:
                logger.warning("Overwriting existing patent simulation: %s", patent_id)

            def factory(config: PatentSimConfig | None = None, **kwargs: Any) -> T:
                cfg = config or PatentSimConfig(patent_id=patent_id)
                if kwargs:
                    cfg_kwargs: dict[str, Any] = {
                        "patent_id": patent_id,
                        "headless": cfg.headless,
                        "dt": cfg.dt,
                        "substeps": cfg.substeps,
                        "resolution": cfg.resolution,
                        "device": cfg.device,
                        "seed": cfg.seed,
                        "parameters": dict(cfg.parameters),
                    }
                    cfg_kwargs.update(kwargs)
                    cfg = PatentSimConfig(**cfg_kwargs)
                return cls(cfg)

            self._factories[patent_id] = factory
            self._metadata[patent_id] = metadata or {}
            logger.debug("Registered patent simulation: %s", patent_id)
            return cls

        return decorator

    def create(
        self,
        patent_id: str,
        config: PatentSimConfig | None = None,
        **kwargs: Any,
    ) -> PatentSimulation:
        """Create a patent simulation instance.

        Args:
            patent_id: Registered patent identifier.
            config: Optional configuration object.
            **kwargs: Optional config overrides.

        Returns:
            A ``PatentSimulation`` instance.

        Raises:
            KeyError: If the patent ID is not registered.
        """
        if patent_id not in self._factories:
            raise KeyError(
                f"Unknown patent simulation: {patent_id}. "
                f"Available: {list(self._factories.keys())}"
            )
        return self._factories[patent_id](config=config, **kwargs)

    def list_patents(self) -> list[str]:
        """Return all registered patent IDs."""
        return list(self._factories.keys())

    def get_metadata(self, patent_id: str) -> dict[str, Any]:
        """Return metadata for a registered patent."""
        return dict(self._metadata.get(patent_id, {}))

    def is_registered(self, patent_id: str) -> bool:
        """Return True if the patent ID is registered."""
        return patent_id in self._factories


# Global default registry
_default_registry: PatentRegistry | None = None


def default_registry() -> PatentRegistry:
    """Get or create the global patent registry."""
    global _default_registry
    if _default_registry is None:
        _default_registry = PatentRegistry()
    return _default_registry


def register_patent(
    patent_id: str,
    metadata: dict[str, Any] | None = None,
) -> Callable[[type[T]], type[T]]:
    """Decorator to register a patent simulation class globally.

    Example:
        >>> from cloud_robotics_sim.patents import register_patent
        >>> @register_patent("US821393", {"title": "Flying-Machine"})
        ... class WrightFlyerSimulation(PatentSimulation):
        ...     pass
    """
    return default_registry().register(patent_id, metadata)


def create_simulation(
    patent_id: str,
    config: PatentSimConfig | None = None,
    **kwargs: Any,
) -> PatentSimulation:
    """Create a simulation from the global registry."""
    return default_registry().create(patent_id, config=config, **kwargs)


def list_patents() -> list[str]:
    """List all patent IDs in the global registry."""
    return default_registry().list_patents()
