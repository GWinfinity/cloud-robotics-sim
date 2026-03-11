"""Component registry system.

Provides a centralized registry for scenes, robots, and tasks,
enabling dynamic environment composition without hardcoded imports.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Generic, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar('T')


class Registry(Generic[T]):
    """Generic registry for named components.

    Provides registration and factory pattern for creating
    configurable components at runtime.

    Example:
        >>> registry = Registry[Scene]()
        >>> @registry.register("living_room")
        ... def create_living_room(**kwargs):
        ...     return LivingRoomScene(**kwargs)
        >>> scene = registry.create("living_room")
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self._factories: dict[str, Callable[..., T]] = {}
        self._metadata: dict[str, dict] = {}

    def register(
        self,
        name: str,
        metadata: dict | None = None,
    ) -> Callable[[Callable[..., T]], Callable[..., T]]:
        """Decorator for registering a component factory.

        Args:
            name: Unique identifier for the component.
            metadata: Optional metadata dictionary.

        Returns:
            Decorator function.
        """
        def decorator(factory: Callable[..., T]) -> Callable[..., T]:
            if name in self._factories:
                logger.warning(f"Overwriting existing {self.name}: {name}")

            self._factories[name] = factory
            self._metadata[name] = metadata or {}
            logger.debug(f"Registered {self.name}: {name}")
            return factory

        return decorator

    def create(self, name: str, **kwargs) -> T:
        """Create a component instance.

        Args:
            name: Registered component name.
            **kwargs: Arguments passed to the factory.

        Returns:
            Created component instance.

        Raises:
            KeyError: If name is not registered.
        """
        if name not in self._factories:
            raise KeyError(
                f"Unknown {self.name}: {name}. "
                f"Available: {list(self._factories.keys())}"
            )

        return self._factories[name](**kwargs)

    def list_components(self) -> list[str]:
        """Get list of registered component names."""
        return list(self._factories.keys())

    def get_metadata(self, name: str) -> dict:
        """Get metadata for a registered component."""
        return self._metadata.get(name, {})


class AssetRegistry:
    """Unified registry for all simulation assets.

    Manages registrations for scenes, robots, and tasks.

    Attributes:
        scenes: Registry for scene components.
        robots: Registry for robot embodiments.
        tasks: Registry for task definitions.
    """

    def __init__(self) -> None:
        self.scenes: Registry[Any] = Registry("scene")
        self.robots: Registry[Any] = Registry("robot")
        self.tasks: Registry[Any] = Registry("task")

    def create_scene(self, name: str, **kwargs) -> Any:
        """Create a scene from registry."""
        return self.scenes.create(name, **kwargs)

    def create_robot(self, name: str, **kwargs) -> Any:
        """Create a robot from registry."""
        return self.robots.create(name, **kwargs)

    def create_task(self, name: str, **kwargs) -> Any:
        """Create a task from registry."""
        return self.tasks.create(name, **kwargs)


# Global default registry
_default_registry: AssetRegistry | None = None


def default_registry() -> AssetRegistry:
    """Get or create the default global registry."""
    global _default_registry
    if _default_registry is None:
        _default_registry = AssetRegistry()
    return _default_registry


def register_scene(
    name: str,
    metadata: dict | None = None,
) -> Callable:
    """Decorator to register a scene factory.

    Example:
        >>> @register_scene("living_room")
        ... class LivingRoom(Scene):
        ...     pass
    """
    return default_registry().scenes.register(name, metadata)


def register_robot(
    name: str,
    metadata: dict | None = None,
) -> Callable:
    """Decorator to register a robot factory."""
    return default_registry().robots.register(name, metadata)


def register_task(
    name: str,
    metadata: dict | None = None,
) -> Callable:
    """Decorator to register a task factory."""
    return default_registry().tasks.register(name, metadata)


# Type aliases for backward compatibility
SceneRegistry = Registry
RobotRegistry = Registry
TaskRegistry = Registry
