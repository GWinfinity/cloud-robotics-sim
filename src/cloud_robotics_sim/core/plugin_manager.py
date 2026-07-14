"""Plugin Manager for Genesis Cloud Sim

Simple plugin system for organizing reusable components.
"""

from __future__ import annotations

import importlib
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


@dataclass
class PluginInfo:
    """Plugin metadata"""

    name: str
    category: str
    version: str
    description: str
    path: Path
    exports: list
    config: dict | None = None


class PluginManager:
    """Simple plugin manager for loading and managing plugins.

    Usage:
        >>> pm = PluginManager()
        >>> pm.discover_plugins()  # Auto-discover all plugins
        >>> plugin = pm.load_plugin('controllers', 'mpc_wbc')
        >>> controller = plugin.MPCWBCController()
    """

    def __init__(self, plugins_dir: str | None = None):
        """Args:
        plugins_dir: Path to plugins directory.
                    Defaults to ../plugins relative to this file.
        """
        if plugins_dir is None:
            # Default: plugins/ directory next to src/
            current_file = Path(__file__).resolve()
            self.plugins_dir = current_file.parent.parent.parent.parent / "plugins"
        else:
            self.plugins_dir = Path(plugins_dir)

        self._plugins: dict[str, dict[str, PluginInfo]] = {}
        self._loaded_modules: dict[str, Any] = {}

    def discover_plugins(self) -> dict[str, list]:
        """Discover all available plugins in the plugins directory.

        Returns:
            Dict mapping category names to lists of plugin names
        """
        discovered: dict[str, list[str]] = {}

        if not self.plugins_dir.exists():
            logger.warning("Plugins directory not found: %s", self.plugins_dir)
            return discovered

        for category_dir in self.plugins_dir.iterdir():
            if not category_dir.is_dir():
                continue

            category = category_dir.name
            discovered[category] = []
            self._plugins[category] = {}

            for plugin_dir in category_dir.iterdir():
                if not plugin_dir.is_dir():
                    continue

                plugin_file = plugin_dir / "plugin.yaml"
                if plugin_file.exists():
                    try:
                        with open(plugin_file) as f:
                            metadata = yaml.safe_load(f)

                        plugin_info = PluginInfo(
                            name=metadata["name"],
                            category=category,
                            version=metadata.get("version", "0.1.0"),
                            description=metadata.get("description", ""),
                            path=plugin_dir,
                            exports=metadata.get("exports", []),
                            config=metadata,
                        )

                        self._plugins[category][metadata["name"]] = plugin_info
                        discovered[category].append(metadata["name"])

                    except Exception as e:
                        logger.error("Error loading plugin from %s: %s", plugin_dir, e)

        return discovered

    def load_plugin(self, category: str, name: str) -> Any:
        """Load a plugin module.

        Args:
            category: Plugin category (e.g., 'controllers', 'envs')
            name: Plugin name

        Returns:
            Loaded module

        Raises:
            ValueError: If plugin not found
        """
        # Check if already loaded
        cache_key = f"{category}.{name}"
        if cache_key in self._loaded_modules:
            return self._loaded_modules[cache_key]

        # Get plugin info
        if category not in self._plugins or name not in self._plugins[category]:
            raise ValueError(f"Plugin not found: {category}/{name}")

        plugin_info = self._plugins[category][name]

        # Add to path and import
        if str(plugin_info.path) not in sys.path:
            sys.path.insert(0, str(plugin_info.path))

        try:
            # Import the plugin module
            module = importlib.import_module(name)
            self._loaded_modules[cache_key] = module
            return module

        except ImportError as e:
            raise ImportError(f"Failed to load plugin {name}: {e}") from e

    def get_plugin_info(self, category: str, name: str) -> PluginInfo:
        """Get plugin metadata"""
        if category not in self._plugins or name not in self._plugins[category]:
            raise ValueError(f"Plugin not found: {category}/{name}")
        return self._plugins[category][name]

    def list_plugins(self, category: str | None = None) -> dict[str, list]:
        """List all available plugins"""
        if category:
            return {category: list(self._plugins.get(category, {}).keys())}
        return {cat: list(plugins.keys()) for cat, plugins in self._plugins.items()}

    def create_plugin_from_template(
        self, category: str, name: str, description: str = "", source_project: str = ""
    ) -> Path:
        """Create a new plugin from template.

        Args:
            category: Plugin category
            name: Plugin name
            description: Plugin description
            source_project: Original project this code comes from

        Returns:
            Path to created plugin directory
        """
        template_dir = self.plugins_dir / "templates" / "basic_plugin"
        plugin_dir = self.plugins_dir / category / name

        if plugin_dir.exists():
            raise ValueError(f"Plugin already exists: {plugin_dir}")

        # Copy template (or create from scratch if no template)
        if template_dir.exists():
            import shutil

            shutil.copytree(template_dir, plugin_dir)
        else:
            plugin_dir.mkdir(parents=True)
            # Create basic structure
            (plugin_dir / "__init__.py").touch()
            (plugin_dir / "README.md").touch()
            (plugin_dir / "plugin.yaml").touch()

        # Update plugin.yaml
        plugin_yaml = plugin_dir / "plugin.yaml"
        with open(plugin_yaml, "w") as f:
            yaml.dump(
                {
                    "name": name,
                    "version": "0.1.0",
                    "description": description,
                    "category": category,
                    "source_project": source_project,
                    "exports": [],
                    "dependencies": [],
                },
                f,
            )

        return plugin_dir


# Global plugin manager instance
_plugin_manager: PluginManager | None = None


def get_plugin_manager() -> PluginManager:
    """Get global plugin manager instance"""
    global _plugin_manager
    if _plugin_manager is None:
        _plugin_manager = PluginManager()
        _plugin_manager.discover_plugins()
    return _plugin_manager


def load_plugin(category: str, name: str) -> Any:
    """Convenience function to load a plugin"""
    return get_plugin_manager().load_plugin(category, name)


def list_plugins(category: str | None = None) -> dict[str, list]:
    """Convenience function to list plugins"""
    return get_plugin_manager().list_plugins(category)
