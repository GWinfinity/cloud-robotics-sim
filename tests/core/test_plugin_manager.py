"""Tests for the plugin manager."""

import tempfile
from pathlib import Path

from cloud_robotics_sim.core.plugin_manager import PluginManager, get_plugin_manager


class TestPluginManager:
    """Tests for PluginManager."""

    def test_init_default_plugins_dir(self):
        """Test initialization with default plugins directory."""
        manager = PluginManager()
        assert manager.plugins_dir.name == "plugins"

    def test_init_custom_plugins_dir(self):
        """Test initialization with a custom plugins directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = PluginManager(plugins_dir=tmpdir)
            assert manager.plugins_dir == Path(tmpdir)

    def test_discover_plugins_empty_dir(self):
        """Test discovering plugins in an empty directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = PluginManager(plugins_dir=tmpdir)
            discovered = manager.discover_plugins()
            assert discovered == {}

    def test_discover_plugins_missing_dir(self):
        """Test discovering plugins when directory does not exist."""
        manager = PluginManager(plugins_dir="/nonexistent/path")
        discovered = manager.discover_plugins()
        assert discovered == {}

    def test_list_plugins_empty(self):
        """Test listing plugins when none are loaded."""
        manager = PluginManager(plugins_dir="/nonexistent/path")
        assert manager.list_plugins() == {}


class TestGlobalPluginManager:
    """Tests for the global plugin manager singleton."""

    def test_get_plugin_manager(self):
        """Test getting the global plugin manager."""
        manager = get_plugin_manager()
        assert isinstance(manager, PluginManager)
