"""Tests for the plugin manager."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import yaml

from cloud_robotics_sim.core.plugin_manager import PluginManager, get_plugin_manager


class TestPluginManagerInit:
    """Tests for PluginManager initialization."""

    def test_default_plugins_dir(self):
        manager = PluginManager()
        assert manager.plugins_dir.name == "plugins"

    def test_custom_plugins_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = PluginManager(plugins_dir=tmpdir)
            assert manager.plugins_dir == Path(tmpdir)


class TestPluginManagerDiscovery:
    """Tests for plugin discovery with real plugin.yaml files."""

    def test_discover_empty_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = PluginManager(plugins_dir=tmpdir)
            assert manager.discover_plugins() == {}

    def test_discover_missing_dir(self):
        manager = PluginManager(plugins_dir="/nonexistent/path")
        assert manager.discover_plugins() == {}

    def test_discover_single_plugin(self, tmp_path):
        """Discover a single plugin with a valid plugin.yaml."""
        plugin_dir = tmp_path / "controllers" / "my_ctrl"
        plugin_dir.mkdir(parents=True)
        (plugin_dir / "plugin.yaml").write_text(
            yaml.dump(
                {
                    "name": "my_ctrl",
                    "version": "1.0.0",
                    "description": "A test controller",
                    "exports": ["MyController"],
                }
            )
        )
        manager = PluginManager(plugins_dir=str(tmp_path))
        discovered = manager.discover_plugins()

        assert "controllers" in discovered
        assert "my_ctrl" in discovered["controllers"]

        info = manager.get_plugin_info("controllers", "my_ctrl")
        assert info.name == "my_ctrl"
        assert info.version == "1.0.0"
        assert info.description == "A test controller"
        assert info.exports == ["MyController"]

    def test_discover_multiple_categories(self, tmp_path):
        """Discover plugins across multiple categories."""
        for category, name in [("controllers", "ctrl_a"), ("envs", "env_b")]:
            d = tmp_path / category / name
            d.mkdir(parents=True)
            (d / "plugin.yaml").write_text(yaml.dump({"name": name}))

        manager = PluginManager(plugins_dir=str(tmp_path))
        discovered = manager.discover_plugins()

        assert set(discovered.keys()) == {"controllers", "envs"}
        assert discovered["controllers"] == ["ctrl_a"]
        assert discovered["envs"] == ["env_b"]

    def test_discover_skips_non_yaml_dirs(self, tmp_path):
        """Directories without plugin.yaml are skipped."""
        (tmp_path / "controllers" / "no_yaml").mkdir(parents=True)
        (tmp_path / "controllers" / "with_yaml").mkdir(parents=True)
        (tmp_path / "controllers" / "with_yaml" / "plugin.yaml").write_text(
            yaml.dump({"name": "with_yaml"})
        )

        manager = PluginManager(plugins_dir=str(tmp_path))
        discovered = manager.discover_plugins()
        assert discovered["controllers"] == ["with_yaml"]

    def test_discover_skips_files(self, tmp_path):
        """Files in category dirs (not directories) are skipped."""
        (tmp_path / "controllers").mkdir()
        (tmp_path / "controllers" / "a_file.txt").write_text("not a plugin")

        manager = PluginManager(plugins_dir=str(tmp_path))
        discovered = manager.discover_plugins()
        assert discovered.get("controllers") == []

    def test_discover_malformed_yaml(self, tmp_path):
        """Malformed plugin.yaml logs an error but doesn't crash."""
        plugin_dir = tmp_path / "controllers" / "bad"
        plugin_dir.mkdir(parents=True)
        (plugin_dir / "plugin.yaml").write_text("{{invalid yaml")

        manager = PluginManager(plugins_dir=str(tmp_path))
        discovered = manager.discover_plugins()
        assert discovered.get("controllers") == []


class TestPluginManagerLoad:
    """Tests for plugin loading."""

    def test_load_undiscovered_plugin_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = PluginManager(plugins_dir=tmpdir)
            with pytest.raises(ValueError, match="Plugin not found"):
                manager.load_plugin("controllers", "nonexistent")

    def test_load_plugin_caches_module(self, tmp_path):
        """Loading the same plugin twice returns the cached module."""
        plugin_dir = tmp_path / "controllers" / "echo"
        plugin_dir.mkdir(parents=True)
        (plugin_dir / "plugin.yaml").write_text(yaml.dump({"name": "echo"}))
        (plugin_dir / "echo.py").write_text("VALUE = 42\n")

        manager = PluginManager(plugins_dir=str(tmp_path))
        manager.discover_plugins()

        mod1 = manager.load_plugin("controllers", "echo")
        mod2 = manager.load_plugin("controllers", "echo")
        assert mod1 is mod2

    def test_load_plugin_imports_module(self, tmp_path):
        """load_plugin returns a module with the expected attribute."""
        # The plugin manager adds plugin_info.path to sys.path and does
        # importlib.import_module(name).  For this to work the importable
        # module must live *inside* the plugin directory (flat file).
        plugin_dir = tmp_path / "controllers" / "greet"
        plugin_dir.mkdir(parents=True)
        (plugin_dir / "plugin.yaml").write_text(yaml.dump({"name": "greet"}))
        (plugin_dir / "greet.py").write_text("HELLO = 'world'\n")

        manager = PluginManager(plugins_dir=str(tmp_path))
        manager.discover_plugins()
        mod = manager.load_plugin("controllers", "greet")
        assert mod.HELLO == "world"


class TestPluginManagerList:
    """Tests for list_plugins."""

    def test_list_empty(self):
        manager = PluginManager(plugins_dir="/nonexistent")
        assert manager.list_plugins() == {}

    def test_list_after_discovery(self, tmp_path):
        d = tmp_path / "controllers" / "c1"
        d.mkdir(parents=True)
        (d / "plugin.yaml").write_text(yaml.dump({"name": "c1"}))

        manager = PluginManager(plugins_dir=str(tmp_path))
        manager.discover_plugins()

        assert manager.list_plugins() == {"controllers": ["c1"]}
        assert manager.list_plugins("controllers") == {"controllers": ["c1"]}
        assert manager.list_plugins("envs") == {"envs": []}


class TestPluginManagerTemplate:
    """Tests for create_plugin_from_template."""

    def test_create_plugin(self, tmp_path):
        manager = PluginManager(plugins_dir=str(tmp_path))
        path = manager.create_plugin_from_template(
            "controllers", "my_new", description="New controller"
        )
        assert path.exists()
        assert (path / "__init__.py").exists()
        assert (path / "plugin.yaml").exists()

        meta = yaml.safe_load((path / "plugin.yaml").read_text())
        assert meta["name"] == "my_new"
        assert meta["description"] == "New controller"

    def test_create_duplicate_raises(self, tmp_path):
        manager = PluginManager(plugins_dir=str(tmp_path))
        manager.create_plugin_from_template("controllers", "dup")
        with pytest.raises(ValueError, match="already exists"):
            manager.create_plugin_from_template("controllers", "dup")


class TestGlobalPluginManager:
    """Tests for the global plugin manager singleton."""

    def test_get_plugin_manager(self):
        manager = get_plugin_manager()
        assert isinstance(manager, PluginManager)
