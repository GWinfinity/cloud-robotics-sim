"""Tests for the `cloud-robotics-sim plugins` CLI and plugin_config storage."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from cloud_robotics_sim.__main__ import main
from cloud_robotics_sim.core import plugin_config
from cloud_robotics_sim.core.plugin_manager import PluginManager


def _make_plugin(
    root: Path,
    category: str,
    name: str,
    description: str = "A test plugin",
    config: dict | None = None,
) -> Path:
    """Create a minimal plugin tree under root/<category>/<name>/."""
    plugin_dir = root / category / name
    plugin_dir.mkdir(parents=True)
    meta: dict = {
        "name": name,
        "version": "1.2.3",
        "description": description,
        "exports": ["Thing"],
        "dependencies": {"required": ["numpy"], "optional": []},
        "entry_points": {"cli": "__main__:main"},
    }
    if config is not None:
        meta["config"] = config
    (plugin_dir / "plugin.yaml").write_text(
        yaml.safe_dump(meta, allow_unicode=True), encoding="utf-8"
    )
    (plugin_dir / "README.md").write_text(
        f"# {name}\n\nIntro paragraph.\n\n## 使用\n\n```bash\npython -m {name} --help\n```\n",
        encoding="utf-8",
    )
    return plugin_dir


@pytest.fixture()
def plugin_root(tmp_path, monkeypatch):
    """Fake plugin tree + isolated user config dir; CLI sees only these."""
    root = tmp_path / "plugins"
    _make_plugin(root, "solvers", "alpha", config={"dt": 0.01, "steps": 10})
    _make_plugin(root, "scenes", "beta")
    # 跨类别重名插件（验证 --category 消歧）
    _make_plugin(root, "solvers", "dup")
    _make_plugin(root, "envs", "dup")
    # 中文描述（钉住 Windows GBK 解码 bug）
    _make_plugin(root, "scenes", "cn", description="中式家居程序化场景生成")

    cfg_dir = tmp_path / "cfg"
    monkeypatch.setenv(plugin_config._CONFIG_DIR_ENV, str(cfg_dir))
    manager = PluginManager(plugins_dir=str(root))
    manager.discover_plugins()
    monkeypatch.setattr(
        "cloud_robotics_sim.core.plugin_manager._plugin_manager",
        manager,
    )
    return root


# ---------------------------------------------------------------------------
# plugin_config storage
# ---------------------------------------------------------------------------


class TestPluginConfig:
    """plugin_config 存取与合并。"""

    def test_roundtrip_set_unset(self, plugin_root):
        plugin_config.set_override("alpha", "dt", 0.02)
        plugin_config.set_override("alpha", "name", "x")
        assert plugin_config.read_overrides("alpha") == {"dt": 0.02, "name": "x"}
        plugin_config.unset_override("alpha", "dt")
        assert plugin_config.read_overrides("alpha") == {"name": "x"}

    def test_empty_unset_removes_file(self, plugin_root, tmp_path):
        plugin_config.set_override("alpha", "k", 1)
        path = plugin_config._overrides_path("alpha")
        assert path.exists()
        plugin_config.unset_override("alpha", "k")
        assert not path.exists()

    def test_merge_override_wins(self):
        merged = plugin_config.merge_config({"a": 1, "b": 2}, {"b": 3})
        assert merged == {"a": 1, "b": 3}

    def test_get_plugin_config_merges_defaults(self, plugin_root):
        plugin_config.set_override("alpha", "dt", 0.05)
        merged = plugin_config.get_plugin_config("alpha", {"dt": 0.01, "steps": 10})
        assert merged == {"dt": 0.05, "steps": 10}

    def test_config_defaults_from_yaml(self):
        assert plugin_config.config_defaults({"config": {"x": 1}}) == {"x": 1}
        assert plugin_config.config_defaults({}) == {}
        assert plugin_config.config_defaults({"config": "not-a-dict"}) == {}
        assert plugin_config.config_defaults(None) == {}

    def test_invalid_name_rejected(self, plugin_root):
        with pytest.raises(ValueError):
            plugin_config.read_overrides("bad/name")


# ---------------------------------------------------------------------------
# CLI: plugins list / info / config
# ---------------------------------------------------------------------------


class TestPluginsList:
    """plugins list 子命令。"""

    def test_lists_all_plugins(self, plugin_root, capsys):
        assert main(["plugins", "list"]) == 0
        out = capsys.readouterr().out
        for name in ("alpha", "beta", "cn", "dup"):
            assert name in out
        assert "1.2.3" in out

    def test_category_filter(self, plugin_root, capsys):
        assert main(["plugins", "list", "--category", "scenes"]) == 0
        out = capsys.readouterr().out
        assert "beta" in out and "alpha" not in out

    def test_utf8_description_intact(self, plugin_root, capsys):
        """中文描述必须原样显示（regression: Windows 默认 GBK 解码）。"""
        assert main(["plugins", "list"]) == 0
        assert "中式家居程序化场景生成" in capsys.readouterr().out


class TestPluginsInfo:
    """plugins info 子命令。"""

    def test_info_shows_metadata_and_usage(self, plugin_root, capsys):
        assert main(["plugins", "info", "alpha"]) == 0
        out = capsys.readouterr().out
        assert "alpha (solvers) v1.2.3" in out
        assert "Thing" in out  # exports
        assert "numpy" in out  # required deps
        assert "dt = 0.01" in out  # config defaults
        assert "python -m alpha --help" in out  # README usage section

    def test_info_unknown_plugin(self, plugin_root, caplog):
        assert main(["plugins", "info", "nope"]) == 1
        assert "not found" in caplog.text

    def test_info_ambiguous_needs_category(self, plugin_root, caplog, capsys):
        assert main(["plugins", "info", "dup"]) == 1
        assert "ambiguous" in caplog.text
        capsys.readouterr()
        assert main(["plugins", "info", "dup", "--category", "envs"]) == 0
        assert "dup (envs)" in capsys.readouterr().out


class TestPluginsConfig:
    """plugins config 子命令。"""

    def test_show_defaults_when_no_overrides(self, plugin_root, capsys):
        assert main(["plugins", "config", "alpha"]) == 0
        out = capsys.readouterr().out
        assert "dt = 0.01  (default)" in out

    def test_set_and_show_override(self, plugin_root, capsys):
        assert (
            main(
                ["plugins", "config", "alpha", "--set", "dt=0.05", "--set", "steps=20"]
            )
            == 0
        )
        out = capsys.readouterr().out
        assert "dt = 0.05  (override)" in out
        assert "steps = 20  (override)" in out
        # 持久化：再读一次仍在
        assert main(["plugins", "config", "alpha"]) == 0
        assert "dt = 0.05  (override)" in capsys.readouterr().out

    def test_unset_restores_default(self, plugin_root, capsys):
        main(["plugins", "config", "alpha", "--set", "dt=0.05"])
        capsys.readouterr()
        assert main(["plugins", "config", "alpha", "--unset", "dt"]) == 0
        out = capsys.readouterr().out
        assert "dt = 0.01  (default)" in out

    def test_defaults_flag_ignores_overrides(self, plugin_root, capsys):
        main(["plugins", "config", "alpha", "--set", "dt=0.05"])
        capsys.readouterr()
        assert main(["plugins", "config", "alpha", "--defaults"]) == 0
        out = capsys.readouterr().out
        assert "dt = 0.01" in out and "override" not in out
