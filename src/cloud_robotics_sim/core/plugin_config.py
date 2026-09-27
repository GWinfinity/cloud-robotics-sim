"""插件用户配置的存取与合并。

配置模型（三层，后者覆盖前者）：

1. **插件自带默认值**：``plugin.yaml`` 的 ``config:`` 段（老格式插件无
   该段则默认为空）；
2. **用户覆盖值**：每个插件一个 YAML 文件，存于用户配置目录
   （``$CRS_PLUGIN_CONFIG_DIR/<name>.yaml``，缺省
   ``~/.cloud-robotics-sim/plugins/<name>.yaml``）；
3. 运行时传入的显式参数（调用方自行合并，本模块不感知）。

插件在运行时用 :func:`get_plugin_config` 读取合并结果；CLI 的
``cloud-robotics-sim plugins config`` 负责查看/设置/删除覆盖值。
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml

_CONFIG_DIR_ENV = "CRS_PLUGIN_CONFIG_DIR"


def get_config_dir() -> Path:
    """用户插件配置目录（可用环境变量覆盖，便于测试与多环境隔离）。"""
    override = os.environ.get(_CONFIG_DIR_ENV)
    if override:
        return Path(override)
    return Path.home() / ".cloud-robotics-sim" / "plugins"


def _overrides_path(name: str) -> Path:
    _validate_name(name)
    return get_config_dir() / f"{name}.yaml"


def _validate_name(name: str) -> None:
    if not name or any(c in name for c in "/\\:"):
        raise ValueError(f"invalid plugin name: {name!r}")


def read_overrides(name: str) -> dict[str, Any]:
    """读取用户覆盖值；无文件或空文件返回 {}。"""
    path = _overrides_path(name)
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"plugin config must be a mapping: {path}")
    return data


def write_overrides(name: str, data: dict[str, Any]) -> Path:
    """整体写入覆盖值（空 dict 时删除文件）。"""
    path = _overrides_path(name)
    if not data:
        if path.exists():
            path.unlink()
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(data, allow_unicode=True, sort_keys=True),
        encoding="utf-8",
    )
    return path


def set_override(name: str, key: str, value: Any) -> Path:
    """设置单个覆盖键（key 为扁平字符串，可含点号，如 ``solver.dt``）。"""
    if not key:
        raise ValueError("config key must not be empty")
    data = read_overrides(name)
    data[key] = value
    return write_overrides(name, data)


def unset_override(name: str, key: str) -> Path:
    """删除单个覆盖键；不存在时静默。"""
    data = read_overrides(name)
    data.pop(key, None)
    return write_overrides(name, data)


def merge_config(
    defaults: dict[str, Any] | None, overrides: dict[str, Any]
) -> dict[str, Any]:
    """浅合并：覆盖值覆盖默认值（同 key 整体替换，不做深度递归——
    配置项通常是扁平标量；嵌套结构请用 JSON 字符串传值）。
    """
    merged: dict[str, Any] = copy.deepcopy(defaults) if defaults else {}
    for key, value in overrides.items():
        merged[key] = value
    return merged


def get_plugin_config(
    name: str,
    defaults: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """插件运行时读取合并配置（默认值 + 用户覆盖值）。"""
    return merge_config(defaults, read_overrides(name))


def config_defaults(plugin_yaml: dict[str, Any] | None) -> dict[str, Any]:
    """从 plugin.yaml 内容提取默认配置段（兼容缺省/非 dict 情况）。"""
    if not plugin_yaml:
        return {}
    section = plugin_yaml.get("config")
    return dict(section) if isinstance(section, dict) else {}
