"""可选依赖守护 helper（新测试文件统一从这里取 skip 标记）。

已有测试文件使用的 `try/except HAS_*` 与 `pytest.importorskip` 风格继续
有效，不强制迁移；本模块供新文件与需要补守护的文件使用。

用法::

    from tests.optional_deps import genesis_only, HAS_TORCH

    pytestmark = genesis_only
"""

from __future__ import annotations

import importlib.util

import pytest


def has_module(name: str) -> bool:
    """模块是否可导入（不实际导入，无副作用）。"""
    return importlib.util.find_spec(name) is not None


HAS_GENESIS = has_module("genesis")
HAS_TORCH = has_module("torch")
HAS_TRIMESH = has_module("trimesh")
HAS_NUMPY = has_module("numpy")
HAS_MCP = has_module("mcp")
HAS_REDIS = has_module("redis")
HAS_XUVDB = has_module("xuvdb")

genesis_only = pytest.mark.skipif(not HAS_GENESIS, reason="genesis-world not installed")
torch_only = pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
xuvdb_only = pytest.mark.skipif(not HAS_XUVDB, reason="xuvdb not installed")
