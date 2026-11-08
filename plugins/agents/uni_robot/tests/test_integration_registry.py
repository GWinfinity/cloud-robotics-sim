"""genesis 集成注册测试：注册后官方 load_integration 可发现本插件。"""

from __future__ import annotations

import sys
from pathlib import Path

from tests.optional_deps import uni_robot_agent_only

pytestmark = uni_robot_agent_only

_TESTS_DIR = Path(__file__).resolve().parent
_CORE_DIR = _TESTS_DIR.parent / "core"
for _dir in (str(_TESTS_DIR), str(_CORE_DIR)):
    if _dir not in sys.path:
        sys.path.insert(0, _dir)

from uni_robot_genesis import integration  # noqa: E402


def test_register_makes_genesis_discoverable():
    """注册后官方 load_integration("genesis") 返回本插件模块。"""
    from uni_robot_agent.integrations import load_integration

    integration.register()
    module = load_integration("genesis")
    assert module is integration


def test_integration_spec_shape():
    """集成 spec 结构：名称/会话入口/工具集与官方通用 gripper 对齐。"""
    spec = integration.get_integration_spec()
    assert spec.name == "genesis"
    assert callable(spec.create_session)
    assert spec.requires_env_url is True
    # 与 robodojo 相同的通用 gripper 工具集（move/grasp/release/capture/...）
    assert "move" in spec.eval_tool_names
    assert "capture_scene" in spec.eval_tool_names


def test_plugin_package_lazy_guard():
    """包级 eager 导入不依赖 uni_robot_agent 之外的懒加载项。"""
    import uni_robot_genesis

    assert uni_robot_genesis.HAS_UNI_ROBOT_AGENT
    assert hasattr(uni_robot_genesis, "SceneSpec")
