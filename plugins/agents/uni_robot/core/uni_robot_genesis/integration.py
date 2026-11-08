"""注册 uni-robot-agent 的 "genesis" 集成。

uni-robot-agent 的 ``load_integration(name)`` 先查内部注册表、再回退到
``uni_robot_agent.integrations.{name}`` 内置命名空间。这里通过预填充注册
表接入（其代码路径显式支持），使官方 CLI/SDK 可以
``--integration genesis`` 直接连接本插件托管的 Genesis 环境服务。

已知限制：CLI 的 ``--episode-timeout-s`` 监督模式会在子进程里重放官方
入口，子进程丢失本注册；需要超时控制时请用外部 timeout 命令包裹本插件
的 ``agent`` 入口（见 README）。
"""

from __future__ import annotations

import sys
from typing import Any

_INTEGRATION_NAME = "genesis"


def create_session(context: Any) -> Any:
    """与 robodojo/robouse 相同的通用 gripper 会话（env_url 连接）。"""
    from uni_robot_agent.integrations.gripper import create_session as build

    return build(context)


def get_integration_spec() -> Any:
    """复用官方通用 gripper 集成：相同工具集与执行策略，无测评专用逻辑。"""
    from uni_robot_agent.integrations.gripper import build_gripper_spec

    return build_gripper_spec(name=_INTEGRATION_NAME, create_session=create_session)


def register() -> None:
    """把本模块注册为 "genesis" 集成（幂等）。"""
    from uni_robot_agent.integrations import _INTEGRATION_REGISTRY

    _INTEGRATION_REGISTRY.setdefault(_INTEGRATION_NAME, sys.modules[__name__])


def agent_main(argv: list[str] | None = None) -> int:
    r"""插件的 agent 侧入口：先注册集成，再转交官方 CLI。

    用法::

        uni-robot-genesis agent run --integration genesis \\
            --env-url http://127.0.0.1:18760 --model YOUR_MODEL_ID \\
            --provider-url https://provider.example/v1 --no-sam3
    """
    register()
    from uni_robot_agent.cli import main as cli_main

    cli_main(argv)
    return 0
