"""uni_robot_genesis：把任意 Genesis 场景暴露为 uni-robot-agent 环境。

通用机器人智能体适配器——不绑定任何测评任务：SceneSpec 描述场景
（机器人/相机/物体）与一段自由文本指令，可选的声明式成功判据；
缺省模式下 episode 不产出成败结论（task_success=None），由调用方评估。

依赖守卫：缺少 uni-robot-agent（pyproject ``agent`` extra）或 genesis 时，
本包仍可导入；仅在使用对应入口时抛出明确错误。
"""

from __future__ import annotations

import importlib
import importlib.util
from typing import Any

__version__ = "0.1.0"

HAS_UNI_ROBOT_AGENT = importlib.util.find_spec("uni_robot_agent") is not None
HAS_GENESIS = importlib.util.find_spec("genesis") is not None

# 无重依赖的模块可以安全 eager 导入
from .judge import GoalJudge, NoGoalJudge, PoseGoalJudge, build_judge  # noqa: E402
from .scene_spec import (  # noqa: E402
    ArmSpec,
    BoxSpec,
    CameraSpec,
    ControlConfig,
    GripperSpec,
    JudgeSpec,
    PoseGoal,
    SceneSpec,
    SimConfig,
)

_LAZY_EXPORTS = {
    "BuiltArm": "built",
    "BuiltScene": "built",
    "GenesisEnvironmentBackend": "backend",
    "build_scene": "builder",
    "GenesisEpisodeHost": "host",
    "serve_main": "host",
    "register": "integration",
    "agent_main": "integration",
    "get_integration_spec": "integration",
}


def _require_extra() -> None:
    if not HAS_UNI_ROBOT_AGENT:
        raise RuntimeError(
            "uni-robot-agent is not installed; install the project 'agent' "
            "extra (uv sync --extra agent) to use the robot-agent adapter"
        )


def __getattr__(name: str) -> Any:
    if name in _LAZY_EXPORTS:
        _require_extra()
        module = importlib.import_module(f".{_LAZY_EXPORTS[name]}", __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(list(globals()) + list(_LAZY_EXPORTS))


__all__ = [
    "HAS_UNI_ROBOT_AGENT",
    "HAS_GENESIS",
    # spec
    "SceneSpec",
    "SimConfig",
    "ControlConfig",
    "ArmSpec",
    "GripperSpec",
    "CameraSpec",
    "BoxSpec",
    "JudgeSpec",
    "PoseGoal",
    # judge
    "GoalJudge",
    "NoGoalJudge",
    "PoseGoalJudge",
    "build_judge",
    # lazy
    *_LAZY_EXPORTS,
]
