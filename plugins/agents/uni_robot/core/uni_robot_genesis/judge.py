"""可插拔成功判据（goal judge）。

通用设计：judge 是可选的。缺省（NoGoalJudge）表示纯执行/数据采集模式，
task_success 恒为 None——由调用方（人或下游流程）自行评估结果，适配器
不内置任何测评逻辑。声明式 PoseGoalJudge 覆盖"物体到达目标点"这类常见
判定，供需要自动成败反馈的场景显式启用。
"""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np

from .scene_spec import JudgeSpec


class GoalJudge(Protocol):
    """episode 结束时评估一次成败；返回 None 表示无法判定。"""

    def evaluate(self, objects: dict[str, Any]) -> bool | None: ...


class NoGoalJudge:
    """通用模式：不判定。task_success 保持 None（不伪造、不 privileged）。"""

    def evaluate(self, objects: dict[str, Any]) -> bool | None:
        return None


class PoseGoalJudge:
    """所有声明目标同时满足（各物体当前位置在目标容差内）即为成功。"""

    def __init__(self, spec: JudgeSpec) -> None:
        self._goals = list(spec.goals)

    def evaluate(self, objects: dict[str, Any]) -> bool | None:
        for goal in self._goals:
            entity = objects.get(goal.object_name)
            if entity is None:
                return None  # 判据对象缺失：无法判定而不是判负
            detach = getattr(entity.get_pos(), "detach", None)
            if callable(detach):
                position = np.asarray(detach().cpu().numpy(), dtype=np.float64).reshape(
                    -1
                )[:3]
            else:
                position = np.asarray(entity.get_pos(), dtype=np.float64).reshape(-1)[
                    :3
                ]
            target = np.asarray(goal.position, dtype=np.float64)
            if not np.isfinite(position).all():
                return None
            if float(np.linalg.norm(position - target)) > goal.tolerance_m:
                return False
        return True


def build_judge(spec: JudgeSpec) -> GoalJudge:
    """按 JudgeSpec 构造 judge；空 spec 退化为 NoGoalJudge。"""
    if spec.enabled:
        return PoseGoalJudge(spec)
    return NoGoalJudge()
