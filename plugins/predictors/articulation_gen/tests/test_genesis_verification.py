"""Genesis 闭环验证测试（需要 genesis-world；缺失时整模块跳过）。

进程内一次 gs.init（GenesisVerifier 懒初始化保证）；每个用例独立
gs.Scene。两个用例：
- 干净合成柜体（2 抽屉 + 1 柜门）：所有关节开合到位 → 全部通过；
- 人为把柜门铰链镜像到错误的棱（几何上必然撞柜体）：闭环验证应判失败
  ——这正是闭环相对纯几何估计多拦下的那类错误。
"""

from __future__ import annotations

import pytest

gs = pytest.importorskip("genesis", reason="genesis-world not installed")  # noqa: E402

from plugins.predictors.articulation_gen import (  # noqa: E402
    ArticulatedPart,
    ArticulationResult,
    JointSpec,
    JointType,
    estimate_joints,
)
from plugins.predictors.articulation_gen.core.verify import (  # noqa: E402
    GenesisVerifier,
)
from plugins.predictors.articulation_gen.tests.test_articulation_gen import (  # noqa: E402
    box_vertices,
)


@pytest.fixture(scope="module")
def verifier() -> GenesisVerifier:
    """模块级共享验证器（懒初始化一次 gs.init）。"""
    return GenesisVerifier(
        backend=gs.cpu, settle_steps=30, ramp_steps=120, hold_steps=30
    )


@pytest.fixture()
def cabinet():
    """合成柜体 + 关节估计结果（1 抽屉 + 1 柜门）。"""
    base = ArticulatedPart(
        "cabinet_body", box_vertices(0, 0, 0.5, 0.8, 0.5, 1.0), kind="base"
    )
    drawer = ArticulatedPart(
        "drawer_top", box_vertices(0, 0.26, 0.78, 0.36, 0.06, 0.18), kind="drawer"
    )
    door = ArticulatedPart(
        "door_left", box_vertices(0, 0.27, 0.22, 0.30, 0.04, 0.40), kind="door"
    )
    parts = [base, drawer, door]
    return parts, estimate_joints(parts)


def test_clean_cabinet_all_joints_pass(verifier, cabinet):
    """干净柜体：抽屉与柜门都应开合到位且无穿透。"""
    parts, result = cabinet
    report = verifier.verify(result, parts)
    assert len(report.checks) == 2  # 1 抽屉 + 1 柜门
    assert report.passed, [c.to_dict() for c in report.failed]


def test_wrong_hinge_side_fails_closed_loop(verifier, cabinet):
    """把铰链镜像到柜体一侧：门会扫进柜体。相邻 link 默认无自碰撞，
    物理接触拦不住它——闭环验证必须靠显式几何穿透检查抓住这个错误。
    """
    parts, result = cabinet
    door = next(j for j in result.joints if j.child == "door_left")
    mirrored_origin = (-door.origin[0], door.origin[1], door.origin[2])
    bad = ArticulationResult(
        base=result.base,
        joints=[
            (
                j
                if j.child != "door_left"
                else JointSpec(
                    name=j.name,
                    parent=j.parent,
                    child=j.child,
                    joint_type=JointType.REVOLUTE,
                    axis=j.axis,
                    origin=mirrored_origin,
                    lower=j.lower,
                    upper=j.upper,
                    confidence=j.confidence,
                )
            )
            for j in result.joints
        ],
    )
    report = verifier.verify(bad, parts)
    door_check = next(c for c in report.checks if c.child == "door_left")
    assert not door_check.passed
    assert "穿透" in door_check.reason
