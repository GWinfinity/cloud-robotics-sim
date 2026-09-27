"""Genesis 闭环验证：把估计出的关节放进物理仿真里实际开合一次。

闭环的含义：验证对象不是估计器的中间量，而是**最终交付物本身**——
``result_to_urdf`` 生成的 URDF 经 Genesis 1.4.0 的 URDF morph 加载为
articulation，逐关节开环位置控制驱动到限位，用**跟踪误差**判定：

- 关节在 ramp 结束后到达目标行程（误差 < max(2% 行程, 1e-3)）；
- 全程无 NaN / 爆炸（发散本质上也会体现为跟踪误差或非有限值）；
- 被碰撞几何挡住的关节（生成网格上最常见的错误）到不了限位 → 判失败；
- **几何穿透检查**：读取子 link 世界位姿，把碰撞盒角点变换回世界系后与
  基座 AABB 比对——相邻 link 在 Genesis 默认不开自碰撞，仅靠物理接触
  抓不到"镜像铰链侧"这类错误，必须显式做几何判定。

通过的关节保持自动接受；失败的连同仿真读数一起降级进复核队列
（由 pipeline.batch_annotate 统一处理），不静默放过。

使用约束（与项目其它 genesis 插件一致）：

- ``gs.init`` 每个进程一次：模块级懒初始化，首个 GenesisVerifier 触发；
- 每次 verify 新建独立 gs.Scene，逐个资产串行验证；
- 需要 genesis-world（核心依赖）；无 GPU 时自动落到 CPU。

::

    verifier = GenesisVerifier()
    report = verifier.verify(result, parts)
    for check in report.failed:
        ...  # 进复核队列
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .export_urdf import result_to_urdf
from .joint_spec import ArticulatedPart, ArticulationResult

_GS_INITED = False


def _ensure_gs_init(backend: Any | None = None, seed: int | None = 0) -> None:
    """进程内只初始化一次 Genesis。"""
    global _GS_INITED
    if _GS_INITED:
        return
    import genesis as gs

    kwargs: dict[str, Any] = {}
    if backend is not None:
        kwargs["backend"] = backend
    if seed is not None:
        kwargs["seed"] = seed
    gs.init(**kwargs)
    _GS_INITED = True


@dataclass
class JointCheck:
    """单个关节的开合验证结果。"""

    joint: str
    child: str
    target: float
    achieved: float
    tracking_error: float
    passed: bool
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "joint": self.joint,
            "child": self.child,
            "target": round(float(self.target), 6),
            "achieved": round(float(self.achieved), 6),
            "tracking_error": round(float(self.tracking_error), 6),
            "passed": self.passed,
            "reason": self.reason,
        }


@dataclass
class VerificationReport:
    """一次验证的完整结果。"""

    checks: list[JointCheck] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)

    @property
    def ok(self) -> list[JointCheck]:
        return [c for c in self.checks if c.passed]

    @property
    def failed(self) -> list[JointCheck]:
        return [c for c in self.checks if not c.passed]

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "checks": [c.to_dict() for c in self.checks],
        }


class GenesisVerifier:
    """URDF → Genesis articulation → 逐关节开合 rollout 验证。

    Args:
        backend: ``gs.cpu`` / ``gs.gpu`` 等；None 时由 Genesis 自行选择。
        dt: 仿真步长（s）。
        settle_steps: 归零后的稳定步数。
        ramp_steps: 每个关节从初始位匀速驱动到限位的步数。
        hold_steps: 到位后再保持的步数（让位置控制收敛）。
        rel_tol / abs_tol: 通过判定的跟踪误差阈值。
    """

    def __init__(
        self,
        backend: Any | None = None,
        dt: float = 0.01,
        settle_steps: int = 60,
        ramp_steps: int = 240,
        hold_steps: int = 60,
        rel_tol: float = 0.02,
        abs_tol: float = 1e-3,
    ) -> None:
        _ensure_gs_init(backend)
        self._dt = dt
        self._settle_steps = settle_steps
        self._ramp_steps = ramp_steps
        self._hold_steps = hold_steps
        self._rel_tol = rel_tol
        self._abs_tol = abs_tol

    # ------------------------------------------------------------------

    def verify(
        self,
        result: ArticulationResult,
        parts: list[ArticulatedPart],
        *,
        object_name: str = "articulated_object",
    ) -> VerificationReport:
        """对估计结果的每个关节做开合验证。

        每个关节单独驱动（其余保持 0），避免门与抽屉同时打开时互相
        碰撞造成假阴性。
        """
        import genesis as gs

        if not result.joints:
            return VerificationReport()
        urdf_text = result_to_urdf(result, parts, object_name=object_name)

        with tempfile.TemporaryDirectory() as tmp:
            urdf_path = Path(tmp) / f"{object_name}.urdf"
            urdf_path.write_text(urdf_text, encoding="utf-8")
            scene = gs.Scene(
                sim_options=gs.options.SimOptions(dt=self._dt),
                show_viewer=False,
            )
            entity = scene.add_entity(gs.morphs.URDF(file=str(urdf_path), fixed=True))
            scene.build()

            dof_index = self._dof_index(entity, [j.name for j in result.joints])
            parts_by_name = {p.name: p for p in parts}
            base_part = parts_by_name[result.base]
            report = VerificationReport()
            for joint in result.joints:
                report.checks.append(
                    self._check_joint(
                        scene, entity, dof_index, joint, parts_by_name, base_part
                    )
                )
            if hasattr(scene, "_destroy"):
                scene._destroy()
        return report

    # ------------------------------------------------------------------

    @staticmethod
    def _dof_index(entity: Any, joint_names: list[str]) -> dict[str, int]:
        """URDF 关节名 → dofs 向量下标。优先用 Genesis Joint 元数据，按名
        后缀匹配（Genesis 可能给 joint 名加实体前缀）；失败则退回文档顺序。
        """
        name_set = set(joint_names)
        found: dict[str, int] = {}
        for j in getattr(entity, "joints", []) or []:
            jname = getattr(j, "name", "") or ""
            match = next((n for n in name_set if jname.endswith(n) or n in jname), None)
            if match is None:
                continue
            idx = getattr(j, "dof_idx", None)
            if idx is None:
                idx = getattr(j, "dof_idx_local", None)
            if idx is not None:
                found[match] = int(np.atleast_1d(idx)[0])
        for order, name in enumerate(joint_names):
            found.setdefault(name, order)
        return found

    @staticmethod
    def _find_link(entity: Any, child_name: str) -> Any | None:
        """按名字子串匹配子 link（Genesis 会给 link 名加实体前缀）。"""
        for link in getattr(entity, "links", []) or []:
            lname = getattr(link, "name", "") or ""
            if child_name in lname or lname.endswith(child_name):
                return link
        return None

    @staticmethod
    def _local_box(
        child_part: ArticulatedPart, joint: Any
    ) -> tuple[np.ndarray, np.ndarray]:
        """Link 系下的碰撞盒参数（中心 + 半边长）。

        URDF 导出走 rpy=0 约定（link 系与世界系同向），部件几何 origin =
        部件中心 - 关节原点，因此 link 系盒中心 = 部件世界 AABB 中心 - 原点。
        """
        mn, mx = child_part.vertices.min(axis=0), child_part.vertices.max(axis=0)
        return (mn + mx) / 2.0 - np.asarray(joint.origin, dtype=np.float64), (
            mx - mn
        ) / 2.0

    @staticmethod
    def _corners_world(
        center_local: np.ndarray,
        half: np.ndarray,
        pos: np.ndarray,
        quat_wxyz: np.ndarray,
    ) -> np.ndarray:
        import trimesh  # 核心依赖，仅本函数使用

        mat = trimesh.transformations.quaternion_matrix(
            np.asarray(quat_wxyz, dtype=np.float64)
        )
        mat[:3, 3] = pos
        signs = np.array(
            [[s1, s2, s3] for s1 in (-1, 1) for s2 in (-1, 1) for s3 in (-1, 1)]
        )
        corners = center_local + signs * half
        return trimesh.transformations.transform_points(corners, mat)

    def _check_joint(
        self,
        scene: Any,
        entity: Any,
        dof_index: dict[str, int],
        joint: Any,
        parts_by_name: dict[str, ArticulatedPart],
        base_part: ArticulatedPart,
    ) -> JointCheck:
        idx = dof_index[joint.name]
        lower, upper = float(joint.lower), float(joint.upper)
        span = max(upper - lower, 1e-9)
        tol = max(self._rel_tol * span, self._abs_tol)

        n_dofs = int(entity.n_dofs)
        home = np.zeros(n_dofs, dtype=np.float64)

        def to_np(q: Any) -> np.ndarray:
            return np.asarray(
                q.cpu().numpy() if hasattr(q, "cpu") else q, dtype=np.float64
            ).reshape(-1)

        def hold(target: np.ndarray, steps: int) -> None:
            entity.control_dofs_position(target)
            for _ in range(steps):
                scene.step()

        # 回零 + 稳定
        if hasattr(entity, "set_dofs_position"):
            entity.set_dofs_position(home)
        hold(home, self._settle_steps)

        # 匀速 ramp 到 upper，再保持收敛
        for step in range(1, self._ramp_steps + 1):
            alpha = step / self._ramp_steps
            target = home.copy()
            target[idx] = lower + span * alpha
            entity.control_dofs_position(target)
            scene.step()
        final_target = home.copy()
        final_target[idx] = upper
        hold(final_target, self._hold_steps)

        q = to_np(entity.get_dofs_position())
        achieved = float(q[idx])
        finite = bool(np.all(np.isfinite(q)))
        error = abs(achieved - upper)
        if not finite:
            return JointCheck(
                joint.name,
                joint.child,
                upper,
                achieved,
                error,
                False,
                "仿真出现非有限值（发散/爆炸）",
            )
        if error > tol:
            return JointCheck(
                joint.name,
                joint.child,
                upper,
                achieved,
                error,
                False,
                f"关节未到达限位（误差 {error:.4f} > 阈值 {tol:.4f}），可能被碰撞几何阻挡",
            )
        # 几何穿透检查：相邻 link 在 Genesis 默认不开自碰撞，镜像铰链侧这类
        # 错误不会体现为跟踪误差；读取子 link 世界位姿，把碰撞盒角点变换回
        # 世界系后与基座 AABB 做穿透判定。
        link = self._find_link(entity, joint.child)
        if link is None:
            return JointCheck(
                joint.name,
                joint.child,
                upper,
                achieved,
                error,
                True,
                "开合到位（未找到子 link，跳过穿透检查）",
            )
        pos = to_np(link.get_pos())
        quat = to_np(link.get_quat())
        center_local, half = self._local_box(parts_by_name[joint.child], joint)
        corners = self._corners_world(center_local, half, pos, quat)
        if not np.all(np.isfinite(corners)):
            return JointCheck(
                joint.name,
                joint.child,
                upper,
                achieved,
                error,
                False,
                "子 link 位姿出现非有限值",
            )
        base_mn, base_mx = base_part.vertices.min(axis=0), base_part.vertices.max(
            axis=0
        )
        clearance = 2e-3
        inside = np.all(corners > base_mn + clearance, axis=1) & np.all(
            corners < base_mx - clearance, axis=1
        )
        if bool(inside.any()):
            return JointCheck(
                joint.name,
                joint.child,
                upper,
                achieved,
                error,
                False,
                f"到位但碰撞盒与基座穿透（{int(inside.sum())}/8 角点深入），关节侧/轴向可能有误",
            )
        return JointCheck(
            joint.name, joint.child, upper, achieved, error, True, "开合到位"
        )

    # ------------------------------------------------------------------


__all__ = ["GenesisVerifier", "JointCheck", "VerificationReport"]
