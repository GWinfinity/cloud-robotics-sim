"""几何关节估计器（纯 numpy，无第三方推理依赖）。

输入是一组**已分割**的 3D 部件（顶点云），输出每个可动部件的关节
类型 / 轴向 / 原点 / 限位 + 置信度。分割可以来自任何来源：

- GLB/USD 的命名节点（``pipeline.parts_from_trimesh_scene``）；
- YOLO + SAM 的 2D mask 反投影回 mesh（``pipeline.annotate_from_images``）；
- PartNet-Mobility / ReplicaCAD 风格的现成部件标签。

估计策略（抽屉/柜门专用，启发式 + 几何验证）：

1. **基座识别**：OBB（PCA 定向包围盒）体积最大的部件为 base。
2. **外露面选择**：对每个非 base 部件，在 6 个 OBB 面中选朝向
   ``部件质心 - 基座质心`` 方向最极端且顶点最聚集（平面性）的面。
3. **类型判定**：``part.kind`` 显式给出时优先；否则按外露面内轮廓的
   高宽比猜——不高过宽的正面 → 抽屉（prismatic），高过宽 → 门（revolute）。
4. **prismatic 拟合**：轴向 = 外露面法向；行程 = 部件深度 × 0.95（封顶
   0.8 m）；方向符号由"平移后不与基座碰撞"的扫描测试决定。
5. **revolute 拟合**：竖直铰链。候选轴为 OBB 左右两条竖直棱；对每条棱
   做旋转扫描（Rodrigues 旋转 OBB 角点，与基座 OBB 粗碰撞检测），取
   "无碰撞自由转角"最大的棱作为铰链，限位 = 该自由角。
6. **置信度** = 面平面性（面上顶点聚集比例）与运动自由度达成率的加权
   和；低于阈值进入人工复核队列。

这套启发式对家具类"方正"部件可靠；对圆润/有机造型应调低
``confidence_threshold`` 并依赖复核队列兜底。
"""

from __future__ import annotations

import numpy as np

from .joint_spec import (
    ArticulatedPart,
    ArticulationResult,
    JointSpec,
    JointType,
    ReviewItem,
)

DEG = np.pi / 180.0

#: 竖直方向（世界系 z 向上，与 Genesis / ReplicaCAD 约定一致）
WORLD_UP = np.array([0.0, 0.0, 1.0])


class NoFreeMotionError(Exception):
    """几何上找不到可行运动。"""


class OBB:
    """PCA 定向包围盒。"""

    def __init__(self, vertices: np.ndarray) -> None:
        centered = vertices - vertices.mean(axis=0)
        cov = (centered.T @ centered) / max(len(vertices) - 1, 1)
        evals, evecs = np.linalg.eigh(cov)
        order = np.argsort(evals)[::-1]
        axes = evecs[:, order]
        if np.linalg.det(axes) < 0:
            axes[:, -1] *= -1.0
        local = centered @ axes
        mn, mx = local.min(axis=0), local.max(axis=0)
        self.axes = axes
        self.half = (mx - mn) / 2.0
        self.center = vertices.mean(axis=0) + axes @ ((mn + mx) / 2.0)

    @property
    def volume(self) -> float:
        return float(8.0 * np.prod(self.half))

    def corners(self) -> np.ndarray:
        """(8, 3) 角点（世界坐标）。"""
        signs = np.array(
            [[s1, s2, s3] for s1 in (-1, 1) for s2 in (-1, 1) for s3 in (-1, 1)]
        )
        return self.center + signs @ (self.axes * self.half).T

    def contains(self, points: np.ndarray, margin: float = 0.0) -> bool:
        """任一点在 OBB 内时返回 True（负 margin 要求点留出间隔）。"""
        local = (np.asarray(points, dtype=np.float64) - self.center) @ self.axes
        return bool(np.any(np.all(np.abs(local) <= self.half + margin, axis=1)))

    def moved(self, rot: np.ndarray, pivot: np.ndarray) -> "MovedOBB":
        """返回绕 pivot 旋转 rot 后的惰性视图。"""
        return MovedOBB(self, rot, pivot)


class MovedOBB:
    """OBB 经刚体变换后的惰性视图（不复制底层数据）。"""

    def __init__(self, obb: OBB, rot: np.ndarray, pivot: np.ndarray) -> None:
        self._obb = obb
        self._rot = rot
        self._pivot = np.asarray(pivot, dtype=np.float64)

    def corners(self) -> np.ndarray:
        return ((self._rot @ (self._obb.corners() - self._pivot).T).T) + self._pivot

    def contains(self, points: np.ndarray, margin: float = 0.0) -> bool:
        pts = np.asarray(points, dtype=np.float64)
        local = (
            ((self._rot.T @ (pts - self._pivot).T).T) - self._obb.center
        ) @ self._obb.axes
        return bool(np.any(np.all(np.abs(local) <= self._obb.half + margin, axis=1)))


def _rotation_about(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rodrigues 旋转矩阵。"""
    a = axis / np.linalg.norm(axis)
    kx, ky, kz = a
    k = np.array([[0.0, -kz, ky], [kz, 0.0, -kx], [-ky, kx, 0.0]])
    return np.eye(3) + np.sin(angle) * k + (1.0 - np.cos(angle)) * (k @ k)


def _collision(
    a_corners: np.ndarray, a: OBB | MovedOBB, b: OBB, margin: float = -0.005
) -> bool:
    """粗碰撞：a 的角点落入 b（或反之）。负 margin 要求二者之间有间隔。"""
    return b.contains(a_corners, margin) or a.contains(b.corners(), margin)


def _outer_face(
    part: ArticulatedPart, obb: OBB, base_obb: OBB
) -> tuple[np.ndarray, np.ndarray, float]:
    """从原始顶点选外露面。

    选面判据：面中心沿法向**超出基座 OBB 表面**（proud）——抽屉/柜门的
    正面必然外露于柜体。多个 proud 面时取超出量×平面性最高者；无 proud
    面（部件完全嵌套/不外露）时退化为质心方向启发式，后续碰撞扫描会把它
    拦进复核队列。

    Returns:
        (normal, face_center, planar_frac)：normal 为指向基座外侧的单位
        法向；face_center 为面上顶点均值；planar_frac ∈ [0, 1] 为贴近该面
        的顶点比例（平面性证据）。
    """
    local = (part.vertices - obb.center) @ obb.axes
    base_center = base_obb.center
    direction = obb.center - base_center
    base_half_along = base_obb.half @ np.abs(base_obb.axes.T @ obb.axes)
    best_proud = -np.inf
    best_fallback = -np.inf
    best: tuple[np.ndarray, np.ndarray, float] | None = None
    fallback: tuple[np.ndarray, np.ndarray, float] | None = None
    for k in range(3):
        for sign in (1.0, -1.0):
            normal = sign * obb.axes[:, k]
            outward = float(np.dot(normal, direction))
            if outward <= 0:
                continue
            dist = np.abs(local[:, k] - sign * obb.half[k])
            frac = float(np.mean(dist < 0.1 * max(obb.half[k], 1e-6)))
            mask = dist < 0.1 * max(obb.half[k], 1e-6)
            face_pts = part.vertices[mask] if mask.any() else part.vertices
            face_center = face_pts.mean(axis=0)
            # 面中心沿法向超出基座表面的距离（在各轴分别比较后的近似）
            beyond = float(np.dot(face_center - base_center, normal)) - float(
                base_half_along[k]
            )
            entry = (normal / np.linalg.norm(normal), face_center, frac)
            if beyond > 0 and beyond * (0.5 + frac) > best_proud:
                best_proud = beyond * (0.5 + frac)
                best = entry
            if outward * (0.5 + frac) > best_fallback:
                best_fallback = outward * (0.5 + frac)
                fallback = entry
    if best is None:
        if fallback is None:  # 部件质心不与基座偏心（完全嵌套）
            raise NoFreeMotionError("部件与基座同心，找不到朝外的面")
        best = fallback
    return best


def _guess_kind(obb: OBB, normal: np.ndarray) -> str:
    """按外露面内轮廓高宽比猜测抽屉 vs 门。

    面内每条 OBB 轴分别取竖直/水平投影，面板高 h = 竖直投影最大值，
    宽 w = 水平投影最大值；不高过宽 → 抽屉（prismatic），高过宽 → 门。
    注意：纯几何对"抽屉前板 vs 矮柜门"存在固有歧义，语义提示
    （part.kind / 节点名 / YOLO 类别）优先。
    """
    h = 0.0
    w = 0.0
    for k in range(3):
        if abs(float(np.dot(obb.axes[:, k], normal))) > 0.9:
            continue
        d = float(np.dot(obb.axes[:, k], WORLD_UP))
        extent = 2.0 * obb.half[k]
        h = max(h, extent * abs(d))
        w = max(w, extent * float(np.sqrt(max(0.0, 1.0 - d * d))))
    if h == 0.0 and w == 0.0:
        return "other"
    return "drawer" if h <= w else "door"


def _fit_prismatic(
    part: ArticulatedPart,
    obb: OBB,
    base_obb: OBB,
    normal: np.ndarray,
    face_center: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, float, float, str]:
    """返回 (axis, origin, lower, upper, free_fraction, rationale)。"""
    depth = float(np.max(np.abs((part.vertices - obb.center) @ normal))) * 2.0
    travel = max(min(0.95 * depth, 0.8), 0.02)
    axis = normal.copy()
    # 方向符号：抽屉嵌在柜体内，微小平移必与基座相交；按"整个部件深度 +
    # 余量"平移，能完全脱离基座的方向即开合方向
    probe_dist = depth + 0.1
    probe = OBB(part.vertices + axis * probe_dist)
    if _collision(probe.corners(), probe, base_obb):
        axis = -axis
        probe = OBB(part.vertices + axis * probe_dist)
        if _collision(probe.corners(), probe, base_obb):
            raise NoFreeMotionError("prismatic 两个平移方向都无法脱离基座")
    return (
        axis,
        face_center,
        0.0,
        travel,
        1.0,
        (f"抽屉滑轨：轴向=外露面法向，行程=深度×0.95（{depth:.3f} m）"),
    )


def _fit_revolute(
    part: ArticulatedPart,
    obb: OBB,
    base_obb: OBB,
    normal: np.ndarray,
    max_angle: float,
    min_angle: float,
) -> tuple[np.ndarray, np.ndarray, float, float, float, str]:
    """返回 (axis, origin, lower, upper, free_fraction, rationale)。"""
    upness = np.abs(obb.axes.T @ WORLD_UP)
    vert_k = int(np.argmax(upness))
    if upness[vert_k] < 0.7:
        raise NoFreeMotionError(
            f"部件无竖直棱（upness={upness[vert_k]:.2f}），暂不支持翻盖/水平铰链"
        )
    # 门面宽度方向 = 垂直于法向与竖直方向的面内方向
    width_dir: np.ndarray | None = None
    half_w = 0.0
    for k in range(3):
        if k == vert_k:
            continue
        candidate = obb.axes[:, k] - float(np.dot(obb.axes[:, k], normal)) * normal
        n = np.linalg.norm(candidate)
        if n < 1e-6:
            continue
        candidate = candidate / n
        extent = float(np.max(np.abs((part.vertices - obb.center) @ candidate)))
        if extent > half_w:
            half_w = extent
            width_dir = candidate
    if width_dir is None:
        raise NoFreeMotionError("无法确定门面宽度方向")

    step = 5.0 * DEG
    best_free = 0.0
    best_pivot: np.ndarray | None = None
    for sign in (1.0, -1.0):
        pivot = obb.center + sign * width_dir * half_w
        free = 0.0
        angle = min_angle
        while angle <= max_angle + 1e-9:
            rot = _rotation_about(WORLD_UP, angle)
            moved = obb.moved(rot, pivot)
            if _collision(moved.corners(), moved, base_obb):
                break
            free = angle
            angle += step
        if free > best_free:
            best_free = free
            best_pivot = pivot
    if best_pivot is None or best_free < min_angle:
        raise NoFreeMotionError("两条候选铰链棱的免费转角均不足")
    return (
        WORLD_UP.copy(),
        best_pivot,
        0.0,
        best_free,
        best_free / max_angle,
        (f"竖直铰链：自由转角 {best_free / DEG:.0f}°（限位取碰撞前最大值）"),
    )


def estimate_joints(
    parts: list[ArticulatedPart],
    *,
    confidence_threshold: float = 0.6,
    max_revolute_deg: float = 110.0,
    min_revolute_deg: float = 10.0,
    planar_weight: float = 0.5,
) -> ArticulationResult:
    """从已分割部件估计关节。

    Args:
        parts: 至少 2 个部件（1 个基座 + 至少 1 个可动部件）。
        confidence_threshold: 低于该置信度的估计进入复核队列而非直接接受。
        max_revolute_deg: revolute 扫描的上限角。
        min_revolute_deg: revolute 被接受所需的最小自由转角。
        planar_weight: 置信度中平面性证据的权重（其余为运动自由度达成率）。

    Returns:
        :class:`ArticulationResult`（joints 按置信度降序）。
    """
    if len(parts) < 2:
        raise ValueError("need >= 2 parts (a base and at least one movable part)")

    obbs = {p.name: OBB(p.vertices) for p in parts}
    base = max(parts, key=lambda p: obbs[p.name].volume)
    base_obb = obbs[base.name]

    result = ArticulationResult(base=base.name)
    for part in parts:
        if part.name == base.name:
            continue
        obb = obbs[part.name]
        try:
            normal, face_center, planar_frac = _outer_face(part, obb, base_obb)
            kind = part.kind or _guess_kind(obb, normal)
            if kind == "drawer":
                axis, origin, lower, upper, free_fraction, rationale = _fit_prismatic(
                    part, obb, base_obb, normal, face_center
                )
                joint_type = JointType.PRISMATIC
            elif kind == "door":
                axis, origin, lower, upper, free_fraction, rationale = _fit_revolute(
                    part,
                    obb,
                    base_obb,
                    normal,
                    max_angle=max_revolute_deg * DEG,
                    min_angle=min_revolute_deg * DEG,
                )
                joint_type = JointType.REVOLUTE
            else:
                raise NoFreeMotionError(
                    f"不支持的部件类型 {kind!r}（支持 drawer/door，可显式指定 part.kind）"
                )
            confidence = (
                planar_weight * planar_frac + (1.0 - planar_weight) * free_fraction
            )
            spec = JointSpec(
                name=f"{base.name}_to_{part.name}",
                parent=base.name,
                child=part.name,
                joint_type=joint_type,
                axis=tuple(float(a) for a in axis),
                origin=tuple(float(o) for o in origin),
                lower=lower,
                upper=upper,
                confidence=confidence,
                rationale=rationale,
            )
            if confidence >= confidence_threshold:
                result.joints.append(spec)
            else:
                result.review.append(
                    ReviewItem(
                        part=part.name,
                        reason=f"置信度 {confidence:.2f} < 阈值 {confidence_threshold}",
                        confidence=confidence,
                    )
                )
        except NoFreeMotionError as exc:
            result.review.append(ReviewItem(part=part.name, reason=str(exc)))

    result.joints.sort(key=lambda j: j.confidence, reverse=True)
    return result


__all__ = ["OBB", "estimate_joints"]
