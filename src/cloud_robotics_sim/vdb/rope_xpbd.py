"""XPBD 粒子链绳缆仿真，碰撞查询走稀疏 SDF 场（xuvdb 或解析场）。

genesis-world 1.4.x 尚无原生绳缆实体（上游 roadmap，见
``Genesis-Embodied-AI/genesis-world`` discussion #2980），本模块补齐这一环：
 rope 芯是教科书式 small-steps XPBD（Macklin et al. 2019）——每子步
预测-解距离约束（拉伸 + 可选弯曲，带柔度 α）-投影到 SDF 等值面外侧，
摩擦按 PBD 位置级库仑模型。约束求解用 Jacobi 批量迭代（GPU 友好），
以更多子步/迭代补偿收敛速度。

碰撞场抽象为 :class:`SdfField` 协议：``sample(points) -> (d, grad)``。
两个实现：

* :class:`AnalyticSdfField` —— 纯 torch 解析原语（地面/球/并集），
  供 CPU 测试与无 xuvdb 环境的降级演示；
* :class:`XuvdbSdfField` —— 包一层 xuvdb ``GpuVolume``：值走 kernel 侧
  三线性采样（``GpuVolume.sample(order=1)``），法向走 ``GpuVolume.sdf_normal``
  （窄带外零向量、可用 ``extend`` 捐赠）。注意 xuvdb 的 numpy 接口意味着
  每个子步有一次 host 往返；绳子规模（百级粒子）下可接受，更大规模应
  把采样也搬进 quadrants/torch kernel（见 troubleshooting 性能记录）。

与 genesis 侧的耦合入口是 :meth:`XpbdRope.step` 的 ``field`` 参数：
把 :func:`cloud_robotics_sim.vdb.core.static_scene.bake_static_scene`
烘出的静态场景网格代入即可，刚体驱动端点用 :meth:`XpbdRope.pin` 每步
更新位置（单向运动学耦合）。
"""

from __future__ import annotations

from typing import Callable, Protocol, Sequence, cast

import numpy as np
import torch

from cloud_robotics_sim.utils.device import get_device

__all__ = [
    "AnalyticSdfField",
    "SdfField",
    "XpbdRope",
    "XuvdbSdfField",
    "ground_field",
    "sphere_field",
    "union_fields",
]


class SdfField(Protocol):
    """碰撞查询协议：世界坐标采样点 -> (有符号距离, 单位法向)。

    约定与 xuvdb level set 一致：负值在物体内部，法向指向外部；
    法向为零向量表示该点无定义（远离窄带），碰撞投影必须跳过。
    """

    def sample(self, points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]: ...


class AnalyticSdfField:
    """由 torch 可调用函数构成的解析 SDF 场（无 xuvdb 依赖）。

    Args:
        fn: ``(n, 3) points -> (d (n,), grad (n, 3))``，全部为 torch 运算。
    """

    def __init__(
        self,
        fn: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
    ) -> None:
        self._fn = fn

    def sample(self, points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self._fn(points)


def ground_field(height: float = 0.0) -> AnalyticSdfField:
    """水平地面 ``z = height`` 的 SDF（上方为正）。"""
    height = float(height)

    def fn(points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        d = points[:, 2] - height
        grad = torch.zeros_like(points)
        grad[:, 2] = 1.0
        return d, grad

    return AnalyticSdfField(fn)


def sphere_field(center: Sequence[float], radius: float) -> AnalyticSdfField:
    """球体 SDF；``center`` 世界坐标，``radius`` 世界单位。"""
    c = torch.as_tensor(tuple(center), dtype=torch.float32).reshape(1, 3)
    radius = float(radius)

    def fn(points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        c_dev = c.to(device=points.device, dtype=points.dtype)
        delta = points - c_dev
        dist = delta.norm(dim=1).clamp_min(1e-12)
        return dist - radius, delta / dist.unsqueeze(1)

    return AnalyticSdfField(fn)


def union_fields(*fields: SdfField) -> AnalyticSdfField:
    """CSG 并集（逐点取最小距离，法向随胜出者），与 xuvdb min-union 语义一致。"""
    if not fields:
        raise ValueError("union_fields requires at least one field")

    def fn(points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        best_d, best_g = fields[0].sample(points)
        for f in fields[1:]:
            d, g = f.sample(points)
            take = d < best_d
            best_d = torch.where(take, d, best_d)
            best_g = torch.where(take.unsqueeze(1), g, best_g)
        return best_d, best_g

    return AnalyticSdfField(fn)


class XuvdbSdfField:
    """xuvdb ``GpuVolume`` 之上的 SDF 场（kernel 侧采样 + 窄带法向）。

    Args:
        volume: :class:`~cloud_robotics_sim.vdb.core.vdb_xuvdb.XuvdbVolume`
            持有的 level set 网格（bake 帧即世界帧）。
        normal_extend: ``sdf_normal`` 的捐赠半径（体素）。窄带半宽通常
            为 3 个体素、碰撞 margin 在其内，默认 1 足够且开销小。

    每个子步的采样经 numpy 进出 xuvdb kernel（host 往返）；批量调用方
    应一次性传入全部粒子（本类内部即如此）。
    """

    def __init__(self, volume: object, normal_extend: int = 1) -> None:
        from cloud_robotics_sim.vdb.core.vdb_xuvdb import (
            XuvdbVolume,
            _require_xuvdb,
        )

        if not isinstance(volume, XuvdbVolume):
            raise TypeError(f"expected XuvdbVolume, got {type(volume)!r}")
        self._xuvdb = _require_xuvdb()
        try:
            self._gpu = self._xuvdb.GpuVolume(volume.grid)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "grid cannot be packed for kernel-side sampling "
                "(needs f32 scalar axis-aligned level set)"
            ) from exc
        self._normal_extend = int(normal_extend)

    def sample(self, points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pts = points.detach().cpu().numpy().astype(np.float32)
        d = np.asarray(self._gpu.sample(pts, order=1), dtype=np.float32)
        n = np.asarray(
            self._gpu.sdf_normal(pts, extend=self._normal_extend),
            dtype=np.float32,
        )
        dev = points.device
        return (
            torch.from_numpy(d).to(dev),
            torch.from_numpy(n).to(dev).to(points.dtype),
        )


class XpbdRope:
    """XPBD 粒子链绳缆（small-steps，Jacobi 批量约束求解）。

    粒子 ``0..n-1`` 沿 ``direction`` 从 ``origin`` 直线展开，相邻静止
    距离 ``segment_length``；拉伸约束柔度 ``stretch_compliance``
    （0 = 不可伸长的硬约束，单位 m/N），弯曲约束为 ``(i, i+2)`` 距离
    约束、``bend_compliance < 0`` 时关闭。

    注意：初始形状不得与碰撞场相交（深埋粒子会被投影一步弹出），
    调用方负责把绳子摆到表面之外。

    Args:
        n_particles: 粒子数（>= 2）。
        segment_length: 相邻粒子静止距离（世界单位）。
        origin: 粒子 0 初始位置。
        direction: 展开方向（自动归一化）。
        mass: 单粒子质量（kg）。
        gravity: 重力加速度向量。
        stretch_compliance: 拉伸柔度 α（m/N）；0 为刚性。
        bend_compliance: 弯曲柔度；负值关闭弯曲约束。
        friction: 库仑摩擦系数（位置级 PBD 近似，静/动不分）。
        margin: 碰撞边距（粒子中心到 SDF 表面的最小允许距离）。
        substeps: 每步子步数（XPBD 的收敛主力，建议 >= 8）。
        solver_iterations: 每子步 Jacobi 迭代数（默认 4）。
        linear_damping: 线性阻尼（1/s），默认 0。
        device: ``"cuda"`` / ``"musa"`` / ``"cpu"``（None 自动）。

    Example:
        >>> rope = XpbdRope(40, 0.03, origin=(0, 0, 1.0))
        >>> rope.pin(0)  # 固定当前位置
        >>> for _ in range(120):
        ...     rope.step(1 / 60, field=union_fields(ground_field()))
    """

    def __init__(
        self,
        n_particles: int,
        segment_length: float,
        *,
        origin: Sequence[float] = (0.0, 0.0, 0.0),
        direction: Sequence[float] = (0.0, 0.0, -1.0),
        mass: float = 0.01,
        gravity: Sequence[float] = (0.0, 0.0, -9.81),
        stretch_compliance: float = 0.0,
        bend_compliance: float = -1.0,
        friction: float = 0.5,
        margin: float = 0.0,
        substeps: int = 8,
        solver_iterations: int = 4,
        linear_damping: float = 0.0,
        device: str | None = None,
    ) -> None:
        if n_particles < 2:
            raise ValueError("n_particles must be >= 2")
        if segment_length <= 0:
            raise ValueError("segment_length must be positive")
        if mass <= 0:
            raise ValueError("mass must be positive")
        if substeps < 1 or solver_iterations < 1:
            raise ValueError("substeps and solver_iterations must be >= 1")
        self.device = torch.device(get_device(device))
        self.n_particles = int(n_particles)
        self.segment_length = float(segment_length)
        self.mass = float(mass)
        self.stretch_compliance = float(stretch_compliance)
        self.bend_compliance = float(bend_compliance)
        self.friction = float(friction)
        self.margin = float(margin)
        self.substeps = int(substeps)
        self.solver_iterations = int(solver_iterations)
        self.linear_damping = float(linear_damping)
        self.gravity = torch.as_tensor(
            tuple(gravity), dtype=torch.float32, device=self.device
        ).reshape(3)

        o = torch.as_tensor(tuple(origin), dtype=torch.float32, device=self.device)
        d = torch.as_tensor(tuple(direction), dtype=torch.float32, device=self.device)
        d = d / d.norm().clamp_min(1e-12)
        offsets = torch.arange(n_particles, dtype=torch.float32, device=self.device)
        self._positions: torch.Tensor = (
            o.unsqueeze(0) + offsets.unsqueeze(1) * d * self.segment_length
        ).contiguous()
        self._velocities: torch.Tensor = torch.zeros_like(self._positions)
        self._prev: torch.Tensor = torch.zeros_like(self._positions)
        self._inv_mass: torch.Tensor = torch.full(
            (n_particles,), 1.0 / self.mass, dtype=torch.float32, device=self.device
        )
        # 边表：拉伸 (i, i+1)；弯曲 (i, i+2)。
        idx = torch.arange(n_particles, dtype=torch.long, device=self.device)
        self._stretch_edges = torch.stack([idx[:-1], idx[1:]], dim=1)
        self._stretch_rest = torch.full(
            (n_particles - 1,), self.segment_length, device=self.device
        )
        self._lam_stretch = torch.zeros(n_particles - 1, device=self.device)
        self._bend_edges: torch.Tensor | None
        self._bend_rest: torch.Tensor | None
        self._lam_bend: torch.Tensor | None
        if self.bend_compliance >= 0.0 and n_particles > 2:
            self._bend_edges = torch.stack([idx[:-2], idx[2:]], dim=1)
            self._bend_rest = torch.full(
                (n_particles - 2,), 2.0 * self.segment_length, device=self.device
            )
            self._lam_bend = torch.zeros(n_particles - 2, device=self.device)
        else:
            self._bend_edges = None
            self._bend_rest = None
            self._lam_bend = None
        self._pins: dict[int, torch.Tensor] = {}

    # ------------------------------------------------------------------ #
    # 状态访问
    # ------------------------------------------------------------------ #
    @property
    def positions(self) -> torch.Tensor:
        """粒子位置 ``(n, 3)``。只读视图；初始摆形请用 :meth:`reposition`。"""
        return self._positions

    @property
    def velocities(self) -> torch.Tensor:
        """粒子速度 ``(n, 3)``。"""
        return self._velocities

    @property
    def rest_length(self) -> float:
        """绳子总静止长度。"""
        return (self.n_particles - 1) * self.segment_length

    def reposition(self, positions: torch.Tensor) -> None:
        """整体重设粒子位置（清零速度），用于摆初始形状。"""
        p = torch.as_tensor(positions, dtype=torch.float32, device=self.device)
        if p.shape != self._positions.shape:
            raise ValueError(
                f"positions shape {tuple(p.shape)} != {tuple(self._positions.shape)}"
            )
        self._positions.copy_(p)
        self._velocities.zero_()

    def pin(
        self, index: int, position: Sequence[float] | torch.Tensor | None = None
    ) -> None:
        """固定粒子（运动学：每个子步末强制到给定位置）。

        ``position=None`` 固定当前位置；之后可用 ``pin(i, new_pos)`` 拖动。
        """
        if not 0 <= index < self.n_particles:
            raise IndexError(f"pin index {index} out of range")
        if position is None:
            pos = self._positions[index].clone()
        else:
            pos = torch.as_tensor(
                tuple(float(v) for v in np.asarray(position).reshape(-1)),
                dtype=torch.float32,
                device=self.device,
            ).reshape(3)
        self._pins[int(index)] = pos
        self._inv_mass[int(index)] = 0.0

    def unpin(self, index: int) -> None:
        """解除固定并恢复动力学质量。"""
        self._pins.pop(int(index), None)
        self._inv_mass[int(index)] = 1.0 / self.mass

    # ------------------------------------------------------------------ #
    # 主循环
    # ------------------------------------------------------------------ #
    def step(self, dt: float, field: SdfField | None = None) -> None:
        """推进一个时间步 ``dt``（内部切 ``substeps`` 个子步）。

        Args:
            dt: 时间步长（s）。
            field: 碰撞 SDF 场；None 表示自由空间（仅重力/约束）。
        """
        if dt <= 0:
            raise ValueError("dt must be positive")
        h = float(dt) / self.substeps
        alpha_s = self.stretch_compliance / (h * h)
        alpha_b = (
            self.bend_compliance / (h * h) if self.bend_compliance >= 0.0 else None
        )
        pin_idx = torch.as_tensor(
            sorted(self._pins), dtype=torch.long, device=self.device
        )
        pin_pos = (
            torch.stack([self._pins[i] for i in sorted(self._pins)])
            if self._pins
            else None
        )
        for _ in range(self.substeps):
            if self.linear_damping > 0.0:
                self._velocities.mul_(max(1.0 - self.linear_damping * h, 0.0))
            self._velocities.add_(h * self.gravity)
            self._prev.copy_(self._positions)
            self._positions.add_(h * self._velocities)
            self._lam_stretch.zero_()
            if self._lam_bend is not None:
                self._lam_bend.zero_()
            for _ in range(self.solver_iterations):
                self._solve_edges(
                    self._stretch_edges, self._stretch_rest, alpha_s, self._lam_stretch
                )
                if (
                    alpha_b is not None
                    and self._bend_edges is not None
                    and self._bend_rest is not None
                    and self._lam_bend is not None
                ):
                    self._solve_edges(
                        self._bend_edges, self._bend_rest, alpha_b, self._lam_bend
                    )
            if field is not None:
                self._solve_collision(field)
            if pin_pos is not None:
                self._positions.index_copy_(0, pin_idx, pin_pos)
            self._velocities.copy_((self._positions - self._prev) / h)

    def _solve_edges(
        self,
        edges: torch.Tensor,
        rest: torch.Tensor,
        alpha_tilde: float,
        lam: torch.Tensor,
    ) -> None:
        i0 = edges[:, 0]
        i1 = edges[:, 1]
        w0 = self._inv_mass[i0]
        w1 = self._inv_mass[i1]
        delta = self._positions[i0] - self._positions[i1]
        dist = delta.norm(dim=1).clamp_min(1e-12)
        c = dist - rest
        denom = w0 + w1 + alpha_tilde
        dl = torch.where(
            denom > 0.0,
            (-c - alpha_tilde * lam) / denom.clamp_min(1e-12),
            torch.zeros_like(c),
        )
        corr = (delta / dist.unsqueeze(1)) * dl.unsqueeze(1)
        self._positions.index_add_(0, i0, corr * w0.unsqueeze(1))
        self._positions.index_add_(0, i1, -(corr * w1.unsqueeze(1)))
        lam.add_(dl)

    def _solve_collision(self, field: SdfField) -> None:
        d, n = field.sample(self._positions)
        valid = n.norm(dim=1) > 1e-6
        contact = (d < self.margin) & valid
        if not bool(contact.any()):
            return
        pen = torch.where(contact, self.margin - d, torch.zeros_like(d))
        corr = n * pen.unsqueeze(1)
        x_new = torch.where(
            contact.unsqueeze(1), self._positions + corr, self._positions
        )
        if self.friction > 0.0:
            disp = x_new - self._prev
            disp_n = (disp * n).sum(dim=1, keepdim=True) * n
            disp_t = disp - disp_n
            len_t = disp_t.norm(dim=1, keepdim=True).clamp_min(1e-12)
            limit = self.friction * pen
            # 位置级库仑：切位移小于 μ*法向修正则全粘住，否则按比例削减。
            scale = torch.clamp(limit.unsqueeze(1) / len_t, max=1.0)
            x_fric = x_new - disp_t * scale
            x_new = torch.where(contact.unsqueeze(1), x_fric, x_new)
        self._positions.copy_(x_new)

    # ------------------------------------------------------------------ #
    # 诊断
    # ------------------------------------------------------------------ #
    def kinetic_energy(self) -> float:
        """当前动能（J）。"""
        return float(0.5 * self.mass * (self._velocities**2).sum())

    def current_length(self) -> torch.Tensor:
        """当前相邻粒子距离之和（含拉伸）。"""
        seg = self._positions[1:] - self._positions[:-1]
        return cast(torch.Tensor, seg.norm(dim=1).sum())

    def stretch_ratio(self) -> float:
        """当前长度 / 静止长度 - 1（>0 为拉伸）。"""
        return float(self.current_length() / self.rest_length - 1.0)

    def diagnostics(self, field: SdfField) -> dict[str, float]:
        """汇总诊断：最大穿透（相对 margin）、拉伸率、动能、最低点。"""
        d, _ = field.sample(self._positions)
        return {
            "max_penetration": float((self.margin - d).clamp_min(0.0).max()),
            "stretch_ratio": self.stretch_ratio(),
            "kinetic_energy": self.kinetic_energy(),
            "min_z": float(self._positions[:, 2].min()),
        }
