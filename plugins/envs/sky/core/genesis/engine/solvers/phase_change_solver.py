"""
相变传热求解器 (Phase Change Solver)

实现固液相变过程的传热计算：
ρ·cp·∂T/∂t = ∇·(k∇T) + ρ·L·∂f/∂t

支持熔化/凝固、潜热释放、相变前沿追踪。
对标 COMSOL Heat Transfer Module > Phase Change。
"""

import numpy as np
import gstaichi as ti
import torch

import genesis as gs
import genesis.utils.array_class as array_class
from genesis.utils.misc import ti_to_torch

from .base_solver import Solver


@ti.data_oriented
class PhaseChangeSolver(Solver):
    """
    相变传热求解器

    求解带有相变的传热方程：
    ρ·∂H/∂t = ∇·(k∇T)
    H = cp·T + f·L  (焓 formulation)

    其中 f 是液相分数：
    f = 0 (T < Ts)  固态
    f = 1 (T > Tl)  液态
    0 < f < 1 (Ts < T < Tl)  糊状区

    Parameters
    ----------
    scene : gs.Scene
    sim : gs.Simulator
    options : PhaseChangeOptions
        - T_solidus: 固相线温度 [K]
        - T_liquidus: 液相线温度 [K]
        - latent_heat: 潜热 [J/kg]
        - cp_solid: 固态比热 [J/(kg·K)]
        - cp_liquid: 液态比热 [J/(kg·K)]
        - k_solid: 固态热导率 [W/(m·K)]
        - k_liquid: 液态热导率 [W/(m·K)]
        - rho_solid: 固态密度 [kg/m³]
        - rho_liquid: 液态密度 [kg/m³]
        - mushy_zone: 糊状区宽度 [K]
    """

    def __init__(self, scene, sim, options):
        super().__init__(scene, sim, options)

        # 相变参数
        self._T_solidus = options.T_solidus
        self._T_liquidus = options.T_liquidus
        self._latent_heat = options.latent_heat
        self._mushy_zone = options.mushy_zone if hasattr(options, 'mushy_zone') else 5.0

        # 材料参数
        self._cp_solid = options.cp_solid if hasattr(options, 'cp_solid') else 1000.0
        self._cp_liquid = options.cp_liquid if hasattr(options, 'cp_liquid') else 1000.0
        self._k_solid = options.k_solid if hasattr(options, 'k_solid') else 1.0
        self._k_liquid = options.k_liquid if hasattr(options, 'k_liquid') else 1.0
        self._rho_solid = options.rho_solid if hasattr(options, 'rho_solid') else 1000.0
        self._rho_liquid = options.rho_liquid if hasattr(options, 'rho_liquid') else 1000.0

        # 网格参数
        self._resolution = options.resolution if hasattr(options, 'resolution') else (64, 64)
        self._dx = options.dx if hasattr(options, 'dx') else 0.001
        self._dt = options.dt if hasattr(options, 'dt') else 0.01

        # 场变量
        self._T = None          # 温度
        self._T_new = None      # 新温度
        self._f_liquid = None   # 液相分数
        self._H = None          # 焓
        self._k_eff = None      # 有效热导率
        self._cp_eff = None     # 有效比热

        # 相变前沿
        self._phase_front = None

    def build(self):
        """构建求解器场变量"""
        super().build()

        nx, ny = self._resolution

        # 温度场
        self._T = ti.field(dtype=gs.ti_float, shape=(nx, ny))
        self._T_new = ti.field(dtype=gs.ti_float, shape=(nx, ny))

        # 液相分数
        self._f_liquid = ti.field(dtype=gs.ti_float, shape=(nx, ny))

        # 焓
        self._H = ti.field(dtype=gs.ti_float, shape=(nx, ny))

        # 有效材料参数
        self._k_eff = ti.field(dtype=gs.ti_float, shape=(nx, ny))
        self._cp_eff = ti.field(dtype=gs.ti_float, shape=(nx, ny))

        # 初始化
        self._init_fields()

    def _init_fields(self):
        """初始化场变量"""
        nx, ny = self._resolution

        # 默认温度 (低于固相线)
        T_np = np.ones((nx, ny), dtype=np.float32) * (self._T_solidus - 10)
        self._T.from_numpy(T_np)
        self._T_new.from_numpy(T_np.copy())

        # 初始液相分数
        f_np = np.zeros((nx, ny), dtype=np.float32)
        self._f_liquid.from_numpy(f_np)

        # 初始焓
        self._update_enthalpy()
        self._update_effective_properties()

    # ============================================================
    # 相变模型
    # ============================================================

    @ti.func
    def _compute_liquid_fraction(self, T: gs.ti_float) -> gs.ti_float:
        """计算液相分数"""
        if T < self._T_solidus:
            return 0.0
        elif T > self._T_liquidus:
            return 1.0
        else:
            # 线性插值 (可改为更平滑的函数)
            return (T - self._T_solidus) / (self._T_liquidus - self._T_solidus)

    @ti.func
    def _compute_enthalpy(self, T: gs.ti_float, f: gs.ti_float) -> gs.ti_float:
        """计算焓 H = cp·T + f·L"""
        cp = self._cp_solid * (1 - f) + self._cp_liquid * f
        return cp * T + f * self._latent_heat

    @ti.func
    def _compute_temperature_from_enthalpy(self, H: gs.ti_float) -> gs.ti_float:
        """从焓计算温度 (迭代求解)"""
        # 简化：假设 cp 为平均值
        cp_avg = (self._cp_solid + self._cp_liquid) / 2

        # 尝试固态
        T_guess = H / self._cp_solid
        if T_guess < self._T_solidus:
            return T_guess

        # 尝试液态
        T_guess = (H - self._latent_heat) / self._cp_liquid
        if T_guess > self._T_liquidus:
            return T_guess

        # 糊状区
        # H = cp_avg * T + f * L, f = (T - Ts) / (Tl - Ts)
        # 解二次方程
        a = 1.0 / (self._T_liquidus - self._T_solidus)
        b = cp_avg - a * self._latent_heat
        c = -H

        if b * b - 4 * a * c >= 0:
            T = (-b + ti.sqrt(b * b - 4 * a * c)) / (2 * a)
            return ti.max(self._T_solidus, ti.min(self._T_liquidus, T))

        return self._T_solidus

    @ti.kernel
    def _update_liquid_fraction(self):
        """更新液相分数"""
        nx, ny = self._resolution[0], self._resolution[1]

        for i, j in ti.ndrange(nx, ny):
            self._f_liquid[i, j] = self._compute_liquid_fraction(self._T[i, j])

    @ti.kernel
    def _update_enthalpy(self):
        """更新焓场"""
        nx, ny = self._resolution[0], self._resolution[1]

        for i, j in ti.ndrange(nx, ny):
            T = self._T[i, j]
            f = self._f_liquid[i, j]
            self._H[i, j] = self._compute_enthalpy(T, f)

    @ti.kernel
    def _update_effective_properties(self):
        """更新有效材料参数"""
        nx, ny = self._resolution[0], self._resolution[1]

        for i, j in ti.ndrange(nx, ny):
            f = self._f_liquid[i, j]

            # 有效热导率 (线性混合)
            self._k_eff[i, j] = self._k_solid * (1 - f) + self._k_liquid * f

            # 有效比热 (含潜热)
            cp = self._cp_solid * (1 - f) + self._cp_liquid * f

            # 在相变区增加等效比热
            if self._T_solidus < self._T[i, j] < self._T_liquidus:
                cp += self._latent_heat / (self._T_liquidus - self._T_solidus)

            self._cp_eff[i, j] = cp

    # ============================================================
    # 传热求解
    # ============================================================

    @ti.kernel
    def _enthalpy_step(self):
        """基于焓方法的显式步进"""
        nx, ny = self._resolution[0], self._resolution[1]
        dx2 = self._dx * self._dx

        for i, j in ti.ndrange(nx, ny):
            if i > 0 and i < nx - 1 and j > 0 and j < ny - 1:
                # 当前焓
                H_old = self._H[i, j]

                # 有效参数
                k = self._k_eff[i, j]
                rho = self._rho_solid * (1 - self._f_liquid[i, j]) + \
                      self._rho_liquid * self._f_liquid[i, j]

                # 热扩散
                alpha = k / (rho * self._cp_eff[i, j])
                r = alpha * self._dt / dx2

                # 邻居温度
                T_ip = self._T[i + 1, j]
                T_im = self._T[i - 1, j]
                T_jp = self._T[i, j + 1]
                T_jm = self._T[i, j - 1]

                # 更新温度
                T_new = self._T[i, j] + r * (
                    T_ip + T_im + T_jp + T_jm - 4.0 * self._T[i, j]
                )

                # 从温度更新焓
                f_new = self._compute_liquid_fraction(T_new)
                H_new = self._compute_enthalpy(T_new, f_new)

                self._T_new[i, j] = T_new
                self._H[i, j] = H_new

    @ti.kernel
    def _swap_temperature(self):
        """交换温度场"""
        nx, ny = self._resolution[0], self._resolution[1]

        for i, j in ti.ndrange(nx, ny):
            self._T[i, j] = self._T_new[i, j]

    # ============================================================
    # 相变前沿追踪
    # ============================================================

    def track_phase_front(self) -> np.ndarray:
        """
        追踪相变前沿位置

        Returns
        -------
        np.ndarray : 相变前沿坐标 (N, 2)
        """
        f = self._f_liquid.to_numpy()
        nx, ny = self._resolution

        # 找到 f ≈ 0.5 的等值线
        front_points = []

        for i in range(nx - 1):
            for j in range(ny - 1):
                # 检查四个角
                f00 = f[i, j]
                f10 = f[i + 1, j]
                f01 = f[i, j + 1]
                f11 = f[i + 1, j + 1]

                # 如果有 crossing
                if (f00 - 0.5) * (f10 - 0.5) < 0 or \
                   (f00 - 0.5) * (f01 - 0.5) < 0 or \
                   (f10 - 0.5) * (f11 - 0.5) < 0 or \
                   (f01 - 0.5) * (f11 - 0.5) < 0:
                    front_points.append([i * self._dx, j * self._dx])

        return np.array(front_points)

    def compute_solid_fraction(self) -> float:
        """计算固相分数"""
        f = self._f_liquid.to_numpy()
        return 1.0 - np.mean(f)

    def compute_melting_rate(self, dt: float) -> float:
        """计算熔化速率 [kg/s]"""
        f_old = self._f_liquid.to_numpy()

        # 更新后
        self._update_liquid_fraction()
        f_new = self._f_liquid.to_numpy()

        # 质量变化
        delta_f = np.mean(f_new - f_old)
        volume = self._dx ** 2 * np.prod(self._resolution)
        rho_avg = (self._rho_solid + self._rho_liquid) / 2
        mass_rate = delta_f * rho_avg * volume / dt

        return mass_rate

    # ============================================================
    # 公共接口
    # ============================================================

    def step(self):
        """单步更新"""
        self._update_effective_properties()
        self._enthalpy_step()
        self._swap_temperature()
        self._update_liquid_fraction()

    def solve(self, n_steps: int = 1000):
        """
        求解相变问题

        Parameters
        ----------
        n_steps : int
            时间步数
        """
        for _ in range(n_steps):
            self.step()

    def get_temperature(self) -> np.ndarray:
        """获取温度场 [K]"""
        return self._T.to_numpy()

    def get_liquid_fraction(self) -> np.ndarray:
        """获取液相分数"""
        return self._f_liquid.to_numpy()

    def get_enthalpy(self) -> np.ndarray:
        """获取焓场 [J/kg]"""
        return self._H.to_numpy()

    def set_temperature(self, T: np.ndarray):
        """设置温度场 [K]"""
        self._T.from_numpy(T.astype(np.float32))
        self._update_liquid_fraction()
        self._update_enthalpy()

    def set_heat_source(self, Q: np.ndarray):
        """设置热源 [W/m³]"""
        # 在焓方程中加入热源
        # 简化：直接加到温度上
        T = self._T.to_numpy()
        T += Q * self._dt / (self._rho_solid * self._cp_solid)
        self._T.from_numpy(T.astype(np.float32))
