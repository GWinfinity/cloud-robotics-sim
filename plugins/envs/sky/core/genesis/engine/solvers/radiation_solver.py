"""
辐射传热求解器 (Radiation Solver)

实现表面-表面辐射传热计算：
Q_rad = ε·σ·(T⁴ - T_amb⁴)  (Stefan-Boltzmann 定律)

支持视角因子计算、灰体/黑体辐射、太阳辐射。
对标 COMSOL Heat Transfer Module > Radiation。
"""

import numpy as np
import gstaichi as ti
import torch

import genesis as gs
import genesis.utils.array_class as array_class
from genesis.utils.misc import ti_to_torch

from .base_solver import Solver


@ti.data_oriented
class RadiationSolver(Solver):
    """
    辐射传热求解器
    
    计算表面间的辐射换热：
    Q_ij = A_i · F_ij · σ · (T_i⁴ - T_j⁴) / (1/ε_i + 1/ε_j - 1)
    
    Parameters
    ----------
    scene : gs.Scene
    sim : gs.Simulator
    options : RadiationOptions
        - emissivity: 表面发射率分布
        - view_factors: 视角因子矩阵
        - T_ambient: 环境温度 [K]
        - sigma: Stefan-Boltzmann 常数
        - solar_flux: 太阳辐射通量 [W/m²]
        - solar_direction: 太阳方向向量
    """
    
    def __init__(self, scene, sim, options):
        super().__init__(scene, sim, options)
        
        # 辐射参数
        self._emissivity = options.emissivity if hasattr(options, 'emissivity') else 0.9
        self._T_ambient = options.T_ambient if hasattr(options, 'T_ambient') else 300.0
        self._sigma = 5.670374419e-8  # Stefan-Boltzmann 常数 [W/(m²·K⁴)]
        
        # 太阳辐射
        self._solar_flux = options.solar_flux if hasattr(options, 'solar_flux') else 0.0
        self._solar_direction = options.solar_direction if hasattr(options, 'solar_direction') else (0, 0, -1)
        
        # 视角因子
        self._view_factors = options.view_factors if hasattr(options, 'view_factors') else None
        
        # 表面网格
        self._resolution = options.resolution if hasattr(options, 'resolution') else (64, 64)
        self._dx = options.dx if hasattr(options, 'dx') else 0.01
        
        # 场变量
        self._T_surface = None    # 表面温度
        self._Q_rad = None        # 辐射热流
        self._epsilon_field = None  # 发射率场
        
        # 视角因子矩阵 (简化：平行平板)
        self._F_matrix = None
    
    def build(self):
        """构建求解器场变量"""
        super().build()
        
        nx, ny = self._resolution
        
        # 表面温度
        self._T_surface = ti.field(dtype=gs.ti_float, shape=(nx, ny))
        
        # 辐射热流
        self._Q_rad = ti.field(dtype=gs.ti_float, shape=(nx, ny))
        
        # 发射率场
        self._epsilon_field = ti.field(dtype=gs.ti_float, shape=(nx, ny))
        
        # 初始化发射率
        self._init_emissivity()
        
        # 计算视角因子
        self._compute_view_factors()
    
    def _init_emissivity(self):
        """初始化发射率分布"""
        nx, ny = self._resolution
        eps_np = np.ones((nx, ny), dtype=np.float32) * self._emissivity
        self._epsilon_field.from_numpy(eps_np)
    
    def _compute_view_factors(self):
        """计算视角因子矩阵"""
        if self._view_factors is not None:
            self._F_matrix = self._view_factors
            return
        
        # 简化：两个无限大平行平板
        nx, ny = self._resolution
        n_surfaces = nx * ny
        
        # 视角因子矩阵 (简化模型)
        # F_ij = cosθ_i · cosθ_j / (π·r²) · dA_j
        self._F_matrix = np.zeros((n_surfaces, n_surfaces), dtype=np.float32)
        
        for i in range(n_surfaces):
            for j in range(n_surfaces):
                if i == j:
                    # 自视角因子 (凹面)
                    self._F_matrix[i, j] = 0.0
                else:
                    # 简化：平行表面，距离为 dx
                    self._F_matrix[i, j] = 1.0 / n_surfaces
        
        # 归一化
        row_sums = self._F_matrix.sum(axis=1)
        self._F_matrix = self._F_matrix / row_sums[:, np.newaxis]
    
    # ============================================================
    # 辐射换热计算
    # ============================================================
    
    @ti.kernel
    def _compute_radiation_heat_flux(self):
        """计算净辐射热流"""
        nx, ny = self._resolution[0], self._resolution[1]
        
        for i, j in ti.ndrange(nx, ny):
            T = self._T_surface[i, j]
            eps = self._epsilon_field[i, j]
            
            # 净辐射: Q = ε·σ·(T⁴ - T_amb⁴)
            q_rad = eps * self._sigma * (T**4 - self._T_ambient**4)
            
            # 太阳辐射 (简化)
            if ti.static(self._solar_flux > 0):
                # 假设表面法向与太阳方向夹角为 0
                q_solar = self._solar_flux * eps
                q_rad += q_solar
            
            self._Q_rad[i, j] = q_rad
    
    def compute_net_radiation(self, T_surface: np.ndarray) -> np.ndarray:
        """
        计算表面净辐射换热
        
        Parameters
        ----------
        T_surface : np.ndarray
            表面温度分布 [K]
        
        Returns
        -------
        np.ndarray : 净辐射热流 [W/m²]
        """
        self._T_surface.from_numpy(T_surface.astype(np.float32))
        self._compute_radiation_heat_flux()
        return self._Q_rad.to_numpy()
    
    def compute_radiation_resistance(self, T_avg: float) -> float:
        """
        计算辐射热阻
        
        R_rad = 1 / (4·ε·σ·T_avg³)
        
        Parameters
        ----------
        T_avg : float
            平均温度 [K]
        
        Returns
        -------
        float : 辐射热阻 [m²·K/W]
        """
        eps = np.mean(self._epsilon_field.to_numpy())
        R_rad = 1.0 / (4 * eps * self._sigma * T_avg**3)
        return R_rad
    
    # ============================================================
    # 视角因子计算
    # ============================================================
    
    @staticmethod
    def compute_view_factor_parallel_plates(W: float, H: float, L: float) -> float:
        """
        计算两个平行矩形平板的视角因子
        
        Parameters
        ----------
        W : float
            宽度
        H : float
            高度
        L : float
            间距
        
        Returns
        -------
        float : 视角因子 F_12
        """
        # 简化公式 (适用于近距离)
        X = W / L
        Y = H / L
        
        F = (2 / (np.pi * X * Y)) * (
            np.log(np.sqrt((1 + X**2) * (1 + Y**2) / (1 + X**2 + Y**2))) +
            X * np.sqrt(1 + Y**2) * np.arctan(X / np.sqrt(1 + Y**2)) +
            Y * np.sqrt(1 + X**2) * np.arctan(Y / np.sqrt(1 + X**2)) -
            X * np.arctan(X) - Y * np.arctan(Y)
        )
        
        return F
    
    @staticmethod
    def compute_view_factor_perpendicular_plates(W: float, H: float, L: float) -> float:
        """
        计算两个垂直矩形平板的视角因子
        
        Parameters
        ----------
        W : float
            公共边长度
        H : float
            平板1高度
        L : float
            平板2高度
        
        Returns
        -------
        float : 视角因子 F_12
        """
        H_over_W = H / W
        L_over_W = L / W
        
        # 简化公式
        F = (1 / (np.pi * H_over_W)) * (
            H_over_W * np.arctan(1 / H_over_W) +
            L_over_W * np.arctan(1 / L_over_W) -
            np.sqrt(H_over_W**2 + L_over_W**2) * np.arctan(1 / np.sqrt(H_over_W**2 + L_over_W**2)) +
            0.25 * np.log(
                (1 + H_over_W**2) * (1 + L_over_W**2) / (1 + H_over_W**2 + L_over_W**2) *
                (H_over_W**2 * (1 + H_over_W**2 + L_over_W**2) / ((1 + H_over_W**2) * (H_over_W**2 + L_over_W**2)))**
                (H_over_W**2) *
                (L_over_W**2 * (1 + H_over_W**2 + L_over_W**2) / ((1 + L_over_W**2) * (H_over_W**2 + L_over_W**2)))**
                (L_over_W**2)
            )
        )
        
        return F
    
    # ============================================================
    # 太阳辐射
    # ============================================================
    
    def compute_solar_radiation(self, surface_normal: np.ndarray, 
                                solar_direction: np.ndarray = None) -> np.ndarray:
        """
        计算太阳辐射入射
        
        Parameters
        ----------
        surface_normal : np.ndarray
            表面法向向量 (nx, ny, 3)
        solar_direction : np.ndarray, optional
            太阳方向向量
        
        Returns
        -------
        np.ndarray : 太阳辐射热流 [W/m²]
        """
        if solar_direction is None:
            solar_direction = np.array(self._solar_direction)
        
        solar_direction = solar_direction / np.linalg.norm(solar_direction)
        
        # 计算入射角余弦
        cos_theta = np.dot(surface_normal, solar_direction)
        cos_theta = np.maximum(cos_theta, 0)  # 只考虑正面照射
        
        # 太阳辐射
        q_solar = self._solar_flux * cos_theta
        
        return q_solar
    
    # ============================================================
    # 公共接口
    # ============================================================
    
    def step(self):
        """单步更新"""
        self._compute_radiation_heat_flux()
    
    def get_heat_flux(self) -> np.ndarray:
        """获取辐射热流 [W/m²]"""
        return self._Q_rad.to_numpy()
    
    def set_surface_temperature(self, T: np.ndarray):
        """设置表面温度 [K]"""
        self._T_surface.from_numpy(T.astype(np.float32))
    
    def set_emissivity(self, epsilon: np.ndarray):
        """设置发射率分布"""
        self._epsilon_field.from_numpy(epsilon.astype(np.float32))
    
    def set_ambient_temperature(self, T_amb: float):
        """设置环境温度 [K]"""
        self._T_ambient = T_amb
    
    def set_solar_flux(self, flux: float, direction: tuple = None):
        """
        设置太阳辐射
        
        Parameters
        ----------
        flux : float
            太阳辐射通量 [W/m²]
        direction : tuple, optional
            太阳方向向量
        """
        self._solar_flux = flux
        if direction is not None:
            self._solar_direction = direction
