"""
热传导求解器

基于 Taichi 的 2D/3D 瞬态和稳态热传导求解器，
支持 Dirichlet/Neumann 边界条件和热源项。

对标 ANSYS Maxwell 的热分析能力。
"""

import numpy as np
import gstaichi as ti
import torch

import genesis as gs
import genesis.utils.array_class as array_class
from genesis.utils.misc import ti_to_torch

from .base_solver import Solver


@ti.data_oriented
class ThermalSolver(Solver):
    """
    热传导求解器
    
    实现 Fourier 热传导方程：
    瞬态: ρ·cp·∂T/∂t = ∇·(k∇T) + Q
    稳态: ∇·(k∇T) + Q = 0
    
    Parameters
    ----------
    scene : gs.Scene
    sim : gs.Simulator
    options : ThermalSolverOptions
        - dim: 2 或 3
        - resolution: 网格分辨率
        - dx: 空间步长
        - dt: 时间步长
        - alpha: 热扩散系数 (k/(ρ·cp))
        - k: 热导率分布
        - rho: 密度分布
        - cp: 比热容分布
        - boundary_temp: Dirichlet 边界温度
        - heat_source: 热源分布 Q
        - solver_type: 'transient' | 'steady'
    """
    
    def __init__(self, scene, sim, options):
        super().__init__(scene, sim, options)
        
        # 热传导参数
        self._dim = options.dim
        self._resolution = options.resolution
        self._dx = options.dx
        self._dt = options.dt
        self._solver_type = options.solver_type
        
        # 材料参数
        self._alpha = options.alpha if hasattr(options, 'alpha') else 1.0
        self._k = options.k if hasattr(options, 'k') else 1.0
        self._rho = options.rho if hasattr(options, 'rho') else 1.0
        self._cp = options.cp if hasattr(options, 'cp') else 1.0
        
        # 边界条件
        self._boundary_temp = options.boundary_temp if hasattr(options, 'boundary_temp') else None
        
        # 热源
        self._heat_source = options.heat_source if hasattr(options, 'heat_source') else None
        
        # 稳定性检查
        if self._solver_type == 'transient':
            self._check_stability()
        
        # 场变量
        self._T = None
        self._T_new = None
        self._source = None
        self._k_field = None
        
        # 稳态迭代参数
        self._max_iter = 10000
        self._tol = 1e-6
    
    def _check_stability(self):
        """检查显式格式的稳定性条件"""
        dt_max = self._dx**2 / (2 * self._dim * self._alpha)
        if self._dt > dt_max:
            gs.logger.warning(
                f"热传导稳定性条件不满足: dt={self._dt:.4e} > dt_max={dt_max:.4e}. "
                f"建议 dt = {0.95 * dt_max:.4e}"
            )
    
    def build(self):
        """构建求解器场变量"""
        super().build()
        
        if self._dim == 2:
            nx, ny = self._resolution
            self._T = ti.field(dtype=gs.ti_float, shape=(nx, ny))
            self._T_new = ti.field(dtype=gs.ti_float, shape=(nx, ny))
            self._source = ti.field(dtype=gs.ti_float, shape=(nx, ny))
            self._k_field = ti.field(dtype=gs.ti_float, shape=(nx, ny))
        else:
            nx, ny, nz = self._resolution
            self._T = ti.field(dtype=gs.ti_float, shape=(nx, ny, nz))
            self._T_new = ti.field(dtype=gs.ti_float, shape=(nx, ny, nz))
            self._source = ti.field(dtype=gs.ti_float, shape=(nx, ny, nz))
            self._k_field = ti.field(dtype=gs.ti_float, shape=(nx, ny, nz))
        
        # 初始化材料
        self._init_materials()
        
        # 初始化热源
        self._init_source()
        
        # 应用边界条件
        self.apply_boundary()
    
    def _init_materials(self):
        """初始化材料参数"""
        if self._dim == 2:
            nx, ny = self._resolution
            k_np = np.ones((nx, ny), dtype=np.float32) * self._k
            self._k_field.from_numpy(k_np)
        else:
            nx, ny, nz = self._resolution
            k_np = np.ones((nx, ny, nz), dtype=np.float32) * self._k
            self._k_field.from_numpy(k_np)
    
    def _init_source(self):
        """初始化热源分布"""
        if self._heat_source is None:
            return
        
        if self._dim == 2:
            nx, ny = self._resolution
            src_np = np.zeros((nx, ny), dtype=np.float32)
            for i in range(nx):
                for j in range(ny):
                    x = i * self._dx
                    y = j * self._dx
                    src_np[i, j] = self._heat_source(x, y)
            self._source.from_numpy(src_np)
        else:
            nx, ny, nz = self._resolution
            src_np = np.zeros((nx, ny, nz), dtype=np.float32)
            for i in range(nx):
                for j in range(ny):
                    for k in range(nz):
                        x = i * self._dx
                        y = j * self._dx
                        z = k * self._dx
                        src_np[i, j, k] = self._heat_source(x, y, z)
            self._source.from_numpy(src_np)
    
    # ============================================================
    # Taichi Kernels - 瞬态求解
    # ============================================================
    
    @ti.kernel
    def _step_transient_2d(self):
        """2D 瞬态热传导显式步进"""
        nx, ny = self._resolution[0], self._resolution[1]
        r = self._alpha * self._dt / (self._dx * self._dx)
        
        for i, j in ti.ndrange(nx, ny):
            if i > 0 and i < nx - 1 and j > 0 and j < ny - 1:
                # FTCS 格式
                self._T_new[i, j] = self._T[i, j] + r * (
                    self._T[i + 1, j] + self._T[i - 1, j] +
                    self._T[i, j + 1] + self._T[i, j - 1] -
                    4.0 * self._T[i, j]
                ) + self._dt * self._source[i, j] / (self._rho * self._cp)
    
    @ti.kernel
    def _step_transient_3d(self):
        """3D 瞬态热传导显式步进"""
        nx, ny, nz = self._resolution[0], self._resolution[1], self._resolution[2]
        r = self._alpha * self._dt / (self._dx * self._dx)
        
        for i, j, k in ti.ndrange(nx, ny, nz):
            if i > 0 and i < nx - 1 and j > 0 and j < ny - 1 and k > 0 and k < nz - 1:
                self._T_new[i, j, k] = self._T[i, j, k] + r * (
                    self._T[i + 1, j, k] + self._T[i - 1, j, k] +
                    self._T[i, j + 1, k] + self._T[i, j - 1, k] +
                    self._T[i, j, k + 1] + self._T[i, j, k - 1] -
                    6.0 * self._T[i, j, k]
                ) + self._dt * self._source[i, j, k] / (self._rho * self._cp)
    
    @ti.kernel
    def _swap_fields_2d(self):
        """交换温度场"""
        for i, j in ti.ndrange(self._resolution[0], self._resolution[1]):
            self._T[i, j] = self._T_new[i, j]
    
    @ti.kernel
    def _swap_fields_3d(self):
        """交换温度场"""
        for i, j, k in ti.ndrange(
            self._resolution[0], self._resolution[1], self._resolution[2]
        ):
            self._T[i, j, k] = self._T_new[i, j, k]
    
    # ============================================================
    # Taichi Kernels - 稳态求解 (Jacobi 迭代)
    # ============================================================
    
    @ti.kernel
    def _step_steady_2d(self) -> gs.ti_float:
        """2D 稳态 Jacobi 迭代一步"""
        nx, ny = self._resolution[0], self._resolution[1]
        max_diff = 0.0
        
        for i, j in ti.ndrange(nx, ny):
            if i > 0 and i < nx - 1 and j > 0 and j < ny - 1:
                T_new = 0.25 * (
                    self._T[i + 1, j] + self._T[i - 1, j] +
                    self._T[i, j + 1] + self._T[i, j - 1] +
                    self._dx * self._dx * self._source[i, j] / self._k_field[i, j]
                )
                diff = ti.abs(T_new - self._T[i, j])
                if diff > max_diff:
                    max_diff = diff
                self._T_new[i, j] = T_new
        
        return max_diff
    
    @ti.kernel
    def _step_steady_3d(self) -> gs.ti_float:
        """3D 稳态 Jacobi 迭代一步"""
        nx, ny, nz = self._resolution[0], self._resolution[1], self._resolution[2]
        max_diff = 0.0
        
        for i, j, k in ti.ndrange(nx, ny, nz):
            if i > 0 and i < nx - 1 and j > 0 and j < ny - 1 and k > 0 and k < nz - 1:
                T_new = (1.0 / 6.0) * (
                    self._T[i + 1, j, k] + self._T[i - 1, j, k] +
                    self._T[i, j + 1, k] + self._T[i, j - 1, k] +
                    self._T[i, j, k + 1] + self._T[i, j, k - 1] +
                    self._dx * self._dx * self._source[i, j, k] / self._k_field[i, j, k]
                )
                diff = ti.abs(T_new - self._T[i, j, k])
                if diff > max_diff:
                    max_diff = diff
                self._T_new[i, j, k] = T_new
        
        return max_diff
    
    # ============================================================
    # 边界条件
    # ============================================================
    
    def apply_boundary(self):
        """应用边界条件"""
        if self._boundary_temp is None:
            return  # Neumann 绝热边界 (默认)
        
        if callable(self._boundary_temp):
            # 空间变化的边界温度
            self._apply_boundary_function()
        else:
            # 恒定边界温度
            if self._dim == 2:
                self._apply_boundary_const_2d(self._boundary_temp)
            else:
                self._apply_boundary_const_3d(self._boundary_temp)
    
    def _apply_boundary_function(self):
        """应用函数型边界条件"""
        if self._dim == 2:
            nx, ny = self._resolution
            T_np = self._T.to_numpy()
            for i in range(nx):
                for j in range(ny):
                    if i == 0 or i == nx - 1 or j == 0 or j == ny - 1:
                        x = i * self._dx
                        y = j * self._dx
                        T_np[i, j] = self._boundary_temp(x, y)
            self._T.from_numpy(T_np)
        else:
            nx, ny, nz = self._resolution
            T_np = self._T.to_numpy()
            for i in range(nx):
                for j in range(ny):
                    for k in range(nz):
                        if (i == 0 or i == nx - 1 or 
                            j == 0 or j == ny - 1 or 
                            k == 0 or k == nz - 1):
                            x = i * self._dx
                            y = j * self._dx
                            z = k * self._dx
                            T_np[i, j, k] = self._boundary_temp(x, y, z)
            self._T.from_numpy(T_np)
    
    @ti.kernel
    def _apply_boundary_const_2d(self, T_bc: gs.ti_float):
        """应用恒定 Dirichlet 边界条件 (2D)"""
        nx, ny = self._resolution[0], self._resolution[1]
        for i in ti.ndrange(nx):
            self._T[i, 0] = T_bc
            self._T[i, ny - 1] = T_bc
        for j in ti.ndrange(ny):
            self._T[0, j] = T_bc
            self._T[nx - 1, j] = T_bc
    
    @ti.kernel
    def _apply_boundary_const_3d(self, T_bc: gs.ti_float):
        """应用恒定 Dirichlet 边界条件 (3D)"""
        nx, ny, nz = self._resolution[0], self._resolution[1], self._resolution[2]
        for i, j in ti.ndrange(nx, ny):
            self._T[i, j, 0] = T_bc
            self._T[i, j, nz - 1] = T_bc
        for i, k in ti.ndrange(nx, nz):
            self._T[i, 0, k] = T_bc
            self._T[i, ny - 1, k] = T_bc
        for j, k in ti.ndrange(ny, nz):
            self._T[0, j, k] = T_bc
            self._T[nx - 1, j, k] = T_bc
    
    # ============================================================
    # 公共接口
    # ============================================================
    
    def step(self):
        """单步更新"""
        if self._solver_type == 'transient':
            if self._dim == 2:
                self._step_transient_2d()
                self._swap_fields_2d()
            else:
                self._step_transient_3d()
                self._swap_fields_3d()
            self.apply_boundary()
        else:
            # 稳态: 执行一次 Jacobi 迭代
            if self._dim == 2:
                max_diff = self._step_steady_2d()
                self._swap_fields_2d()
            else:
                max_diff = self._step_steady_3d()
                self._swap_fields_3d()
            self.apply_boundary()
            return max_diff
    
    def solve_steady(self, max_iter: int = None, tol: float = None) -> int:
        """
        求解稳态问题
        
        Returns
        -------
        int : 实际迭代次数
        """
        if self._solver_type != 'steady':
            raise ValueError("solver_type must be 'steady' for solve_steady()")
        
        max_iter = max_iter or self._max_iter
        tol = tol or self._tol
        
        for i in range(max_iter):
            max_diff = self.step()
            if max_diff < tol:
                gs.logger.info(f"Steady state converged in {i} iterations")
                return i
        
        gs.logger.warning(f"Steady state not converged after {max_iter} iterations")
        return max_iter
    
    def get_temperature(self) -> np.ndarray:
        """获取温度场"""
        return self._T.to_numpy()
    
    def set_temperature(self, T: np.ndarray):
        """设置温度场"""
        self._T.from_numpy(T.astype(np.float32))
    
    def compute_heat_flux(self) -> np.ndarray:
        """计算热流密度 q = -k∇T"""
        T = self._T.to_numpy()
        k = self._k_field.to_numpy()
        
        if self._dim == 2:
            nx, ny = self._resolution
            qx = np.zeros((nx, ny), dtype=np.float32)
            qy = np.zeros((nx, ny), dtype=np.float32)
            
            for i in range(1, nx - 1):
                for j in range(1, ny - 1):
                    qx[i, j] = -k[i, j] * (T[i + 1, j] - T[i - 1, j]) / (2 * self._dx)
                    qy[i, j] = -k[i, j] * (T[i, j + 1] - T[i, j - 1]) / (2 * self._dx)
            
            return np.stack([qx, qy], axis=-1)
        else:
            nx, ny, nz = self._resolution
            qx = np.zeros((nx, ny, nz), dtype=np.float32)
            qy = np.zeros((nx, ny, nz), dtype=np.float32)
            qz = np.zeros((nx, ny, nz), dtype=np.float32)
            
            for i in range(1, nx - 1):
                for j in range(1, ny - 1):
                    for k in range(1, nz - 1):
                        qx[i, j, k] = -k[i, j, k] * (T[i + 1, j, k] - T[i - 1, j, k]) / (2 * self._dx)
                        qy[i, j, k] = -k[i, j, k] * (T[i, j + 1, k] - T[i, j - 1, k]) / (2 * self._dx)
                        qz[i, j, k] = -k[i, j, k] * (T[i, j, k + 1] - T[i, j, k - 1]) / (2 * self._dx)
            
            return np.stack([qx, qy, qz], axis=-1)
    
    def compute_thermal_stress(self, E: float, nu: float, alpha_t: float) -> np.ndarray:
        """
        计算热应力 (简化模型)
        σ = E·α_t·ΔT / (1-ν)
        
        Parameters
        ----------
        E : float
            杨氏模量 [Pa]
        nu : float
            泊松比
        alpha_t : float
            热膨胀系数 [1/K]
        """
        T = self.get_temperature()
        T_ref = np.mean(T)  # 参考温度
        dT = T - T_ref
        
        stress = E * alpha_t * dT / (1 - nu)
        return stress
