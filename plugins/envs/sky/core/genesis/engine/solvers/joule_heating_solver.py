"""
焦耳热求解器 (Joule Heating Solver)

耦合电流场和热传导方程：
电流场: ∇·(σ∇V) = 0
热源: Q = σ|∇V|²
热传导: ρ·cp·∂T/∂t = ∇·(k∇T) + Q

支持电阻加热、绕组温升、PCB 热分析。
对标 COMSOL Multiphysics > Joule Heating。
"""

import numpy as np
import gstaichi as ti
import torch

import genesis as gs
import genesis.utils.array_class as array_class
from genesis.utils.misc import ti_to_torch

from .base_solver import Solver


@ti.data_oriented
class JouleHeatingSolver(Solver):
    """
    焦耳热求解器
    
    求解电流场和热传导的耦合问题：
    1. 电流场: ∇·(σ∇V) = 0, J = -σ∇V
    2. 热源: Q = J·E = σ|∇V|²
    3. 热传导: ρ·cp·∂T/∂t = ∇·(k∇T) + Q
    
    Parameters
    ----------
    scene : gs.Scene
    sim : gs.Simulator
    options : JouleHeatingOptions
        - resolution: 网格分辨率
        - dx: 空间步长
        - dt: 时间步长
        - sigma: 电导率分布 [S/m]
        - k: 热导率分布 [W/(m·K)]
        - rho: 密度分布 [kg/m³]
        - cp: 比热容分布 [J/(kg·K)]
        - voltage_boundary: 电压边界条件
        - current_boundary: 电流边界条件
        - max_iter: 电流场最大迭代次数
        - tol: 收敛容差
    """
    
    def __init__(self, scene, sim, options):
        super().__init__(scene, sim, options)
        
        # 求解参数
        self._resolution = options.resolution
        self._dx = options.dx
        self._dt = options.dt
        self._max_iter = options.max_iter
        self._tol = options.tol
        
        # 材料参数
        self._sigma = options.sigma if hasattr(options, 'sigma') else 1e6
        self._k = options.k if hasattr(options, 'k') else 1.0
        self._rho = options.rho if hasattr(options, 'rho') else 1.0
        self._cp = options.cp if hasattr(options, 'cp') else 1.0
        
        # 边界条件
        self._voltage_boundary = options.voltage_boundary if hasattr(options, 'voltage_boundary') else None
        self._current_boundary = options.current_boundary if hasattr(options, 'current_boundary') else None
        
        # 场变量
        self._V = None          # 电势 [V]
        self._J = None          # 电流密度 [A/m²] (3分量)
        self._Q = None          # 热源 [W/m³]
        self._T = None          # 温度 [K]
        self._T_new = None      # 新温度 (用于显式迭代)
        
        # 材料场
        self._sigma_field = None
        self._k_field = None
        
        # 迭代状态
        self._converged = False
        self._iter_count = 0
    
    def build(self):
        """构建求解器场变量"""
        super().build()
        
        nx, ny, nz = self._resolution
        
        # 电势场
        self._V = ti.field(dtype=gs.ti_float, shape=(nx, ny, nz))
        
        # 电流密度
        self._J = ti.Vector.field(3, dtype=gs.ti_float, shape=(nx, ny, nz))
        
        # 热源
        self._Q = ti.field(dtype=gs.ti_float, shape=(nx, ny, nz))
        
        # 温度场
        self._T = ti.field(dtype=gs.ti_float, shape=(nx, ny, nz))
        self._T_new = ti.field(dtype=gs.ti_float, shape=(nx, ny, nz))
        
        # 材料场
        self._sigma_field = ti.field(dtype=gs.ti_float, shape=(nx, ny, nz))
        self._k_field = ti.field(dtype=gs.ti_float, shape=(nx, ny, nz))
        
        # 初始化材料
        self._init_materials()
        
        # 初始化温度
        self._init_temperature()
    
    def _init_materials(self):
        """初始化材料分布"""
        nx, ny, nz = self._resolution
        
        # 电导率
        sigma_np = np.ones((nx, ny, nz), dtype=np.float32)
        if isinstance(self._sigma, np.ndarray):
            sigma_np *= self._sigma
        else:
            sigma_np *= self._sigma
        self._sigma_field.from_numpy(sigma_np)
        
        # 热导率
        k_np = np.ones((nx, ny, nz), dtype=np.float32)
        if isinstance(self._k, np.ndarray):
            k_np *= self._k
        else:
            k_np *= self._k
        self._k_field.from_numpy(k_np)
    
    def _init_temperature(self):
        """初始化温度场"""
        nx, ny, nz = self._resolution
        T_np = np.ones((nx, ny, nz), dtype=np.float32) * 300.0  # 默认 300K
        self._T.from_numpy(T_np)
        self._T_new.from_numpy(T_np.copy())
    
    # ============================================================
    # 电流场求解: ∇·(σ∇V) = 0
    # ============================================================
    
    @ti.kernel
    def _solve_electric_potential_step(self) -> gs.ti_float:
        """电势场 Jacobi 迭代一步"""
        nx, ny, nz = self._resolution[0], self._resolution[1], self._resolution[2]
        max_diff = 0.0
        
        for i, j, k in ti.ndrange(nx, ny, nz):
            if i > 0 and i < nx - 1 and j > 0 and j < ny - 1 and k > 0 and k < nz - 1:
                # 获取相邻电导率
                sigma_ip = self._sigma_field[i + 1, j, k]
                sigma_im = self._sigma_field[i - 1, j, k]
                sigma_jp = self._sigma_field[i, j + 1, k]
                sigma_jm = self._sigma_field[i, j - 1, k]
                sigma_kp = self._sigma_field[i, j, k + 1]
                sigma_km = self._sigma_field[i, j, k - 1]
                
                # 调和平均电导率
                sigma_x = 2.0 * sigma_ip * sigma_im / (sigma_ip + sigma_im + 1e-10)
                sigma_y = 2.0 * sigma_jp * sigma_jm / (sigma_jp + sigma_jm + 1e-10)
                sigma_z = 2.0 * sigma_kp * sigma_km / (sigma_kp + sigma_km + 1e-10)
                
                # Jacobi 更新
                V_new = (
                    sigma_x * (self._V[i + 1, j, k] + self._V[i - 1, j, k]) +
                    sigma_y * (self._V[i, j + 1, k] + self._V[i, j - 1, k]) +
                    sigma_z * (self._V[i, j, k + 1] + self._V[i, j, k - 1])
                ) / (2.0 * (sigma_x + sigma_y + sigma_z))
                
                diff = ti.abs(V_new - self._V[i, j, k])
                if diff > max_diff:
                    max_diff = diff
                
                self._V[i, j, k] = V_new
        
        return max_diff
    
    @ti.kernel
    def _compute_current_density(self):
        """计算电流密度 J = -σ∇V"""
        nx, ny, nz = self._resolution[0], self._resolution[1], self._resolution[2]
        
        for i, j, k in ti.ndrange(nx, ny, nz):
            if i > 0 and i < nx - 1 and j > 0 and j < ny - 1 and k > 0 and k < nz - 1:
                sigma = self._sigma_field[i, j, k]
                
                # J = -σ∇V
                Jx = -sigma * (self._V[i + 1, j, k] - self._V[i - 1, j, k]) / (2 * self._dx)
                Jy = -sigma * (self._V[i, j + 1, k] - self._V[i, j - 1, k]) / (2 * self._dx)
                Jz = -sigma * (self._V[i, j, k + 1] - self._V[i, j, k - 1]) / (2 * self._dx)
                
                self._J[i, j, k] = ti.Vector([Jx, Jy, Jz])
    
    @ti.kernel
    def _compute_heat_source(self):
        """计算焦耳热 Q = σ|∇V|² = |J|²/σ"""
        nx, ny, nz = self._resolution[0], self._resolution[1], self._resolution[2]
        
        for i, j, k in ti.ndrange(nx, ny, nz):
            if i > 0 and i < nx - 1 and j > 0 and j < ny - 1 and k > 0 and k < nz - 1:
                sigma = self._sigma_field[i, j, k]
                
                # |∇V|²
                dVdx = (self._V[i + 1, j, k] - self._V[i - 1, j, k]) / (2 * self._dx)
                dVdy = (self._V[i, j + 1, k] - self._V[i, j - 1, k]) / (2 * self._dx)
                dVdz = (self._V[i, j, k + 1] - self._V[i, j, k - 1]) / (2 * self._dx)
                
                grad_V2 = dVdx**2 + dVdy**2 + dVdz**2
                
                # Q = σ|∇V|²
                self._Q[i, j, k] = sigma * grad_V2
    
    # ============================================================
    # 热传导求解: ρ·cp·∂T/∂t = ∇·(k∇T) + Q
    # ============================================================
    
    @ti.kernel
    def _solve_heat_step(self):
        """热传导显式步进"""
        nx, ny, nz = self._resolution[0], self._resolution[1], self._resolution[2]
        
        # 热扩散系数
        alpha = self._k / (self._rho * self._cp)
        r = alpha * self._dt / (self._dx * self._dx)
        
        for i, j, k in ti.ndrange(nx, ny, nz):
            if i > 0 and i < nx - 1 and j > 0 and j < ny - 1 and k > 0 and k < nz - 1:
                # FTCS 格式
                self._T_new[i, j, k] = self._T[i, j, k] + r * (
                    self._T[i + 1, j, k] + self._T[i - 1, j, k] +
                    self._T[i, j + 1, k] + self._T[i, j - 1, k] +
                    self._T[i, j, k + 1] + self._T[i, j, k - 1] -
                    6.0 * self._T[i, j, k]
                ) + self._dt * self._Q[i, j, k] / (self._rho * self._cp)
    
    @ti.kernel
    def _swap_temperature(self):
        """交换温度场"""
        nx, ny, nz = self._resolution[0], self._resolution[1], self._resolution[2]
        
        for i, j, k in ti.ndrange(nx, ny, nz):
            self._T[i, j, k] = self._T_new[i, j, k]
    
    # ============================================================
    # 边界条件
    # ============================================================
    
    def apply_voltage_boundary(self):
        """应用电压边界条件"""
        if self._voltage_boundary is None:
            return
        
        # 解析边界条件
        for boundary, value in self._voltage_boundary.items():
            self._apply_voltage_boundary_kernel(boundary, value)
    
    @ti.kernel
    def _apply_voltage_boundary_kernel(self, boundary: ti.i32, value: gs.ti_float):
        """应用电压边界"""
        nx, ny, nz = self._resolution[0], self._resolution[1], self._resolution[2]
        
        # boundary: 0=x_min, 1=x_max, 2=y_min, 3=y_max, 4=z_min, 5=z_max
        if boundary == 0:  # x_min
            for j, k in ti.ndrange(ny, nz):
                self._V[0, j, k] = value
        elif boundary == 1:  # x_max
            for j, k in ti.ndrange(ny, nz):
                self._V[nx - 1, j, k] = value
        elif boundary == 2:  # y_min
            for i, k in ti.ndrange(nx, nz):
                self._V[i, 0, k] = value
        elif boundary == 3:  # y_max
            for i, k in ti.ndrange(nx, nz):
                self._V[i, ny - 1, k] = value
        elif boundary == 4:  # z_min
            for i, j in ti.ndrange(nx, ny):
                self._V[i, j, 0] = value
        elif boundary == 5:  # z_max
            for i, j in ti.ndrange(nx, ny):
                self._V[i, j, nz - 1] = value
    
    def apply_temperature_boundary(self):
        """应用温度边界条件"""
        # 默认绝热边界 (Neumann)
        # 可在子类中扩展
        pass
    
    # ============================================================
    # 损耗和功率计算
    # ============================================================
    
    def compute_total_power(self) -> float:
        """计算总焦耳热功率 P = ∫ Q dV"""
        Q = self._Q.to_numpy()
        dV = self._dx ** 3
        P = np.sum(Q) * dV
        return P
    
    def compute_resistance(self) -> float:
        """计算等效电阻 R = V/I"""
        # 从边界条件获取电压和电流
        if self._voltage_boundary is None:
            return 0.0
        
        # 计算总电流
        J = self._J.to_numpy()
        # 通过某个截面的电流
        J_mag = np.linalg.norm(J, axis=-1)
        # 简化：取平均电流密度乘以面积
        I = np.mean(J_mag) * (self._dx ** 2)
        
        # 电压差
        V_values = list(self._voltage_boundary.values()) if isinstance(self._voltage_boundary, dict) else [0]
        if len(V_values) >= 2:
            V_diff = abs(V_values[1] - V_values[0])
        else:
            V_diff = V_values[0] if V_values else 0
        
        R = V_diff / I if I > 0 else 0
        return R
    
    def compute_max_temperature(self) -> float:
        """获取最高温度"""
        T = self._T.to_numpy()
        return np.max(T)
    
    def compute_average_temperature(self) -> float:
        """获取平均温度"""
        T = self._T.to_numpy()
        return np.mean(T)
    
    # ============================================================
    # 公共接口
    # ============================================================
    
    def solve_electric_field(self) -> int:
        """
        求解电流场
        
        Returns
        -------
        int : 实际迭代次数
        """
        for i in range(self._max_iter):
            max_diff = self._solve_electric_potential_step()
            self.apply_voltage_boundary()
            
            if max_diff < self._tol:
                self._converged = True
                gs.logger.info(f"Electric field converged in {i} iterations")
                break
        
        # 计算电流密度和热源
        self._compute_current_density()
        self._compute_heat_source()
        
        return i
    
    def solve_thermal(self, n_steps: int = 100):
        """
        求解热传导 (瞬态)
        
        Parameters
        ----------
        n_steps : int
            时间步数
        """
        for _ in range(n_steps):
            self._solve_heat_step()
            self._swap_temperature()
            self.apply_temperature_boundary()
    
    def solve_coupled(self, n_thermal_steps: int = 100) -> dict:
        """
        求解耦合的焦耳热问题
        
        1. 求解电流场
        2. 计算焦耳热
        3. 求解热传导
        
        Returns
        -------
        dict : {'electric_iter': int, 'max_temperature': float, 'total_power': float}
        """
        # 求解电流场
        elec_iter = self.solve_electric_field()
        
        # 求解热传导
        self.solve_thermal(n_thermal_steps)
        
        return {
            'electric_iter': elec_iter,
            'max_temperature': self.compute_max_temperature(),
            'average_temperature': self.compute_average_temperature(),
            'total_power': self.compute_total_power(),
            'resistance': self.compute_resistance()
        }
    
    def step(self):
        """单步更新 (用于耦合仿真)"""
        # 电流场
        self._solve_electric_potential_step()
        self.apply_voltage_boundary()
        self._compute_current_density()
        self._compute_heat_source()
        
        # 热场
        self._solve_heat_step()
        self._swap_temperature()
        self.apply_temperature_boundary()
    
    def get_voltage(self) -> np.ndarray:
        """获取电势分布 [V]"""
        return self._V.to_numpy()
    
    def get_current_density(self) -> np.ndarray:
        """获取电流密度 [A/m²]"""
        return self._J.to_numpy()
    
    def get_heat_source(self) -> np.ndarray:
        """获取热源分布 [W/m³]"""
        return self._Q.to_numpy()
    
    def get_temperature(self) -> np.ndarray:
        """获取温度分布 [K]"""
        return self._T.to_numpy()
    
    def set_voltage_boundary(self, boundary: str, value: float):
        """
        设置电压边界条件
        
        Parameters
        ----------
        boundary : str
            'x_min', 'x_max', 'y_min', 'y_max', 'z_min', 'z_max'
        value : float
            边界电压 [V]
        """
        boundary_map = {
            'x_min': 0, 'x_max': 1,
            'y_min': 2, 'y_max': 3,
            'z_min': 4, 'z_max': 5
        }
        
        if boundary in boundary_map:
            self._apply_voltage_boundary_kernel(boundary_map[boundary], value)
    
    def set_conductivity(self, sigma: np.ndarray):
        """设置电导率分布"""
        self._sigma_field.from_numpy(sigma.astype(np.float32))
    
    def set_temperature(self, T: np.ndarray):
        """设置温度分布"""
        self._T.from_numpy(T.astype(np.float32))
