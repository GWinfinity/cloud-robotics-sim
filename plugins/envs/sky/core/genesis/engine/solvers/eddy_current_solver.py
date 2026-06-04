"""
涡流求解器 (Eddy Current Solver)

基于复数矢量磁势方法求解时谐电磁场问题：
∇×(1/μ ∇×A) + jωσA = Js

支持集肤效应、邻近效应、铁损计算。
对标 COMSOL AC/DC Module > Magnetic Fields > Eddy Currents。
"""

import numpy as np
import gstaichi as ti
import torch

import genesis as gs
import genesis.utils.array_class as array_class
from genesis.utils.misc import ti_to_torch

from .base_solver import Solver


@ti.data_oriented
class EddyCurrentSolver(Solver):
    """
    涡流求解器
    
    求解 Maxwell 方程组的涡流近似 (低频近似，忽略位移电流)：
    ∇×E = -jωB          (Faraday 定律，频域)
    ∇×H = J = Js + σE   (Ampère 定律，含涡流)
    B = ∇×A, E = -jωA   (势函数定义)
    
    最终方程：
    ∇×(1/μ ∇×A) + jωσA = Js
    
    Parameters
    ----------
    scene : gs.Scene
    sim : gs.Simulator
    options : EddyCurrentOptions
        - frequency: 激励频率 [Hz]
        - resolution: 网格分辨率
        - dx: 空间步长
        - max_iter: 最大迭代次数
        - tol: 收敛容差
        - mu_r: 相对磁导率分布
        - sigma: 电导率分布 [S/m]
        - Js: 源电流密度 [A/m²]
        - skin_depth_factor: 网格细化因子 (基于集肤深度)
    """
    
    def __init__(self, scene, sim, options):
        super().__init__(scene, sim, options)
        
        # 求解参数
        self._frequency = options.frequency
        self._omega = 2.0 * np.pi * self._frequency
        self._resolution = options.resolution
        self._dx = options.dx
        self._max_iter = options.max_iter
        self._tol = options.tol
        
        # 物理常数
        self._mu0 = 4 * np.pi * 1e-7
        
        # 材料参数
        self._mu_r = options.mu_r if hasattr(options, 'mu_r') else 1000.0
        self._sigma = options.sigma if hasattr(options, 'sigma') else 1e6
        self._Js = options.Js if hasattr(options, 'Js') else None
        
        # 集肤深度计算
        self._skin_depth = self._compute_skin_depth()
        
        # 场变量 (复数：实部和虚部分开存储)
        # A = Ar + jAi
        self._Ar = None       # 矢量磁势实部
        self._Ai = None       # 矢量磁势虚部
        self._Br = None       # B 场实部
        self._Bi = None       # B 场虚部
        self._Er = None       # E 场实部
        self._Ei = None       # E 场虚部
        self._Jr = None       # 涡流密度实部
        self._Ji = None       # 涡流密度虚部
        
        # 材料场
        self._mu = None
        self._sigma_field = None
        
        # 迭代状态
        self._converged = False
        self._iter_count = 0
    
    def _compute_skin_depth(self) -> float:
        """计算集肤深度 δ = √(2/(ωμσ))"""
        mu = self._mu0 * self._mu_r
        sigma = self._sigma if isinstance(self._sigma, float) else np.mean(self._sigma)
        delta = np.sqrt(2.0 / (self._omega * mu * sigma))
        return delta
    
    def build(self):
        """构建求解器场变量"""
        super().build()
        
        nx, ny, nz = self._resolution
        
        # 矢量磁势 (复数，3分量)
        self._Ar = ti.Vector.field(3, dtype=gs.ti_float, shape=(nx, ny, nz))
        self._Ai = ti.Vector.field(3, dtype=gs.ti_float, shape=(nx, ny, nz))
        
        # B 场
        self._Br = ti.Vector.field(3, dtype=gs.ti_float, shape=(nx, ny, nz))
        self._Bi = ti.Vector.field(3, dtype=gs.ti_float, shape=(nx, ny, nz))
        
        # E 场
        self._Er = ti.Vector.field(3, dtype=gs.ti_float, shape=(nx, ny, nz))
        self._Ei = ti.Vector.field(3, dtype=gs.ti_float, shape=(nx, ny, nz))
        
        # 涡流密度
        self._Jr = ti.Vector.field(3, dtype=gs.ti_float, shape=(nx, ny, nz))
        self._Ji = ti.Vector.field(3, dtype=gs.ti_float, shape=(nx, ny, nz))
        
        # 材料场
        self._mu = ti.field(dtype=gs.ti_float, shape=(nx, ny, nz))
        self._sigma_field = ti.field(dtype=gs.ti_float, shape=(nx, ny, nz))
        
        # 初始化材料
        self._init_materials()
        
        # 初始化源电流
        if self._Js is not None:
            self._Js_field = ti.Vector.field(3, dtype=gs.ti_float, shape=(nx, ny, nz))
            self._init_source_current()
    
    def _init_materials(self):
        """初始化材料分布"""
        nx, ny, nz = self._resolution
        
        # 磁导率
        mu_np = np.ones((nx, ny, nz), dtype=np.float32) * self._mu0
        if isinstance(self._mu_r, np.ndarray):
            mu_np *= self._mu_r
        else:
            mu_np *= self._mu_r
        self._mu.from_numpy(mu_np)
        
        # 电导率
        sigma_np = np.ones((nx, ny, nz), dtype=np.float32)
        if isinstance(self._sigma, np.ndarray):
            sigma_np *= self._sigma
        else:
            sigma_np *= self._sigma
        self._sigma_field.from_numpy(sigma_np)
    
    def _init_source_current(self):
        """初始化源电流密度"""
        if isinstance(self._Js, np.ndarray):
            self._Js_field.from_numpy(self._Js.astype(np.float32))
        else:
            nx, ny, nz = self._resolution
            Js_np = np.zeros((nx, ny, nz, 3), dtype=np.float32)
            if isinstance(self._Js, (list, tuple)):
                Js_np[:, :, :] = self._Js
            self._Js_field.from_numpy(Js_np)
    
    # ============================================================
    # 复数矢量磁势求解
    # ∇×(1/μ ∇×A) + jωσA = Js
    # 简化为：∇²A - jωμσA = -μJs (Coulomb 规范)
    # ============================================================
    
    @ti.kernel
    def _solve_eddy_current_step(self) -> gs.ti_float:
        """涡流方程 Jacobi 迭代一步 (简化：仅 z 分量，实部和虚部耦合)"""
        nx, ny, nz = self._resolution[0], self._resolution[1], self._resolution[2]
        max_diff = 0.0
        
        for i, j, k in ti.ndrange(nx, ny, nz):
            if i > 0 and i < nx - 1 and j > 0 and j < ny - 1 and k > 0 and k < nz - 1:
                mu = self._mu[i, j, k]
                sigma = self._sigma_field[i, j, k]
                
                # 耦合系数
                beta = self._omega * mu * sigma * self._dx * self._dx
                
                # 源项
                Js_r = 0.0
                Js_i = 0.0
                if ti.static(self._Js is not None):
                    Js_r = mu * self._Js_field[i, j, k][2] * self._dx * self._dx
                    Js_i = 0.0  # 假设源电流为实数
                
                # Ar_z 更新 (实部)
                Ar_new = (
                    self._Ar[i + 1, j, k][2] + self._Ar[i - 1, j, k][2] +
                    self._Ar[i, j + 1, k][2] + self._Ar[i, j - 1, k][2] +
                    self._Ar[i, j, k + 1][2] + self._Ar[i, j, k - 1][2] +
                    beta * self._Ai[i, j, k][2] + Js_r
                ) / 6.0
                
                # Ai_z 更新 (虚部)
                Ai_new = (
                    self._Ai[i + 1, j, k][2] + self._Ai[i - 1, j, k][2] +
                    self._Ai[i, j + 1, k][2] + self._Ai[i, j - 1, k][2] +
                    self._Ai[i, j, k + 1][2] + self._Ai[i, j, k - 1][2] -
                    beta * self._Ar[i, j, k][2] + Js_i
                ) / 6.0
                
                diff_r = ti.abs(Ar_new - self._Ar[i, j, k][2])
                diff_i = ti.abs(Ai_new - self._Ai[i, j, k][2])
                
                if diff_r > max_diff:
                    max_diff = diff_r
                if diff_i > max_diff:
                    max_diff = diff_i
                
                self._Ar[i, j, k][2] = Ar_new
                self._Ai[i, j, k][2] = Ai_new
        
        return max_diff
    
    @ti.kernel
    def _compute_fields(self):
        """从矢量磁势计算 B, E, J 场"""
        nx, ny, nz = self._resolution[0], self._resolution[1], self._resolution[2]
        
        for i, j, k in ti.ndrange(nx, ny, nz):
            if i > 0 and i < nx - 1 and j > 0 and j < ny - 1 and k > 0 and k < nz - 1:
                # B = ∇×A (实部和虚部分开)
                # Br_z = ∂Ar_y/∂x - ∂Ar_x/∂y
                Br_z = (self._Ar[i + 1, j, k][1] - self._Ar[i - 1, j, k][1]) / (2 * self._dx) - \
                       (self._Ar[i, j + 1, k][0] - self._Ar[i, j - 1, k][0]) / (2 * self._dx)
                
                Bi_z = (self._Ai[i + 1, j, k][1] - self._Ai[i - 1, j, k][1]) / (2 * self._dx) - \
                       (self._Ai[i, j + 1, k][0] - self._Ai[i, j - 1, k][0]) / (2 * self._dx)
                
                self._Br[i, j, k] = ti.Vector([0.0, 0.0, Br_z])
                self._Bi[i, j, k] = ti.Vector([0.0, 0.0, Bi_z])
                
                # E = -jωA
                # Er = ωAi, Ei = -ωAr
                sigma = self._sigma_field[i, j, k]
                self._Er[i, j, k] = ti.Vector([0.0, 0.0, self._omega * self._Ai[i, j, k][2]])
                self._Ei[i, j, k] = ti.Vector([0.0, 0.0, -self._omega * self._Ar[i, j, k][2]])
                
                # J = σE
                self._Jr[i, j, k] = ti.Vector([0.0, 0.0, sigma * self._Er[i, j, k][2]])
                self._Ji[i, j, k] = ti.Vector([0.0, 0.0, sigma * self._Ei[i, j, k][2]])
    
    # ============================================================
    # 损耗计算
    # ============================================================
    
    def compute_joule_loss(self) -> float:
        """
        计算焦耳损耗 (涡流损耗)
        P_joule = 0.5 ∫ σ|E|² dV = 0.5 ∫ σ|ωA|² dV
        """
        Er = self._Er.to_numpy()
        Ei = self._Ei.to_numpy()
        sigma = self._sigma_field.to_numpy()
        
        # |E|² = Er² + Ei²
        E2 = np.sum(Er**2 + Ei**2, axis=-1)
        
        # 积分
        dV = self._dx ** 3
        P = 0.5 * np.sum(sigma * E2) * dV
        
        return P
    
    def compute_hysteresis_loss(self, Kh: float = 50.0, alpha: float = 1.6) -> float:
        """
        计算磁滞损耗 (Steinmetz 方程)
        P_hyst = Kh · f · B_max^alpha
        
        Parameters
        ----------
        Kh : float
            磁滞损耗系数 [W/(m³·Hz·T^alpha)]
        alpha : float
            Steinmetz 指数 (通常 1.6-2.2)
        """
        Br = self._Br.to_numpy()
        Bi = self._Bi.to_numpy()
        
        # B_max = √(Br² + Bi²)
        B_mag = np.sqrt(np.sum(Br**2 + Bi**2, axis=-1))
        B_max = np.max(B_mag)
        
        # 体积
        V = self._dx ** 3 * np.prod(self._resolution)
        
        P = Kh * self._frequency * (B_max ** alpha) * V
        
        return P
    
    def compute_core_loss(self, Kh: float = 50.0, Ke: float = 0.0001, 
                          alpha: float = 1.6) -> dict:
        """
        计算总铁芯损耗 (磁滞 + 涡流)
        P_core = P_hyst + P_eddy = Kh·f·B^alpha + Ke·(f·B)²
        
        Returns
        -------
        dict : {'hysteresis': float, 'eddy_current': float, 'total': float}
        """
        P_hyst = self.compute_hysteresis_loss(Kh, alpha)
        P_eddy = self.compute_joule_loss()
        
        return {
            'hysteresis': P_hyst,
            'eddy_current': P_eddy,
            'total': P_hyst + P_eddy
        }
    
    def compute_force_on_conductor(self, conductor_mask: np.ndarray) -> np.ndarray:
        """
        计算磁场对导体的作用力 (Lorentz 力)
        F = ∫ J×B dV
        
        Parameters
        ----------
        conductor_mask : np.ndarray
            导体区域掩码
        """
        Jr = self._Jr.to_numpy()
        Ji = self._Ji.to_numpy()
        Br = self._Br.to_numpy()
        Bi = self._Bi.to_numpy()
        
        # 时均 Lorentz 力: F = 0.5 Re(J × B*)
        # 简化: F = Jr × Br + Ji × Bi
        
        # J × B (z 分量)
        fx = Jr[:, :, :, 1] * Br[:, :, :, 2] - Jr[:, :, :, 2] * Br[:, :, :, 1]
        fy = Jr[:, :, :, 2] * Br[:, :, :, 0] - Jr[:, :, :, 0] * Br[:, :, :, 2]
        fz = Jr[:, :, :, 0] * Br[:, :, :, 1] - Jr[:, :, :, 1] * Br[:, :, :, 0]
        
        # 积分
        dV = self._dx ** 3
        Fx = np.sum(fx * conductor_mask) * dV
        Fy = np.sum(fy * conductor_mask) * dV
        Fz = np.sum(fz * conductor_mask) * dV
        
        return np.array([Fx, Fy, Fz])
    
    # ============================================================
    # 公共接口
    # ============================================================
    
    def solve(self) -> int:
        """
        求解涡流问题
        
        Returns
        -------
        int : 实际迭代次数
        """
        for i in range(self._max_iter):
            max_diff = self._solve_eddy_current_step()
            
            if max_diff < self._tol:
                self._converged = True
                self._iter_count = i
                gs.logger.info(f"Eddy current converged in {i} iterations")
                break
        
        # 计算场量
        self._compute_fields()
        
        return self._iter_count
    
    def step(self):
        """单步更新"""
        self._solve_eddy_current_step()
        self._compute_fields()
    
    def get_B(self) -> np.ndarray:
        """获取磁通密度 B [T] (复数：实部+虚部)"""
        Br = self._Br.to_numpy()
        Bi = self._Bi.to_numpy()
        return Br + 1j * Bi
    
    def get_E(self) -> np.ndarray:
        """获取电场强度 E [V/m] (复数)"""
        Er = self._Er.to_numpy()
        Ei = self._Ei.to_numpy()
        return Er + 1j * Ei
    
    def get_J(self) -> np.ndarray:
        """获取电流密度 J [A/m²] (复数)"""
        Jr = self._Jr.to_numpy()
        Ji = self._Ji.to_numpy()
        return Jr + 1j * Ji
    
    def get_A(self) -> np.ndarray:
        """获取矢量磁势 A [Wb/m] (复数)"""
        Ar = self._Ar.to_numpy()
        Ai = self._Ai.to_numpy()
        return Ar + 1j * Ai
    
    def get_B_magnitude(self) -> np.ndarray:
        """获取 B 场幅值 |B| [T]"""
        Br = self._Br.to_numpy()
        Bi = self._Bi.to_numpy()
        return np.sqrt(np.sum(Br**2 + Bi**2, axis=-1))
    
    def get_current_density_magnitude(self) -> np.ndarray:
        """获取电流密度幅值 |J| [A/m²]"""
        Jr = self._Jr.to_numpy()
        Ji = self._Ji.to_numpy()
        return np.sqrt(np.sum(Jr**2 + Ji**2, axis=-1))
    
    def get_skin_depth(self) -> float:
        """获取集肤深度 [m]"""
        return self._skin_depth
    
    def set_frequency(self, frequency: float):
        """设置激励频率"""
        self._frequency = frequency
        self._omega = 2.0 * np.pi * frequency
        self._skin_depth = self._compute_skin_depth()
    
    def set_conductivity(self, sigma: np.ndarray):
        """设置电导率分布"""
        self._sigma_field.from_numpy(sigma.astype(np.float32))
    
    def set_source_current(self, Js: np.ndarray):
        """设置源电流密度"""
        if self._Js_field is None:
            nx, ny, nz = self._resolution
            self._Js_field = ti.Vector.field(3, dtype=gs.ti_float, shape=(nx, ny, nz))
        self._Js_field.from_numpy(Js.astype(np.float32))
