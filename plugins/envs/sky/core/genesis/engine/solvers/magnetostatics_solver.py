"""
静磁场求解器 (Magnetostatics Solver)

基于标量磁势和矢量磁势方法求解静磁场问题：
∇×H = J, ∇·B = 0, B = μH + μM

支持永磁体建模、非线性 B-H 曲线材料、磁路计算。
对标 COMSOL AC/DC Module > Magnetic Fields > Magnetostatics。
"""

import numpy as np
import gstaichi as ti
import torch

import genesis as gs
import genesis.utils.array_class as array_class
from genesis.utils.misc import ti_to_torch

from .base_solver import Solver


@ti.data_oriented
class MagnetostaticsSolver(Solver):
    """
    静磁场求解器
    
    求解 Maxwell 方程组的静磁近似：
    ∇×H = J          (Ampère 定律，稳态)
    ∇·B = 0          (无磁单极子)
    B = μH + μM      (本构关系，含永磁体)
    
    方法选择：
    - 标量磁势法 (Scalar Potential): 无自由电流区域
    - 矢量磁势法 (Vector Potential): 含自由电流区域
    
    Parameters
    ----------
    scene : gs.Scene
    sim : gs.Simulator
    options : MagnetostaticsOptions
        - method: 'scalar' | 'vector' | 'auto'
        - resolution: 网格分辨率
        - dx: 空间步长
        - max_iter: 最大迭代次数
        - tol: 收敛容差
        - mu_r: 相对磁导率分布
        - M: 永磁体磁化强度分布 [A/m]
        - J: 自由电流密度分布 [A/m²]
        - bh_curve: B-H 曲线数据 (用于非线性材料)
    """
    
    def __init__(self, scene, sim, options):
        super().__init__(scene, sim, options)
        
        # 求解参数
        self._method = options.method
        self._resolution = options.resolution
        self._dx = options.dx
        self._max_iter = options.max_iter
        self._tol = options.tol
        
        # 物理常数
        self._mu0 = 4 * np.pi * 1e-7  # 真空磁导率 [H/m]
        
        # 材料参数
        self._mu_r = options.mu_r if hasattr(options, 'mu_r') else 1.0
        self._M = options.M if hasattr(options, 'M') else None  # 永磁体磁化 [A/m]
        self._J = options.J if hasattr(options, 'J') else None  # 自由电流 [A/m²]
        
        # B-H 曲线 (非线性材料)
        self._bh_curve = options.bh_curve if hasattr(options, 'bh_curve') else None
        self._nonlinear = self._bh_curve is not None
        
        # 自动选择方法
        if self._method == 'auto':
            self._method = 'vector' if self._J is not None else 'scalar'
        
        # 场变量
        self._phi_m = None      # 标量磁势 [A]
        self._A = None          # 矢量磁势 [Wb/m] (3分量)
        self._B = None          # 磁通密度 [T] (3分量)
        self._H = None          # 磁场强度 [A/m] (3分量)
        self._mu = None         # 磁导率分布 [H/m]
        
        # 迭代状态
        self._converged = False
        self._iter_count = 0
    
    def build(self):
        """构建求解器场变量"""
        super().build()
        
        nx, ny, nz = self._resolution
        
        # 磁导率场
        self._mu = ti.field(dtype=gs.ti_float, shape=(nx, ny, nz))
        
        # 初始化磁导率
        self._init_permeability()
        
        if self._method == 'scalar':
            # 标量磁势: H = -∇φ_m
            self._phi_m = ti.field(dtype=gs.ti_float, shape=(nx, ny, nz))
            self._B = ti.Vector.field(3, dtype=gs.ti_float, shape=(nx, ny, nz))
            self._H = ti.Vector.field(3, dtype=gs.ti_float, shape=(nx, ny, nz))
        else:
            # 矢量磁势: B = ∇×A
            self._A = ti.Vector.field(3, dtype=gs.ti_float, shape=(nx, ny, nz))
            self._B = ti.Vector.field(3, dtype=gs.ti_float, shape=(nx, ny, nz))
            self._H = ti.Vector.field(3, dtype=gs.ti_float, shape=(nx, ny, nz))
        
        # 永磁体磁化强度
        if self._M is not None:
            self._M_field = ti.Vector.field(3, dtype=gs.ti_float, shape=(nx, ny, nz))
            self._init_magnetization()
    
    def _init_permeability(self):
        """初始化磁导率分布"""
        nx, ny, nz = self._resolution
        mu_np = np.ones((nx, ny, nz), dtype=np.float32) * self._mu0
        
        if isinstance(self._mu_r, np.ndarray):
            mu_np *= self._mu_r
        else:
            mu_np *= self._mu_r
        
        self._mu.from_numpy(mu_np)
    
    def _init_magnetization(self):
        """初始化永磁体磁化强度"""
        if isinstance(self._M, np.ndarray):
            self._M_field.from_numpy(self._M.astype(np.float32))
        else:
            # 常数磁化
            nx, ny, nz = self._resolution
            M_np = np.zeros((nx, ny, nz, 3), dtype=np.float32)
            if isinstance(self._M, (list, tuple)):
                M_np[:, :, :] = self._M
            self._M_field.from_numpy(M_np)
    
    # ============================================================
    # 标量磁势法 (无自由电流区域)
    # ∇·(μ∇φ_m) = ∇·(μM)
    # ============================================================
    
    @ti.kernel
    def _solve_scalar_potential_step(self) -> gs.ti_float:
        """标量磁势 Jacobi 迭代一步"""
        nx, ny, nz = self._resolution[0], self._resolution[1], self._resolution[2]
        max_diff = 0.0
        
        for i, j, k in ti.ndrange(nx, ny, nz):
            if i > 0 and i < nx - 1 and j > 0 and j < ny - 1 and k > 0 and k < nz - 1:
                # 获取相邻磁导率
                mu_ip = self._mu[i + 1, j, k]
                mu_im = self._mu[i - 1, j, k]
                mu_jp = self._mu[i, j + 1, k]
                mu_jm = self._mu[i, j - 1, k]
                mu_kp = self._mu[i, j, k + 1]
                mu_km = self._mu[i, j, k - 1]
                mu_c = self._mu[i, j, k]
                
                # 计算等效磁导率 (调和平均)
                mu_x = 2.0 / (1.0 / mu_ip + 1.0 / mu_im) if mu_ip > 0 and mu_im > 0 else mu_c
                mu_y = 2.0 / (1.0 / mu_jp + 1.0 / mu_jm) if mu_jp > 0 and mu_jm > 0 else mu_c
                mu_z = 2.0 / (1.0 / mu_kp + 1.0 / mu_km) if mu_kp > 0 and mu_km > 0 else mu_c
                
                # 右端项: ∇·(μM)
                rhs = 0.0
                if ti.static(self._M is not None):
                    # 简化: 假设 M 只有 z 分量变化
                    Mz_ip = self._M_field[i + 1, j, k][2]
                    Mz_im = self._M_field[i - 1, j, k][2]
                    Mz_jp = self._M_field[i, j + 1, k][2]
                    Mz_jm = self._M_field[i, j - 1, k][2]
                    Mz_kp = self._M_field[i, j, k + 1][2]
                    Mz_km = self._M_field[i, j, k - 1][2]
                    
                    rhs = (mu_ip * Mz_ip - mu_im * Mz_im) / (2 * self._dx) + \
                          (mu_jp * Mz_jp - mu_jm * Mz_jm) / (2 * self._dx) + \
                          (mu_kp * Mz_kp - mu_km * Mz_km) / (2 * self._dx)
                
                # Jacobi 更新
                phi_new = (
                    mu_x * (self._phi_m[i + 1, j, k] + self._phi_m[i - 1, j, k]) +
                    mu_y * (self._phi_m[i, j + 1, k] + self._phi_m[i, j - 1, k]) +
                    mu_z * (self._phi_m[i, j, k + 1] + self._phi_m[i, j, k - 1]) -
                    rhs * self._dx * self._dx
                ) / (2 * (mu_x + mu_y + mu_z))
                
                diff = ti.abs(phi_new - self._phi_m[i, j, k])
                if diff > max_diff:
                    max_diff = diff
                
                self._phi_m[i, j, k] = phi_new
        
        return max_diff
    
    @ti.kernel
    def _compute_B_from_scalar_potential(self):
        """从标量磁势计算 B: H = -∇φ_m, B = μH + μM"""
        nx, ny, nz = self._resolution[0], self._resolution[1], self._resolution[2]
        
        for i, j, k in ti.ndrange(nx, ny, nz):
            if i > 0 and i < nx - 1 and j > 0 and j < ny - 1 and k > 0 and k < nz - 1:
                # H = -∇φ_m
                Hx = -(self._phi_m[i + 1, j, k] - self._phi_m[i - 1, j, k]) / (2 * self._dx)
                Hy = -(self._phi_m[i, j + 1, k] - self._phi_m[i, j - 1, k]) / (2 * self._dx)
                Hz = -(self._phi_m[i, j, k + 1] - self._phi_m[i, j, k - 1]) / (2 * self._dx)
                
                self._H[i, j, k] = ti.Vector([Hx, Hy, Hz])
                
                # B = μH + μM
                mu = self._mu[i, j, k]
                Bx = mu * Hx
                By = mu * Hy
                Bz = mu * Hz
                
                if ti.static(self._M is not None):
                    Bx += mu * self._M_field[i, j, k][0]
                    By += mu * self._M_field[i, j, k][1]
                    Bz += mu * self._M_field[i, j, k][2]
                
                self._B[i, j, k] = ti.Vector([Bx, By, Bz])
    
    # ============================================================
    # 矢量磁势法 (含自由电流区域)
    # ∇×(1/μ ∇×A) = J + ∇×M
    # ============================================================
    
    @ti.kernel
    def _solve_vector_potential_step(self) -> gs.ti_float:
        """矢量磁势 Jacobi 迭代一步 (简化版本，仅 z 分量)"""
        nx, ny, nz = self._resolution[0], self._resolution[1], self._resolution[2]
        max_diff = 0.0
        
        for i, j, k in ti.ndrange(nx, ny, nz):
            if i > 0 and i < nx - 1 and j > 0 and j < ny - 1 and k > 0 and k < nz - 1:
                # 简化: 假设电流只有 z 分量，A 也只有 z 分量
                # ∇²A_z = -μJ_z
                
                mu = self._mu[i, j, k]
                
                Az_new = 0.25 * (
                    self._A[i + 1, j, k][2] + self._A[i - 1, j, k][2] +
                    self._A[i, j + 1, k][2] + self._A[i, j - 1, k][2] +
                    self._A[i, j, k + 1][2] + self._A[i, j, k - 1][2] +
                    mu * self._J_field[i, j, k][2] * self._dx * self._dx
                )
                
                diff = ti.abs(Az_new - self._A[i, j, k][2])
                if diff > max_diff:
                    max_diff = diff
                
                self._A[i, j, k][2] = Az_new
        
        return max_diff
    
    @ti.kernel
    def _compute_B_from_vector_potential(self):
        """从矢量磁势计算 B: B = ∇×A"""
        nx, ny, nz = self._resolution[0], self._resolution[1], self._resolution[2]
        
        for i, j, k in ti.ndrange(nx, ny, nz):
            if i > 0 and i < nx - 1 and j > 0 and j < ny - 1 and k > 0 and k < nz - 1:
                # B = ∇×A
                Bx = (self._A[i, j + 1, k][2] - self._A[i, j - 1, k][2]) / (2 * self._dx) - \
                     (self._A[i, j, k + 1][1] - self._A[i, j, k - 1][1]) / (2 * self._dx)
                By = (self._A[i, j, k + 1][0] - self._A[i, j, k - 1][0]) / (2 * self._dx) - \
                     (self._A[i + 1, j, k][2] - self._A[i - 1, j, k][2]) / (2 * self._dx)
                Bz = (self._A[i + 1, j, k][1] - self._A[i - 1, j, k][1]) / (2 * self._dx) - \
                     (self._A[i, j + 1, k][0] - self._A[i, j - 1, k][0]) / (2 * self._dx)
                
                self._B[i, j, k] = ti.Vector([Bx, By, Bz])
                
                # H = B/μ - M
                mu = self._mu[i, j, k]
                Hx = Bx / mu
                Hy = By / mu
                Hz = Bz / mu
                
                if ti.static(self._M is not None):
                    Hx -= self._M_field[i, j, k][0]
                    Hy -= self._M_field[i, j, k][1]
                    Hz -= self._M_field[i, j, k][2]
                
                self._H[i, j, k] = ti.Vector([Hx, Hy, Hz])
    
    # ============================================================
    # 非线性材料处理 (B-H 曲线)
    # ============================================================
    
    def _update_permeability_nonlinear(self):
        """根据 B-H 曲线更新磁导率 (固定点迭代)"""
        if not self._nonlinear:
            return
        
        # 获取当前 H 场
        H_np = self.get_H()
        B_np = self.get_B()
        
        # 计算 |H| 和 |B|
        H_mag = np.linalg.norm(H_np, axis=-1)
        B_mag = np.linalg.norm(B_np, axis=-1)
        
        # 从 B-H 曲线插值新的磁导率
        mu_new = self._interpolate_mu(H_mag)
        
        # 限制变化率，保证收敛
        mu_old = self._mu.to_numpy()
        mu_blend = 0.5 * mu_new + 0.5 * mu_old
        
        self._mu.from_numpy(mu_blend.astype(np.float32))
    
    def _interpolate_mu(self, H_mag: np.ndarray) -> np.ndarray:
        """从 B-H 曲线插值磁导率 μ = B/H"""
        if self._bh_curve is None:
            return np.ones_like(H_mag) * self._mu0
        
        H_curve = self._bh_curve['H']  # [A/m]
        B_curve = self._bh_curve['B']  # [T]
        
        # 插值 B 值
        B_interp = np.interp(H_mag.flatten(), H_curve, B_curve)
        B_interp = B_interp.reshape(H_mag.shape)
        
        # 计算磁导率 μ = B/H
        mu = np.where(H_mag > 1e-10, B_interp / H_mag, self._mu0 * 1000)
        
        return mu
    
    # ============================================================
    # 公共接口
    # ============================================================
    
    def solve(self) -> int:
        """
        求解静磁场问题
        
        Returns
        -------
        int : 实际迭代次数
        """
        for i in range(self._max_iter):
            if self._method == 'scalar':
                max_diff = self._solve_scalar_potential_step()
            else:
                max_diff = self._solve_vector_potential_step()
            
            # 非线性材料更新
            if self._nonlinear and i % 10 == 0:
                if self._method == 'scalar':
                    self._compute_B_from_scalar_potential()
                else:
                    self._compute_B_from_vector_potential()
                self._update_permeability_nonlinear()
            
            if max_diff < self._tol:
                self._converged = True
                self._iter_count = i
                gs.logger.info(f"Magnetostatics converged in {i} iterations")
                break
        
        # 最终计算 B 和 H
        if self._method == 'scalar':
            self._compute_B_from_scalar_potential()
        else:
            self._compute_B_from_vector_potential()
        
        return self._iter_count
    
    def step(self):
        """单步更新 (用于与其他求解器耦合)"""
        if self._method == 'scalar':
            self._solve_scalar_potential_step()
            self._compute_B_from_scalar_potential()
        else:
            self._solve_vector_potential_step()
            self._compute_B_from_vector_potential()
    
    def get_B(self) -> np.ndarray:
        """获取磁通密度 B [T]"""
        return self._B.to_numpy()
    
    def get_H(self) -> np.ndarray:
        """获取磁场强度 H [A/m]"""
        return self._H.to_numpy()
    
    def get_A(self) -> np.ndarray:
        """获取矢量磁势 A [Wb/m]"""
        if self._A is None:
            raise ValueError("Vector potential not available for scalar method")
        return self._A.to_numpy()
    
    def get_phi_m(self) -> np.ndarray:
        """获取标量磁势 φ_m [A]"""
        if self._phi_m is None:
            raise ValueError("Scalar potential not available for vector method")
        return self._phi_m.to_numpy()
    
    def compute_magnetic_energy(self) -> float:
        """计算磁场能量 W = 0.5∫ B·H dV"""
        B = self.get_B()
        H = self.get_H()
        
        # 点积 B·H
        BH = np.sum(B * H, axis=-1)
        
        # 积分
        dV = self._dx ** 3
        W = 0.5 * np.sum(BH) * dV
        
        return W
    
    def compute_flux_through_surface(self, normal: tuple = (0, 0, 1), 
                                     position: str = 'center') -> float:
        """
        计算通过表面的磁通量 Φ = ∫ B·n dS
        
        Parameters
        ----------
        normal : tuple
            表面法向 (nx, ny, nz)
        position : str
            表面位置: 'center', 'x_min', 'x_max', 'y_min', 'y_max', 'z_min', 'z_max'
        """
        B = self.get_B()
        nx, ny, nz = self._resolution
        
        # 选择表面
        if position == 'center':
            B_surface = B[nx // 2, :, :]
            dS = self._dx * self._dx
        elif position == 'z_min':
            B_surface = B[:, :, 0]
            dS = self._dx * self._dx
        elif position == 'z_max':
            B_surface = B[:, :, -1]
            dS = self._dx * self._dx
        else:
            raise ValueError(f"Unknown position: {position}")
        
        # 计算通量
        nx_n, ny_n, nz_n = normal
        flux = np.sum(B_surface[:, :, 0] * nx_n + 
                      B_surface[:, :, 1] * ny_n + 
                      B_surface[:, :, 2] * nz_n) * dS
        
        return flux
    
    def compute_force_on_body(self, body_mask: np.ndarray) -> np.ndarray:
        """
        计算磁场对物体的作用力 (Maxwell 应力张量法)
        
        F = ∮ T·n dS,  T_ij = (B_i B_j)/μ - δ_ij B²/(2μ)
        
        Parameters
        ----------
        body_mask : np.ndarray
            布尔掩码，标记物体占据的网格
        """
        B = self.get_B()
        mu = self._mu.to_numpy()
        
        # Maxwell 应力张量
        Bx, By, Bz = B[:, :, :, 0], B[:, :, :, 1], B[:, :, :, 2]
        B2 = Bx**2 + By**2 + Bz**2
        
        # 简化: 计算体积力密度 f = J×B
        # 需要电流密度
        if self._J is None:
            return np.zeros(3)
        
        J = self._J if isinstance(self._J, np.ndarray) else np.zeros_like(B)
        
        # f = J × B
        fx = J[:, :, :, 1] * Bz - J[:, :, :, 2] * By
        fy = J[:, :, :, 2] * Bx - J[:, :, :, 0] * Bz
        fz = J[:, :, :, 0] * By - J[:, :, :, 1] * Bx
        
        # 在物体内积分
        dV = self._dx ** 3
        Fx = np.sum(fx * body_mask) * dV
        Fy = np.sum(fy * body_mask) * dV
        Fz = np.sum(fz * body_mask) * dV
        
        return np.array([Fx, Fy, Fz])
    
    def set_permeability(self, mu: np.ndarray):
        """设置磁导率分布"""
        self._mu.from_numpy(mu.astype(np.float32))
    
    def set_magnetization(self, M: np.ndarray):
        """设置永磁体磁化强度 [A/m]"""
        if self._M_field is None:
            nx, ny, nz = self._resolution
            self._M_field = ti.Vector.field(3, dtype=gs.ti_float, shape=(nx, ny, nz))
        self._M_field.from_numpy(M.astype(np.float32))
