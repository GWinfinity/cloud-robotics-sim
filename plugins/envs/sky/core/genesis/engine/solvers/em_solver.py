"""
电磁场 FDTD 求解器

基于 Taichi 的 2D TE 模式和 3D 全波 FDTD 求解器，
支持 PML 吸收边界、多种激励源。

对标 ANSYS Maxwell 的瞬态电磁场求解能力。
"""

import numpy as np
import gstaichi as ti
import torch

import genesis as gs
import genesis.utils.array_class as array_class
from genesis.utils.misc import ti_to_torch

from .base_solver import Solver


@ti.data_oriented
class EMSolver(Solver):
    """
    电磁场 FDTD 求解器
    
    实现 Maxwell 方程组的时域有限差分法求解：
    ∇×E = -∂B/∂t,  ∇×H = ∂D/∂t + J
    
    Parameters
    ----------
    scene : gs.Scene
    sim : gs.Simulator
    options : EMSolverOptions
        - dim: 2 或 3
        - resolution: 网格分辨率
        - dx: 空间步长
        - dt: 时间步长 (需满足 CFL 条件)
        - boundary: 'pml' | 'pec' | 'pmc'
        - pml_layers: PML 层数
        - source_type: 'point' | 'plane_wave' | 'gaussian_pulse'
        - source_position: 源位置
        - wavelength: 激励源波长
        - epsilon_r: 相对介电常数分布
        - mu_r: 相对磁导率分布
        - sigma: 电导率分布
    """
    
    def __init__(self, scene, sim, options):
        super().__init__(scene, sim, options)
        
        # 电磁场参数
        self._dim = options.dim
        self._resolution = options.resolution
        self._dx = options.dx
        self._dt = options.dt
        self._boundary = options.boundary
        self._pml_layers = options.pml_layers
        self._source_type = options.source_type
        self._source_position = options.source_position
        self._wavelength = options.wavelength
        
        # 物理常数 (SI 单位)
        self._c = 299792458.0  # 光速
        self._eps0 = 8.854187817e-12  # 真空介电常数
        self._mu0 = 4 * np.pi * 1e-7  # 真空磁导率
        
        # CFL 条件检查
        self._check_cfl()
        
        # 材料参数
        self._epsilon_r = options.epsilon_r if hasattr(options, 'epsilon_r') else None
        self._mu_r = options.mu_r if hasattr(options, 'mu_r') else None
        self._sigma = options.sigma if hasattr(options, 'sigma') else None
        
        # 源参数
        self._omega = 2.0 * np.pi * self._c / self._wavelength
        bandwidth = 0.1
        self._sigma_t = self._wavelength / (2.0 * np.pi * self._c * bandwidth)
        self._t0 = 4.0 * self._sigma_t
        
        # PML 参数
        self._init_pml_profiles()
        
        # 场变量将在 build() 中初始化
        self._Ex = None
        self._Ey = None
        self._Ez = None
        self._Hx = None
        self._Hy = None
        self._Hz = None
        
        # 材料场
        self._eps = None
        self._mu = None
        self._sigma_field = None
        
        # PML 电导率场
        self._sigma_x = None
        self._sigma_y = None
        self._sigma_z = None
    
    def _check_cfl(self):
        """检查 CFL 稳定性条件"""
        dt_max = self._dx / (self._c * np.sqrt(self._dim))
        if self._dt > dt_max:
            gs.logger.warning(
                f"EM CFL 条件不满足: dt={self._dt:.4e} > dt_max={dt_max:.4e}. "
                f"建议 dt = {0.99 * dt_max:.4e}"
            )
    
    def _init_pml_profiles(self):
        """初始化 PML 电导率分布"""
        if self._boundary != 'pml':
            self._pml_sigma_x = None
            self._pml_sigma_y = None
            self._pml_sigma_z = None
            return
        
        # 理论最大电导率
        m = 3.0
        sigma_max = -np.log(1e-6) * (m + 1) * self._eps0 * self._c \
                    / (2.0 * self._pml_layers * self._dx)
        
        def _profile(n):
            s = np.zeros(n, dtype=np.float32)
            for i in range(n):
                if i < self._pml_layers:
                    d = float(self._pml_layers - i) / self._pml_layers
                    s[i] = sigma_max * d**m
                elif i >= n - self._pml_layers:
                    d = float(i - (n - self._pml_layers) + 1) / self._pml_layers
                    s[i] = sigma_max * d**m
            return s
        
        self._pml_sigma_x = _profile(self._resolution[0])
        self._pml_sigma_y = _profile(self._resolution[1])
        if self._dim == 3:
            self._pml_sigma_z = _profile(self._resolution[2])
    
    def build(self):
        """构建求解器场变量"""
        super().build()
        
        if self._dim == 2:
            nx, ny = self._resolution
            # 2D TE 模式: Ez, Hx, Hy
            self._Ez = ti.field(dtype=gs.ti_float, shape=(nx, ny))
            self._Hx = ti.field(dtype=gs.ti_float, shape=(nx, ny))
            self._Hy = ti.field(dtype=gs.ti_float, shape=(nx, ny))
            
            # 材料场
            self._eps = ti.field(dtype=gs.ti_float, shape=(nx, ny))
            self._mu = ti.field(dtype=gs.ti_float, shape=(nx, ny))
            self._sigma_field = ti.field(dtype=gs.ti_float, shape=(nx, ny))
            
            # PML 场
            if self._boundary == 'pml':
                self._sigma_x = ti.field(dtype=gs.ti_float, shape=(nx,))
                self._sigma_y = ti.field(dtype=gs.ti_float, shape=(ny,))
                self._sigma_x.from_numpy(self._pml_sigma_x)
                self._sigma_y.from_numpy(self._pml_sigma_y)
        
        else:  # 3D
            nx, ny, nz = self._resolution
            self._Ex = ti.field(dtype=gs.ti_float, shape=(nx, ny, nz))
            self._Ey = ti.field(dtype=gs.ti_float, shape=(nx, ny, nz))
            self._Ez = ti.field(dtype=gs.ti_float, shape=(nx, ny, nz))
            self._Hx = ti.field(dtype=gs.ti_float, shape=(nx, ny, nz))
            self._Hy = ti.field(dtype=gs.ti_float, shape=(nx, ny, nz))
            self._Hz = ti.field(dtype=gs.ti_float, shape=(nx, ny, nz))
            
            # 材料场
            self._eps = ti.field(dtype=gs.ti_float, shape=(nx, ny, nz))
            self._mu = ti.field(dtype=gs.ti_float, shape=(nx, ny, nz))
            self._sigma_field = ti.field(dtype=gs.ti_float, shape=(nx, ny, nz))
            
            # PML 场
            if self._boundary == 'pml':
                self._sigma_x = ti.field(dtype=gs.ti_float, shape=(nx,))
                self._sigma_y = ti.field(dtype=gs.ti_float, shape=(ny,))
                self._sigma_z = ti.field(dtype=gs.ti_float, shape=(nz,))
                self._sigma_x.from_numpy(self._pml_sigma_x)
                self._sigma_y.from_numpy(self._pml_sigma_y)
                self._sigma_z.from_numpy(self._pml_sigma_z)
        
        # 初始化材料
        self._init_materials()
    
    def _init_materials(self):
        """初始化材料分布"""
        if self._dim == 2:
            nx, ny = self._resolution
            eps_np = np.ones((nx, ny), dtype=np.float32) * self._eps0
            mu_np = np.ones((nx, ny), dtype=np.float32) * self._mu0
            sigma_np = np.zeros((nx, ny), dtype=np.float32)
            
            if self._epsilon_r is not None:
                eps_np *= self._epsilon_r
            if self._mu_r is not None:
                mu_np *= self._mu_r
            if self._sigma is not None:
                sigma_np = self._sigma
            
            self._eps.from_numpy(eps_np)
            self._mu.from_numpy(mu_np)
            self._sigma_field.from_numpy(sigma_np)
        else:
            nx, ny, nz = self._resolution
            eps_np = np.ones((nx, ny, nz), dtype=np.float32) * self._eps0
            mu_np = np.ones((nx, ny, nz), dtype=np.float32) * self._mu0
            sigma_np = np.zeros((nx, ny, nz), dtype=np.float32)
            
            if self._epsilon_r is not None:
                eps_np *= self._epsilon_r
            if self._mu_r is not None:
                mu_np *= self._mu_r
            if self._sigma is not None:
                sigma_np = self._sigma
            
            self._eps.from_numpy(eps_np)
            self._mu.from_numpy(mu_np)
            self._sigma_field.from_numpy(sigma_np)
    
    # ============================================================
    # Taichi Kernels - 2D TE Mode
    # ============================================================
    
    @ti.kernel
    def _update_H_2d(self):
        """更新 2D TE 模式的 H 场"""
        for i, j in ti.ndrange(self._resolution[0] - 1, self._resolution[1] - 1):
            # Hx(i, j+0.5) 更新
            self._Hx[i, j] -= (self._dt / self._mu[i, j]) * (
                self._Ez[i, j + 1] - self._Ez[i, j]
            ) / self._dx
            
            # Hy(i+0.5, j) 更新
            self._Hy[i, j] += (self._dt / self._mu[i, j]) * (
                self._Ez[i + 1, j] - self._Ez[i, j]
            ) / self._dx
    
    @ti.kernel
    def _update_E_2d(self):
        """更新 2D TE 模式的 E 场"""
        for i, j in ti.ndrange(self._resolution[0] - 1, self._resolution[1] - 1):
            if i > 0 and j > 0:
                # Ez(i, j) 更新
                dHy = (self._Hy[i, j] - self._Hy[i - 1, j]) / self._dx
                dHx = (self._Hx[i, j] - self._Hx[i, j - 1]) / self._dx
                
                self._Ez[i, j] += (self._dt / self._eps[i, j]) * (dHy - dHx)
                
                # 电导率衰减 (PML)
                if ti.static(self._boundary == 'pml'):
                    sigma_avg = (self._sigma_x[i] + self._sigma_y[j]) * 0.5
                    self._Ez[i, j] *= ti.exp(-sigma_avg * self._dt / self._eps[i, j])
    
    @ti.kernel
    def _apply_source_2d(self, t: gs.ti_float):
        """应用 2D 激励源"""
        sx, sy = self._source_position
        if self._source_type == 'point':
            # 点源: 正弦激励
            self._Ez[sx, sy] = ti.sin(self._omega * t)
        elif self._source_type == 'gaussian_pulse':
            # 高斯脉冲
            envelope = ti.exp(-(t - self._t0)**2 / (2 * self._sigma_t**2))
            self._Ez[sx, sy] = envelope * ti.sin(self._omega * t)
    
    # ============================================================
    # Taichi Kernels - 3D Full Wave
    # ============================================================
    
    @ti.kernel
    def _update_H_3d(self):
        """更新 3D H 场"""
        nx, ny, nz = self._resolution[0], self._resolution[1], self._resolution[2]
        for i, j, k in ti.ndrange(nx - 1, ny - 1, nz - 1):
            # Hx(i, j+0.5, k+0.5)
            self._Hx[i, j, k] -= (self._dt / self._mu[i, j, k]) * (
                (self._Ez[i, j + 1, k] - self._Ez[i, j, k]) -
                (self._Ey[i, j, k + 1] - self._Ey[i, j, k])
            ) / self._dx
            
            # Hy(i+0.5, j, k+0.5)
            self._Hy[i, j, k] -= (self._dt / self._mu[i, j, k]) * (
                (self._Ex[i, j, k + 1] - self._Ex[i, j, k]) -
                (self._Ez[i + 1, j, k] - self._Ez[i, j, k])
            ) / self._dx
            
            # Hz(i+0.5, j+0.5, k)
            self._Hz[i, j, k] -= (self._dt / self._mu[i, j, k]) * (
                (self._Ey[i + 1, j, k] - self._Ey[i, j, k]) -
                (self._Ex[i, j + 1, k] - self._Ex[i, j, k])
            ) / self._dx
    
    @ti.kernel
    def _update_E_3d(self):
        """更新 3D E 场"""
        nx, ny, nz = self._resolution[0], self._resolution[1], self._resolution[2]
        for i, j, k in ti.ndrange(nx - 1, ny - 1, nz - 1):
            if i > 0 and j > 0 and k > 0:
                # Ex(i+0.5, j, k)
                dHz = (self._Hz[i, j, k] - self._Hz[i, j - 1, k]) / self._dx
                dHy = (self._Hy[i, j, k] - self._Hy[i, j, k - 1]) / self._dx
                self._Ex[i, j, k] += (self._dt / self._eps[i, j, k]) * (dHz - dHy)
                
                # Ey(i, j+0.5, k)
                dHx = (self._Hx[i, j, k] - self._Hx[i, j, k - 1]) / self._dx
                dHz = (self._Hz[i, j, k] - self._Hz[i - 1, j, k]) / self._dx
                self._Ey[i, j, k] += (self._dt / self._eps[i, j, k]) * (dHx - dHz)
                
                # Ez(i, j, k+0.5)
                dHy = (self._Hy[i, j, k] - self._Hy[i - 1, j, k]) / self._dx
                dHx = (self._Hx[i, j, k] - self._Hx[i, j - 1, k]) / self._dx
                self._Ez[i, j, k] += (self._dt / self._eps[i, j, k]) * (dHy - dHx)
                
                # PML 衰减
                if ti.static(self._boundary == 'pml'):
                    sigma_avg = (self._sigma_x[i] + self._sigma_y[j] + self._sigma_z[k]) / 3.0
                    decay = ti.exp(-sigma_avg * self._dt / self._eps[i, j, k])
                    self._Ex[i, j, k] *= decay
                    self._Ey[i, j, k] *= decay
                    self._Ez[i, j, k] *= decay
    
    @ti.kernel
    def _apply_source_3d(self, t: gs.ti_float):
        """应用 3D 激励源"""
        sx, sy, sz = self._source_position
        if self._source_type == 'point':
            self._Ez[sx, sy, sz] = ti.sin(self._omega * t)
        elif self._source_type == 'gaussian_pulse':
            envelope = ti.exp(-(t - self._t0)**2 / (2 * self._sigma_t**2))
            self._Ez[sx, sy, sz] = envelope * ti.sin(self._omega * t)
    
    # ============================================================
    # 公共接口
    # ============================================================
    
    def step(self, t: float):
        """单步更新"""
        if self._dim == 2:
            self._update_H_2d()
            self._update_E_2d()
            self._apply_source_2d(t)
        else:
            self._update_H_3d()
            self._update_E_3d()
            self._apply_source_3d(t)
    
    def get_field(self, field_name: str = 'Ez') -> np.ndarray:
        """获取场数据"""
        if self._dim == 2:
            if field_name == 'Ez':
                return self._Ez.to_numpy()
            elif field_name == 'Hx':
                return self._Hx.to_numpy()
            elif field_name == 'Hy':
                return self._Hy.to_numpy()
        else:
            if field_name == 'Ex':
                return self._Ex.to_numpy()
            elif field_name == 'Ey':
                return self._Ey.to_numpy()
            elif field_name == 'Ez':
                return self._Ez.to_numpy()
            elif field_name == 'Hx':
                return self._Hx.to_numpy()
            elif field_name == 'Hy':
                return self._Hy.to_numpy()
            elif field_name == 'Hz':
                return self._Hz.to_numpy()
        
        raise ValueError(f"Unknown field name: {field_name}")
    
    def compute_poynting_vector(self) -> np.ndarray:
        """计算 Poynting 矢量 S = E × H"""
        if self._dim == 2:
            Ez = self._Ez.to_numpy()
            Hx = self._Hx.to_numpy()
            Hy = self._Hy.to_numpy()
            # Sx = -Ez * Hy, Sy = Ez * Hx
            Sx = -Ez[:-1, :-1] * Hy[:-1, :-1]
            Sy = Ez[:-1, :-1] * Hx[:-1, :-1]
            return np.stack([Sx, Sy], axis=-1)
        else:
            Ex = self._Ex.to_numpy()
            Ey = self._Ey.to_numpy()
            Ez = self._Ez.to_numpy()
            Hx = self._Hx.to_numpy()
            Hy = self._Hy.to_numpy()
            Hz = self._Hz.to_numpy()
            # S = E × H
            Sx = Ey * Hz - Ez * Hy
            Sy = Ez * Hx - Ex * Hz
            Sz = Ex * Hy - Ey * Hx
            return np.stack([Sx, Sy, Sz], axis=-1)
    
    def compute_energy_density(self) -> np.ndarray:
        """计算电磁能量密度 w = 0.5*(ε|E|² + μ|H|²)"""
        if self._dim == 2:
            Ez = self._Ez.to_numpy()
            Hx = self._Hx.to_numpy()
            Hy = self._Hy.to_numpy()
            eps = self._eps.to_numpy()
            mu = self._mu.to_numpy()
            
            w_e = 0.5 * eps * Ez**2
            w_m = 0.5 * mu * (Hx**2 + Hy**2)
            return w_e + w_m
        else:
            Ex = self._Ex.to_numpy()
            Ey = self._Ey.to_numpy()
            Ez = self._Ez.to_numpy()
            Hx = self._Hx.to_numpy()
            Hy = self._Hy.to_numpy()
            Hz = self._Hz.to_numpy()
            eps = self._eps.to_numpy()
            mu = self._mu.to_numpy()
            
            E2 = Ex**2 + Ey**2 + Ez**2
            H2 = Hx**2 + Hy**2 + Hz**2
            
            w_e = 0.5 * eps * E2
            w_m = 0.5 * mu * H2
            return w_e + w_m
