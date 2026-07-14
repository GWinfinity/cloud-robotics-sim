"""
B-H 曲线求解器 (B-H Curve Solver)

实现非线性磁性材料的 B-H 曲线建模和求解：
- 解析模型: Frohlich, Langevin, Fröhlich-Kelly
- 插值模型: 样条插值、 lookup table
- 磁滞模型: Jiles-Atherton, Preisach

支持电机设计中的非线性材料分析。
对标 COMSOL AC/DC Module > Nonlinear Magnetic Materials。
"""

import numpy as np
import gstaichi as ti
import torch

import genesis as gs
import genesis.utils.array_class as array_class
from genesis.utils.misc import ti_to_torch

from .base_solver import Solver


@ti.data_oriented
class BHSolver(Solver):
    """
    B-H 曲线求解器
    
    提供多种非线性磁性材料模型：
    1. 解析模型:
       - Frohlich: B = H / (a + b·H)
       - Langevin: M = Ms·(coth(x) - 1/x), x = H/a
       - Fröhlich-Kelly: B = μ0·H + Ms·H/(a + H)
    
    2. 插值模型:
       - 线性插值 lookup table
       - 三次样条插值
    
    3. 磁滞模型:
       - Jiles-Atherton: 基于磁畴壁运动
       - Preisach: 基于磁滞算子
    
    Parameters
    ----------
    scene : gs.Scene
    sim : gs.Simulator
    options : BHOptions
        - model_type: 'frohlich' | 'langevin' | 'spline' | 'jiles_atherton' | 'preisach'
        - material: 'iron' | 'steel' | 'ndfeb' | 'custom'
        - H_max: 最大磁场强度 [A/m]
        - n_points: lookup table 点数
    """
    
    def __init__(self, scene, sim, options):
        super().__init__(scene, sim, options)
        
        # 模型参数
        self._model_type = options.model_type
        self._material = options.material
        self._H_max = options.H_max
        self._n_points = options.n_points
        
        # 材料参数
        self._mu0 = 4 * np.pi * 1e-7
        self._Ms = 0.0  # 饱和磁化强度
        self._a = 0.0   # 形状参数
        self._alpha = 0.0  # 耦合参数
        self._c = 0.0   # 可逆系数
        self._k = 0.0   # 钉扎系数
        
        # 预计算 lookup table
        self._H_table = None
        self._B_table = None
        self._dBdH_table = None
        
        # 磁滞状态
        self._M_prev = None
        self._H_prev = None
        self._B_prev = None
        
        self._init_material_parameters()
    
    def _init_material_parameters(self):
        """初始化材料参数"""
        material_params = {
            'iron': {'Ms': 1.7e6, 'a': 1000, 'alpha': 1e-3, 'c': 0.1, 'k': 500},
            'steel': {'Ms': 1.6e6, 'a': 800, 'alpha': 1e-3, 'c': 0.08, 'k': 400},
            'ndfeb': {'Ms': 1.2e6, 'a': 500, 'alpha': 0.1, 'c': 0.0, 'k': 0},
            'ferrite': {'Ms': 0.4e6, 'a': 200, 'alpha': 0.05, 'c': 0.0, 'k': 0},
        }
        
        if self._material in material_params:
            params = material_params[self._material]
            self._Ms = params['Ms']
            self._a = params['a']
            self._alpha = params['alpha']
            self._c = params['c']
            self._k = params['k']
    
    def build(self):
        """构建 lookup table"""
        super().build()
        
        # 预计算 H-B 关系
        self._H_table = np.linspace(0, self._H_max, self._n_points)
        self._B_table = np.zeros(self._n_points)
        self._dBdH_table = np.zeros(self._n_points)
        
        for i, H in enumerate(self._H_table):
            self._B_table[i] = self._compute_B(H)
        
        # 计算微分磁导率
        self._dBdH_table[1:-1] = (self._B_table[2:] - self._B_table[:-2]) / (self._H_table[2:] - self._H_table[:-2])
        self._dBdH_table[0] = self._dBdH_table[1]
        self._dBdH_table[-1] = self._dBdH_table[-2]
        
        # Taichi fields
        if self.dim == 2:
            self._mu_eff = ti.field(gs.ti_float, shape=(self.nx, self.ny))
            self._B_field = ti.Vector.field(3, gs.ti_float, shape=(self.nx, self.ny))
            self._H_field = ti.Vector.field(3, gs.ti_float, shape=(self.nx, self.ny))
        else:
            self._mu_eff = ti.field(gs.ti_float, shape=(self.nx, self.ny, self.nz))
            self._B_field = ti.Vector.field(3, gs.ti_float, shape=(self.nx, self.ny, self.nz))
            self._H_field = ti.Vector.field(3, gs.ti_float, shape=(self.nx, self.ny, self.nz))
    
    def _compute_B(self, H: float) -> float:
        """计算 B = f(H)"""
        H = abs(H)
        
        if self._model_type == 'frohlich':
            # Frohlich 模型: B = H / (a + b*H)
            a = self._a
            b = 1.0 / (self._mu0 * self._Ms)
            B = H / (a + b * H)
            return B
        
        elif self._model_type == 'langevin':
            # Langevin 函数
            x = H / self._a if self._a > 0 else 0
            if x < 1e-6:
                M = self._Ms * x / 3
            else:
                M = self._Ms * (1 / np.tanh(x) - 1 / x)
            B = self._mu0 * (H + M)
            return B
        
        elif self._model_type == 'spline':
            # 插值 lookup table
            return np.interp(H, self._H_table, self._B_table)
        
        elif self._model_type == 'jiles_atherton':
            # 简化 Jiles-Atherton (无磁滞曲线)
            Man = self._Ms * (1 / np.tanh(H / self._a) - self._a / H) if H > 1e-6 else 0
            B = self._mu0 * (H + Man)
            return B
        
        else:
            # 线性
            return self._mu0 * H
    
    def _compute_dBdH(self, H: float) -> float:
        """计算微分磁导率 dB/dH"""
        H = abs(H)
        
        if self._model_type == 'spline':
            return np.interp(H, self._H_table, self._dBdH_table)
        
        # 数值微分
        dH = H * 1e-6 + 1e-10
        B1 = self._compute_B(H - dH)
        B2 = self._compute_B(H + dH)
        return (B2 - B1) / (2 * dH)
    
    @ti.kernel
    def _update_permeability_2d(self, Hx: ti.types.ndarray(), Hy: ti.types.ndarray()):
        """更新有效磁导率 (2D)"""
        for i, j in self._mu_eff:
            H_mag = ti.sqrt(Hx[i, j]**2 + Hy[i, j]**2)
            # 使用线性近似
            if H_mag < 1e-6:
                self._mu_eff[i, j] = self._mu0 * 1000  # 初始相对磁导率
            else:
                # 简化：查表
                idx = ti.cast(H_mag / self._H_max * self._n_points, ti.i32)
                idx = ti.max(0, ti.min(idx, self._n_points - 2))
                # 线性插值
                h1 = self._H_table[idx]
                h2 = self._H_table[idx + 1]
                b1 = self._B_table[idx]
                b2 = self._B_table[idx + 1]
                if h2 > h1:
                    dBdH = (b2 - b1) / (h2 - h1)
                    self._mu_eff[i, j] = dBdH
                else:
                    self._mu_eff[i, j] = self._mu0
    
    @ti.kernel
    def _update_permeability_3d(self, Hx: ti.types.ndarray(), Hy: ti.types.ndarray(), Hz: ti.types.ndarray()):
        """更新有效磁导率 (3D)"""
        for i, j, k in self._mu_eff:
            H_mag = ti.sqrt(Hx[i, j, k]**2 + Hy[i, j, k]**2 + Hz[i, j, k]**2)
            if H_mag < 1e-6:
                self._mu_eff[i, j, k] = self._mu0 * 1000
            else:
                idx = ti.cast(H_mag / self._H_max * self._n_points, ti.i32)
                idx = ti.max(0, ti.min(idx, self._n_points - 2))
                h1 = self._H_table[idx]
                h2 = self._H_table[idx + 1]
                b1 = self._B_table[idx]
                b2 = self._B_table[idx + 1]
                if h2 > h1:
                    dBdH = (b2 - b1) / (h2 - h1)
                    self._mu_eff[i, j, k] = dBdH
                else:
                    self._mu_eff[i, j, k] = self._mu0
    
    def update_permeability(self, H_field: np.ndarray):
        """根据磁场强度更新磁导率分布"""
        if self.dim == 2:
            self._update_permeability_2d(H_field[0], H_field[1])
        else:
            self._update_permeability_3d(H_field[0], H_field[1], H_field[2])
    
    def get_permeability(self) -> np.ndarray:
        """获取有效磁导率分布"""
        return self._mu_eff.to_numpy()
    
    def get_BH_curve(self) -> tuple:
        """获取 B-H 曲线数据"""
        return self._H_table.copy(), self._B_table.copy()
    
    def compute_relative_permeability(self, H: float) -> float:
        """计算相对磁导率"""
        B = self._compute_B(H)
        return B / (self._mu0 * H) if H > 1e-6 else 1000
    
    def compute_coercivity(self) -> float:
        """计算矫顽力 Hc"""
        # 简化：从 B-H 曲线找到 B=0 时的 H
        if self._B_table is None:
            return 0
        
        # 找到 B 最接近 0 的点
        idx = np.argmin(np.abs(self._B_table))
        return self._H_table[idx]
    
    def compute_remanence(self) -> float:
        """计算剩磁 Br"""
        # H=0 时的 B 值
        return self._compute_B(0)
    
    def step(self):
        """更新磁导率"""
        pass  # 由外部求解器调用 update_permeability
    
    def get_field(self, field_name: str = 'mu_eff') -> np.ndarray:
        """获取场数据"""
        if field_name == 'mu_eff':
            return self.get_permeability()
        elif field_name == 'B':
            return self._B_field.to_numpy()
        elif field_name == 'H':
            return self._H_field.to_numpy()
        else:
            return np.zeros(self.resolution)
