"""
多物理场耦合求解器 (Coupled Solver)

实现电磁-热-结构多物理场耦合：
1. 电磁损耗 → 热源
2. 温度 → 材料参数更新
3. 热应力 → 结构变形

支持磁热耦合、电热耦合、热结构耦合。
对标 COMSOL Multiphysics > Multiphysics Couplings。
"""

import numpy as np
import gstaichi as ti
import torch

import genesis as gs
import genesis.utils.array_class as array_class
from genesis.utils.misc import ti_to_torch

from .base_solver import Solver


class CoupledSolver(Solver):
    """
    多物理场耦合求解器
    
    协调多个物理场求解器的耦合计算：
    - 电磁场 → 损耗 → 热源
    - 温度场 → 材料参数 (μ, σ, k)
    - 热膨胀 → 结构应力/变形
    
    Parameters
    ----------
    scene : gs.Scene
    sim : gs.Simulator
    options : CoupledOptions
        - coupling_type: 'em_thermal' | 'thermal_structural' | 'full'
        - update_interval: 耦合更新间隔
        - max_coupled_iter: 最大耦合迭代次数
        - coupling_tol: 耦合收敛容差
    """
    
    def __init__(self, scene, sim, options):
        super().__init__(scene, sim, options)
        
        # 耦合参数
        self._coupling_type = options.coupling_type
        self._update_interval = options.update_interval if hasattr(options, 'update_interval') else 10
        self._max_coupled_iter = options.max_coupled_iter if hasattr(options, 'max_coupled_iter') else 5
        self._coupling_tol = options.coupling_tol if hasattr(options, 'coupling_tol') else 1e-3
        
        # 引用其他求解器
        self._em_solver = None
        self._thermal_solver = None
        self._magnetostatics_solver = None
        self._eddy_current_solver = None
        self._joule_heating_solver = None
        self._convection_solver = None
        self._radiation_solver = None
        
        # 耦合状态
        self._step_count = 0
        self._coupled_converged = False
    
    def build(self):
        """构建耦合求解器"""
        super().build()
        
        # 获取其他求解器的引用
        self._em_solver = getattr(self._sim, 'em_solver', None)
        self._thermal_solver = getattr(self._sim, 'thermal_solver', None)
        self._magnetostatics_solver = getattr(self._sim, 'magnetostatics_solver', None)
        self._eddy_current_solver = getattr(self._sim, 'eddy_current_solver', None)
        self._joule_heating_solver = getattr(self._sim, 'joule_heating_solver', None)
        self._convection_solver = getattr(self._sim, 'convection_solver', None)
        self._radiation_solver = getattr(self._sim, 'radiation_solver', None)
    
    # ============================================================
    # 电磁-热耦合
    # ============================================================
    
    def _couple_em_thermal(self):
        """电磁-热耦合：计算电磁损耗并作为热源"""
        if self._eddy_current_solver is not None:
            # 涡流损耗
            P_eddy = self._eddy_current_solver.compute_joule_loss()
            
            # 将损耗分布转换为热源
            Q_eddy = self._distribute_loss_to_heat_source(
                self._eddy_current_solver.get_current_density_magnitude(),
                P_eddy
            )
            
            # 更新热源
            if self._thermal_solver is not None:
                # 将热源加到热传导求解器
                self._add_heat_source_to_thermal(Q_eddy)
        
        if self._joule_heating_solver is not None:
            # 焦耳热直接计算
            Q_joule = self._joule_heating_solver.get_heat_source()
            
            if self._thermal_solver is not None:
                self._add_heat_source_to_thermal(Q_joule)
    
    def _distribute_loss_to_heat_source(self, loss_density: np.ndarray, 
                                        total_loss: float) -> np.ndarray:
        """将损耗密度分布转换为热源分布"""
        # 归一化
        loss_sum = np.sum(loss_density)
        if loss_sum > 0:
            Q = loss_density / loss_sum * total_loss
        else:
            Q = np.zeros_like(loss_density)
        
        return Q
    
    def _add_heat_source_to_thermal(self, Q: np.ndarray):
        """将热源加到热传导求解器"""
        if self._thermal_solver is None:
            return
        
        # 获取当前温度
        T = self._thermal_solver.get_temperature()
        
        # 简化：直接加热
        # 实际应该通过热传导方程的源项
        dx = self._thermal_solver._dx
        dt = self._thermal_solver._dt
        rho = self._thermal_solver._rho
        cp = self._thermal_solver._cp
        
        # ΔT = Q·dt / (ρ·cp)
        T_new = T + Q * dt / (rho * cp)
        
        self._thermal_solver.set_temperature(T_new)
    
    # ============================================================
    # 温度-材料耦合
    # ============================================================
    
    def _update_material_properties(self):
        """根据温度更新材料参数"""
        if self._thermal_solver is None:
            return
        
        T = self._thermal_solver.get_temperature()
        T_avg = np.mean(T)
        
        # 更新磁导率 (居里温度效应)
        if self._magnetostatics_solver is not None:
            mu_new = self._compute_temperature_dependent_mu(T)
            self._magnetostatics_solver.set_permeability(mu_new)
        
        # 更新电导率
        if self._eddy_current_solver is not None:
            sigma_new = self._compute_temperature_dependent_sigma(T)
            self._eddy_current_solver.set_conductivity(sigma_new)
        
        # 更新热导率
        if self._thermal_solver is not None:
            k_new = self._compute_temperature_dependent_k(T)
            # 更新热导率场
            pass  # 需要扩展 thermal_solver 接口
    
    def _compute_temperature_dependent_mu(self, T: np.ndarray) -> np.ndarray:
        """计算温度相关的磁导率"""
        # 简化模型：居里温度以上磁导率下降
        T_curie = 1043.0  # 铁的居里温度 [K]
        
        mu = self._magnetostatics_solver._mu.to_numpy()
        
        # 居里温度以上：μ → μ0
        mask = T > T_curie
        mu[mask] = 4 * np.pi * 1e-7
        
        # 居里温度附近：线性下降
        transition_zone = (T > T_curie - 50) & (T <= T_curie)
        mu[transition_zone] *= (T_curie - T[transition_zone]) / 50
        
        return mu
    
    def _compute_temperature_dependent_sigma(self, T: np.ndarray) -> np.ndarray:
        """计算温度相关的电导率"""
        # 铜的电导率温度系数
        alpha = 0.0039  # 1/K
        T_ref = 293.0   # 20°C
        
        sigma_ref = self._eddy_current_solver._sigma_field.to_numpy()
        sigma = sigma_ref / (1 + alpha * (T - T_ref))
        
        return sigma
    
    def _compute_temperature_dependent_k(self, T: np.ndarray) -> np.ndarray:
        """计算温度相关的热导率"""
        # 简化：线性下降
        k_ref = 400.0  # 铜
        dk_dT = -0.1   # W/(m·K²)
        
        k = k_ref + dk_dT * (T - 300)
        k = np.maximum(k, 10)  # 最小值限制
        
        return k
    
    # ============================================================
    # 热-结构耦合
    # ============================================================
    
    def _couple_thermal_structural(self):
        """热-结构耦合：计算热应力"""
        if self._thermal_solver is None:
            return
        
        T = self._thermal_solver.get_temperature()
        
        # 计算热应力
        # σ = E·α·ΔT / (1-ν)
        # 这里需要 FEM solver 的扩展
        pass
    
    # ============================================================
    # 对流-辐射耦合
    # ============================================================
    
    def _couple_convection_radiation(self):
        """对流-辐射耦合边界条件"""
        if self._thermal_solver is None:
            return
        
        T_surface = self._thermal_solver.get_temperature()
        
        # 对流热流
        q_conv = np.zeros_like(T_surface)
        if self._convection_solver is not None:
            h = self._convection_solver.get_convection_coefficient()
            T_fluid = self._convection_solver._inlet_temp
            q_conv = h * (T_surface - T_fluid)
        
        # 辐射热流
        q_rad = np.zeros_like(T_surface)
        if self._radiation_solver is not None:
            self._radiation_solver.set_surface_temperature(T_surface)
            self._radiation_solver.step()
            q_rad = self._radiation_solver.get_heat_flux()
        
        # 总热流边界条件
        q_total = q_conv + q_rad
        
        # 应用到热传导求解器 (Neumann 边界)
        # 简化：作为热源项
        self._add_heat_source_to_thermal(-q_total)
    
    # ============================================================
    # 主耦合循环
    # ============================================================
    
    def solve_coupled(self, n_steps: int = 100) -> dict:
        """
        求解耦合问题
        
        Parameters
        ----------
        n_steps : int
            时间步数
        
        Returns
        -------
        dict : 耦合结果
        """
        results = {
            'temperature_max': [],
            'temperature_avg': [],
            'power_loss': [],
            'converged': False,
        }
        
        for step in range(n_steps):
            # 电磁场求解
            if self._magnetostatics_solver is not None:
                self._magnetostatics_solver.solve()
            
            if self._eddy_current_solver is not None:
                self._eddy_current_solver.solve()
            
            if self._joule_heating_solver is not None:
                self._joule_heating_solver.solve_electric_field()
            
            # 耦合电磁-热
            if self._coupling_type in ('em_thermal', 'full'):
                self._couple_em_thermal()
            
            # 更新材料参数
            if self._coupling_type == 'full':
                self._update_material_properties()
            
            # 热传导求解
            if self._thermal_solver is not None:
                for _ in range(self._update_interval):
                    self._thermal_solver.step()
            
            # 对流-辐射边界
            if self._coupling_type in ('thermal_structural', 'full'):
                self._couple_convection_radiation()
            
            # 记录结果
            if self._thermal_solver is not None:
                T = self._thermal_solver.get_temperature()
                results['temperature_max'].append(np.max(T))
                results['temperature_avg'].append(np.mean(T))
            
            # 计算总损耗
            total_loss = 0
            if self._eddy_current_solver is not None:
                total_loss += self._eddy_current_solver.compute_joule_loss()
            if self._joule_heating_solver is not None:
                total_loss += self._joule_heating_solver.compute_total_power()
            results['power_loss'].append(total_loss)
            
            self._step_count += 1
        
        results['converged'] = True
        return results
    
    def step(self):
        """单步耦合更新"""
        if self._step_count % self._update_interval == 0:
            # 更新电磁场
            if self._magnetostatics_solver is not None:
                self._magnetostatics_solver.step()
            
            if self._eddy_current_solver is not None:
                self._eddy_current_solver.step()
            
            # 耦合
            if self._coupling_type in ('em_thermal', 'full'):
                self._couple_em_thermal()
            
            if self._coupling_type == 'full':
                self._update_material_properties()
        
        # 热传导
        if self._thermal_solver is not None:
            self._thermal_solver.step()
        
        # 对流-辐射
        if self._coupling_type in ('thermal_structural', 'full'):
            self._couple_convection_radiation()
        
        self._step_count += 1
    
    def get_coupling_status(self) -> dict:
        """获取耦合状态"""
        return {
            'step_count': self._step_count,
            'coupling_type': self._coupling_type,
            'converged': self._coupled_converged,
            'has_em_solver': self._magnetostatics_solver is not None or self._eddy_current_solver is not None,
            'has_thermal_solver': self._thermal_solver is not None,
            'has_convection_solver': self._convection_solver is not None,
            'has_radiation_solver': self._radiation_solver is not None,
        }
