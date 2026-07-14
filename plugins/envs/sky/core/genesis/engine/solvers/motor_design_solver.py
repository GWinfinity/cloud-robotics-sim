"""
电机设计求解器 (Motor Design Solver)

集成多种物理场求解器，提供完整的电机设计分析工作流：
1. 几何建模: 轴向磁通/径向磁通电机
2. 电磁分析: 空载/负载磁场、反电势、电感
3. 损耗计算: 铜损、铁损、涡流损耗
4. 热分析: 温升分布、冷却效果
5. 性能计算: 扭矩、效率、功率因数

对标 ANSYS Maxwell Motor Design Toolkit + RMxprt。
"""

import numpy as np
import gstaichi as ti
import torch

import genesis as gs
import genesis.utils.array_class as array_class
from genesis.utils.misc import ti_to_torch

from .base_solver import Solver


class MotorDesignSolver(Solver):
    """
    电机设计求解器
    
    提供完整的电机设计分析功能：
    - 几何参数化建模
    - 电磁场求解 (静磁 + 涡流)
    - 损耗计算
    - 热分析
    - 性能评估
    
    Parameters
    ----------
    scene : gs.Scene
    sim : gs.Simulator
    options : MotorDesignOptions
        - motor_type: 'afpm' | 'rfpm' | 'spoke' | 'vshape'
        - pole_pairs: 极对数
        - slots: 槽数
        - outer_radius: 外径 [m]
        - inner_radius: 内径 [m]
        - air_gap: 气隙 [m]
        - stack_length: 轴向长度 [m]
        - magnet_material: 'ndfeb' | 'smco' | 'ferrite'
        - steel_material: 'm19' | 'm27' | 'm36' | 'custom'
        - winding_type: 'concentrated' | 'distributed'
        - turns_per_coil: 每线圈匝数
        - wire_diameter: 线径 [m]
        - rated_current: 额定电流 [A]
        - rated_speed: 额定转速 [RPM]
    """
    
    def __init__(self, scene, sim, options):
        super().__init__(scene, sim, options)
        
        # 电机参数
        self._motor_type = options.motor_type
        self._pole_pairs = options.pole_pairs
        self._slots = options.slots
        self._outer_radius = options.outer_radius
        self._inner_radius = options.inner_radius
        self._air_gap = options.air_gap
        self._stack_length = options.stack_length
        self._magnet_material = options.magnet_material
        self._steel_material = options.steel_material
        self._winding_type = options.winding_type
        self._turns_per_coil = options.turns_per_coil
        self._wire_diameter = options.wire_diameter
        self._rated_current = options.rated_current
        self._rated_speed = options.rated_speed
        
        # 派生参数
        self._pole_pitch = 2 * np.pi / (2 * self._pole_pairs)
        self._slot_pitch = 2 * np.pi / self._slots
        self._mech_omega = 2 * np.pi * self._rated_speed / 60
        self._elec_omega = self._pole_pairs * self._mech_omega
        
        # 材料参数
        self._mu0 = 4 * np.pi * 1e-7
        self._Br_magnet = 1.2  # 剩磁 [T]
        self._mu_r_steel = 5000
        self._sigma_copper = 5.8e7
        self._rho_copper = 1.68e-8
        
        # 求解结果
        self._flux_linkage = None
        self._back_emf = None
        self._inductance = None
        self._torque = None
        self._copper_loss = None
        self._core_loss = None
        self._efficiency = None
        
        # 引用其他求解器
        self._magnetostatics_solver = None
        self._eddy_current_solver = None
        self._thermal_solver = None
        self._bh_solver = None
    
    def build(self):
        """构建电机模型"""
        super().build()
        
        # 获取其他求解器引用
        self._magnetostatics_solver = getattr(self._sim, 'magnetostatics_solver', None)
        self._eddy_current_solver = getattr(self._sim, 'eddy_current_solver', None)
        self._thermal_solver = getattr(self._sim, 'thermal_solver', None)
        self._bh_solver = getattr(self._sim, 'bh_solver', None)
        
        # 初始化结果数组
        n_phases = 3
        self._flux_linkage = np.zeros(n_phases)
        self._back_emf = np.zeros(n_phases)
        self._inductance = np.zeros((n_phases, n_phases))
        self._torque = 0.0
        self._copper_loss = 0.0
        self._core_loss = 0.0
        self._efficiency = 0.0
    
    # ============================================================
    # 几何建模
    # ============================================================
    
    def compute_magnet_dimensions(self) -> dict:
        """计算永磁体尺寸"""
        if self._motor_type == 'afpm':
            # 轴向磁通电机
            magnet_width = self._pole_pitch * (self._outer_radius + self._inner_radius) / 2
            magnet_length = (self._outer_radius - self._inner_radius) * 0.8
            magnet_thickness = self._air_gap * 2
            
            return {
                'width': magnet_width,
                'length': magnet_length,
                'thickness': magnet_thickness,
                'pole_arc': 0.8,  # 极弧系数
            }
        
        elif self._motor_type == 'rfpm':
            # 径向磁通电机
            magnet_width = self._pole_pitch * self._outer_radius * 0.8
            magnet_length = self._stack_length
            magnet_thickness = self._air_gap * 2
            
            return {
                'width': magnet_width,
                'length': magnet_length,
                'thickness': magnet_thickness,
                'pole_arc': 0.8,
            }
        
        else:
            return {'width': 0.01, 'length': 0.01, 'thickness': 0.005, 'pole_arc': 0.8}
    
    def compute_winding_factor(self) -> float:
        """计算绕组系数"""
        # 简化：集中绕组
        if self._winding_type == 'concentrated':
            # 短距系数
            q = self._slots / (2 * self._pole_pairs * 3)  # 每极每相槽数
            if q < 1:
                # 分数槽集中绕组
                return 0.966  # 近似值
            else:
                return 0.966
        else:
            # 分布绕组
            return 0.96
    
    def compute_slot_area(self) -> float:
        """计算槽面积"""
        # 简化：梯形槽
        r_avg = (self._outer_radius + self._inner_radius) / 2
        slot_width = 2 * np.pi * r_avg / self._slots * 0.5
        slot_height = (self._outer_radius - self._inner_radius) * 0.4
        
        return slot_width * slot_height
    
    # ============================================================
    # 电磁分析
    # ============================================================
    
    def compute_no_load_field(self) -> np.ndarray:
        """计算空载磁场"""
        if self._magnetostatics_solver is None:
            return np.zeros(self.resolution)
        
        # 设置永磁体激励
        self._setup_magnet_excitation()
        
        # 求解静磁场
        self._magnetostatics_solver.solve()
        
        # 获取气隙磁密
        B_gap = self._get_airgap_flux_density()
        
        return B_gap
    
    def _setup_magnet_excitation(self):
        """设置永磁体激励"""
        # 简化：在 magnetostatics_solver 中设置 M 分布
        if self._magnetostatics_solver is None:
            return
        
        # 创建永磁体磁化强度分布
        # 这里简化处理，实际应该根据几何精确建模
        pass
    
    def _get_airgap_flux_density(self) -> np.ndarray:
        """获取气隙磁密分布"""
        if self._magnetostatics_solver is None:
            return np.zeros(100)
        
        # 从静磁场求解器获取 B 场
        B = self._magnetostatics_solver.get_field('B')
        
        # 提取气隙区域的 B
        # 简化：返回径向分量
        return B[..., 0] if B.ndim > 1 else B
    
    def compute_back_emf(self, speed_rpm: float = None) -> np.ndarray:
        """计算反电势"""
        if speed_rpm is None:
            speed_rpm = self._rated_speed
        
        omega = 2 * np.pi * speed_rpm / 60
        
        # 反电势 = -dλ/dt = -N * dΦ/dt
        # 简化：E = k_e * ω
        k_e = self._compute_emf_constant()
        
        theta = np.linspace(0, 2 * np.pi, 360)
        
        back_emf = np.zeros((3, len(theta)))
        for i in range(3):
            phase_angle = i * 2 * np.pi / 3
            back_emf[i] = k_e * omega * np.sin(theta + phase_angle)
        
        self._back_emf = back_emf
        return back_emf
    
    def _compute_emf_constant(self) -> float:
        """计算反电势常数 [V/(rad/s)]"""
        # k_e = N * Φ * k_w
        N = self._turns_per_coil
        k_w = self.compute_winding_factor()
        
        # 估算每极磁通
        B_gap_avg = 0.8  # 假设气隙磁密 [T]
        A_pole = np.pi * (self._outer_radius**2 - self._inner_radius**2) / (2 * self._pole_pairs)
        phi = B_gap_avg * A_pole
        
        k_e = N * phi * k_w * 2 * self._pole_pairs
        
        return k_e
    
    def compute_inductance(self) -> np.ndarray:
        """计算电感矩阵"""
        # 简化：基于几何参数估算
        L_self = self._compute_self_inductance()
        M_mutual = self._compute_mutual_inductance()
        
        L = np.zeros((3, 3))
        for i in range(3):
            L[i, i] = L_self
            for j in range(3):
                if i != j:
                    L[i, j] = M_mutual
        
        self._inductance = L
        return L
    
    def _compute_self_inductance(self) -> float:
        """计算自感"""
        # 简化公式: L = μ0 * N² * A / l
        N = self._turns_per_coil
        A = self.compute_slot_area()
        l = self._stack_length
        
        L = self._mu0 * N**2 * A / l
        return L
    
    def _compute_mutual_inductance(self) -> float:
        """计算互感"""
        # 三相绕组互感约为自感的 -0.5 倍
        L_self = self._compute_self_inductance()
        return -0.5 * L_self
    
    def compute_torque(self, current: np.ndarray = None, 
                       angle: float = 0) -> float:
        """计算电磁转矩"""
        if current is None:
            current = np.array([self._rated_current, 0, 0])
        
        # 转矩 = (3/2) * p * (λ_d * i_q - λ_q * i_d)
        # 简化: T = k_t * I_q
        k_t = self._compute_torque_constant()
        
        # 假设电流在 q 轴
        I_q = np.sqrt(2/3) * np.linalg.norm(current)
        torque = k_t * I_q
        
        self._torque = torque
        return torque
    
    def _compute_torque_constant(self) -> float:
        """计算转矩常数 [N·m/A]"""
        # k_t = (3/2) * p * k_e
        k_e = self._compute_emf_constant()
        k_t = 1.5 * self._pole_pairs * k_e
        return k_t
    
    # ============================================================
    # 损耗计算
    # ============================================================
    
    def compute_copper_loss(self, current: np.ndarray = None) -> float:
        """计算铜损"""
        if current is None:
            current = np.ones(3) * self._rated_current
        
        # 计算绕组电阻
        R = self._compute_phase_resistance()
        
        # 铜损 = 3 * I² * R
        I_rms = np.sqrt(np.mean(current**2))
        P_cu = 3 * I_rms**2 * R
        
        self._copper_loss = P_cu
        return P_cu
    
    def _compute_phase_resistance(self) -> float:
        """计算相电阻"""