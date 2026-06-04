"""
对流换热求解器 (Convection Solver)

实现强制对流和自然对流的传热计算：
强制对流: Nu = f(Re, Pr) - 液冷、风冷
自然对流: Nu = f(Gr, Pr) - 散热器、自然冷却

支持液冷管道网络、散热器翅片、对流换热系数计算。
对标 COMSOL Heat Transfer Module > Heat Transfer in Fluids。
"""

import numpy as np
import gstaichi as ti
import torch

import genesis as gs
import genesis.utils.array_class as array_class
from genesis.utils.misc import ti_to_torch

from .base_solver import Solver


@ti.data_oriented
class ConvectionSolver(Solver):
    """
    对流换热求解器
    
    计算对流换热系数和流体温度分布：
    强制对流: q = h·(T_surface - T_fluid)
    自然对流: q = h·(T_surface - T_ambient)
    
    其中 h = Nu·k_fluid / L_characteristic
    
    Parameters
    ----------
    scene : gs.Scene
    sim : gs.Simulator
    options : ConvectionOptions
        - convection_type: 'forced' | 'natural' | 'mixed'
        - fluid: 'water' | 'air' | 'oil' | custom
        - flow_rate: 体积流量 [m³/s]
        - inlet_temp: 进口温度 [K]
        - channel_geometry: 流道几何参数
        - fan_speed: 风扇转速 [RPM] (风冷)
        - gravity: 重力方向 (自然对流)
    """
    
    def __init__(self, scene, sim, options):
        super().__init__(scene, sim, options)
        
        # 对流类型
        self._convection_type = options.convection_type
        self._fluid_type = options.fluid
        
        # 流体参数
        self._flow_rate = options.flow_rate if hasattr(options, 'flow_rate') else 0.001
        self._inlet_temp = options.inlet_temp if hasattr(options, 'inlet_temp') else 300.0
        self._outlet_temp = self._inlet_temp
        
        # 流道几何
        self._channel_diameter = options.channel_diameter if hasattr(options, 'channel_diameter') else 0.01
        self._channel_length = options.channel_length if hasattr(options, 'channel_length') else 0.1
        self._n_channels = options.n_channels if hasattr(options, 'n_channels') else 1
        
        # 风冷参数
        self._fan_speed = options.fan_speed if hasattr(options, 'fan_speed') else 0
        self._fin_height = options.fin_height if hasattr(options, 'fin_height') else 0.02
        self._fin_thickness = options.fin_thickness if hasattr(options, 'fin_thickness') else 0.002
        self._fin_spacing = options.fin_spacing if hasattr(options, 'fin_spacing') else 0.005
        
        # 自然对流参数
        self._gravity = options.gravity if hasattr(options, 'gravity') else (0, 0, -9.81)
        self._orientation = options.orientation if hasattr(options, 'orientation') else 'vertical'
        
        # 流体物性
        self._fluid_props = self._get_fluid_properties()
        
        # 计算对流换热系数
        self._h = self._compute_convection_coefficient()
        
        # 场变量
        self._T_fluid = None      # 流体温度
        self._T_surface = None    # 表面温度
        self._heat_flux = None    # 热流密度
        self._h_field = None      # 局部换热系数
        
        # 管道网络
        self._pipe_network = options.pipe_network if hasattr(options, 'pipe_network') else None
    
    def _get_fluid_properties(self) -> dict:
        """获取流体物性参数"""
        fluids = {
            'water': {
                'rho': 998.0,      # kg/m³
                'cp': 4182.0,      # J/(kg·K)
                'k': 0.598,        # W/(m·K)
                'mu': 1.002e-3,    # Pa·s
                'Pr': 7.0,         # Prandtl number
            },
            'air': {
                'rho': 1.225,
                'cp': 1005.0,
                'k': 0.0257,
                'mu': 1.81e-5,
                'Pr': 0.71,
            },
            'oil': {
                'rho': 850.0,
                'cp': 2000.0,
                'k': 0.15,
                'mu': 0.05,
                'Pr': 100.0,
            },
        }
        
        if self._fluid_type in fluids:
            return fluids[self._fluid_type]
        else:
            # 自定义流体
            return {
                'rho': getattr(self.options, 'fluid_rho', 1000.0),
                'cp': getattr(self.options, 'fluid_cp', 4182.0),
                'k': getattr(self.options, 'fluid_k', 0.6),
                'mu': getattr(self.options, 'fluid_mu', 1e-3),
                'Pr': getattr(self.options, 'fluid_Pr', 7.0),
            }
    
    def _compute_convection_coefficient(self) -> float:
        """计算对流换热系数 h [W/(m²·K)]"""
        props = self._fluid_props
        
        if self._convection_type == 'forced':
            return self._compute_forced_convection(props)
        elif self._convection_type == 'natural':
            return self._compute_natural_convection(props)
        else:  # mixed
            h_forced = self._compute_forced_convection(props)
            h_natural = self._compute_natural_convection(props)
            # 混合对流: h = (h_forced³ + h_natural³)^(1/3)
            return (h_forced**3 + h_natural**3)**(1/0.333)
    
    def _compute_forced_convection(self, props: dict) -> float:
        """计算强制对流换热系数"""
        rho = props['rho']
        k = props['k']
        mu = props['mu']
        Pr = props['Pr']
        
        # 流速
        A_channel = np.pi * (self._channel_diameter / 2)**2
        velocity = self._flow_rate / (A_channel * self._n_channels)
        
        # Reynolds number
        Re = rho * velocity * self._channel_diameter / mu
        
        # Nusselt number - 根据流态选择关联式
        if Re < 2300:  # 层流
            # 充分发展层流: Nu = 3.66 (常壁温) 或 4.36 (常热流)
            Nu = 3.66
            # 入口效应修正
            Gz = Re * Pr * self._channel_diameter / self._channel_length
            if Gz > 10:
                Nu = 3.66 + 0.0668 * Gz / (1 + 0.04 * Gz**0.67)
        else:  # 湍流
            # Dittus-Boelter: Nu = 0.023·Re^0.8·Pr^n
            n = 0.4 if self._inlet_temp > 300 else 0.3  # 加热/冷却
            Nu = 0.023 * Re**0.8 * Pr**n
            # Gnielinski 修正 (更精确)
            f = (0.79 * np.log(Re) - 1.64)**(-2)
            Nu = (f / 8) * (Re - 1000) * Pr / (1 + 12.7 * (f / 8)**0.5 * (Pr**(2/0.333) - 1))
        
        # 换热系数
        h = Nu * k / self._channel_diameter
        
        return h
    
    def _compute_natural_convection(self, props: dict) -> float:
        """计算自然对流换热系数"""
        rho = props['rho']
        k = props['k']
        mu = props['mu']
        Pr = props['Pr']
        
        # 热膨胀系数 (近似)
        beta = 1.0 / self._inlet_temp  # 理想气体近似
        
        # 特征长度
        if self._orientation == 'vertical':
            L = self._channel_length
        else:  # horizontal
            L = self._channel_diameter
        
        # Grashof number
        delta_T = 20.0  # 假设温差
        g = abs(self._gravity[2])
        Gr = g * beta * delta_T * L**3 / (mu / rho)**2
        
        # Rayleigh number
        Ra = Gr * Pr
        
        # Nusselt number
        if Ra < 1e9:  # 层流
            if self._orientation == 'vertical':
                Nu = 0.59 * Ra**0.25
            else:  # horizontal cylinder
                Nu = 0.53 * Ra**0.25
        else:  # 湍流
            Nu = 0.1 * Ra**(1/0.333)
        
        # 换热系数
        h = Nu * k / L
        
        return h
    
    def build(self):
        """构建求解器场变量"""
        super().build()
        
        # 这里可以构建更复杂的流场计算
        # 简化版本：使用集总参数法
        pass
    
    # ============================================================
    # 散热器设计计算
    # ============================================================
    
    def compute_fin_efficiency(self, h: float, k_fin: float = 200.0) -> float:
        """
        计算翅片效率
        
        Parameters
        ----------
        h : float
            对流换热系数
        k_fin : float
            翅片材料热导率 (铝: ~200, 铜: ~400)
        
        Returns
        -------
        float : 翅片效率 η_fin
        """
        # 矩形翅片效率
        m = np.sqrt(2 * h / (k_fin * self._fin_thickness))
        Lc = self._fin_height + self._fin_thickness / 2  # 修正高度
        
        eta = np.tanh(m * Lc) / (m * Lc)
        
        return eta
    
    def compute_heat_sink_thermal_resistance(self) -> dict:
        """
        计算散热器总热阻
        
        Returns
        -------
        dict : {'convection': R_conv, 'conduction': R_cond, 'total': R_total}
        """
        # 翅片表面积
        A_fin_single = 2 * self._fin_height * self._channel_length
        n_fins = int(self._channel_length / (self._fin_thickness + self._fin_spacing))
        A_fins_total = n_fins * A_fin_single
        
        # 基板面积
        A_base = self._channel_length * self._channel_length
        
        # 翅片效率
        eta_fin = self.compute_fin_efficiency(self._h)
        
        # 总表面积 (含翅片和基板)
        A_total = eta_fin * A_fins_total + A_base
        
        # 对流热阻
        R_conv = 1.0 / (self._h * A_total)
        
        # 导热热阻 (基板)
        t_base = 0.005  # 基板厚度 5mm
        k_base = 200.0  # 铝
        R_cond = t_base / (k_base * A_base)
        
        return {
            'convection': R_conv,
            'conduction': R_cond,
            'total': R_conv + R_cond,
            'fin_efficiency': eta_fin,
            'surface_area': A_total,
        }
    
    # ============================================================
    # 液冷系统计算
    # ============================================================
    
    def compute_cooling_performance(self, heat_load: float, T_surface: float) -> dict:
        """
        计算冷却系统性能
        
        Parameters
        ----------
        heat_load : float
            热负荷 [W]
        T_surface : float
            表面温度 [K]
        
        Returns
        -------
        dict : 冷却性能参数
        """
        props = self._fluid_props
        
        # 质量流量
        mdot = props['rho'] * self._flow_rate
        
        # 温升
        delta_T_fluid = heat_load / (mdot * props['cp'])
        
        # 出口温度
        T_out = self._inlet_temp + delta_T_fluid
        
        # 对数平均温差 (LMTD)
        delta_T1 = T_surface - self._inlet_temp
        delta_T2 = T_surface - T_out
        if abs(delta_T1 - delta_T2) < 0.01:
            LMTD = delta_T1
        else:
            LMTD = (delta_T1 - delta_T2) / np.log(delta_T1 / delta_T2)
        
        # 所需换热面积
        A_required = heat_load / (self._h * LMTD)
        
        # 压降 (Darcy-Weisbach)
        velocity = self._flow_rate / (np.pi * (self._channel_diameter/2)**2)
        Re = props['rho'] * velocity * self._channel_diameter / props['mu']
        
        if Re < 2300:
            f = 64 / Re  # 层流
        else:
            f = 0.316 / Re**0.25  # Blasius
        
        delta_P = f * (self._channel_length / self._channel_diameter) * \
                  props['rho'] * velocity**2 / 2
        
        # 泵功率
        pump_power = self._flow_rate * delta_P / 0.7  # 假设泵效率 70%
        
        return {
            'outlet_temperature': T_out,
            'temperature_rise': delta_T_fluid,
            'LMTD': LMTD,
            'required_area': A_required,
            'pressure_drop': delta_P,
            'pump_power': pump_power,
            'Reynolds_number': Re,
            'heat_transfer_coefficient': self._h,
            'flow_velocity': velocity,
        }
    
    def design_cooling_system(self, max_temp: float, heat_load: float) -> dict:
        """
        设计冷却系统参数
        
        Parameters
        ----------
        max_temp : float
            允许最高温度 [K]
        heat_load : float
            热负荷 [W]
        
        Returns
        -------
        dict : 设计参数
        """
        # 计算所需流量
        props = self._fluid_props
        delta_T_max = max_temp - self._inlet_temp
        mdot_min = heat_load / (props['cp'] * delta_T_max)
        flow_rate_min = mdot_min / props['rho']
        
        # 计算所需换热面积
        A_min = heat_load / (self._h * delta_T_max * 0.5)
        
        # 计算所需流道长度
        perimeter = np.pi * self._channel_diameter
        L_min = A_min / (perimeter * self._n_channels)
        
        return {
            'min_flow_rate': flow_rate_min,
            'min_channel_length': L_min,
            'min_surface_area': A_min,
            'recommended_channels': max(1, int(np.ceil(A_min / (perimeter * 0.1)))),
        }
    
    # ============================================================
    # 公共接口
    # ============================================================
    
    def get_convection_coefficient(self) -> float:
        """获取对流换热系数 [W/(m²·K)]"""
        return self._h
    
    def set_flow_rate(self, flow_rate: float):
        """设置流量并重新计算换热系数"""
        self._flow_rate = flow_rate
        self._h = self._compute_convection_coefficient()
    
    def set_inlet_temperature(self, temp: float):
        """设置进口温度"""
        self._inlet_temp = temp
    
    def get_fluid_properties(self) -> dict:
        """获取流体物性"""
        return self._fluid_props.copy()
    
    def step(self):
        """单步更新 (简化版本)"""
        # 在完整实现中，这里会更新流场和温度场
        pass
