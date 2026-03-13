"""
MPC + WBC Combined Controller

来源: openloong-dyn-control
组合 MPC (高层规划) + WBC (底层执行)
"""

import numpy as np
from typing import Optional, Dict, Tuple
from .mpc_controller import MPCController
from .wbc_controller import WBCController
from .gait_scheduler import GaitScheduler


class MPCWBCController:
    """
    MPC + WBC 组合控制器
    
    架构:
    - MPC: 生成质心轨迹和接触力
    - WBC: 将接触力转化为关节力矩
    - GaitScheduler: 管理步态时序
    """
    
    def __init__(
        self,
        num_dofs: int = 19,
        dt: float = 0.005,
        gait_frequency: float = 1.25,
        mass: float = 77.35,
        use_mpc: bool = True,
        use_wbc: bool = True
    ):
        """
        Args:
            num_dofs: 机器人自由度
            dt: 控制周期
            gait_frequency: 步态频率 (Hz)
            mass: 机器人质量
            use_mpc: 是否使用 MPC
            use_wbc: 是否使用 WBC
        """
        self.dt = dt
        self.num_dofs = num_dofs
        self.use_mpc = use_mpc
        self.use_wbc = use_wbc
        
        # 初始化子控制器
        if use_mpc:
            self.mpc = MPCController(dt=dt, mass=mass)
            self.mpc.enable()
        
        if use_wbc:
            self.wbc = WBCController(num_dofs=num_dofs, dt=dt)
        
        # 步态调度器
        self.gait_scheduler = GaitScheduler(frequency=gait_frequency)
        
        # 状态
        self.time = 0.0
        self.swing_leg = 'left'  # 当前摆动腿
    
    def update(
        self,
        base_pos: np.ndarray,
        base_rpy: np.ndarray,
        base_lin_vel: np.ndarray,
        base_ang_vel: np.ndarray,
        left_foot_pos: np.ndarray,
        right_foot_pos: np.ndarray,
        joint_pos: np.ndarray,
        joint_vel: np.ndarray,
        target_vel: np.ndarray
    ) -> np.ndarray:
        """
        更新控制器并计算输出
        
        Args:
            base_pos: 基座位置 [x, y, z]
            base_rpy: 基座姿态 [roll, pitch, yaw]
            base_lin_vel: 基座线速度
            base_ang_vel: 基座角速度
            left_foot_pos: 左脚位置
            right_foot_pos: 右脚位置
            joint_pos: 关节位置
            joint_vel: 关节速度
            target_vel: 目标速度 [vx, vy, yaw_rate]
        
        Returns:
            关节力矩或目标位置
        """
        # 更新步态
        phase = self.gait_scheduler.update(self.dt)
        self.swing_leg = self.gait_scheduler.get_swing_leg()
        
        com_pos = base_pos
        
        # MPC: 计算接触力
        if self.use_mpc:
            self.mpc.set_state(base_pos, base_rpy, 
                             base_lin_vel, base_ang_vel)
            self.mpc.set_foot_positions(left_foot_pos, right_foot_pos, com_pos)
            
            # 生成期望轨迹
            desired_traj = self._generate_desired_trajectory(
                com_pos, target_vel
            )
            self.mpc.set_desired_trajectory(desired_traj)
            
            contact_forces = self.mpc.compute()
        else:
            # 简化: 只补偿重力
            contact_forces = np.zeros(12)
            contact_forces[2] = -self.mpc.m * self.mpc.g / 2
            contact_forces[8] = -self.mpc.m * self.mpc.g / 2
        
        # WBC: 计算关节力矩
        if self.use_wbc:
            self.wbc.clear_tasks()
            
            # 添加质心任务
            J_com, xdd_com = self.wbc.compute_com_task(
                com_pos, base_lin_vel,
                com_pos + target_vel[:3] * 0.1,  # 简单目标
                target_vel[:3],
                contact_forces
            )
            self.wbc.add_task('com', J_com, xdd_com, priority=0)
            
            # 添加摆动脚任务
            if self.swing_leg == 'left':
                swing_pos = left_foot_pos
                stance_pos = right_foot_pos
            else:
                swing_pos = right_foot_pos
                stance_pos = left_foot_pos
            
            desired_swing_pos = stance_pos + target_vel[:3] * 0.3
            desired_swing_pos[2] = 0.1  # 抬高
            
            J_foot, xdd_foot = self.wbc.compute_swing_foot_task(
                swing_pos, np.zeros(3),
                desired_swing_pos, np.zeros(3)
            )
            self.wbc.add_task('swing_foot', J_foot, xdd_foot, priority=1)
            
            # 计算力矩
            tau = self.wbc.compute_torques(joint_pos, joint_vel, contact_forces)
        else:
            tau = np.zeros(self.num_dofs)
        
        self.time += self.dt
        
        return tau
    
    def _generate_desired_trajectory(
        self,
        current_pos: np.ndarray,
        target_vel: np.ndarray
    ) -> np.ndarray:
        """
        生成期望质心轨迹
        
        Args:
            current_pos: 当前位置
            target_vel: 目标速度 [vx, vy, yaw_rate]
        
        Returns:
            期望状态轨迹 [N*12]
        """
        N = self.mpc.N
        nx = self.mpc.nx
        trajectory = np.zeros(nx * N)
        
        for i in range(N):
            # 简单前向积分
            t = i * self.dt
            pos = current_pos + target_vel[:3] * t
            
            # 填充状态 [rpy(3), pos(3), ang_vel(3), lin_vel(3)]
            trajectory[i*nx + 3:i*nx + 6] = pos  # 位置
            trajectory[i*nx + 9:i*nx + 12] = target_vel[:3]  # 线速度
        
        return trajectory
    
    def set_gait_frequency(self, frequency: float):
        """设置步态频率"""
        self.gait_scheduler.frequency = frequency
    
    def get_contact_forces(self) -> np.ndarray:
        """获取当前接触力"""
        if self.use_mpc:
            return self.mpc.Fr_ff.copy()
        return np.zeros(12)
