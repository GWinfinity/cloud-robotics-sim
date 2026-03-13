"""
WBC (Whole-Body Control) Controller

来源: openloong-dyn-control
核心算法: 基于零空间投影的全身优先级控制
"""

import numpy as np
from typing import Dict, List, Optional, Tuple


class WBCController:
    """
    全身控制器 (Whole-Body Control)
    
    使用零空间投影实现多任务优先级控制。
    任务栈 (从高到低优先级):
    1. 质心动量跟踪
    2. 摆动脚轨迹跟踪
    3. 上身姿态保持
    4. 关节极限避免
    """
    
    def __init__(
        self,
        num_dofs: int = 19,
        dt: float = 0.005,
        kp_com: float = 100.0,
        kd_com: float = 20.0
    ):
        """
        Args:
            num_dofs: 机器人自由度数量
            dt: 控制周期
            kp_com: 质心位置比例增益
            kd_com: 质心位置微分增益
        """
        self.num_dofs = num_dofs
        self.dt = dt
        
        # PD 增益
        self.kp_com = kp_com
        self.kd_com = kd_com
        self.kp_swing = 400.0
        self.kd_swing = 40.0
        self.kp_orientation = 100.0
        self.kd_orientation = 20.0
        
        # 任务列表
        self.tasks: List[Dict] = []
        
        # 惯性矩阵 (简化，实际需要机器人模型)
        self.M = np.eye(num_dofs) * 10.0
        self.M_inv = np.linalg.inv(self.M)
    
    def add_task(
        self,
        name: str,
        jacobian: np.ndarray,
        desired_accel: np.ndarray,
        weight: float = 1.0,
        priority: int = 0
    ):
        """
        添加任务
        
        Args:
            name: 任务名称
            jacobian: 任务雅可比矩阵
            desired_accel: 期望加速度
            weight: 任务权重
            priority: 优先级 (0 最高)
        """
        self.tasks.append({
            'name': name,
            'J': jacobian,
            'xdd': desired_accel,
            'weight': weight,
            'priority': priority
        })
        
        # 按优先级排序
        self.tasks.sort(key=lambda t: t['priority'])
    
    def compute_torques(
        self,
        joint_pos: np.ndarray,
        joint_vel: np.ndarray,
        contact_forces: np.ndarray
    ) -> np.ndarray:
        """
        计算关节力矩
        
        Args:
            joint_pos: 当前关节位置
            joint_vel: 当前关节速度
            contact_forces: 接触力/力矩 [FL(6), FR(6)]
        
        Returns:
            关节力矩
        """
        # 初始化
        tau = np.zeros(self.num_dofs)
        N = np.eye(self.num_dofs)  # 零空间投影矩阵
        
        # 遍历任务 (按优先级)
        for task in self.tasks:
            J = task['J']
            xdd = task['xdd']
            
            # 计算任务空间动力学
            J_pinv = self._pseudoinverse(J @ N)
            
            # 计算关节加速度
            qdd = J_pinv @ (xdd - J @ joint_vel)
            
            # 计算力矩
            tau_task = self.M @ qdd
            
            # 累加力矩
            tau += N @ tau_task
            
            # 更新零空间投影矩阵
            N = N @ (np.eye(self.num_dofs) - J_pinv @ J)
        
        # 添加接触力补偿 (简化)
        tau += self._contact_force_compensation(contact_forces)
        
        return tau
    
    def _pseudoinverse(self, J: np.ndarray, damping: float = 0.01) -> np.ndarray:
        """计算阻尼伪逆"""
        return J.T @ np.linalg.inv(J @ J.T + damping**2 * np.eye(J.shape[0]))
    
    def _contact_force_compensation(self, contact_forces: np.ndarray) -> np.ndarray:
        """接触力补偿"""
        # 简化实现: 将接触力映射到关节
        # 实际需要接触雅可比
        return np.zeros(self.num_dofs)
    
    def compute_com_task(
        self,
        com_pos: np.ndarray,
        com_vel: np.ndarray,
        desired_com_pos: np.ndarray,
        desired_com_vel: np.ndarray,
        contact_forces: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算质心任务
        
        Returns:
            (雅可比, 期望加速度)
        """
        # 质心雅可比 (简化)
        J_com = np.zeros((3, self.num_dofs))
        J_com[:, :6] = np.eye(3)  # 假设前6个是浮动基
        
        # 期望加速度 (PD控制)
        pos_err = desired_com_pos - com_pos
        vel_err = desired_com_vel - com_vel
        
        # 从接触力计算期望加速度
        total_force = contact_forces[:3] + contact_forces[6:9]
        desired_accel = total_force / 77.35  # 质量
        
        # PD修正
        desired_accel += self.kp_com * pos_err + self.kd_com * vel_err
        
        return J_com, desired_accel
    
    def compute_swing_foot_task(
        self,
        swing_foot_pos: np.ndarray,
        swing_foot_vel: np.ndarray,
        desired_pos: np.ndarray,
        desired_vel: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算摆动脚任务
        
        Returns:
            (雅可比, 期望加速度)
        """
        # 摆动脚雅可比 (简化)
        J_foot = np.zeros((3, self.num_dofs))
        # 实际需要运动学计算
        
        # 期望加速度
        pos_err = desired_pos - swing_foot_pos
        vel_err = desired_vel - swing_foot_vel
        
        desired_accel = (self.kp_swing * pos_err + 
                        self.kd_swing * vel_err)
        
        return J_foot, desired_accel
    
    def clear_tasks(self):
        """清除所有任务"""
        self.tasks = []
