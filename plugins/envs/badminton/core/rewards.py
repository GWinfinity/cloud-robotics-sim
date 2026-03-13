"""
羽毛球奖励函数
"""

import numpy as np
from typing import Dict, Tuple


def compute_hit_reward(
    hit: bool,
    hit_speed: float,
    target_speed: float = 15.0,
    weights: Dict = None
) -> float:
    """
    计算击球奖励
    
    Args:
        hit: 是否击中
        hit_speed: 击球速度
        target_speed: 目标速度
        weights: 奖励权重
        
    Returns:
        奖励值
    """
    if weights is None:
        weights = {'hit': 10.0, 'speed': 1.0}
    
    reward = 0.0
    
    if hit:
        reward += weights['hit']
        
        # 速度奖励
        speed_ratio = min(hit_speed / target_speed, 2.0)
        reward += weights['speed'] * speed_ratio
    
    return reward


def compute_landing_reward(
    landing_pos: np.ndarray,
    ideal_landing: np.ndarray,
    court_bounds: Dict,
    weights: Dict = None
) -> float:
    """
    计算落点奖励
    
    Args:
        landing_pos: 实际落点
        ideal_landing: 理想落点
        court_bounds: 场地边界
        weights: 奖励权重
        
    Returns:
        奖励值
    """
    if weights is None:
        weights = {'landing': 5.0, 'out_of_bounds': -2.0}
    
    # 检查是否在场地内
    x, y = landing_pos[0], landing_pos[1]
    in_bounds = (
        court_bounds['x_min'] <= x <= court_bounds['x_max'] and
        court_bounds['y_min'] <= y <= court_bounds['y_max']
    )
    
    if not in_bounds:
        return weights.get('out_of_bounds', -2.0)
    
    # 距离奖励
    distance = np.linalg.norm(landing_pos[:2] - ideal_landing[:2])
    reward = weights['landing'] * np.exp(-distance / 2.0)
    
    return reward


def compute_footwork_reward(
    robot_pos: np.ndarray,
    target_pos: np.ndarray,
    robot_vel: np.ndarray,
    weights: Dict = None
) -> float:
    """
    计算步法奖励
    
    Args:
        robot_pos: 机器人位置
        target_pos: 目标位置
        robot_vel: 机器人速度
        weights: 奖励权重
        
    Returns:
        奖励值
    """
    if weights is None:
        weights = {'position': 1.0, 'velocity': 0.3, 'energy': -0.001}
    
    # 位置奖励
    distance = np.linalg.norm(robot_pos[:2] - target_pos[:2])
    position_reward = np.exp(-distance)
    
    # 速度朝向目标的奖励
    if np.linalg.norm(robot_vel) > 0.01:
        direction_to_target = (target_pos[:2] - robot_pos[:2])
        if np.linalg.norm(direction_to_target) > 0.01:
            direction_to_target /= np.linalg.norm(direction_to_target)
            velocity_norm = robot_vel[:2] / (np.linalg.norm(robot_vel[:2]) + 1e-8)
            alignment = np.dot(direction_to_target, velocity_norm)
            velocity_reward = max(0, alignment)
        else:
            velocity_reward = 0.0
    else:
        velocity_reward = 0.0
    
    # 能量惩罚
    energy_penalty = np.sum(robot_vel ** 2)
    
    reward = (
        weights['position'] * position_reward +
        weights['velocity'] * velocity_reward +
        weights['energy'] * energy_penalty
    )
    
    return reward
