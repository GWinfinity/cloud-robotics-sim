"""
奖励函数实现 - 专门用于人形机器人跌倒保护
"""

import numpy as np
from typing import Dict, List, Tuple


def compute_triangle_reward(
    body_positions: Dict,
    body_contacts: Dict,
    torso_height: float,
    target_height: float = 0.8
) -> float:
    """
    计算三角形结构奖励 - 核心奖励函数
    
    鼓励机器人形成三角形支撑结构来保护关键部位
    
    Args:
        body_positions: 身体部位位置
        body_contacts: 身体部位接触状态
        torso_height: 躯干高度
        target_height: 目标躯干高度
        
    Returns:
        reward: 三角形结构奖励
    """
    reward = 0.0
    
    # 1. 基础支撑点检测
    support_points = []
    
    # 检查手和脚是否接触地面
    for part_name, is_contact in body_contacts.items():
        if is_contact and part_name in ['hand_l', 'hand_r', 'foot_l', 'foot_r']:
            support_points.append(body_positions[part_name])
    
    # 2. 三角形形成奖励
    num_supports = len(support_points)
    
    if num_supports >= 3:
        # 形成了至少3点支撑
        reward += 1.0
        
        # 3. 计算支撑多边形的面积 (稳定性指标)
        if num_supports >= 3:
            # 计算凸包面积
            try:
                from scipy.spatial import ConvexHull
                points_2d = np.array([[p[0], p[1]] for p in support_points])
                hull = ConvexHull(points_2d)
                base_area = hull.volume  # 2D 凸包的面积
                
                # 奖励大面积的稳定支撑
                reward += min(base_area * 0.5, 1.0)
            except:
                # scipy 不可用，使用简化计算
                # 计算包围盒面积
                if len(support_points) >= 3:
                    xs = [p[0] for p in support_points]
                    ys = [p[1] for p in support_points]
                    base_area = (max(xs) - min(xs)) * (max(ys) - min(ys))
                    reward += min(base_area * 0.5, 1.0)
    
    # 4. 关键部位高度奖励
    # 躯干和头部保持较高位置
    height_ratio = torso_height / target_height
    if height_ratio > 0.5:
        reward += 0.5 * min(height_ratio, 1.0)
    
    # 5. 重心投影奖励
    # 计算重心在支撑多边形内的奖励
    if num_supports >= 3:
        # 简化的重心投影检查
        com_projection_in_support = check_com_in_support_polygon(
            body_positions.get('com', np.zeros(3)),
            support_points
        )
        if com_projection_in_support:
            reward += 0.5
    
    return reward


def compute_impact_penalty(
    contact_forces: Dict[str, float],
    critical_parts: List[str] = None,
    force_threshold: float = 100.0
) -> float:
    """
    计算冲击惩罚
    
    惩罚高冲击力的接触，特别是关键部位
    
    Args:
        contact_forces: 各部位的接触力 {部位名称: 力的大小}
        critical_parts: 关键部位列表 (如头部、躯干)
        force_threshold: 力的阈值
        
    Returns:
        penalty: 冲击惩罚 (正值表示惩罚)
    """
    if critical_parts is None:
        critical_parts = ['head', 'torso', 'pelvis']
    
    penalty = 0.0
    
    for part_name, force in contact_forces.items():
        if force > force_threshold:
            # 基础惩罚
            base_penalty = (force - force_threshold) / force_threshold
            
            # 关键部位惩罚加倍
            if part_name in critical_parts:
                base_penalty *= 2.0
            
            penalty += base_penalty
    
    return min(penalty, 10.0)  # 限制最大惩罚


def compute_orientation_reward(
    torso_quat: np.ndarray,
    target_up: np.ndarray = None
) -> float:
    """
    计算姿态奖励
    
    鼓励躯干保持直立
    
    Args:
        torso_quat: 躯干四元数 [x, y, z, w]
        target_up: 目标向上方向
        
    Returns:
        reward: 姿态奖励
    """
    if target_up is None:
        target_up = np.array([0, 0, 1])
    
    # 将四元数转换为旋转矩阵或直接使用
    # 简化为检查躯干的高度
    # 实际实现中应使用四元数计算与目标姿态的相似度
    
    # 这里我们假设 torso_quat 已经是某种姿态表示
    # 实际实现需要根据 Genesis 的接口调整
    
    return 0.0  # 占位


def compute_energy_efficiency(
    actions: np.ndarray,
    joint_velocities: np.ndarray,
    joint_torques: np.ndarray = None
) -> float:
    """
    计算能量效率奖励
    
    鼓励平滑、节能的动作
    
    Args:
        actions: 动作
        joint_velocities: 关节速度
        joint_torques: 关节力矩 (可选)
        
    Returns:
        penalty: 能量惩罚 (负值)
    """
    # 动作幅值惩罚
    action_magnitude = np.mean(actions ** 2)
    
    # 速度惩罚
    velocity_magnitude = np.mean(joint_velocities ** 2)
    
    # 力矩惩罚 (如果有)
    torque_penalty = 0.0
    if joint_torques is not None:
        torque_penalty = np.mean(joint_torques ** 2)
    
    # 综合能量惩罚
    energy_penalty = action_magnitude + 0.1 * velocity_magnitude + 0.01 * torque_penalty
    
    return -energy_penalty


def compute_joint_protection_reward(
    joint_positions: np.ndarray,
    joint_velocities: np.ndarray,
    joint_limits: Dict
) -> float:
    """
    计算关节保护奖励
    
    避免关节超限和过速
    
    Args:
        joint_positions: 关节位置
        joint_velocities: 关节速度
        joint_limits: 关节限制 {'lower': [], 'upper': [], 'velocity': []}
        
    Returns:
        reward: 关节保护奖励 (负值表示惩罚)
    """
    penalty = 0.0
    
    # 位置限制检查
    lower_limits = joint_limits.get('lower', [])
    upper_limits = joint_limits.get('upper', [])
    
    for i, (pos, lower, upper) in enumerate(zip(joint_positions, lower_limits, upper_limits)):
        # 接近限制时给予惩罚
        lower_margin = pos - lower
        upper_margin = upper - pos
        
        # 使用软约束
        soft_margin = 0.1  # 弧度
        if lower_margin < soft_margin:
            penalty += (soft_margin - lower_margin) ** 2
        if upper_margin < soft_margin:
            penalty += (soft_margin - upper_margin) ** 2
    
    # 速度限制检查
    velocity_limits = joint_limits.get('velocity', [])
    for i, (vel, limit) in enumerate(zip(joint_velocities, velocity_limits)):
        if abs(vel) > limit * 0.8:  # 超过 80% 限制
            penalty += (abs(vel) - limit * 0.8) ** 2
    
    return -penalty


def check_com_in_support_polygon(
    com: np.ndarray,
    support_points: List[np.ndarray]
) -> bool:
    """
    检查重心投影是否在支撑多边形内
    
    使用射线法判断点是否在多边形内
    
    Args:
        com: 重心位置 [x, y, z]
        support_points: 支撑点列表
        
    Returns:
        inside: 是否在支撑多边形内
    """
    if len(support_points) < 3:
        return False
    
    # 提取 2D 投影
    x, y = com[0], com[1]
    polygon = [(p[0], p[1]) for p in support_points]
    
    # 射线法
    n = len(polygon)
    inside = False
    
    p1x, p1y = polygon[0]
    for i in range(n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    
    return inside


def compute_falling_style_reward(
    body_contacts: Dict[str, bool],
    contact_sequence: List[str],
    target_sequence: List[str] = None
) -> float:
    """
    计算跌倒风格奖励
    
    鼓励特定的跌倒模式，如:
    - 先用手臂着地缓冲
    - 侧身倒地保护头部
    
    Args:
        body_contacts: 身体部位接触状态
        contact_sequence: 接触顺序
        target_sequence: 目标接触顺序
        
    Returns:
        reward: 风格奖励
    """
    if target_sequence is None:
        # 理想顺序: 手/脚 -> 手臂/腿 -> 躯干 (避免头部先着地)
        target_sequence = ['hand_l', 'hand_r', 'foot_l', 'foot_r', 'torso']
    
    reward = 0.0
    
    # 检查头部是否先着地 (应该避免)
    if 'head' in contact_sequence:
        head_index = contact_sequence.index('head')
        # 如果头部是第一个或第二个着地的，给予惩罚
        if head_index < 2:
            reward -= 1.0
    
    # 奖励手臂先着地缓冲
    hand_contact_time = float('inf')
    for hand in ['hand_l', 'hand_r']:
        if hand in contact_sequence:
            hand_contact_time = min(hand_contact_time, contact_sequence.index(hand))
    
    if hand_contact_time < 3:  # 手臂较早着地
        reward += 0.5
    
    return reward
