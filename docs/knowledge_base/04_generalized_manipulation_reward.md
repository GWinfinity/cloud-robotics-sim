# 通用操作奖励函数 (Generalized Manipulation Reward)

## 1. 算法原理

### 1.1 论文引用

**核心论文**: "Sim-to-Real Reinforcement Learning for Vision-Based Dexterous Manipulation on Humanoids" (2024)

**相关论文**:
- "Learning Dexterous Manipulation from Exemplar Videos" (Hand-Object Interaction)
- "Generalized Robot Manipulation through Contact-Aware Rewards"

### 1.2 核心思想

#### 问题背景
传统操作任务奖励设计的问题：
- **任务特定**: 每个任务需要单独设计奖励函数
- **调参困难**: 需要大量试错调整权重
- **迁移性差**: 在新任务上需要重新设计

#### 通用奖励公式

```
通用奖励 = 接触奖励 + 物体目标奖励 + 手部目标奖励 + 正则化项

R = w_contact * R_contact + w_object * R_object + w_hand * R_hand + R_regularization
```

**设计原则**:
1. **接触中心**: 鼓励稳定的多点接触
2. **目标导向**: 基于物体目标状态的稀疏奖励
3. **手部引导**: 引导手部到达合适位置
4. **物理可行**: 考虑能量、平滑性等约束

### 1.3 奖励组件详解

#### 1. 接触奖励 (Contact Reward)

```python
class ContactReward:
    """
    接触奖励设计
    
    核心思想: 鼓励手掌和手指与物体接触
    """
    
    def compute(self, state):
        reward = 0.0
        
        # 手掌接触奖励
        for hand in ['left', 'right']:
            palm_force = state[f'{hand}_palm_contact_force']
            if palm_force > threshold_palm:
                reward += 0.5  # 手掌接触基础奖励
            
            # 手指接触奖励
            finger_forces = state[f'{hand}_finger_contact_forces']
            for force in finger_forces:
                if force > threshold_finger:
                    reward += 0.1  # 每个手指接触奖励
            
            # 稳定抓取奖励 (多手指同时接触)
            num_touching = sum(f > threshold_finger for f in finger_forces)
            if num_touching >= 3:  # 至少3个手指
                reward += stable_grasp_bonus
        
        return reward
```

#### 2. 物体目标奖励 (Object Goal Reward)

```
任务类型:
├── Grasp-and-Reach: 抓取物体并移动到目标位置
├── Box Lift: 提升箱子到指定高度
├── Bimanual Handover: 双手交接物体
└── In-Hand Rotation: 手中旋转物体

通用接口:
R_object = task.compute_object_reward(state, goal)
```

#### 3. 手部目标奖励 (Hand Goal Reward)

```python
def compute_hand_goal_reward(hand_pos, object_pos, target_offset=None):
    """
    手部目标奖励
    
    鼓励手部到达物体周围合适位置
    """
    # 手到物体的距离
    distance = np.linalg.norm(hand_pos - object_pos)
    
    # 接近奖励 (指数衰减)
    approach_reward = np.exp(-distance * k_approach)
    
    # 到达奖励 (稀疏)
    if distance < position_threshold:
        reach_reward = reach_bonus
    else:
        reach_reward = 0.0
    
    return approach_reward + reach_reward
```

### 1.4 架构设计

```
┌─────────────────────────────────────────────────────────────┐
│                  通用奖励函数系统                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  传感器输入                                                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ - 接触力传感器 (手掌、手指)                           │   │
│  │ - 物体位姿 (位置、姿态)                               │   │
│  │ - 手部位姿                                            │   │
│  │ - 关节状态                                            │   │
│  └─────────────────────────────────────────────────────┘   │
│                         ↓                                   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ 接触奖励计算                                          │   │
│  │ - 手掌接触检测                                        │   │
│  │ - 手指接触计数                                        │   │
│  │ - 稳定抓取判断                                        │   │
│  └─────────────────────────────────────────────────────┘   │
│                         ↓                                   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ 任务特定奖励 (可配置)                                  │   │
│  │ - Grasp-and-Reach: 物体到目标距离                     │   │
│  │ - Box Lift: 提升高度                                  │   │
│  │ - Handover: 交接点到达                                │   │
│  └─────────────────────────────────────────────────────┘   │
│                         ↓                                   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ 手部引导奖励                                          │   │
│  │ - 手到物体距离                                        │   │
│  │ - 手到目标位置                                        │   │
│  └─────────────────────────────────────────────────────┘   │
│                         ↓                                   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ 正则化项                                              │   │
│  │ - 能量消耗                                            │   │
│  │ - 动作平滑性                                          │   │
│  │ - 关节极限                                            │   │
│  └─────────────────────────────────────────────────────┘   │
│                         ↓                                   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ 加权求和 → 最终奖励                                   │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. 使用场景

### 2.1 适用任务

| 任务类型 | 适用性 | 说明 |
|---------|-------|------|
| **抓取放置** | ⭐⭐⭐⭐⭐ | 最基础应用 |
| **双手交接** | ⭐⭐⭐⭐⭐ | 需要协调双手接触 |
| **箱体提升** | ⭐⭐⭐⭐ | 需要保持物体稳定 |
| **插销装配** | ⭐⭐⭐⭐ | 需要精确接触 |
| **开门抽屉** | ⭐⭐⭐ | 需要考虑 articulation |
| **柔性物体** | ⭐⭐ | 需要变形建模 |

### 2.2 任务配置示例

```python
# Grasp-and-Reach 配置
grasp_reach_config = {
    'task_type': 'grasp_and_reach',
    'success_threshold': 0.05,  # 5cm 精度
    'target_position': [0.5, 0.0, 0.5],
    'rewards': {
        'contact': {'weight': 2.0},
        'object_goal': {'weight': 3.0},
        'hand_goal': {'weight': 1.5}
    }
}

# Box Lift 配置
box_lift_config = {
    'task_type': 'box_lift',
    'lift_height_threshold': 0.3,  # 提升 30cm
    'stability_threshold': 0.1,    # 速度阈值
    'rewards': {
        'contact': {'weight': 2.0},
        'object_goal': {'weight': 3.0},
        'hand_goal': {'weight': 1.0}
    }
}

# Bimanual Handover 配置
handover_config = {
    'task_type': 'bimanual_handover',
    'handover_position': [0.0, 0.0, 0.8],
    'release_threshold': 0.5,      # 释放接触力阈值
    'rewards': {
        'contact': {'weight': 2.0},
        'object_goal': {'weight': 3.0},
        'hand_goal': {'weight': 2.0}
    }
}
```

### 2.3 限制与注意事项

#### 接触检测精度
- 需要精确的接触力传感器或碰撞检测
- 接触力阈值需要根据任务调整
- 传感器噪声会影响奖励稳定性

#### 物体建模要求
- 需要准确的物体质量和惯性参数
- 摩擦系数影响抓取稳定性
- 对于未知物体需要在线估计

#### 奖励稀疏性
- 早期训练可能难以获得正奖励
- 建议结合课程学习或 shaping
- 可能需要预训练接触策略

### 2.4 与其他奖励设计对比

| 方法 | 设计难度 | 迁移性 | 最终性能 | 适用场景 |
|-----|---------|-------|---------|---------|
| **通用奖励** | 中 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 多种操作任务 |
| **特定奖励** | 高 | ⭐ | ⭐⭐⭐⭐⭐ | 单一复杂任务 |
| **稀疏奖励** | 低 | ⭐⭐⭐ | ⭐⭐ | 简单任务 |
| **演示奖励** | 低 | ⭐⭐ | ⭐⭐⭐ | 有专家演示 |

---

## 3. 代码示例

### 3.1 最小可运行示例

```python
import numpy as np
import genesis as gs
from typing import Dict, Callable

# ==================== 通用奖励函数 ====================

class GeneralizedRewardFunction:
    """
    通用操作奖励函数
    
    适用于多种操作任务的核心奖励计算
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.task_type = config['task_type']
        
        # 奖励权重
        self.contact_weight = config.get('contact', {}).get('weight', 2.0)
        self.object_goal_weight = config.get('object_goal', {}).get('weight', 3.0)
        self.hand_goal_weight = config.get('hand_goal', {}).get('weight', 1.5)
        
        # 正则化权重
        self.energy_penalty = config.get('energy_penalty', -0.001)
        self.action_smoothness = config.get('action_smoothness', -0.01)
        self.joint_limit_penalty = config.get('joint_limit_penalty', -0.1)
        
        # 任务特定配置
        self.task_reward_fn = self._get_task_reward_fn()
    
    def _get_task_reward_fn(self) -> Callable:
        """获取任务特定的奖励函数"""
        task_fns = {
            'grasp_and_reach': self._compute_grasp_reach_reward,
            'box_lift': self._compute_box_lift_reward,
            'bimanual_handover': self._compute_handover_reward
        }
        return task_fns.get(self.task_type)
    
    def compute_reward(
        self,
        state: Dict,
        action: np.ndarray,
        last_action: np.ndarray = None
    ) -> float:
        """
        计算通用奖励
        
        Args:
            state: 环境状态字典
            action: 当前动作
            last_action: 上一帧动作 (用于平滑性计算)
        
        Returns:
            标量奖励值
        """
        reward = 0.0
        
        # 1. 接触奖励
        contact_reward = self._compute_contact_reward(state)
        reward += self.contact_weight * contact_reward
        
        # 2. 物体目标奖励 (任务特定)
        object_reward = self.task_reward_fn(state)
        reward += self.object_goal_weight * object_reward
        
        # 3. 手部目标奖励
        hand_reward = self._compute_hand_goal_reward(state)
        reward += self.hand_goal_weight * hand_reward
        
        # 4. 正则化项
        reward += self.energy_penalty * np.sum(action ** 2)
        
        if last_action is not None:
            action_diff = action - last_action
            reward += self.action_smoothness * np.sum(action_diff ** 2)
        
        return reward
    
    def _compute_contact_reward(self, state: Dict) -> float:
        """
        接触奖励
        
        鼓励手掌和手指与物体接触
        """
        reward = 0.0
        
        palm_threshold = self.config.get('contact', {}).get('palm_threshold', 1.0)
        finger_threshold = self.config.get('contact', {}).get('finger_threshold', 0.5)
        stable_bonus = self.config.get('contact', {}).get('stable_bonus', 1.0)
        
        for hand in ['left', 'right']:
            # 手掌接触
            palm_force = state.get(f'{hand}_palm_contact_force', 0.0)
            if palm_force > palm_threshold:
                reward += 0.5
            
            # 手指接触
            finger_forces = state.get(f'{hand}_finger_contact_forces', [0.0] * 5)
            num_touching = 0
            for force in finger_forces:
                if force > finger_threshold:
                    reward += 0.1
                    num_touching += 1
            
            # 稳定抓取奖励 (至少3个手指)
            if num_touching >= 3:
                reward += stable_bonus
        
        return reward
    
    def _compute_grasp_reach_reward(self, state: Dict) -> float:
        """Grasp-and-Reach 任务奖励"""
        obj_pos = state.get('object_position', np.zeros(3))
        target_pos = state.get('target_position', np.zeros(3))
        
        # 物体到目标的距离
        distance = np.linalg.norm(obj_pos - target_pos)
        
        # 距离奖励 (指数衰减)
        reward = np.exp(-distance * 5)
        
        # 成功奖励
        threshold = self.config.get('success_threshold', 0.05)
        if distance < threshold:
            reward += 10.0
        
        return reward
    
    def _compute_box_lift_reward(self, state: Dict) -> float:
        """Box Lift 任务奖励"""
        obj_pos = state.get('object_position', np.zeros(3))
        obj_vel = state.get('object_velocity', np.zeros(3))
        obj_height = obj_pos[2]
        
        threshold = self.config.get('lift_height_threshold', 0.3)
        
        # 提升高度奖励 (归一化)
        reward = min(obj_height / threshold, 1.0)
        
        # 成功奖励
        if obj_height > threshold:
            reward += 5.0
            
            # 稳定性奖励 (速度小)
            if np.linalg.norm(obj_vel) < 0.1:
                reward += 5.0
        
        return reward
    
    def _compute_handover_reward(self, state: Dict) -> float:
        """Bimanual Handover 任务奖励"""
        reward = 0.0
        
        obj_pos = state.get('object_position', np.zeros(3))
        handover_pos = np.array(self.config.get('handover_position', [0.0, 0.0, 0.8]))
        
        # 物体接近交接点
        distance = np.linalg.norm(obj_pos - handover_pos)
        reward += np.exp(-distance * 3)
        
        # 左手释放
        left_contact = state.get('left_palm_contact_force', 0.0)
        if left_contact < 0.5:
            reward += 2.0
        
        # 右手抓取
        right_contact = state.get('right_palm_contact_force', 0.0)
        if right_contact > 1.0:
            reward += 2.0
        
        # 物体稳定
        obj_vel = state.get('object_velocity', np.zeros(3))
        if np.linalg.norm(obj_vel) < 0.1 and right_contact > 1.0:
            reward += 5.0
        
        return reward
    
    def _compute_hand_goal_reward(self, state: Dict) -> float:
        """手部目标奖励"""
        reward = 0.0
        
        obj_pos = state.get('object_position', np.zeros(3))
        position_threshold = self.config.get('hand_goal_threshold', 0.05)
        
        for hand in ['left', 'right']:
            hand_pos = state.get(f'{hand}_hand_position', np.zeros(3))
            distance = np.linalg.norm(hand_pos - obj_pos)
            
            # 接近奖励
            reward += np.exp(-distance * 3)
            
            # 到达奖励
            if distance < position_threshold:
                reward += 1.0
        
        return reward


# ==================== Genesis 环境集成 ====================

class ManipulationEnv:
    """
    使用通用奖励的操作环境
    """
    
    def __init__(self, task_config):
        # 初始化 Genesis
        gs.init(backend=gs.cuda)
        
        self.scene = gs.Scene()
        
        # 添加机器人和物体
        self.robot = self.scene.add_entity(
            gs.morphs.MJCF(file='xml/franka/panda.xml')
        )
        self.object = self.scene.add_entity(
            gs.morphs.Box(size=(0.05, 0.05, 0.05))
        )
        
        self.scene.build()
        
        # 初始化奖励函数
        self.reward_fn = GeneralizedRewardFunction(task_config)
        
        self.last_action = None
    
    def get_state(self) -> Dict:
        """获取环境状态"""
        # 获取接触力 (需要传感器)
        left_palm_force = self.get_contact_force('left_palm')
        left_finger_forces = [self.get_contact_force(f'left_finger_{i}') for i in range(5)]
        
        right_palm_force = self.get_contact_force('right_palm')
        right_finger_forces = [self.get_contact_force(f'right_finger_{i}') for i in range(5)]
        
        # 获取位姿
        object_pos = self.object.get_pos().cpu().numpy()
        left_hand_pos = self.robot.get_link('left_hand').get_pos().cpu().numpy()
        right_hand_pos = self.robot.get_link('right_hand').get_pos().cpu().numpy()
        
        return {
            'left_palm_contact_force': left_palm_force,
            'left_finger_contact_forces': left_finger_forces,
            'right_palm_contact_force': right_palm_force,
            'right_finger_contact_forces': right_finger_forces,
            'object_position': object_pos,
            'object_velocity': self.object.get_vel().cpu().numpy(),
            'left_hand_position': left_hand_pos,
            'right_hand_position': right_hand_pos,
            'target_position': self.target_position
        }
    
    def step(self, action):
        """环境步进"""
        # 应用动作
        self.robot.control_dofs_position(action)
        self.scene.step()
        
        # 获取状态和奖励
        state = self.get_state()
        reward = self.reward_fn.compute_reward(
            state, action, self.last_action
        )
        
        self.last_action = action.copy()
        
        # 检查成功
        done = self.check_success(state)
        
        return state, reward, done, {}


# ==================== 多任务训练示例 ====================

def train_multi_task():
    """多任务训练示例"""
    
    tasks = [
        {'task_type': 'grasp_and_reach', 'target_position': [0.5, 0.0, 0.5]},
        {'task_type': 'box_lift', 'lift_height_threshold': 0.3},
        {'task_type': 'bimanual_handover', 'handover_position': [0.0, 0.0, 0.8]}
    ]
    
    # 使用相同的网络架构，不同的奖励函数
    policy = PPO(obs_dim=100, action_dim=14)
    
    for task_config in tasks:
        # 为每个任务创建环境
        env = ManipulationEnv(task_config)
        
        # 训练
        for episode in range(1000):
            obs = env.reset()
            episode_reward = 0
            
            for step in range(200):
                action = policy.select_action(obs)
                obs, reward, done, _ = env.step(action)
                episode_reward += reward
                
                if done:
                    break
            
            print(f"Task: {task_config['task_type']}, "
                  f"Episode: {episode}, Reward: {episode_reward:.2f}")
```

---

## 4. 超参数指南

### 4.1 奖励权重配置

| 参数 | 推荐值 | 说明 |
|-----|-------|------|
| `contact_weight` | 2.0 | 接触奖励权重 |
| `object_goal_weight` | 3.0 | 物体目标权重 (通常最高) |
| `hand_goal_weight` | 1.5 | 手部引导权重 |
| `energy_penalty` | -0.001 | 能量惩罚系数 |
| `action_smoothness` | -0.01 | 动作平滑性系数 |

### 4.2 接触阈值

```python
# 接触力阈值 (需要根据传感器调整)
contact_thresholds = {
    'palm_threshold': 1.0,      # 手掌接触阈值 (牛顿)
    'finger_threshold': 0.5,    # 手指接触阈值 (牛顿)
    'stable_grasp_threshold': 3  # 稳定抓取所需手指数
}

# 距离阈值
distance_thresholds = {
    'success_distance': 0.05,    # 任务成功距离 (5cm)
    'hand_goal_distance': 0.05,  # 手部到达距离
    'approach_decay': 3.0        # 接近奖励衰减系数
}
```

### 4.3 任务特定参数

| 任务 | 关键参数 | 推荐值 |
|-----|---------|-------|
| **Grasp-and-Reach** | `success_threshold` | 0.05 m |
| **Box Lift** | `lift_height_threshold` | 0.3 m |
| | `stability_threshold` | 0.1 m/s |
| **Handover** | `release_threshold` | 0.5 N |
| | `grasp_threshold` | 1.0 N |

### 4.4 训练技巧

```python
# 课程学习配置
curriculum = {
    'stage1': {
        'contact_weight': 5.0,  # 早期侧重接触学习
        'object_goal_weight': 1.0
    },
    'stage2': {
        'contact_weight': 2.0,  # 后期侧重任务完成
        'object_goal_weight': 3.0
    }
}

# 奖励缩放
reward_scale = 0.1  # 防止Q值爆炸
```

### 4.5 故障排查

| 现象 | 可能原因 | 解决方案 |
|-----|---------|---------|
| **从不接触物体** | 接触奖励权重太低或手部引导不够 | 增大 contact_weight 和 hand_goal_weight |
| **接触但不移动** | 物体目标奖励未激活 | 检查物体状态计算，增大 object_goal_weight |
| **动作抖动** | 平滑性惩罚不够 | 增大 action_smoothness 惩罚 |
| **能量消耗过大** | 能量惩罚不够 | 增大 energy_penalty |
| **训练不稳定** | 奖励值太大 | 使用 reward_scale 缩放 |

---

## 5. 进阶技巧

### 5.1 自适应权重
```python
class AdaptiveRewardWeights:
    """根据训练进度自适应调整权重"""
    
    def update_weights(self, episode, success_rate):
        if success_rate < 0.3:
            # 早期: 侧重接触
            self.contact_weight = 5.0
            self.object_goal_weight = 1.0
        elif success_rate < 0.7:
            # 中期: 平衡
            self.contact_weight = 2.0
            self.object_goal_weight = 2.0
        else:
            # 后期: 侧重任务
            self.contact_weight = 1.0
            self.object_goal_weight = 4.0
```

### 5.2 多物体支持
```python
class MultiObjectReward:
    """支持多物体的奖励计算"""
    
    def compute(self, state):
        total_reward = 0.0
        
        for obj_id in self.object_ids:
            obj_state = self.extract_object_state(state, obj_id)
            obj_reward = self.compute_object_reward(obj_state)
            total_reward += obj_reward * self.object_weights[obj_id]
        
        return total_reward
```

### 5.3 奖励塑形
```python
def potential_based_shaping(state, next_state, gamma):
    """
    基于势能的奖励塑形
    保证最优策略不变
    """
    potential = -np.linalg.norm(state['object_position'] - state['target_position'])
    next_potential = -np.linalg.norm(next_state['object_position'] - next_state['target_position'])
    
    shaping = gamma * next_potential - potential
    return shaping
```
