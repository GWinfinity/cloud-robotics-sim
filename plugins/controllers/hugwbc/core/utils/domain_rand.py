"""
Domain Randomization for HugWBC

域随机化实现，提高策略泛化能力。

物理参数采样与写入统一委托 ``cloud_robotics_sim.core.domain_rand``
（seeded、向量化、经 genesis 公开 setter 逐 env 写入）；本类保留
HugWBC 既有的 numpy 参数数组 API（``params`` / ``get_params``）。
"""
from __future__ import annotations

import numpy as np

from cloud_robotics_sim.core.domain_rand import (
    DomainRandomizer as _CoreRandomizer,
)
from cloud_robotics_sim.core.domain_rand import ParamSpec, draw_specs


class DomainRandomizer:
    """
    域随机化器

    对物理参数进行随机化，提高策略鲁棒性。质量/摩擦/电机强度/关节阻尼
    的采样委托核心实现；质心位移无 genesis 公开 setter 映射，仅保留为
    参数数组供上层消费。
    """

    def __init__(self, config: dict, num_envs: int):
        self.config = config
        self.num_envs = num_envs

        # 随机化范围
        self.friction_range = config.get('friction_range', [0.5, 1.5])
        self.mass_scale_range = config.get('mass_scale_range', [0.9, 1.1])
        self.com_displacement_range = config.get('com_displacement_range', [-0.05, 0.05])
        self.motor_strength_range = config.get('motor_strength_range', [0.9, 1.1])
        self.joint_damping_range = config.get('joint_damping_range', [0.9, 1.1])

        # 核心采样规格（不绑定实体,纯采样走 draw_specs）
        self._param_specs = [
            ParamSpec("friction", "friction_ratio", *self.friction_range, scale="absolute"),
            ParamSpec("mass_scale", "mass", *self.mass_scale_range),
            ParamSpec("motor_strength", "kp", *self.motor_strength_range),
            ParamSpec("joint_damping", "damping", *self.joint_damping_range, scale="absolute"),
        ]

        # 当前随机化参数
        self.params = self._generate_params()

    def _generate_params(self) -> dict:
        """生成随机参数（核心 draw_specs,基线语义不变）"""
        sampled = draw_specs(self._param_specs, self.num_envs)
        com_lo, com_hi = self.com_displacement_range
        rng = np.random.default_rng()
        return {
            'friction': sampled['friction'].reshape(-1),
            'mass_scale': sampled['mass_scale'].reshape(-1),
            'com_displacement': rng.uniform(
                com_lo, com_hi, (self.num_envs, 3)
            ),
            'motor_strength': sampled['motor_strength'].reshape(-1),
            'joint_damping': sampled['joint_damping'].reshape(-1),
        }

    def randomize(self, env_ids=None):
        """
        重新随机化参数

        Args:
            env_ids: 需要随机化的环境索引，None 表示全部
        """
        fresh = self._generate_params()
        if env_ids is None:
            env_ids = range(self.num_envs)
        for key, values in fresh.items():
            self.params[key][list(env_ids)] = values[list(env_ids)]

    def get_params(self, env_idx: int) -> dict:
        """获取指定环境的参数"""
        return {k: v[env_idx] for k, v in self.params.items()}

    def apply_to_sim(self, scene, robot, env_ids=None):
        """
        将随机化参数应用到仿真

        首次调用时以已构建的机器人为基线绑定核心随机化器（需要场景以
        ``batch_links_info=True, batch_dofs_info=True`` 构建）,随后按
        ``env_ids`` 逐 env 写入 mass / friction_ratio / 电机 kp·kv / 阻尼。

        Args:
            scene: gs.Scene（保留以兼容旧调用,当前实现经 robot 写入）
            robot: 已构建的 RigidEntity
            env_ids: 需要写入的环境索引,None 表示全部
        """
        if not hasattr(self, "_core") or self._core is None:
            motor_dof_idx = tuple(
                joint.dof_start for joint in robot.joints[1:]
            )
            kp_lo, kp_hi = self.motor_strength_range
            specs = [
                ParamSpec(
                    "friction", "friction_ratio", *self.friction_range,
                    scale="absolute",
                ),
                ParamSpec("mass_scale", "mass", *self.mass_scale_range),
                ParamSpec(
                    "motor_kp", "kp", kp_lo, kp_hi, indices=motor_dof_idx,
                ),
                ParamSpec(
                    "motor_kv", "kv", kp_lo, kp_hi, indices=motor_dof_idx,
                ),
                ParamSpec(
                    "joint_damping", "damping", *self.joint_damping_range,
                    scale="absolute",
                ),
            ]
            self._core = _CoreRandomizer(
                robot, specs, seed=self.config.get('seed')
            )
        return self._core.sample(env_ids)


class Curriculum:
    """
    课程学习管理器
    
    逐渐增加任务难度
    """
    
    def __init__(self, config: dict):
        self.config = config
        
        # 难度级别
        self.levels = config.get('levels', [
            {'name': 'easy', 'max_force': 50, 'command_range': 0.5},
            {'name': 'medium', 'max_force': 100, 'command_range': 1.0},
            {'name': 'hard', 'max_force': 150, 'command_range': 1.5},
        ])
        
        self.current_level = 0
        self.success_threshold = config.get('success_threshold', 0.8)
        self.success_history = []
        self.window_size = config.get('window_size', 100)
    
    def get_current_params(self) -> dict:
        """获取当前难度参数"""
        return self.levels[self.current_level]
    
    def update(self, success_rate: float):
        """
        更新课程进度
        
        Args:
            success_rate: 最近的成功率
        """
        self.success_history.append(success_rate)
        if len(self.success_history) > self.window_size:
            self.success_history.pop(0)
        
        # 检查是否升级
        if len(self.success_history) >= self.window_size:
            avg_success = np.mean(self.success_history[-self.window_size:])
            if avg_success > self.success_threshold and self.current_level < len(self.levels) - 1:
                self.current_level += 1
                print(f"Curriculum: Level up to {self.levels[self.current_level]['name']}")
                self.success_history = []
    
    def get_command_range_scale(self) -> float:
        """获取当前命令范围缩放因子"""
        return self.levels[self.current_level]['command_range']
    
    def reset(self):
        """重置课程"""
        self.current_level = 0
        self.success_history = []
