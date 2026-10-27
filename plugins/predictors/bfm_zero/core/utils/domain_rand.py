"""Domain Randomization Utilities

物理参数随机化委托 ``cloud_robotics_sim.core.domain_rand``（seeded、向量化、
经 genesis 公开 setter 写入）;观测/动作噪声保留本类实现。
"""

from __future__ import annotations

import numpy as np

from cloud_robotics_sim.core.domain_rand import (
    DomainRandomizer as _CoreRandomizer,
)
from cloud_robotics_sim.core.domain_rand import ParamSpec


class DomainRandomization:
    """域随机化管理器"""

    def __init__(self, config: dict):
        self.config = config
        self.enabled = config.get('enabled', False)
        self._core = None

    def apply_to_simulation(self, scene, robot):
        """应用随机化到仿真

        质量/摩擦经核心 DomainRandomizer 写入 genesis 实体（场景需以
        ``batch_links_info=True, batch_dofs_info=True`` 构建）;首次调用时
        以机器人为基线惰性绑定。
        """
        if not self.enabled:
            return

        if self._core is None:
            specs = []
            if 'mass_scale' in self.config:
                specs.append(
                    ParamSpec("mass", "mass", *self.config['mass_scale'])
                )
            if 'friction_scale' in self.config:
                specs.append(
                    ParamSpec(
                        "friction", "friction_ratio", *self.config['friction_scale'],
                        scale="absolute",
                    )
                )
            if specs:
                self._core = _CoreRandomizer(robot, specs)
        if self._core is not None:
            self._core.sample()

        if 'gravity_perturbation' in self.config:
            # 重力扰动保留为参数输出（genesis 重力为场景级,不随 env 随机化）
            np.random.randn(3) * self.config['gravity_perturbation']

    def add_observation_noise(self, obs: np.ndarray) -> np.ndarray:
        """添加观测噪声"""
        if not self.enabled:
            return obs

        noise_scale = self.config.get('observation_noise', 0.0)
        noise = np.random.randn(*obs.shape) * noise_scale
        return obs + noise

    def add_action_noise(self, action: np.ndarray) -> np.ndarray:
        """添加动作噪声"""
        if not self.enabled:
            return action

        noise_scale = self.config.get('action_noise', 0.0)
        noise = np.random.randn(*action.shape) * noise_scale
        return np.clip(action + noise, -1, 1)
