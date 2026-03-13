"""
A/B Test for Badminton Environment Migration

对比旧版 genesis-humanoid-badminton 和 Plugin 实现
"""

import sys
from pathlib import Path

# 添加路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

# 旧版实现路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent.parent / 'genesis-humanoid-badminton'))

import numpy as np
from cloud_robotics_sim.core.ab_test_framework import ABTestRunner


def create_legacy_env():
    """创建旧版环境"""
    try:
        from envs.badminton_env import BadmintonEnv
        return BadmintonEnv(
            config_path=None,
            num_envs=1,
            curriculum_stage=1
        )
    except Exception as e:
        print(f"Warning: Could not load legacy env: {e}")
        # 返回模拟环境
        return MockBadmintonEnv()


def create_plugin_env():
    """创建 Plugin 环境"""
    from cloud_robotics_sim.plugins.envs.badminton import BadmintonEnv
    return BadmintonEnv(
        config_path=None,
        num_envs=1,
        curriculum_stage=1
    )


class MockBadmintonEnv:
    """模拟环境 (当旧版不可用时)"""
    def __init__(self):
        self.name = "legacy_mock"
        self.obs_dim = 75
        self.action_dim = 12
        self.step_count = 0
    
    def reset(self):
        self.step_count = 0
        return np.random.randn(self.obs_dim)
    
    def step(self, action):
        self.step_count += 1
        obs = np.random.randn(self.obs_dim)
        reward = np.random.rand()
        done = self.step_count > 500
        info = {'hit': np.random.rand() > 0.9}
        return obs, reward, done, info


class EnvWrapper:
    """环境包装器，统一接口"""
    def __init__(self, env, name):
        self.env = env
        self.name = name
        self.total_reward = 0
        self.hits = 0
    
    def __call__(self, action_generator):
        """运行一个回合"""
        obs = self.env.reset()
        self.total_reward = 0
        self.hits = 0
        
        for step in range(1000):
            action = action_generator(obs)
            
            try:
                obs, reward, done, info = self.env.step(action)
                self.total_reward += reward
                
                if info.get('hit', False):
                    self.hits += 1
                
                if done:
                    break
            except Exception as e:
                raise RuntimeError(f"Step failed: {e}")
        
        return {
            'total_reward': self.total_reward,
            'hits': self.hits,
            'episode_length': step + 1
        }


def run_ab_test():
    """运行 A/B 测试"""
    print("=" * 60)
    print("A/B Test: Legacy vs Plugin Badminton Environment")
    print("=" * 60)
    
    # 创建环境
    print("\nCreating environments...")
    try:
        legacy_env_raw = create_legacy_env()
        legacy_env = EnvWrapper(legacy_env_raw, "legacy")
        print("  Legacy env: OK")
    except Exception as e:
        print(f"  Legacy env: Failed ({e})")
        legacy_env = EnvWrapper(MockBadmintonEnv(), "legacy_mock")
    
    try:
        plugin_env_raw = create_plugin_env()
        plugin_env = EnvWrapper(plugin_env_raw, "plugin")
        print("  Plugin env: OK")
    except Exception as e:
        print(f"  Plugin env: Failed ({e})")
        return
    
    # 创建 A/B 测试运行器
    runner = ABTestRunner(
        variant_a_name="legacy_badminton",
        variant_a_fn=legacy_env,
        variant_b_name="plugin_badminton",
        variant_b_fn=plugin_env,
        output_dir="ab_test_results/badminton",
        warmup_steps=5
    )
    
    # 动作生成器 (随机策略)
    def action_generator(obs):
        return np.random.randn(12) * 0.5  # 12 DOF
    
    # 定义测试函数
    def test_fn(env_wrapper):
        return env_wrapper(action_generator)
    
    # 定义自定义指标
    def collect_metrics(result):
        return {
            'total_reward': result['total_reward'],
            'hits': result['hits'],
            'episode_length': result['episode_length']
        }
    
    # 运行测试
    print("\nRunning A/B tests...")
    num_episodes = 100
    
    for i in range(num_episodes):
        try:
            results = runner.run_both(test_fn, collect_metrics)
            
            if (i + 1) % 20 == 0:
                print(f"  Completed {i + 1}/{num_episodes} episodes")
                summary = runner.results.summary()
                print(f"    A Success: {summary['variant_a']['success_rate']:.1%}, "
                      f"B Success: {summary['variant_b']['success_rate']:.1%}")
        except Exception as e:
            print(f"  Error at episode {i}: {e}")
    
    # 生成报告
    print("\n" + "=" * 60)
    print("Test Complete!")
    print("=" * 60)
    
    report = runner.generate_report(detailed=True)
    print(report)
    
    # 保存报告
    report_path = runner.save_report()
    print(f"\nReport saved to: {report_path}")
    
    # 迁移建议
    recommendation = runner.recommend_migration()
    print("\n" + "=" * 60)
    print("Migration Recommendation")
    print("=" * 60)
    print(f"Recommend: {recommendation['recommend']}")
    print(f"Confidence: {recommendation['confidence']:.1%}")
    print(f"Reason: {recommendation['reason']}")
    
    if recommendation['cautions']:
        print("\nCautions:")
        for caution in recommendation['cautions']:
            print(f"  - {caution}")


if __name__ == '__main__':
    run_ab_test()
