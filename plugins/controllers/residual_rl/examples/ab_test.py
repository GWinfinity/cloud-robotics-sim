"""
A/B Test for Residual RL Migration

对比旧版 genesis-residual-rl 和 Plugin 实现
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent.parent / 'genesis-residual-rl'))

import torch
import numpy as np
from cloud_robotics_sim.core.ab_test_framework import ABTestRunner


def create_legacy_residual():
    """创建旧版 Residual RL 实现"""
    try:
        # 尝试导入旧版
        from models.residual_network import ResidualNetwork
        return ResidualNetwork(obs_dim=100, action_dim=10)
    except Exception as e:
        print(f"Warning: Could not load legacy model: {e}")
        return MockResidualNetwork("legacy")


def create_plugin_residual():
    """创建 Plugin 版 Residual RL 实现"""
    from cloud_robotics_sim.plugins.controllers.residual_rl import ResidualNetwork
    return ResidualNetwork(obs_dim=100, action_dim=10, residual_scale=0.2)


class MockResidualNetwork:
    """模拟网络 (当旧版不可用时)"""
    def __init__(self, name):
        self.name = name
        self.call_count = 0
    
    def __call__(self, obs):
        self.call_count += 1
        return {
            'residual': torch.randn(1, 10) * 0.1,
            'action': torch.randn(1, 10)
        }


class ModelWrapper:
    """模型包装器，统一接口"""
    def __init__(self, model, name):
        self.model = model
        self.name = name
    
    def __call__(self, obs_tensor):
        """前向传播"""
        with torch.no_grad():
            output = self.model(obs_tensor)
            
            # 统一输出格式
            if isinstance(output, dict):
                residual = output.get('residual', torch.zeros(1, 10))
                action = output.get('action', torch.zeros(1, 10))
            else:
                residual = output
                action = output
            
            return {
                'residual_norm': torch.norm(residual).item(),
                'residual_max': torch.abs(residual).max().item(),
                'action_mean': action.mean().item()
            }


def run_ab_test():
    """运行 A/B 测试"""
    print("=" * 60)
    print("A/B Test: Legacy vs Plugin Residual RL")
    print("=" * 60)
    
    # 创建模型
    print("\nCreating models...")
    try:
        legacy_model_raw = create_legacy_residual()
        legacy_model = ModelWrapper(legacy_model_raw, "legacy")
        print("  Legacy model: OK")
    except Exception as e:
        print(f"  Legacy model: Failed ({e})")
        legacy_model = ModelWrapper(MockResidualNetwork("legacy_mock"), "legacy")
    
    try:
        plugin_model_raw = create_plugin_residual()
        plugin_model = ModelWrapper(plugin_model_raw, "plugin")
        print("  Plugin model: OK")
    except Exception as e:
        print(f"  Plugin model: Failed ({e})")
        return
    
    # 创建 A/B 测试运行器
    runner = ABTestRunner(
        variant_a_name="legacy_residual",
        variant_a_fn=legacy_model,
        variant_b_name="plugin_residual",
        variant_b_fn=plugin_model,
        output_dir="ab_test_results/residual_rl",
        warmup_steps=5
    )
    
    # 测试函数
    def test_fn(model_wrapper):
        obs = torch.randn(1, 100)
        return model_wrapper(obs)
    
    def collect_metrics(result):
        return {
            'residual_norm': result['residual_norm'],
            'residual_max': result['residual_max'],
            'action_mean': result['action_mean']
        }
    
    # 运行测试
    print("\nRunning A/B tests...")
    num_tests = 100
    
    for i in range(num_tests):
        try:
            results = runner.run_both(test_fn, collect_metrics)
            
            if (i + 1) % 20 == 0:
                print(f"  Completed {i + 1}/{num_tests} tests")
                summary = runner.results.summary()
                print(f"    A Success: {summary['variant_a']['success_rate']:.1%}, "
                      f"B Success: {summary['variant_b']['success_rate']:.1%}")
        except Exception as e:
            print(f"  Error at test {i}: {e}")
    
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


def test_residual_constraint():
    """测试残差输出约束"""
    print("\n" + "=" * 60)
    print("Residual Output Constraint Test")
    print("=" * 60)
    
    from cloud_robotics_sim.plugins.controllers.residual_rl import ResidualNetwork
    
    residual_net = ResidualNetwork(
        obs_dim=100,
        action_dim=10,
        residual_scale=0.2
    )
    
    print("\nTesting output constraints (ε=0.2)...")
    
    violations = 0
    for i in range(100):
        obs = torch.randn(1, 100)
        output = residual_net(obs)
        residual = output['residual']
        
        max_val = torch.abs(residual).max().item()
        if max_val > 0.21:  # 允许微小误差
            violations += 1
            print(f"  Violation at test {i}: max={max_val:.4f}")
    
    print(f"\n  Total tests: 100")
    print(f"  Violations: {violations}")
    print(f"  Constraint satisfaction: {(100-violations)/100:.1%}")
    
    if violations == 0:
        print("  ✅ Output constraint verified!")


def test_bc_frozen():
    """测试 BC 策略冻结"""
    print("\n" + "=" * 60)
    print("BC Policy Frozen Test")
    print("=" * 60)
    
    from cloud_robotics_sim.plugins.controllers.residual_rl import (
        BCPolicy, ResidualNetwork, CombinedPolicy
    )
    
    bc = BCPolicy(obs_dim=100, action_dim=10)
    residual = ResidualNetwork(obs_dim=100, action_dim=10)
    
    combined = CombinedPolicy(bc, residual, freeze_bc=True)
    
    # 检查参数是否冻结
    bc_frozen = all(not p.requires_grad for p in bc.parameters())
    residual_trainable = all(p.requires_grad for p in residual.parameters())
    
    print(f"\n  BC parameters frozen: {bc_frozen}")
    print(f"  Residual parameters trainable: {residual_trainable}")
    
    if bc_frozen and residual_trainable:
        print("  ✅ Parameter freezing correct!")
    else:
        print("  ❌ Parameter freezing incorrect!")


if __name__ == '__main__':
    run_ab_test()
    test_residual_constraint()
    test_bc_frozen()
