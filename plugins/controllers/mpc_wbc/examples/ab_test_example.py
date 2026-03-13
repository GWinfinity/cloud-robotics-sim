"""
A/B Test Example: MPC-WBC Controller Migration

演示如何使用 A/B 测试框架对比旧版实现和 Plugin 实现
"""

import numpy as np
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from cloud_robotics_sim.core.ab_test_framework import ABTestRunner, GradualMigration


def create_mock_legacy_controller():
    """模拟旧版控制器"""
    class LegacyController:
        def __init__(self):
            self.name = "legacy"
            self.call_count = 0
        
        def __call__(self, state):
            """旧版实现"""
            self.call_count += 1
            
            # 模拟一些处理
            import time
            time.sleep(0.001)  # 1ms 延迟
            
            # 模拟偶尔的失败
            if self.call_count % 50 == 0:
                raise RuntimeError("Legacy occasional error")
            
            return {
                'torque': np.random.randn(19) * 10,
                'contact_forces': np.random.randn(12) * 100
            }
    
    return LegacyController()


def create_mock_plugin_controller():
    """模拟 Plugin 控制器"""
    from cloud_robotics_sim.plugins.controllers.mpc_wbc import MPCWBCController
    
    # 使用真实的 Plugin 控制器
    controller = MPCWBCController(
        num_dofs=19,
        dt=0.005,
        gait_frequency=1.25,
        mass=77.35
    )
    
    def wrapper(state):
        """包装为与旧版相同的接口"""
        result = controller.update(
            base_pos=state['base_pos'],
            base_rpy=state['base_rpy'],
            base_lin_vel=state['base_lin_vel'],
            base_ang_vel=state['base_ang_vel'],
            left_foot_pos=state['left_foot_pos'],
            right_foot_pos=state['right_foot_pos'],
            joint_pos=state['joint_pos'],
            joint_vel=state['joint_vel'],
            target_vel=state['target_vel']
        )
        
        return {
            'torque': result,
            'contact_forces': controller.get_contact_forces()
        }
    
    return wrapper


def run_ab_test():
    """运行 A/B 测试"""
    print("=" * 60)
    print("A/B Test: Legacy vs MPC-WBC Plugin")
    print("=" * 60)
    
    # 创建测试控制器
    legacy = create_mock_legacy_controller()
    plugin = create_mock_plugin_controller()
    
    # 创建 A/B 测试运行器
    runner = ABTestRunner(
        variant_a_name="legacy_implementation",
        variant_a_fn=legacy,
        variant_b_name="mpc_wbc_plugin",
        variant_b_fn=plugin,
        output_dir="ab_test_results/mpc_wbc",
        warmup_steps=5
    )
    
    # 生成测试状态
    def generate_state():
        return {
            'base_pos': np.array([0.0, 0.0, 1.0]),
            'base_rpy': np.array([0.0, 0.0, 0.0]),
            'base_lin_vel': np.array([0.5, 0.0, 0.0]),
            'base_ang_vel': np.array([0.0, 0.0, 0.0]),
            'left_foot_pos': np.array([0.0, 0.1, 0.0]),
            'right_foot_pos': np.array([0.0, -0.1, 0.0]),
            'joint_pos': np.zeros(19),
            'joint_vel': np.zeros(19),
            'target_vel': np.array([1.0, 0.0, 0.0])
        }
    
    # 定义测试函数
    def test_fn(controller_fn):
        state = generate_state()
        return controller_fn(state)
    
    # 定义自定义指标收集
    def collect_metrics(result):
        return {
            'torque_norm': np.linalg.norm(result['torque']),
            'contact_force_norm': np.linalg.norm(result['contact_forces'])
        }
    
    # 运行测试
    print("\nRunning A/B tests...")
    num_tests = 100
    
    for i in range(num_tests):
        results = runner.run_both(test_fn, collect_metrics)
        
        if (i + 1) % 20 == 0:
            print(f"  Completed {i + 1}/{num_tests} tests")
            
            # 实时查看中间结果
            summary = runner.results.summary()
            print(f"    A Success: {summary['variant_a']['success_rate']:.1%}, "
                  f"B Success: {summary['variant_b']['success_rate']:.1%}")
    
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


def run_gradual_migration():
    """演示渐进式迁移"""
    print("\n" + "=" * 60)
    print("Gradual Migration Demo")
    print("=" * 60)
    
    # 创建控制器
    legacy = create_mock_legacy_controller()
    plugin = create_mock_plugin_controller()
    
    # 创建渐进式迁移控制器
    migration = GradualMigration(
        legacy_fn=legacy,
        plugin_fn=plugin,
        initial_plugin_ratio=0.0,  # 开始时全部使用旧版
        min_samples_before_increase=50,
        success_threshold=0.95
    )
    
    # 模拟训练过程
    print("\nPhase 1: 0% Plugin (All Legacy)")
    for episode in range(50):
        fn = migration.select_implementation()
        is_plugin = (fn is plugin)
        
        try:
            state = {
                'base_pos': np.array([0.0, 0.0, 1.0]),
                'base_rpy': np.array([0.0, 0.0, 0.0]),
                'base_lin_vel': np.array([0.5, 0.0, 0.0]),
                'base_ang_vel': np.array([0.0, 0.0, 0.0]),
                'left_foot_pos': np.array([0.0, 0.1, 0.0]),
                'right_foot_pos': np.array([0.0, -0.1, 0.0]),
                'joint_pos': np.zeros(19),
                'joint_vel': np.zeros(19),
                'target_vel': np.array([1.0, 0.0, 0.0])
            }
            fn(state)
            migration.update_metrics(is_plugin, success=True)
        except:
            migration.update_metrics(is_plugin, success=False)
    
    status = migration.get_status()
    print(f"  Plugin samples: {status['plugin_stats']['samples']}")
    print(f"  Plugin success rate: {status['plugin_stats']['success_rate']:.1%}")
    
    # 尝试增加比例
    print("\nAttempting to increase plugin ratio to 10%...")
    success = migration.increase_plugin_ratio(0.1)
    
    if success:
        print("\nPhase 2: 10% Plugin")
        for episode in range(50):
            fn = migration.select_implementation()
            is_plugin = (fn is plugin)
            
            try:
                fn({})
                migration.update_metrics(is_plugin, success=True)
            except:
                migration.update_metrics(is_plugin, success=False)
        
        status = migration.get_status()
        print(f"  Current ratio: {status['plugin_ratio']:.1%}")
        print(f"  Plugin success rate: {status['plugin_stats']['success_rate']:.1%}")
    
    print("\n" + "=" * 60)
    print("Gradual migration allows you to:")
    print("  1. Start with 0% plugin (safe)")
    print("  2. Monitor success rate")
    print("  3. Increase ratio gradually (10% -> 25% -> 50% -> 100%)")
    print("  4. Rollback immediately if issues detected")
    print("=" * 60)


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'migration':
        run_gradual_migration()
    else:
        run_ab_test()
