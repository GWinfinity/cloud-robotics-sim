"""
A/B Testing Framework for Plugin Migration

用于插件迁移的 A/B 测试框架，支持新旧实现对比测试。
"""

import time
import json
import numpy as np
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass, field
from pathlib import Path
import traceback


@dataclass
class TestMetrics:
    """测试指标"""
    # 性能指标
    latency_ms: float = 0.0           # 执行延迟
    memory_mb: float = 0.0            # 内存使用
    
    # 功能指标
    success: bool = True              # 是否成功
    error_message: str = ""           # 错误信息
    
    # 业务指标 (根据插件类型定制)
    custom_metrics: Dict[str, float] = field(default_factory=dict)
    
    # 时间戳
    timestamp: float = field(default_factory=time.time)


@dataclass
class ABTestResult:
    """A/B 测试结果"""
    variant_a: str                    # A 版本名称 (通常是旧版)
    variant_b: str                    # B 版本名称 (通常是新版 plugin)
    
    metrics_a: List[TestMetrics] = field(default_factory=list)
    metrics_b: List[TestMetrics] = field(default_factory=list)
    
    def summary(self) -> Dict[str, Any]:
        """生成测试摘要"""
        def _avg(metrics_list: List[TestMetrics], key: str):
            values = [getattr(m, key) for m in metrics_list if getattr(m, key) is not None]
            return np.mean(values) if values else 0.0
        
        def _success_rate(metrics_list: List[TestMetrics]):
            if not metrics_list:
                return 0.0
            return sum(1 for m in metrics_list if m.success) / len(metrics_list)
        
        return {
            'variant_a': {
                'name': self.variant_a,
                'samples': len(self.metrics_a),
                'success_rate': _success_rate(self.metrics_a),
                'avg_latency_ms': _avg(self.metrics_a, 'latency_ms'),
            },
            'variant_b': {
                'name': self.variant_b,
                'samples': len(self.metrics_b),
                'success_rate': _success_rate(self.metrics_b),
                'avg_latency_ms': _avg(self.metrics_b, 'latency_ms'),
            },
            'improvement': {
                'success_rate_delta': _success_rate(self.metrics_b) - _success_rate(self.metrics_a),
                'latency_delta_percent': (
                    (_avg(self.metrics_a, 'latency_ms') - _avg(self.metrics_b, 'latency_ms'))
                    / (_avg(self.metrics_a, 'latency_ms') + 1e-6) * 100
                )
            }
        }


class ABTestRunner:
    """
    A/B 测试运行器
    
    支持新旧实现对比测试，自动收集指标，生成报告。
    
    Usage:
        >>> runner = ABTestRunner(
        ...     variant_a_name="legacy",
        ...     variant_a_fn=old_controller,
        ...     variant_b_name="plugin",
        ...     variant_b_fn=new_plugin_controller
        ... )
        >>> 
        >>> for _ in range(100):
        ...     obs = env.reset()
        ...     runner.run_both(lambda fn: fn(obs))
        >>>
        >>> report = runner.generate_report()
        >>> print(report)
    """
    
    def __init__(
        self,
        variant_a_name: str,
        variant_a_fn: Callable,
        variant_b_name: str,
        variant_b_fn: Callable,
        output_dir: Optional[str] = None,
        warmup_steps: int = 10
    ):
        """
        Args:
            variant_a_name: A 版本名称 (旧版)
            variant_a_fn: A 版本函数
            variant_b_name: B 版本名称 (新版 plugin)
            variant_b_fn: B 版本函数
            output_dir: 报告输出目录
            warmup_steps: 预热步数 (不计入统计)
        """
        self.variant_a_name = variant_a_name
        self.variant_a_fn = variant_a_fn
        self.variant_b_name = variant_b_name
        self.variant_b_fn = variant_b_fn
        
        self.output_dir = Path(output_dir) if output_dir else Path("ab_test_results")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.warmup_steps = warmup_steps
        self.step_count = 0
        
        self.results = ABTestResult(
            variant_a=variant_a_name,
            variant_b=variant_b_name
        )
        
        self._a_warmed_up = False
        self._b_warmed_up = False
    
    def run_single(
        self,
        variant: str,  # 'a' or 'b'
        test_fn: Callable,
        collect_custom_metrics: Optional[Callable] = None
    ) -> TestMetrics:
        """
        运行单个测试
        
        Args:
            variant: 'a' 或 'b'
            test_fn: 测试函数，接收 variant_fn 作为参数
            collect_custom_metrics: 可选的自定义指标收集函数
        
        Returns:
            测试指标
        """
        fn = self.variant_a_fn if variant == 'a' else self.variant_b_fn
        
        metrics = TestMetrics()
        
        try:
            # 记录开始时间
            start_time = time.perf_counter()
            
            # 执行测试
            result = test_fn(fn)
            
            # 记录延迟
            metrics.latency_ms = (time.perf_counter() - start_time) * 1000
            
            # 收集自定义指标
            if collect_custom_metrics:
                metrics.custom_metrics = collect_custom_metrics(result)
            
            metrics.success = True
            
        except Exception as e:
            metrics.success = False
            metrics.error_message = str(e)
            if hasattr(e, '__traceback__'):
                metrics.error_message += f"\n{traceback.format_exc()}"
        
        # 记录结果 (如果不是预热阶段)
        self.step_count += 1
        if self.step_count > self.warmup_steps:
            if variant == 'a':
                self.results.metrics_a.append(metrics)
            else:
                self.results.metrics_b.append(metrics)
        
        return metrics
    
    def run_both(
        self,
        test_fn: Callable,
        collect_custom_metrics: Optional[Callable] = None,
        random_order: bool = True
    ) -> Dict[str, TestMetrics]:
        """
        同时运行 A/B 两个版本
        
        Args:
            test_fn: 测试函数
            collect_custom_metrics: 自定义指标收集函数
            random_order: 是否随机执行顺序 (避免时间偏差)
        
        Returns:
            {'a': metrics_a, 'b': metrics_b}
        """
        import random
        
        order = ['a', 'b'] if not random_order or random.random() > 0.5 else ['b', 'a']
        
        results = {}
        for variant in order:
            metrics = self.run_single(variant, test_fn, collect_custom_metrics)
            results[variant] = metrics
        
        return results
    
    def generate_report(self, detailed: bool = False) -> str:
        """生成测试报告"""
        summary = self.results.summary()
        
        lines = [
            "=" * 60,
            "A/B Test Report",
            "=" * 60,
            "",
            f"Total Steps: {self.step_count} (warmup: {self.warmup_steps})",
            "",
            "Variant A (Legacy):",
            f"  Name: {summary['variant_a']['name']}",
            f"  Samples: {summary['variant_a']['samples']}",
            f"  Success Rate: {summary['variant_a']['success_rate']:.2%}",
            f"  Avg Latency: {summary['variant_a']['avg_latency_ms']:.2f} ms",
            "",
            "Variant B (Plugin):",
            f"  Name: {summary['variant_b']['name']}",
            f"  Samples: {summary['variant_b']['samples']}",
            f"  Success Rate: {summary['variant_b']['success_rate']:.2%}",
            f"  Avg Latency: {summary['variant_b']['avg_latency_ms']:.2f} ms",
            "",
            "Improvement (B vs A):",
            f"  Success Rate Delta: {summary['improvement']['success_rate_delta']:+.2%}",
            f"  Latency Delta: {summary['improvement']['latency_delta_percent']:+.1f}%",
            "",
        ]
        
        # 失败案例分析
        if detailed:
            lines.extend([
                "Failure Analysis:",
                "-" * 40,
            ])
            
            failed_a = [m for m in self.results.metrics_a if not m.success]
            failed_b = [m for m in self.results.metrics_b if not m.success]
            
            if failed_a:
                lines.append(f"\nVariant A Failures ({len(failed_a)}):")
                for i, m in enumerate(failed_a[:5], 1):
                    lines.append(f"  {i}. {m.error_message[:100]}...")
            
            if failed_b:
                lines.append(f"\nVariant B Failures ({len(failed_b)}):")
                for i, m in enumerate(failed_b[:5], 1):
                    lines.append(f"  {i}. {m.error_message[:100]}...")
        
        lines.append("=" * 60)
        
        return "\n".join(lines)
    
    def save_report(self, filename: Optional[str] = None):
        """保存报告到文件"""
        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"ab_test_report_{timestamp}.txt"
        
        report_path = self.output_dir / filename
        
        with open(report_path, 'w') as f:
            f.write(self.generate_report(detailed=True))
        
        # 同时保存 JSON 数据
        json_path = report_path.with_suffix('.json')
        with open(json_path, 'w') as f:
            json.dump(self.results.summary(), f, indent=2)
        
        print(f"Report saved to: {report_path}")
        return report_path
    
    def recommend_migration(self) -> Dict[str, Any]:
        """
        基于测试结果给出迁移建议
        
        Returns:
            {
                'recommend': bool,      # 是否建议迁移
                'confidence': float,    # 置信度
                'reason': str,          # 原因
                'cautions': List[str]   # 注意事项
            }
        """
        summary = self.results.summary()
        
        a_success = summary['variant_a']['success_rate']
        b_success = summary['variant_b']['success_rate']
        latency_improvement = summary['improvement']['latency_delta_percent']
        
        recommendation = {
            'recommend': False,
            'confidence': 0.0,
            'reason': '',
            'cautions': []
        }
        
        # 成功率检查
        if b_success < a_success - 0.05:  # B 成功率低 5% 以上
            recommendation['reason'] = f"Plugin success rate ({b_success:.1%}) is significantly lower than legacy ({a_success:.1%})"
            recommendation['cautions'].append("Investigate failure cases before migration")
            return recommendation
        
        # 样本量检查
        if summary['variant_b']['samples'] < 100:
            recommendation['cautions'].append("Sample size is small, consider more tests")
        
        # 推荐迁移
        recommendation['recommend'] = True
        recommendation['confidence'] = min(b_success / (a_success + 1e-6), 1.0)
        
        if b_success >= a_success:
            recommendation['reason'] = f"Plugin matches or exceeds legacy performance ({b_success:.1%} vs {a_success:.1%})"
        else:
            recommendation['reason'] = f"Plugin performance is acceptable ({b_success:.1%} vs {a_success:.1%})"
        
        if latency_improvement < -20:  # 延迟增加超过 20%
            recommendation['cautions'].append(f"Significant latency increase ({latency_improvement:+.1f}%), monitor performance")
        
        return recommendation


class GradualMigration:
    """
    渐进式迁移控制器
    
    支持按流量比例逐步切换到新实现。
    
    Usage:
        >>> migration = GradualMigration(
        ...     legacy_fn=old_controller,
        ...     plugin_fn=new_controller,
        ...     initial_plugin_ratio=0.0
        ... )
        >>> 
        >>> for episode in range(1000):
        ...     fn = migration.select_implementation()
        ...     result = fn(obs)
        ...     migration.update_metrics(success=True)
        >>> 
        >>> migration.increase_plugin_ratio(0.1)  # 增加 10%
    """
    
    def __init__(
        self,
        legacy_fn: Callable,
        plugin_fn: Callable,
        initial_plugin_ratio: float = 0.0,
        min_samples_before_increase: int = 100,
        success_threshold: float = 0.95
    ):
        """
        Args:
            legacy_fn: 旧版实现
            plugin_fn: 新版 plugin 实现
            initial_plugin_ratio: 初始 plugin 流量比例 (0-1)
            min_samples_before_increase: 增加比例前最小样本数
            success_threshold: 成功率阈值
        """
        self.legacy_fn = legacy_fn
        self.plugin_fn = plugin_fn
        
        self.plugin_ratio = initial_plugin_ratio
        self.min_samples = min_samples_before_increase
        self.success_threshold = success_threshold
        
        # 统计
        self.plugin_samples = 0
        self.plugin_successes = 0
        self.legacy_samples = 0
        self.legacy_successes = 0
        
        self.history = []
    
    def select_implementation(self) -> Callable:
        """选择实现版本"""
        import random
        
        if random.random() < self.plugin_ratio:
            return self.plugin_fn
        else:
            return self.legacy_fn
    
    def update_metrics(self, is_plugin: bool, success: bool):
        """更新指标"""
        if is_plugin:
            self.plugin_samples += 1
            if success:
                self.plugin_successes += 1
        else:
            self.legacy_samples += 1
            if success:
                self.legacy_successes += 1
    
    def can_increase_ratio(self, increase_amount: float = 0.1) -> bool:
        """检查是否可以增加 plugin 比例"""
        if self.plugin_ratio >= 1.0:
            return False
        
        if self.plugin_samples < self.min_samples:
            return False
        
        plugin_success_rate = self.plugin_successes / (self.plugin_samples + 1e-6)
        
        if plugin_success_rate < self.success_threshold:
            return False
        
        return True
    
    def increase_plugin_ratio(self, amount: float = 0.1):
        """增加 plugin 流量比例"""
        if not self.can_increase_ratio(amount):
            print(f"Cannot increase plugin ratio yet.")
            print(f"  Current ratio: {self.plugin_ratio:.1%}")
            print(f"  Plugin samples: {self.plugin_samples}")
            print(f"  Plugin success rate: {self.plugin_successes/(self.plugin_samples+1e-6):.1%}")
            return False
        
        old_ratio = self.plugin_ratio
        self.plugin_ratio = min(1.0, self.plugin_ratio + amount)
        
        # 重置统计
        self.plugin_samples = 0
        self.plugin_successes = 0
        
        print(f"Plugin ratio increased: {old_ratio:.1%} -> {self.plugin_ratio:.1%}")
        return True
    
    def get_status(self) -> Dict[str, Any]:
        """获取当前状态"""
        return {
            'plugin_ratio': self.plugin_ratio,
            'legacy_ratio': 1.0 - self.plugin_ratio,
            'plugin_stats': {
                'samples': self.plugin_samples,
                'successes': self.plugin_successes,
                'success_rate': self.plugin_successes / (self.plugin_samples + 1e-6)
            },
            'legacy_stats': {
                'samples': self.legacy_samples,
                'successes': self.legacy_successes,
                'success_rate': self.legacy_successes / (self.legacy_samples + 1e-6)
            }
        }
