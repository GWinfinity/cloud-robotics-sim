# Plugin Migration Guide (A/B Testing)

使用 A/B 测试框架渐进式迁移代码到 Plugin。

## 迁移流程

```
Step 1: 提取 Plugin
    │
    ▼
Step 2: A/B 测试 (并行运行)
    │
    ├── 对比成功率
    ├── 对比性能
    └── 对比业务指标
    │
    ▼
Step 3: 渐进上线 (流量切换)
    │
    ├── 0% → 10% → 25% → 50% → 100%
    └── 每步验证成功率
    │
    ▼
Step 4: 完全切换
    │
    └── 删除旧代码
```

## Step 1: 提取 Plugin

### 1.1 创建 Plugin 目录

```bash
# 使用模板创建
cd genesis-cloud-sim/plugins
mkdir -p controllers/<your_plugin>/{core,configs,examples,tests}
```

### 1.2 迁移核心代码

从原项目复制核心实现到 `core/` 目录，确保：
- ✅ 删除项目特定的路径/导入
- ✅ 使用相对导入
- ✅ 添加清晰的文档字符串
- ✅ 保持接口兼容

### 1.3 创建 Plugin 元信息

```yaml
# plugin.yaml
name: your_controller
version: 0.1.0
description: Description of your controller
category: controllers
source_project: original_project_name
```

## Step 2: A/B 测试

### 2.1 编写测试脚本

```python
# test_migration.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from cloud_robotics_sim.core.ab_test_framework import ABTestRunner


def main():
    # 1. 导入旧版实现
    sys.path.insert(0, '/path/to/original/project')
    from original_module import OldController
    
    # 2. 导入新版 Plugin
    from cloud_robotics_sim.plugins.controllers.your_controller import NewController
    
    old_controller = OldController()
    new_controller = NewController()
    
    # 3. 创建 A/B 测试运行器
    runner = ABTestRunner(
        variant_a_name="original_implementation",
        variant_a_fn=old_controller,
        variant_b_name="plugin_implementation", 
        variant_b_fn=new_controller,
        output_dir="ab_test_results/your_controller",
        warmup_steps=10
    )
    
    # 4. 运行测试
    for episode in range(1000):
        obs = env.reset()
        
        # 同时运行两个版本
        results = runner.run_both(
            test_fn=lambda fn: fn(obs),
            collect_custom_metrics=lambda result: {
                'reward': result.get('reward', 0),
                'episode_length': result.get('length', 0)
            }
        )
        
        if episode % 100 == 0:
            print(f"Episode {episode}: A={results['a'].success}, B={results['b'].success}")
    
    # 5. 生成报告
    print(runner.generate_report(detailed=True))
    runner.save_report()
    
    # 6. 获取迁移建议
    recommendation = runner.recommend_migration()
    print(f"\nRecommendation: {recommendation}")


if __name__ == '__main__':
    main()
```

### 2.2 解读测试结果

```
A/B Test Report
============================================================

Variant A (Legacy):
  Name: original_implementation
  Samples: 1000
  Success Rate: 98.5%
  Avg Latency: 2.34 ms

Variant B (Plugin):
  Name: plugin_implementation
  Samples: 1000
  Success Rate: 99.1%          <- 成功率对比
  Avg Latency: 1.89 ms         <- 性能对比

Improvement (B vs A):
  Success Rate Delta: +0.6%    <- 提升
  Latency Delta: +19.2%        <- 加速

============================================================
```

### 2.3 决策标准

| 指标 | 可接受 | 优秀 | 不可接受 |
|-----|-------|------|---------|
| 成功率差 | ±2% | 优于旧版 | <-5% |
| 延迟变化 | ±20% | 优于旧版 30%+ | >50% |
| 错误类型 | 无新增 | - | 新增严重错误 |

## Step 3: 渐进上线

### 3.1 使用 GradualMigration

```python
from cloud_robotics_sim.core.ab_test_framework import GradualMigration

# 创建渐进式迁移控制器
migration = GradualMigration(
    legacy_fn=old_controller,
    plugin_fn=new_controller,
    initial_plugin_ratio=0.0,      # 开始时 0% plugin
    min_samples_before_increase=100,
    success_threshold=0.95
)

# 训练循环
for episode in range(total_episodes):
    # 自动选择实现 (根据当前比例)
    controller_fn = migration.select_implementation()
    is_plugin = (controller_fn is new_controller)
    
    # 运行一回合
    obs = env.reset()
    success = True
    
    for step in range(max_steps):
        try:
            action = controller_fn(obs)
            obs, reward, done, _ = env.step(action)
            
            if done:
                break
        except Exception as e:
            success = False
            break
    
    # 更新统计
    migration.update_metrics(is_plugin, success)
    
    # 定期尝试增加比例
    if episode % 500 == 0 and episode > 0:
        print(f"\nEpisode {episode}:")
        print(migration.get_status())
        
        # 尝试增加到 10% -> 25% -> 50% -> 100%
        current_ratio = migration.plugin_ratio
        if current_ratio < 0.1:
            migration.increase_plugin_ratio(0.1)    # 到 10%
        elif current_ratio < 0.25:
            migration.increase_plugin_ratio(0.15)   # 到 25%
        elif current_ratio < 0.5:
            migration.increase_plugin_ratio(0.25)   # 到 50%
        elif current_ratio < 1.0:
            migration.increase_plugin_ratio(0.5)    # 到 100%
```

### 3.2 监控和回滚

```python
# 每回合检查
status = migration.get_status()

# 如果 plugin 成功率太低，暂停增加
if status['plugin_stats']['success_rate'] < 0.90:
    print("WARNING: Plugin success rate dropped below 90%")
    print("Pausing ratio increase until stability returns")
    # 保持当前比例，不增加

# 如果严重问题，可以手动回滚
if status['plugin_stats']['success_rate'] < 0.80:
    print("CRITICAL: Serious degradation detected")
    print("Rolling back to 0% plugin")
    migration.plugin_ratio = 0.0
```

## Step 4: 完全切换

当 plugin_ratio 达到 1.0 (100%) 且稳定运行一段时间后：

### 4.1 清理旧代码

```python
# 修改使用处
# 从:
if use_legacy:
    from old_project import Controller
else:
    from plugins import Controller

# 改为:
from cloud_robotics_sim.plugins.controllers.your_controller import Controller
```

### 4.2 更新文档

- 更新 README，说明已迁移
- 记录迁移过程中的问题和解决方案
- 更新依赖列表

## 常见问题和解决方案

### Q1: Plugin 和旧版结果不一致

**诊断**: 
```python
# 添加详细对比
result_a = old_controller(state)
result_b = new_controller(state)

print(f"Difference: {np.abs(result_a - result_b).max()}")
print(f"A: {result_a[:5]}")
print(f"B: {result_b[:5]}")
```

**常见原因**:
- 随机种子不同 → 同步种子
- 初始化参数不同 → 检查默认值
- 数值精度问题 → 确认数据类型

### Q2: Plugin 性能更差

**优化步骤**:
1. 使用 `cProfile` 分析热点
2. 检查是否有不必要的复制
3. 考虑使用 Numba/JIT 加速
4. 批处理操作

### Q3: 依赖冲突

**解决**:
```yaml
# plugin.yaml
# 明确指定依赖版本
dependencies:
  - numpy>=1.20,<2.0
  - scipy>=1.7
```

## 迁移检查清单

### 提取阶段
- [ ] 核心代码已迁移
- [ ] 单元测试通过
- [ ] 示例代码可运行
- [ ] README 文档完整

### A/B 测试阶段
- [ ] A/B 测试脚本完成
- [ ] 至少 1000 次测试
- [ ] 成功率差异 < 2%
- [ ] 性能差异可接受

### 上线阶段
- [ ] 渐进比例: 0% → 10% → 25% → 50% → 100%
- [ ] 每步验证成功率
- [ ] 监控日志无异常
- [ ] 回滚方案就绪

### 清理阶段
- [ ] 旧代码已删除
- [ ] 文档已更新
- [ ] 团队成员已通知
- [ ] 版本已标记

## 示例迁移计划

以 `residual_rl` 为例：

| Week | Task | Milestone |
|-----|------|-----------|
| 1 | 提取代码到 plugin | Plugin 可用 |
| 2 | A/B 测试 | 成功率对比报告 |
| 3 | 修复差异 | 测试通过 |
| 4 | 10% 上线 | 监控稳定 |
| 5 | 50% 上线 | 监控稳定 |
| 6 | 100% 上线 | 完全切换 |
| 7 | 清理旧代码 | 迁移完成 |

---

*记住: 渐进式迁移 = 降低风险 + 可回滚 + 数据驱动决策*
