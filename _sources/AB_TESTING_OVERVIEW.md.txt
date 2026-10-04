# A/B Testing Migration Strategy

## 方案概览

使用 A/B 测试框架实现渐进式 Plugin 迁移，确保：
- ✅ 低风险：新旧实现并行运行
- ✅ 可回滚：随时切回旧版
- ✅ 数据驱动：基于指标决策

## 核心组件

### 1. ABTestRunner - A/B 测试运行器

对比新旧实现的核心工具。

```python
from cloud_robotics_sim.core.ab_test_framework import ABTestRunner

runner = ABTestRunner(
    variant_a_name="legacy",      # 旧版
    variant_a_fn=old_controller,
    variant_b_name="plugin",      # 新版
    variant_b_fn=new_controller,
    warmup_steps=10
)

# 同时运行两个版本
for _ in range(1000):
    results = runner.run_both(
        test_fn=lambda fn: fn(state),
        collect_custom_metrics=lambda r: {'reward': r['reward']}
    )

# 生成报告
print(runner.generate_report())
```

**输出指标**:
- 成功率对比
- 延迟对比
- 自定义业务指标
- 错误类型分析

### 2. GradualMigration - 渐进式迁移

按流量比例逐步切换到新实现。

```python
from cloud_robotics_sim.core.ab_test_framework import GradualMigration

migration = GradualMigration(
    legacy_fn=old_controller,
    plugin_fn=new_controller,
    initial_plugin_ratio=0.0  # 从 0% 开始
)

# 自动选择实现
for episode in range(10000):
    fn = migration.select_implementation()
    is_plugin = (fn is new_controller)
    
    # 运行...
    success = run_episode(fn)
    
    # 更新统计
    migration.update_metrics(is_plugin, success)
    
    # 定期增加比例
    if episode % 500 == 0:
        migration.increase_plugin_ratio(0.1)  # 10% -> 20% -> ...
```

### 3. MigrationTool - 自动化迁移工具

将现有项目自动转换为 Plugin 结构。

```bash
python plugins/migrate_project.py \
    --source /path/to/original/project \
    --name residual_rl \
    --category controllers
```

自动生成：
- Plugin 目录结构
- plugin.yaml
- __init__.py
- README.md 模板
- A/B 测试模板

## 迁移流程

```
┌─────────────────────────────────────────────────────────────┐
│                    完整迁移流程                              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Step 1: 准备                                                │
│  ├── 使用 MigrationTool 创建 Plugin 结构                     │
│  ├── 迁移核心代码到 plugins/<category>/<name>/              │
│  └── 完善 README 和配置                                      │
│                                                             │
│  Step 2: A/B 测试                                            │
│  ├── 编写 A/B 测试脚本                                       │
│  ├── 并行运行 1000+ 次测试                                   │
│  ├── 对比成功率、延迟、业务指标                              │
│  └── 生成测试报告                                            │
│                                                             │
│  Step 3: 决策                                                │
│  ├── 如果 Plugin 优于旧版 → 进入 Step 4                      │
│  ├── 如果性能相当 → 可选迁移                                 │
│  └── 如果 Plugin 较差 → 修复后重新测试                       │
│                                                             │
│  Step 4: 渐进上线                                            │
│  ├── 使用 GradualMigration                                   │
│  ├── 流量比例: 0% → 10% → 25% → 50% → 100%                 │
│  ├── 每步验证成功率                                          │
│  └── 发现问题立即回滚                                        │
│                                                             │
│  Step 5: 完成                                                │
│  ├── Plugin 比例达到 100%                                    │
│  ├── 稳定运行一段时间后                                       │
│  ├── 删除旧代码                                              │
│  └── 更新文档                                                │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## 快速开始

### 1. 迁移第一个 Plugin

```bash
# 1. 进入插件目录
cd genesis-cloud-sim/plugins

# 2. 使用迁移工具
python migrate_project.py \
    --source ../../openloong-dyn-control \
    --name mpc_wbc \
    --category controllers

# 3. 按提示选择要迁移的文件

# 4. 完善生成的文件
vim controllers/mpc_wbc/README.md
vim controllers/mpc_wbc/__init__.py

# 5. 运行 A/B 测试
python controllers/mpc_wbc/examples/ab_test.py

# 6. 查看报告
cat ab_test_results/mpc_wbc/ab_test_report_*.txt
```

### 2. 渐进上线

```python
# gradual_rollout.py
from cloud_robotics_sim.core.ab_test_framework import GradualMigration
from cloud_robotics_sim.plugins.controllers.mpc_wbc import MPCWBCController

# 导入旧版 (从原项目)
import sys
sys.path.insert(0, '/path/to/openloong')
from genesis_sim.openloong_controller import OldController

# 创建迁移控制器
migration = GradualMigration(
    legacy_fn=OldController(),
    plugin_fn=MPCWBCController(),
    initial_plugin_ratio=0.0
)

# 训练循环
for episode in range(10000):
    # 自动选择实现
    controller = migration.select_implementation()
    
    # 运行
    success = run_episode(controller)
    
    # 更新统计
    is_plugin = (controller is migration.plugin_fn)
    migration.update_metrics(is_plugin, success)
    
    # 定期增加比例
    if episode % 500 == 0:
        status = migration.get_status()
        print(f"Episode {episode}: {status}")
        
        # 尝试增加比例
        migration.increase_plugin_ratio(0.1)
```

## 文件结构

```
genesis-cloud-sim/
├── src/cloud_robotics_sim/core/
│   ├── plugin_manager.py           # 插件管理
│   └── ab_test_framework.py        # ★ A/B 测试框架
│
├── plugins/
│   ├── README.md
│   ├── MIGRATION_GUIDE.md          # ★ 迁移指南
│   ├── migrate_project.py          # ★ 迁移工具
│   │
│   └── controllers/mpc_wbc/        # ★ 示例 Plugin
│       ├── README.md
│       ├── plugin.yaml
│       ├── __init__.py
│       ├── core/                   # 核心实现
│       ├── configs/                # 配置
│       ├── examples/
│       │   ├── basic_usage.py
│       │   └── ab_test.py          # ★ A/B 测试示例
│       └── tests/
│
└── docs/
    └── AB_TESTING_OVERVIEW.md      # ★ 本文档
```

## 下一步

1. **阅读详细指南**: [MIGRATION_GUIDE.md](../plugins/MIGRATION_GUIDE.md)
2. **运行示例**: `python plugins/controllers/mpc_wbc/examples/ab_test_example.py`
3. **迁移第一个项目**: 使用 `migrate_project.py`
4. **渐进上线**: 按照 0% → 10% → 25% → 50% → 100% 的步骤

## 最佳实践

### DO (推荐)
- ✅ 始终运行 A/B 测试后再上线
- ✅ 保持渐进式，不要一次性 100%
- ✅ 监控每一步的成功率
- ✅ 准备回滚方案
- ✅ 记录迁移过程中的问题

### DON'T (避免)
- ❌ 直接替换，不经过测试
- ❌ 跳过 A/B 测试直接上线
- ❌ 忽略失败案例分析
- ❌ 在没有监控的情况下增加比例
- ❌ 删除旧代码前不做验证

---

*这个方案的核心: 数据驱动 + 渐进式 + 可回滚 = 安全的知识沉淀*
