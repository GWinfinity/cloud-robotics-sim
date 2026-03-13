# Project 2: Badminton Environment Migration Summary

## 迁移完成 ✅

已将 `genesis-humanoid-badminton` 迁移为 Plugin: `plugins/envs/badminton/`

---

## 迁移内容

### 核心文件 (来自原项目)

| 文件 | 原位置 | 说明 |
|-----|-------|------|
| `badminton_env.py` | `envs/badminton_env.py` | 主环境类 |
| `shuttlecock.py` | `envs/shuttlecock.py` | 羽毛球物理模型 |
| `curriculum.py` | `envs/curriculum.py` | 课程学习系统 |
| `ekf.py` | `utils/ekf.py` | EKF 轨迹预测器 |
| `rewards.py` | `utils/rewards.py` | 奖励函数 |

### 新增文件 (Plugin 规范)

| 文件 | 说明 |
|-----|------|
| `plugin.yaml` | Plugin 元信息 |
| `__init__.py` | 插件入口，导出主要类 |
| `README.md` | 知识文档 |
| `configs/stage1_footwork.yaml` | Stage 1 配置 |
| `configs/stage3_full.yaml` | Stage 3 配置 |
| `examples/basic_usage.py` | 使用示例 |
| `examples/ab_test.py` | A/B 测试脚本 |

---

## 快速使用

### 1. 直接使用

```python
from cloud_robotics_sim.plugins.envs.badminton import BadmintonEnv

env = BadmintonEnv(curriculum_stage=1)
obs = env.reset()
```

### 2. 运行 A/B 测试

```bash
cd genesis-cloud-sim
python plugins/envs/badminton/examples/ab_test.py
```

### 3. 运行示例

```bash
python plugins/envs/badminton/examples/basic_usage.py
```

---

## 核心特性

### 三阶段课程学习

```python
# Stage 1: 步法训练 (冻结上肢)
env.set_curriculum_stage(1)

# Stage 2: 挥拍训练 (冻结下肢)
env.set_curriculum_stage(2)

# Stage 3: 全身协调 (全部解冻)
env.set_curriculum_stage(3)
```

### 羽毛球物理模型

- **高速**: 羽毛压缩，阻力小 (飞行远)
- **低速**: 羽毛展开，阻力大 (减速快)
- **翻转**: 低速时的翻转效应

### EKF 轨迹预测

```python
from cloud_robotics_sim.plugins.envs.badminton import EKFPredictor

predictor = EKFPredictor()
predictor.update(ball_position)
landing = predictor.predict_landing()
```

---

## 与旧版对比

| 特性 | 旧版 (genesis-humanoid-badminton) | Plugin 版 |
|-----|----------------------------------|-----------|
| 导入 | `from envs.badminton_env import *` | `from cloud_robotics_sim.plugins.envs.badminton import *` |
| 结构 | 独立项目 | 统一 Plugin 结构 |
| 文档 | 分散 | 集中 README.md |
| 配置 | 硬编码 | YAML 配置文件 |
| A/B 测试 | 无 | 内置测试脚本 |

---

## 下一步：A/B 测试 & 渐进上线

### 1. 运行 A/B 测试

```bash
python plugins/envs/badminton/examples/ab_test.py
```

检查指标:
- 成功率对比
- 延迟对比
- 奖励一致性

### 2. 渐进上线

```python
from cloud_robotics_sim.core.ab_test_framework import GradualMigration

migration = GradualMigration(
    legacy_fn=LegacyBadmintonEnv,
    plugin_fn=BadmintonEnv,
    initial_plugin_ratio=0.0
)

# 0% → 10% → 25% → 50% → 100%
for episode in range(10000):
    controller = migration.select_implementation()
    # ...
    migration.increase_plugin_ratio(0.1)
```

### 3. 完成切换

当 Plugin 稳定运行后:
- 删除原 `genesis-humanoid-badminton` 项目
- 更新文档引用

---

## 已完成迁移项目总结

| # | 项目 | 类别 | 状态 |
|---|------|------|------|
| 1 | mpc_wbc | controllers | ✅ 完成 |
| 2 | badminton | envs | ✅ 完成 |

---

*下一步建议: 迁移 residual_rl (controllers) 或 table_tennis (envs)*
