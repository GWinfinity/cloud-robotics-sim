# Genesis Cloud Sim Plugins

可沉淀的知识模块 - 从各个项目提取的核心实现。

## 快速开始

```python
# 方式1: 直接导入
from cloud_robotics_sim.plugins.controllers.mpc_wbc import MPCWBCController

controller = MPCWBCController(config_path="configs/default.yaml")

# 方式2: 动态加载
from cloud_robotics_sim.core.plugin_manager import PluginManager

pm = PluginManager()
pm.discover_plugins()
plugin = pm.load_plugin('controllers', 'mpc_wbc')
```

## 可用插件

### Controllers (控制器)

| 插件 | 来源 | 描述 | 状态 |
|-----|------|------|------|
| [mpc_wbc](controllers/mpc_wbc/) | openloong | MPC + WBC 全身控制 | ✅ 可用 |

### Envs (环境)

| 插件 | 来源 | 描述 | 状态 |
|-----|------|------|------|
| (待添加) | genesis-badminton | 羽毛球环境 | 📝 计划 |
| (待添加) | genesis-table-tennis | 乒乓球环境 | 📝 计划 |

### Predictors (预测器)

| 插件 | 来源 | 描述 | 状态 |
|-----|------|------|------|
| (待添加) | genesis-table-tennis | EKF + 物理预测器 | 📝 计划 |

### Sim2Real

| 插件 | 来源 | 描述 | 状态 |
|-----|------|------|------|
| (待添加) | genesis-sim2real | 域随机化 | 📝 计划 |
| (待添加) | genesis-sim2real | Real2Sim 自动调参 | 📝 计划 |

## 创建新插件

```bash
# 1. 复制模板
cp -r templates/basic_plugin controllers/my_controller

# 2. 更新 plugin.yaml
cd controllers/my_controller
vim plugin.yaml

# 3. 实现核心功能
cd core
vim my_controller.py

# 4. 编写 README
vim README.md

# 5. 提交 PR
```

## 插件开发指南

### 目录结构

```
my_plugin/
├── README.md              # 知识文档 (必须)
├── plugin.yaml            # 插件元信息 (必须)
├── __init__.py            # 插件入口 (必须)
├── core/                  # 核心实现
│   └── *.py
├── configs/               # 预设配置
│   └── *.yaml
├── examples/              # 使用示例
│   └── *.py
└── tests/                 # 单元测试
    └── test_*.py
```

### plugin.yaml 模板

```yaml
name: my_plugin
version: 0.1.0
description: Description of my plugin
category: controllers  # or envs, predictors, sim2real
source_project: original_project_name

author:
  name: Your Name
  email: email@example.com

dependencies:
  - numpy>=1.20
  - genesis-world>=0.2.0

exports:
  - MyClass
  - my_function

tags:
  - tag1
  - tag2
```

### README.md 模板

见 [templates/basic_plugin/README.md](templates/basic_plugin/)

## 设计原则

1. **单一职责**: 每个插件只做一件事
2. **文档优先**: 先写 README，再写代码
3. **可测试**: 必须包含单元测试
4. **可配置**: 支持 YAML/JSON 配置
5. **向后兼容**: 版本升级不破坏 API

## 知识沉淀流程

```
发现可沉淀代码
       ↓
提取核心实现 → plugins/<category>/<name>/
       ↓
编写 README → 记录算法原理
       ↓
创建示例 → 放在 examples/
       ↓
添加配置 → 提供默认参数
       ↓
代码审查 → 确保质量
       ↓
合并到主分支
```
