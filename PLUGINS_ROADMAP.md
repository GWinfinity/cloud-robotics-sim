# Genesis Cloud Sim - 插件化沉淀路线图

## 核心思路

> **Plugin = 可沉淀的知识模块**
> 
> 把其他项目的核心实现提取为独立 plugin，在 genesis-cloud-sim 内部渐进式整合。

## 为什么用 Plugin？

```
传统方式                    Plugin 方式
─────────────────────────────────────────────────────────
复制粘贴代码        →       独立模块，清晰边界
深度耦合            →       接口隔离，可插拔
难以测试            →       独立测试，快速迭代
知识分散            →       统一位置，文档沉淀
```

## Plugin 目录结构

```
genesis-cloud-sim/
├── src/cloud_robotics_sim/          # 现有核心
│   ├── core/                         # 基础抽象 (最小化)
│   │   ├── base_controller.py       # 控制器接口
│   │   ├── base_env.py              # 环境接口
│   │   └── plugin_manager.py        # 插件管理
│   └── ...
│
├── plugins/                          # ★ 沉淀的知识模块
│   ├── README.md
│   │
│   ├── controllers/                  # 控制器插件
│   │   ├── mpc_wbc/                  # 来自 openloong
│   │   │   ├── README.md            # 知识文档
│   │   │   ├── controller.py        # 核心实现
│   │   │   ├── config.yaml          # 默认配置
│   │   │   └── examples/            # 使用示例
│   │   │
│   │   ├── residual_rl/              # 来自 residual-rl
│   │   │   ├── README.md
│   │   │   ├── residual_network.py
│   │   │   ├── residual_sac.py
│   │   │   └── examples/
│   │   │
│   │   └── hugwbc/                   # 来自 hugwbc-genesis
│   │       ├── README.md
│   │       ├── locomotion_controller.py
│   │       └── examples/
│   │
│   ├── envs/                         # 环境插件
│   │   ├── ball_sports/              # 球类运动
│   │   │   ├── badminton/            # 来自 genesis-badminton
│   │   │   │   ├── README.md
│   │   │   │   ├── shuttlecock_physics.py
│   │   │   │   ├── badminton_env.py
│   │   │   │   └── curriculum.py
│   │   │   │
│   │   │   └── table_tennis/         # 来自 genesis-table-tennis
│   │   │       ├── README.md
│   │   │       ├── dual_predictor.py
│   │   │       └── table_tennis_env.py
│   │   │
│   │   └── manipulation/             # 操作环境
│   │       └── generalized_reward.py # 来自 sim2real-dexterous
│   │
│   ├── predictors/                   # 预测器插件
│   │   ├── ekf_predictor/            # EKF轨迹预测
│   │   │   ├── README.md
│   │   │   └── ekf.py
│   │   │
│   │   └── physics_predictor/        # 物理预测器
│   │       ├── README.md
│   │       └── ball_physics.py
│   │
│   └── sim2real/                     # Sim2Real工具
│       ├── domain_randomization/
│       │   ├── README.md
│       │   └── randomizer.py
│       │
│       └── real2sim_tuning/
│           ├── README.md
│           └── auto_tuner.py
│
└── docs/plugins/                     # 插件文档
    ├── 01_how_to_use.md
    ├── 02_how_to_create.md
    └── architecture/                 # 架构决策记录
        └── 001_plugin_system.md
```

## Plugin 规范

### 1. 目录结构标准

```
plugins/<category>/<plugin_name>/
├── README.md              # 必须: 知识文档 (算法原理 + 使用指南)
├── __init__.py            # 必须: 插件入口
├── plugin.yaml            # 必须: 插件元信息
│
├── core/                  # 核心实现
│   └── *.py
│
├── configs/               # 预设配置
│   └── *.yaml
│
├── examples/              # 使用示例
│   └── *.py
│
└── tests/                 # 单元测试
    └── test_*.py
```

### 2. 必备文件模板

#### README.md 模板

```markdown
# Plugin: <名称>

## 来源
- **原始项目**: <项目名称>
- **提取日期**: YYYY-MM-DD
- **版本**: x.x.x

## 核心功能
一句话描述这个插件是做什么的。

## 快速开始
```python
from cloud_robotics_sim.plugins.controllers.mpc_wbc import MPCWBCController

controller = MPCWBCController(config)
```

## 算法原理
简要描述算法原理，参考论文。

## 配置参数
| 参数 | 类型 | 默认值 | 说明 |
|-----|------|-------|------|
| ... | ... | ... | ... |

## 示例
见 [examples/](examples/) 目录。

## Changelog
- 2024-03-15: 初始导入
```

#### plugin.yaml 模板

```yaml
name: mpc_wbc
version: 0.1.0
description: MPC + WBC controller for humanoid locomotion

category: controllers
source_project: openloong-dyn-control

author: 
  name: <Your Name>
  email: <email@example.com>

dependencies:
  - numpy>=1.20
  - scipy>=1.7
  - genesis-world>=0.2.0

exports:
  - MPCWBCController
  - GaitScheduler
  - WBCConfig

tags:
  - humanoid
  - locomotion
  - mpc
  - wbc
```

### 3. 代码规范

```python
# plugins/controllers/mpc_wbc/__init__.py

"""
MPC + WBC Controller Plugin

来源: openloong-dyn-control
核心实现: MPC模型预测控制 + WBC全身控制
"""

__version__ = "0.1.0"
__source__ = "openloong-dyn-control"

from .controller import MPCWBCController
from .gait_scheduler import GaitScheduler
from .config import WBCConfig

__all__ = ['MPCWBCController', 'GaitScheduler', 'WBCConfig']

# 注册到插件系统
from cloud_robotics_sim.core.plugin_manager import register_plugin

register_plugin(
    name='mpc_wbc',
    category='controllers',
    exports=__all__,
    config_class=WBCConfig
)
```

## 实施路线图

### Phase 1: 基础设施 (Week 1-2)

```
任务:
1. 创建 plugins/ 目录结构
2. 实现 PluginManager (简单版)
3. 创建第一个示例 plugin (MPC-WBC)

交付物:
- plugins/README.md (插件开发指南)
- cloud_robotics_sim/core/plugin_manager.py
- plugins/controllers/mpc_wbc/ (示例)
```

### Phase 2: 核心沉淀 (Week 3-6)

```
按优先级逐个迁移:

P0 (核心 - 2周内完成):
├── controllers/mpc_wbc/          # 来自 openloong
├── controllers/residual_rl/       # 来自 residual-rl
└── envs/ball_sports/badminton/    # 来自 genesis-badminton

P1 (重要 - 2-4周):
├── envs/ball_sports/table_tennis/ # 来自 genesis-table-tennis
├── predictors/ekf/                # EKF预测器
└── sim2real/domain_randomization/ # 来自 sim2real-dexterous

P2 (增强 - 4-6周):
├── controllers/hugwbc/
├── envs/manipulation/
└── sim2real/real2sim_tuning/
```

### Phase 3: 优化完善 (Week 7-8)

```
任务:
1. 统一配置格式
2. 编写完整文档
3. 添加单元测试
4. 创建示例集合

交付物:
- docs/plugins/ (完整文档)
- examples/01-10/ (使用示例)
- tests/plugins/ (测试套件)
```

## 使用方式

### 对用户

```python
# 方式 1: 直接导入
from cloud_robotics_sim.plugins.controllers.mpc_wbc import MPCWBCController

controller = MPCWBCController(config_path="configs/mpc_wbc.yaml")

# 方式 2: 通过插件管理器动态加载
from cloud_robotics_sim.core.plugin_manager import PluginManager

pm = PluginManager()
plugin = pm.load_plugin('controllers', 'mpc_wbc')
controller = plugin.MPCWBCController()

# 方式 3: 配置驱动
from cloud_robotics_sim import load_env

env = load_env("configs/tasks/h1_walking_with_mpc.yaml")
# 配置文件中指定使用 mpc_wbc 控制器
```

### 对开发者

```python
# 创建新插件步骤:

# 1. 复制模板
# cp -r plugins/templates/basic_plugin plugins/<category>/<my_plugin>

# 2. 实现核心功能
# plugins/<category>/<my_plugin>/core/my_module.py

# 3. 编写文档
# plugins/<category>/<my_plugin>/README.md

# 4. 注册插件
# 在 __init__.py 中调用 register_plugin()

# 5. 提交 PR
```

## 知识沉淀流程

```
发现可沉淀代码
       ↓
提取核心实现 → 放入 plugins/<category>/<name>/
       ↓
编写 README → 记录算法原理和使用方法
       ↓
创建示例 → 放在 examples/ 目录
       ↓
添加配置 → 提供默认参数
       ↓
代码审查 → 确保质量和一致性
       ↓
合并到主分支
       ↓
更新文档索引
```

## 与现有代码的关系

```
┌─────────────────────────────────────────────────────────────┐
│                    现有代码 (保持不变)                        │
│  src/cloud_robotics_sim/                                     │
│  ├── core/composer.py                                       │
│  ├── core/scene.py                                          │
│  └── ...                                                    │
└─────────────────────────────────────────────────────────────┘
                            ↑ 使用
┌─────────────────────────────────────────────────────────────┐
│                    沉淀的 Plugin                              │
│  plugins/                                                   │
│  ├── controllers/mpc_wbc/ ← 来自 openloong                  │
│  ├── controllers/residual_rl/ ← 来自 residual-rl            │
│  └── envs/ball_sports/ ← 来自 badminton/table-tennis       │
└─────────────────────────────────────────────────────────────┘
                            ↑ 逐步
┌─────────────────────────────────────────────────────────────┐
│                    原始项目 (可以删除)                        │
│  - openloong-dyn-control/                                   │
│  - genesis-residual-rl/                                     │
│  - genesis-humanoid-badminton/                              │
│  - ...                                                      │
└─────────────────────────────────────────────────────────────┘
```

## FAQ

**Q: Plugin 和独立包的区别？**
A: Plugin 是 genesis-cloud-sim 的内部模块，共享版本号，一起发布。独立包需要单独维护版本。

**Q: 什么时候应该把代码提取为 Plugin？**
A: 满足以下条件时：
1. 代码来自其他项目，有沉淀价值
2. 功能相对独立，有清晰边界
3. 可能被多个场景复用
4. 有明确的算法原理和使用方法

**Q: Plugin 之间可以依赖吗？**
A: 可以，但尽量避免循环依赖。推荐通过核心接口解耦。

**Q: 如何确保 Plugin 质量？**
A: 每个 Plugin 必须有：
1. README.md 文档
2. 至少一个可运行的示例
3. 基本的单元测试
4. 代码审查通过

---

*这个方案的核心: 渐进式沉淀，先收集再优化，不破坏现有结构。*
