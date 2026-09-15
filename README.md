# 云端机器人仿真平台

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Documentation](https://img.shields.io/badge/docs-latest-blue.svg)](https://gwinfinity.github.io/cloud-robotics-sim/)

基于 [Genesis](https://genesis-world.readthedocs.io/) 物理引擎构建的云原生机器人仿真平台，面向可扩展的强化学习与模仿学习研究。

## 特性

- **云原生架构** - Kubernetes 原生部署（[deploy/kubernetes/](deploy/kubernetes/)），KEDA 按任务队列自动扩缩容，空闲缩到 0
- **可组合设计** - 场景（Scene）、机器人（Robot）、任务（Task）可自由组合
- **大规模并行** - 可同时训练多达 4,096 个并行环境
- **Agent 就绪** - 内置技能注册表与任务执行引擎
- **兼容 Gymnasium** - 可与主流 RL/IL 库无缝集成

## 快速开始

### 安装

```bash
# 克隆仓库
git clone https://github.com/GWinfinity/cloud-robotics-sim.git
cd cloud-robotics-sim

# 使用 pip 安装
pip install -e ".[dev]"

# 或使用 uv（推荐，可复现锁定依赖）
uv sync --extra dev
```

### 基本用法

```python
from cloud_robotics_sim import (
    EnvironmentComposer,
    ComposerConfig,
    SceneConfig,
    ObjectLibrary,
)

# 创建一个简单的抓取放置环境
composer = EnvironmentComposer(ComposerConfig(headless=False))

# 组合并运行
obs, info = env.reset()
for _ in range(100):
    action = env.action_space.sample()
    obs, reward, done, trunc, info = env.step(action)
    if done or trunc:
        obs, info = env.reset()
```

### 命令行工具

```bash
# 训练
cloud-robotics-sim train --config configs/franka_pickplace.yaml

# 评估
cloud-robotics-sim eval --checkpoint checkpoints/latest.pt --num-episodes 100

# 交互式 Agent
cloud-robotics-sim agent --goal "pick up the red cube"

# 运行测试
cloud-robotics-sim test
```

## 架构

```
┌─────────────────────────────────────────────────────────────┐
│                    运行时层 (Runtime)                        │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────────┐   │
│  │  技能注册表  │  │  任务执行器  │  │  回放缓冲区      │   │
│  └─────────────┘  └──────────────┘  └──────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   环境层 (Environment)                       │
│                    Genesis 物理引擎                          │
│  ┌──────────┐  ┌────────────┐  ┌────────────────────────┐  │
│  │   场景   │  │   机器人   │  │         任务           │  │
│  │  组件    │  │  本体实现  │  │    （目标 + 奖励）     │  │
│  └──────────┘  └────────────┘  └────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   学习层 (Learning)                          │
│  ┌─────────────────┐  ┌──────────────────────────────────┐ │
│  │  RL 算法        │  │    模仿学习                      │ │
│  │  (PPO, SAC)     │  │    （行为克隆、扩散策略）        │ │
│  └─────────────────┘  └──────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## 项目结构

```
cloud-robotics-sim/
├── src/
│   └── cloud_robotics_sim/     # 主包
│       ├── core/               # 核心组件
│       │   ├── composer.py     # 环境组合
│       │   ├── scene.py        # 场景定义
│       │   ├── embodiment.py   # 机器人实现
│       │   ├── task.py         # 任务定义
│       │   ├── registry.py     # 组件注册表
│       │   └── vectorized.py   # 并行环境
│       ├── runtime/            # Agent 运行时
│       └── learning/           # RL/IL 框架
├── configs/                    # 配置文件
├── tests/                      # 测试套件
├── examples/                   # 示例脚本（索引见 examples/README.md）
└── docs/                       # 文档
```

## 支持的机器人

| 机器人 | 类型 | 自由度 | 状态 |
|--------|------|--------|------|
| Franka Emika Panda | 协作机械臂 | 7+1 | ✅ 完全支持 |
| Universal Robots UR5 | 工业机械臂 | 6 | ✅ 完全支持 |
| 移动操作机器人 | 移动底盘 + 机械臂 | 10+ | 🚧 实验性支持 |

## 支持的任务

- **抓取放置（Pick and Place）** - 抓取物体并放置到目标位置
- **导航（Navigation）** - 在避开障碍物的同时到达目标位置
- **到达（Reach）** - 将末端执行器移动到目标位姿

## 文档

📖 **完整文档**: [https://gwinfinity.github.io/cloud-robotics-sim/](https://gwinfinity.github.io/cloud-robotics-sim/)

- [架构概述](docs/architecture/overview.md)
- [快速上手指南](docs/guides/quickstart.md)
- [安装指南](docs/guides/installation.md)
- [配置指南](docs/guides/configuration.md)
- [Kubernetes 部署指南](docs/guides/kubernetes.md)
- [Agent 接口指南](docs/guides/agent.md)
- [API 参考](docs/api/core.md)
- [贡献指南](CONTRIBUTING.md)

## 环境要求

- Python 3.10+
- CUDA 11.8+（用于 GPU 加速）
- Genesis World 0.4+
- PyTorch 2.0+

## 开发

```bash
# 安装开发依赖
pip install -e ".[dev]"

# 运行代码检查
ruff check src/
black --check src/

# 运行测试
pytest tests/ -v

# 构建文档
cd docs && make html
```

## 引用

如果在研究中使用了本平台，请引用：

```bibtex
@software{cloud_robotics_sim,
  title = {Cloud Robotics Simulation Platform},
  author = {Cloud Robotics Team},
  year = {2025},
  url = {https://github.com/your-org/cloud-robotics-sim}
}
```

## 许可证

本项目基于 Apache License 2.0 许可证开源 - 详见 [LICENSE](LICENSE) 文件。

## 致谢

- [Genesis](https://genesis-world.readthedocs.io/) - 底层物理引擎
- [LeRobot](https://github.com/huggingface/lerobot) - 学习框架的设计灵感
- [Gymnasium](https://gymnasium.farama.org/) - 强化学习环境接口标准

## 参与贡献

我们欢迎贡献！详情请参阅[贡献指南](CONTRIBUTING.md)。

## 支持

- 📧 邮箱: support@cloudrobotics.dev
- 💬 讨论区: [GitHub Discussions](https://github.com/your-org/cloud-robotics-sim/discussions)
- 🐛 问题反馈: [GitHub Issues](https://github.com/your-org/cloud-robotics-sim/issues)
