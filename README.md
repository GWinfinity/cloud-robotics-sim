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
- **多物理场求解器** - 插件化网格场求解器：热传导（FTCS）、焦耳热（电势-热源耦合）、时域声学（leapfrog 波动方程），随 `scene.step()` 一起积分，支持麦克风录音与 SPL 频谱后处理（[plugins/solvers/](plugins/solvers/)）

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

安装后提供 `crs`（短别名）和 `cloud-robotics-sim` 两个等价的命令：

```bash
# 训练
crs train --config configs/franka_pickplace.yaml

# 评估
crs eval --checkpoint checkpoints/latest.pt --num-episodes 100

# 交互式 Agent
crs agent --goal "pick up the red cube"

# 运行测试
crs test
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
├── plugins/                    # 插件（求解器 / 环境 / 控制器 / 数据集，见 plugins/ 各 README）
├── tests/                      # 测试套件
├── examples/                   # 示例脚本（索引见 examples/README.md）
└── docs/                       # 文档
```

## 支持的机器人

| 机器人 | 类型 | 自由度 | 状态 |
|--------|------|--------|------|
| Franka Emika Panda | 协作机械臂 | 7+1 | ✅ 支持（模型自动解析，见[配置指南](docs/guides/configuration.md)） |
| Universal Robots UR5 | 工业机械臂 | 6 | ✅ 支持（模型自动解析） |
| 移动操作机器人 | 移动底盘 + 机械臂 | 10+ | 🚧 实验性支持 |

### 人形机器人（实验性）

以下人形机器人由 `plugins/` 下的环境/控制器插件支持（G1 有完整的关节配置、PD
增益和力矩限制定义）：

| 机器人 | 自由度 | 插件入口 | 状态 |
|--------|--------|----------|------|
| Unitree G1 | 29 | `plugins/controllers/wbc_lab/`、`plugins/envs/table_tennis/`、`plugins/predictors/bfm_zero/` | 🚧 实验性 |
| Unitree H1 | — | `plugins/controllers/hugwbc/`（全身控制 + PPO） | 🚧 实验性 |
| Fourier GR1 | 32 | `plugins/datasets/dreamdojo/`（数据管线，见 `examples/migration/dreamdojo_example.py`） | 🚧 实验性 |
| OpenLoong | — | `plugins/controllers/openloong/`（步行控制） | 🚧 实验性 |

> **注意**：人形机器人模型资产暂未随仓库分发，运行时自动回退到 Genesis 内置的
> 21 自由度 `humanoid.xml` 占位模型；插件中的关节/任务配置按上表对应的真实机型
> 编写。可运行示例：`plugins/controllers/hugwbc/examples/basic_usage.py`、
> `plugins/envs/humanoid_falling/`。

## 支持的任务

- **抓取放置（Pick and Place）** - 抓取物体并放置到目标位置
- **导航（Navigation）** - 在避开障碍物的同时到达目标位置
- **到达（Reach）** - 将末端执行器移动到目标位姿

## 多物理场场求解器

`plugins/solvers/` 提供三个结构一致的网格场求解器插件，在 `scene.build()` 前
`install()` 注入、随仿真循环一起积分（[plugins/solvers/README.md](plugins/solvers/README.md)）：

| 求解器 | 物理 | 数值方法 | 亮点 |
|---|---|---|---|
| `thermal` | 热传导 | 显式 FTCS + 能量守恒实体耦合 | 刚体-网格双向换热，GB 标准器件（马弗炉）底层模型 |
| `joule_heating` | 焦耳热 | Jacobi 电势求解 → `Q = σ\|∇V\|²` | 可注入 thermal 求解器（`couple_to_thermal=True`），材料参数可微 |
| `acoustics` | 线性声学 | leapfrog 波动方程 + CFL 校验 | 海绵层/刚性壁边界、单极子声源、刚体振动发声（单向流固耦合）、虚拟麦克风 + SPL 频谱后处理 |

```python
import genesis as gs
from plugins.solvers.acoustics import AcousticsOptions, install

gs.init(backend=gs.cpu)
scene = gs.Scene(sim_options=gs.options.SimOptions(dt=4e-6), show_viewer=False)
scene.add_entity(gs.morphs.Plane())

acoustics = install(scene, AcousticsOptions(resolution=(200, 200), dx=0.0025))
acoustics.add_source(position=(0.5, 0.5, 0.0),
                     signal=lambda t: 5.0 * __import__("math").sin(2 * 3.1416 * 200 * t))
mic = acoustics.add_probe((0.7, 0.5, 0.0))
scene.build()
for _ in range(1200):
    scene.step()
print(mic.spl(dt=4e-6))  # 整体声压级 (dB)
```

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
- Genesis World 1.4+
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

- 📧 邮箱: guoweist@foxmail.com
- 💬 讨论区: [GitHub Discussions](https://github.com/your-org/cloud-robotics-sim/discussions)
- 🐛 问题反馈: [GitHub Issues](https://github.com/your-org/cloud-robotics-sim/issues)
