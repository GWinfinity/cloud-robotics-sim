# sky Plugin

来源: [genesis-sky](../../genesis-sky)

Genesis: 通用物理引擎和机器人仿真平台。

## 核心功能

### 1. 多物理求解器
Genesis 集成了多种物理求解器：

| 求解器 | 描述 | 适用场景 |
|--------|------|----------|
| Rigid | 刚体和关节体 | 机器人、机械臂 |
| MPM | 材料点法 | 可变形物体、颗粒材料 |
| SPH | 光滑粒子流体 | 液体、气体 |
| FEM | 有限元法 | 弹性体、软体机器人 |
| PBD | 基于位置的动力学 | 布料、绳索 |
| Stable Fluid | 稳定流体 | 烟雾、火焰 |

### 2. 高性能
- **速度**: 单RTX 4090上Franka机器人超过4300万FPS
- **跨平台**: Linux, macOS, Windows
- **多后端**: CPU, NVIDIA CUDA, AMD ROCm, Apple Metal

### 3. 可微性
- MPM求解器和Tool Solver支持可微仿真
- 支持梯度传播和优化

### 4. 照片级渲染
- 实时光追渲染
- PBR材质系统
- 全局光照

## 快速开始

### 基础场景

```python
from cloud_robotics_sim.plugins.envs.sky import Scene

# 创建场景
scene = Scene()

# 添加地面
plane = scene.add_entity(
    morph=gs.morphs.Plane(),
    material=gs.materials.Rigid()
)

# 添加球体
sphere = scene.add_entity(
    morph=gs.morphs.Sphere(radius=0.1),
    material=gs.materials.Rigid(),
    pos=(0, 0, 1.0)
)

# 构建场景
scene.build()

# 运行仿真
for _ in range(1000):
    scene.step()
```

### 机器人仿真

```python
# 加载机器人
robot = scene.add_entity(
    morph=gs.morphs.MJCF(file='xml/franka_emika_panda/panda.xml'),
    pos=(0, 0, 0.5)
)

# 控制关节
robot.control_dofs_position([0.5, -0.5, 0, 0, 0, 0, 0])
```

### 可变形物体 (MPM)

```python
# 创建可变形物体
elastic_cube = scene.add_entity(
    morph=gs.morphs.Box(size=(0.1, 0.1, 0.1)),
    material=gs.materials.MPM.Elastic(
        E=1e5,  # 杨氏模量
        nu=0.3  # 泊松比
    )
)
```

### 流体仿真 (SPH)

```python
# 创建流体
water = scene.add_entity(
    morph=gs.morphs.Box(size=(0.5, 0.5, 0.5)),
    material=gs.materials.SPH.Liquid(
        density=1000,
        viscosity=0.01
    )
)
```

## 算法原理

### 统一物理框架

```
Genesis 架构:
├── Scene (场景)
│   ├── 实体管理
│   ├── 求解器协调
│   └── 渲染管理
├── Solvers (求解器)
│   ├── Rigid Solver
│   ├── MPM Solver
│   ├── SPH Solver
│   ├── FEM Solver
│   └── PBD Solver
├── Materials (材质)
│   └── 物理属性定义
└── Rendering (渲染)
    ├── 光栅化
    └── 光追
```

### MPM (Material Point Method)

```python
# MPM 双网格方法
# 粒子 → 网格 → 粒子

# 1. 粒子到网格 (P2G)
# 将粒子属性转移到背景网格

# 2. 网格更新
# 在网格上求解动量方程

# 3. 网格到粒子 (G2P)
# 将网格速度插值回粒子

# 应用: 雪、沙子、弹性体
```

### SPH (Smoothed Particle Hydrodynamics)

```python
# 无网格拉格朗日方法
# 粒子间通过核函数交互

# 密度估计
rho_i = sum(m_j * W(r_ij, h))

# 压力计算
P = k * (rho - rho_0)

# 应用: 水、油、血液
```

## 配置参数

### 场景配置

```yaml
# scene_config.yaml
scene:
  sim_options:
    dt: 0.01
    substeps: 10
  
  viewer_options:
    camera_pos: [3, 3, 3]
    camera_lookat: [0, 0, 0]
    res: [1280, 720]

solvers:
  rigid:
    enabled: true
    iterations: 10
  
  mpm:
    enabled: false
    grid_resolution: 64
  
  sph:
    enabled: false
    particle_radius: 0.01
```

### 渲染配置

```yaml
# render_config.yaml
render:
  renderer: ray_tracing  # 或 rasterization
  
  ray_tracing:
    samples_per_pixel: 128
    max_bounces: 4
  
  lighting:
    ambient: [0.1, 0.1, 0.1]
    sun_direction: [1, -1, -1]
```

## 示例

见 [examples/](examples/) 目录：

- `differentiable_push.py` - 可微仿真示例
- `elastic_dragon.py` - 可变形物体
- `pbd_liquid.py` - 基于位置的动力学液体
- `smoke.py` - 烟雾仿真

## 与其他仿真器对比

| 特性 | Genesis | PyBullet | MuJoCo | Isaac Gym |
|------|---------|----------|--------|-----------|
| 速度 | 4300万FPS | 低 | 中 | 高 |
| 多物理 | ✅ | ❌ | ❌ | ❌ |
| 可微性 | 部分 | ❌ | ❌ | 部分 |
| 光追渲染 | ✅ | ❌ | ❌ | ❌ |
| 开源 | ✅ | ✅ | ✅ | ❌ |

## 引用

```bibtex
@software{genesis2024,
  title={Genesis: A Generative Physics Engine for General Purpose Robotics},
  author={Genesis Embodied AI Team},
  year={2024}
}
```

## 相关链接

- [官网](https://genesis-embodied-ai.github.io/)
- [文档](https://genesis-world.readthedocs.io/)
- [GitHub](https://github.com/Genesis-Embodied-AI/Genesis)

## Changelog

- **2026-03-13**: 从 genesis-sky 迁移到 genesis-cloud-sim plugins
- **2026-01-08**: Genesis v0.3.11 发布
