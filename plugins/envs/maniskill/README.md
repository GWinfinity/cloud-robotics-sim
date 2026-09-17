# maniskill Plugin

来源: [genesis-maniskill](../../genesis-maniskill)

基于 Genesis 物理引擎的机器人操作仿真平台，整合 RoboCasa 厨房场景和 ManiSkill API 设计。

## 核心功能

### 1. Genesis 后端
- 高性能 GPU 加速物理仿真
- 支持数千个环境并行训练
- 实时渲染和传感器模拟

### 2. RoboCasa 场景
**Kitchen 厨房环境**:
- 6种布局: G-shaped, U-shaped, L-shaped等
- 8种风格: 现代、工业、地中海等
- 2500+ 厨房物品

**TableTop 桌面环境**:
- 基础桌面操作
- 可自定义物体配置

**ReplicaCAD 公寓场景**（ManiSkill ReplicaCAD_SceneManipulation 等价）:
- 90 个真实公寓场景配置（`apt_0..5`、`v3_sc*_staging_*`）
- ModelScope 数据集（~289MB）自动下载解压
- 刚体物件 + 关节家具（冰箱、橱柜等 URDF）

### 3. ManiSkill API
- 熟悉的 Gymnasium 接口
- 易于从 ManiSkill2/3 迁移
- 标准化任务定义

### 4. 统一资产管理
支持从标准格式直接加载任意机器人：
```python
from cloud_robotics_sim.plugins.envs.maniskill import RobotLoader

# 从 URDF 加载
robot = RobotLoader.from_urdf(scene, "urdf/my_robot.urdf")

# 使用预设
robot = RobotLoader.from_preset(scene, "franka")
```

## 快速开始

### 基础使用

```python
from cloud_robotics_sim.plugins.envs.maniskill import KitchenEnv

# 创建厨房环境
env = KitchenEnv(
    scene_type="kitchen",
    robot="franka",
    task="pick_place",
    num_envs=16,
    headless=False
)

# 运行仿真
obs, info = env.reset()
for _ in range(100):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
```

### 桌面操作

```python
from cloud_robotics_sim.plugins.envs.maniskill import TableTopEnv

# 创建桌面环境
env = TableTopEnv(
    task="pick_cube",
    robot="ur5",
    num_envs=32
)

obs, info = env.reset()
# ... 训练代码
```

### ReplicaCAD 场景（ManiSkill ReplicaCAD_SceneManipulation 等价）

支持 ManiSkill 的 ReplicaCAD 公寓场景数据集（ModelScope 镜像
`jessy888/ManiSkill_replica_cad_dataset`，Apache-2.0，约 289MB，含 90 个场景配置：
`apt_0..apt_5` 与 `v3_sc*_staging_*`）。数据集不随仓库分发，首次使用时自动从
ModelScope 下载并解压到 `assets/maniskill/`（gitignored）；也可以手动预取：

```bash
python -m genesis_maniskill.datasets.replicacad_assets          # 下载 + 解压
python -m genesis_maniskill.datasets.replicacad_assets --check-only
```

环境变量：`CRS_REPLICACAD_ASSETS` 覆盖数据集根目录；
`CRS_REPLICACAD_AUTO_DOWNLOAD=0` 禁用自动下载（缺失时抛 `FileNotFoundError`）。

```python
from genesis_maniskill.envs.replica_cad_env import ReplicaCADEnv

# 与 ManiSkill 的 ReplicaCAD_SceneManipulation-v1 对应的场景
env = ReplicaCADEnv(
    scene_name="apt_0",        # 90 个场景之一，见 ReplicaCADSceneBuilder.list_available_scenes()
    robot_uid="franka",
    task_type="pick_place",
    scene_config={"include_doors": False},   # 与 ManiSkill 一致默认跳过门
)
obs, info = env.reset()
for _ in range(50):
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
env.close()
```

或直接查看场景（不需要机器人）：

```bash
python plugins/envs/maniskill/examples/replica_cad_scene.py --scene apt_0 --list-scenes
python plugins/envs/maniskill/examples/replica_cad_scene.py --scene apt_0 --save-img
python plugins/envs/maniskill/examples/replica_cad_scene.py --scene apt_0 --human   # 交互查看
```

实现要点（语义对齐 ManiSkill 的 `ReplicaCADSceneBuilder`）：

- 所有资产为 habitat Y-up 坐标，实例姿态统一左乘 RotX(+90°) 转到 Genesis Z-up；
  数据集四元数 (x,y,z,w) 转为 Genesis 的 (w,x,y,z)。
- `DYNAMIC` 物体用数据集中预分解的凸包碰撞 GLB（`objects/convex/*_cv_decomp.glb`）
  作为物理网格以避免昂贵的运行时凸分解，渲染表面为略 facet 化的原表面；
  `STATIC` 物体以原始 render GLB 非凸固定网格加入。质量经 `object_config.json` 的
  `mass` 在 `scene.build()` 后通过 `set_mass()` 落实。
- 关节家具从 `urdf/<name>/<name>.urdf` 加载（mesh 为同目录相对路径）。
- 目前仅支持 `num_envs=1`（GPU 并行场景构建待后续支持）。

### 自定义任务

```python
from cloud_robotics_sim.plugins.envs.maniskill import PickPlaceTask

# 创建自定义拾取放置任务
task = PickPlaceTask(
    env=env,
    object="apple",
    target_location="plate",
    success_threshold=0.05
)
```

## 支持的任务

### 基础操作任务
| 任务 | 描述 |
|------|------|
| pick_place | 拾取并放置物体 |
| open_drawer | 打开抽屉 |
| push | 推动物体到目标位置 |
| stack | 堆叠多个物体 |

### 厨房任务
| 任务 | 描述 |
|------|------|
| prepare_food | 多阶段食物准备 |
| cleanup | 清理台面物品 |
| organize_cabinet | 整理橱柜物品 |

### 桌面任务
| 任务 | 描述 |
|------|------|
| insert | 插入操作（如插入孔洞） |
| sort | 按颜色/类型分类 |
| assembly | 装配多个部件 |

### 移动操作
| 任务 | 描述 |
|------|------|
| mobile_manipulation | 导航+操作组合任务 |

## 支持的机器人

| 机器人 | 类型 | DOF |
|--------|------|-----|
| Franka Emika Panda | 协作臂 | 7 |
| Universal Robots UR5 | 工业臂 | 6 |
| Kinova Gen3 | 超轻量臂 | 7 |
| UFACTORY xArm | 轻量臂 | 5/6/7 |
| Fetch | 移动机械臂 | 8 |
| Unitree G1 | 人形机器人 | 29 |

## 算法原理

### 并行环境架构

```
Genesis ManiSkill
├── Scene Manager
│   ├── Kitchen/TableTop Scene
│   ├── Robot Assets
│   └── Object Assets
├── Task Controller
│   ├── Task Logic
│   ├── Reward Function
│   └── Success Criteria
├── Sensor System
│   ├── RGB Camera
│   ├── Depth Camera
│   └── Force/Torque
└── Vectorized Environment
    └── num_envs parallel instances
```

### 任务定义接口

```python
class BaseTask:
    def reset(self, env_idx):
        """重置任务状态"""
        pass
    
    def compute_reward(self, obs, action, info):
        """计算奖励"""
        pass
    
    def check_success(self, obs, info):
        """检查任务完成"""
        pass
    
    def get_observation(self, env_idx):
        """获取观测"""
        pass
```

## 配置参数

### 环境配置

```yaml
# kitchen_config.yaml
scene:
  type: kitchen
  layout_id: 0  # 0-5: G, U, L, etc.
  style_id: 0   # 0-7: modern, industrial, etc.
  
robot:
  type: franka
  init_joint_positions: [0, -0.785, 0, -2.356, 0, 1.571, 0.785]
  
task:
  type: pick_place
  object: apple
  target: plate
  
training:
  num_envs: 1024
  episode_length: 200
```

### 桌面配置

```yaml
# tabletop_config.yaml
scene:
  type: tabletop
  table_size: [1.0, 0.6, 0.02]
  
objects:
  - type: cube
    color: red
    size: 0.05
    position: random
  - type: cylinder
    color: blue
    size: [0.03, 0.08]
    position: random
```

## 数据集迁移

### RoboCasa 数据集转换

```bash
python -m cloud_robotics_sim.plugins.envs.maniskill.scripts.convert_robocasa \
    --source /path/to/robocasa_dataset.hdf5 \
    --target /path/to/output
```

### ManiSkill 数据集转换

```bash
python -m cloud_robotics_sim.plugins.envs.maniskill.scripts.convert_maniskill \
    --source /path/to/maniskill_data \
    --target /path/to/output
```

### 使用转换后的数据

```python
from cloud_robotics_sim.plugins.envs.maniskill import TrajectoryDataset

# 加载转换后的数据集
dataset = TrajectoryDataset.load("/path/to/converted_dataset")

# 转换为 PyTorch DataLoader
torch_dataset = dataset.to_torch_dataset(obs_key="state")
dataloader = DataLoader(torch_dataset, batch_size=32)
```

## 示例

见 [examples/](examples/) 目录：

- `basic_kitchen.py` - 厨房环境基础示例
- `basic_tabletop.py` - 桌面环境基础示例
- `advanced_task_example.py` - 高级任务示例
- `train_with_converted_data.py` - 使用迁移数据训练
- `demo_all_robots.py` - 演示所有机器人
- `demo_all_tasks.py` - 演示所有任务

## 迁移指南

### 从 RoboCasa 迁移

```python
# RoboCasa (旧)
from robocasa.environments import KitchenEnv
env = KitchenEnv(layout_id=0, style_id=0)

# Genesis ManiSkill (新)
from cloud_robotics_sim.plugins.envs.maniskill import KitchenEnv
env = KitchenEnv(scene_type="kitchen", layout_id=0, style_id=0)
```

### 从 ManiSkill 迁移

```python
# ManiSkill (旧)
import gymnasium as gym
env = gym.make("PickCube-v1", num_envs=16)

# Genesis ManiSkill (新)
from cloud_robotics_sim.plugins.envs.maniskill import TableTopEnv
env = TableTopEnv(task="pick_cube", num_envs=16)
```

## 性能指标

| 指标 | Genesis ManiSkill | ManiSkill3 | RoboCasa |
|------|-------------------|------------|----------|
| 并行环境数 | 4096+ | 2048 | 256 |
| FPS (1024 envs) | 50000+ | 30000 | 5000 |
| 渲染速度 | 实时 | 实时 | 离线 |

## 引用

```bibtex
@article{genesis2024,
  title={Genesis: A Generative and Universal Physics Engine 
         for Robotics and Beyond},
  author={Genesis Authors},
  year={2024}
}

@inproceedings{robocasa2024,
  title={RoboCasa: Large-Scale Simulation of Everyday Tasks 
         for Generalist Robots},
  author={Nasiriany, Soroush and others},
  booktitle={RSS},
  year={2024}
}

@article{maniskill3,
  title={ManiSkill3: GPU Parallelized Robotics Simulation and 
         Rendering for Generalizable Embodied AI},
  author={Tao, Stone and others},
  journal={RSS},
  year={2025}
}
```

## Changelog

- **2026-03-13**: 从 genesis-maniskill 迁移到 genesis-cloud-sim plugins
- **2026-02-15**: 初始版本，整合 RoboCasa 和 ManiSkill
