# Genesis-Cloud-Sim 迁移至 MT Lambda 平台技术方案评估

> 评估日期：2026-06-11
> 评估对象：`genesis-cloud-sim` → 摩尔线程 MT Lambda 全栈具身智能仿真平台
> 目标场景：具身应用 / MT Lambda 底座移植 / 行业解决方案 / VLA·WAM·BFM 大模型训练微调

---

## 一、双平台技术栈对比

| 维度 | Genesis-Cloud-Sim (当前) | MT Lambda (目标) |
|------|------------------------|-----------------|
| **GPU 架构** | NVIDIA CUDA (Taichi 后端) | 摩尔线程 MUSA (全功能 GPU) |
| **物理引擎** | Genesis (自研，Taichi 加速) | MuJoCo-Warp-musa / Newton-musa / AlphaCore |
| **渲染引擎** | Genesis 内置 (OpenGL) | MT Photon / 3DGS / MTAGR |
| **AI 框架** | PyTorch (CUDA) | Torch-MUSA + muSolver/muFFT |
| **并行仿真** | Genesis GPU batch (设计占位) | MuJoCo-Warp / MJX 批量并行 |
| **模型格式** | URDF / MJCF / glTF | URDF / MJCF (兼容) |
| **场景描述** | 程序化 `gs.morphs.*` | MuJoCo XML / 场景图 / USD |
| **控制接口** | `entity.control_dofs_position()` | `mjData.ctrl` / MJX 等效 API |
| **上层平台** | 自研 cloud-robotics-sim | MT Lambda-Lab + MT Lambda-Sim |

### 关键结论
- **兼容点**：两者都支持 URDF/MJCF 机器人描述格式，模型资产可复用。
- **差异点**：场景构建 API、物理状态查询 API、渲染 API、并行化范式完全不同。
- **最核心差异**：Genesis 使用自研物理引擎 + 特有高级 API；MT Lambda 基于 MuJoCo 生态 + Warp GPU 加速。

---

## 二、代码耦合度量化分析

### 2.1 Genesis 依赖分布

```
src/cloud_robotics_sim/
├── core/
│   ├── composer.py      ████████████████████  direct gs.init, gs.Scene, gs.options
│   ├── scene.py         ████████████████████  gs.morphs.*, gs.surfaces.*, gs.lights.*
│   ├── embodiment.py    ████████████████████  gs.morphs.MJCF/URDF, entity.control_*
│   ├── vectorized.py    ████████████          gs.init(backend=gs.backends.CUDA)
│   └── task.py          ████                  仅通过 Scene/Robot 间接使用 (低耦合)
├── utils/
│   ├── genesis_compat.py ████████████████████ 722行，纯Genesis工具集
│   ├── rendering.py      ████████████████████ 直接调用Genesis渲染
│   └── camera.py         ████████████████     Genesis Viewer/相机API
└── plugins/ (13个插件)
    └── 7086处 gs.* 引用, 20+文件直接 import genesis
```

### 2.2 耦合严重度分级

| 模块 | 严重度 | 说明 |
|------|--------|------|
| `core/scene.py` | 🔴 **极高** | 房间结构、物体生成、光照全部调用 `gs.morphs`/`gs.surfaces`/`gs.lights` |
| `core/embodiment.py` | 🔴 **极高** | 机器人加载(`gs.morphs.MJCF/URDF`)、关节控制(`control_dofs_position`)、状态读取(`get_qpos`) |
| `core/composer.py` | 🔴 **极高** | 引擎初始化(`gs.init`)、场景创建(`gs.Scene`)、仿真步进(`gs_scene.step`) |
| `utils/genesis_compat.py` | 🔴 **极高** | 物体查询、URDF解析、状态提取、接触处理，全部基于Genesis内部数据结构 |
| `utils/rendering.py` | 🟠 **高** | 材质、光追、纹理、录屏依赖Genesis渲染系统 |
| `utils/camera.py` | 🟠 **高** | Viewer创建、相机pose、射线生成依赖Genesis |
| `core/task.py` | 🟡 **中** | 通过 `hasattr(entity, "get_pos")` 弱耦合，但假设了Genesis实体接口 |
| `core/vectorized.py` | 🟡 **中** | 目前为占位实现，但设计意图依赖Genesis GPU batch API |
| `plugins/*` | 🔴 **极高** | 7086处引用，大量直接调用Genesis求解器(RigidSolver/MPMSolver等) |

---

## 三、核心问题：能不能"只替换底层"？

### 直接回答：不能直接"只替换底层"，但可以通过**引入 Backend 抽象层**实现等效效果。

### 原因分析

当前 `genesis-cloud-sim` **没有物理后端抽象层**。Scene、RobotEmbodiment、Task 等虽有抽象基类(ABC)，但：

1. **ABC 只定义了高层业务接口**，没有定义仿真底层接口（如"如何创建一个Box"、"如何加载URDF"、"如何执行位置控制"）
2. **所有子类的实现层直接嵌入 Genesis API**，例如：
   ```python
   # scene.py 第249行
   floor = self.scene.add_entity(
       morph=gs.morphs.Box(size=(width, depth, thickness), ...),
       surface=gs.surfaces.Default(color=...)
   )
   
   # embodiment.py 第184行
   self.entity = scene.add_entity(
       morph=gs.morphs.MJCF(file="franka_emika_panda/panda.xml")
   )
   
   # embodiment.py 第220行
   self.entity.control_dofs_position(scaled_action[:-1])
   ```

3. **类型注解都写死了 `gs.Scene`**：
   ```python
   def build(self, gs_scene: gs.Scene) -> Scene: ...
   def spawn(self, scene: gs.Scene, position=...) -> RobotEmbodiment: ...
   ```

### 如果要"只替换底层"，必须满足的前提

| 前提条件 | 当前状态 | 改造工作量 |
|---------|---------|-----------|
| 物理后端抽象接口 | ❌ 不存在 | 需新增 `Backend` / `Simulator` ABC |
| 实体抽象接口 | ❌ 不存在 | 需新增 `Entity` / `Articulation` ABC |
| 场景构建与引擎解耦 | ❌ 深度耦合 | 需将 `gs.morphs.*` 封装为通用原语 |
| 渲染后端抽象 | ❌ 不存在 | 需新增 `Renderer` ABC |
| 插件隔离层 | ❌ 插件直接调用 Genesis | 需新增插件适配规范 |

**结论**：必须先做"中间加一层"的架构改造，才能"只替换底层"。

---

## 四、迁移难度综合评估

### 4.1 总体难度：🔴 高 (8/10)

| 维度 | 评分 | 说明 |
|------|------|------|
| 物理引擎迁移 | 9/10 | Genesis自研引擎 ↔ MuJoCo-Warp，API范式完全不同 |
| 场景系统迁移 | 8/10 | Genesis程序化生成 ↔ MuJoCo XML/场景图，需重写构建逻辑 |
| 机器人控制迁移 | 7/10 | `control_dofs_position` ↔ `mjData.ctrl`，控制接口映射 |
| 渲染系统迁移 | 7/10 | Genesis OpenGL ↔ MT Photon/3DGS，材质/光照模型差异 |
| 并行仿真迁移 | 6/10 | 占位实现，可重新基于 MJX/Warp 设计，反而灵活 |
| 大模型训练链路 | 5/10 | Torch-MUSA 替代 PyTorch-CUDA，模型代码改动有限 |
| 插件生态迁移 | 9/10 | 7086处引用，13个插件需逐一审查和适配 |
| 资产复用度 | 3/10 | URDF/MJCF 模型可复用，场景、材质、纹理需重做 |

### 4.2 工作量估算（人月）

| 阶段 | 内容 | 预估人月 |
|------|------|---------|
| **Phase 1: 抽象层建设** | 设计 Backend/Entity/Renderer ABC，重构 core/ 接口 | 4-6 |
| **Phase 2: Genesis Adapter** | 将现有逻辑下沉为 GenesisBackend，验证功能无损 | 3-4 |
| **Phase 3: MT Lambda Backend** | 实现 MTLambdaBackend（MuJoCo-Warp-musa + MT Photon） | 6-8 |
| **Phase 4: 工具库迁移** | genesis_compat.py → mt_compat.py，rendering/camera 适配 | 2-3 |
| **Phase 5: 插件适配** | 13个插件逐一评估，高价值插件优先迁移 | 8-12 |
| **Phase 6: 向量化实现** | 基于 MJX/Warp 实现真正的 GPU 并行环境 | 3-4 |
| **Phase 7: 大模型链路验证** | VLA/WAM/BFM 训练/微调/推理在 Torch-MUSA 上验证 | 4-6 |
| **Phase 8: 测试与对齐** | ABTestFramework 对比新旧实现，物理一致性验证 | 3-4 |
| **合计** | | **33-47 人月** |

> 注：若团队已有 MuJoCo/MJX 经验，可压缩至 25-35 人月；若 MT Lambda 提供 Python SDK/Adapter，可再压缩 30%。

---

## 五、技术方案设计

### 5.1 推荐架构：分层解耦 + Backend 插件化

```
┌─────────────────────────────────────────────────────────────┐
│  Application Layer                                          │
│  行业解决方案 / VLA·WAM·BFM 训练 / 策略部署                    │
├─────────────────────────────────────────────────────────────┤
│  Learning Layer (cloud_robotics_sim.learning)               │
│  RL / IL / VLA 训练框架 (Gymnasium 接口兼容)                  │
├─────────────────────────────────────────────────────────────┤
│  Core Layer (cloud_robotics_sim.core)                       │
│  EnvironmentComposer · Scene · RobotEmbodiment · Task        │
│  ── 业务逻辑保持不动，与引擎解耦 ──                             │
├─────────────────────────────────────────────────────────────┤
│  Backend Abstract Layer (cloud_robotics_sim.backend)  ← 新增  │
│  SimulatorBackend (ABC)                                     │
│  ├── create_scene() / add_entity() / add_light()            │
│  ├── load_urdf() / load_mjcf()                              │
│  ├── step() / reset()                                       │
│  └── render() / get_camera()                                │
│  EntityBackend (ABC)                                        │
│  ├── set_qpos() / get_qpos() / get_qvel()                   │
│  ├── set_pos() / get_pos()                                  │
│  └── control_dofs_position() / apply_force()                │
├─────────────────────────────────────────────────────────────┤
│  Backend Implementations (cloud_robotics_sim.backends)  ← 新增 │
│  ├── genesis_backend.py     # 现有逻辑下沉                  │
│  ├── mt_lambda_backend.py   # MuJoCo-Warp-musa + MT Photon  │
│  └── mujoco_backend.py      # 可选：标准 MuJoCo (调试/对比)  │
├─────────────────────────────────────────────────────────────┤
│  Physics Engine                                             │
│  Genesis Engine         MuJoCo-Warp-musa         Newton     │
│  (Taichi/CUDA)          (Warp/MUSA)              (MUSA)     │
├─────────────────────────────────────────────────────────────┤
│  GPU Hardware                                               │
│  NVIDIA CUDA                         摩尔线程 MUSA          │
└─────────────────────────────────────────────────────────────┘
```

### 5.2 详细改造方案

#### Step 1: 新增 Backend 抽象层 (4-6 周)

创建 `src/cloud_robotics_sim/backend/` 模块：

```python
# backend/base.py
from abc import ABC, abstractmethod
from typing import Any
import numpy as np

class SimulatorBackend(ABC):
    """物理仿真后端抽象接口."""
    
    @abstractmethod
    def initialize(self, use_gpu: bool = True, **kwargs) -> None: ...
    
    @abstractmethod
    def create_scene(self, dt: float, substeps: int, 
                     headless: bool = False, **kwargs) -> "SceneBackend": ...
    
    @abstractmethod
    def create_box(self, size, pos, quat=None, color=None, 
                   static=True) -> "EntityBackend": ...
    
    @abstractmethod
    def create_sphere(self, radius, pos, quat=None, 
                      color=None) -> "EntityBackend": ...
    
    @abstractmethod
    def load_urdf(self, file: str, pos, **kwargs) -> "EntityBackend": ...
    
    @abstractmethod
    def load_mjcf(self, file: str, pos, **kwargs) -> "EntityBackend": ...

class SceneBackend(ABC):
    """场景后端抽象接口."""
    
    @abstractmethod
    def add_entity(self, entity: "EntityBackend") -> None: ...
    
    @abstractmethod
    def add_light(self, light_type: str, **kwargs) -> None: ...
    
    @abstractmethod
    def step(self) -> None: ...
    
    @abstractmethod
    def build(self) -> None: ...

class EntityBackend(ABC):
    """实体后端抽象接口."""
    
    @abstractmethod
    def set_qpos(self, qpos: np.ndarray) -> None: ...
    
    @abstractmethod
    def get_qpos(self) -> np.ndarray: ...
    
    @abstractmethod
    def get_qvel(self) -> np.ndarray: ...
    
    @abstractmethod
    def set_pos(self, pos: tuple) -> None: ...
    
    @abstractmethod
    def get_pos(self) -> np.ndarray: ...
    
    @abstractmethod
    def control_dofs_position(self, targets: np.ndarray) -> None: ...
```

#### Step 2: 重构 core/ 模块解耦引擎 (4-6 周)

**`composer.py` 改造**：
```python
# 改造前
import genesis as gs
# gs.init(backend=gs.backends.CUDA)
# gs_scene = gs.Scene(...)

# 改造后
from cloud_robotics_sim.backend import get_backend

backend = get_backend("mt_lambda")  # or "genesis"
backend.initialize(use_gpu=True)
scene_backend = backend.create_scene(dt=0.01, substeps=10)
```

**`scene.py` 改造**：
```python
# ObjectSpawn.spawn() 改造前
morph = gs.morphs.Box(size=self.size, pos=self.position, quat=self.orientation)
surface = gs.surfaces.Default(color=self.color)
entity = scene.add_entity(morph=morph, surface=surface)

# 改造后
entity = backend.create_box(
    size=self.size, pos=self.position, quat=self.orientation, 
    color=self.color, static=self.static
)
scene.add_entity(entity)
```

**`embodiment.py` 改造**：
```python
# FrankaPanda.spawn() 改造前
self.entity = scene.add_entity(morph=gs.morphs.MJCF(file="..."), pos=pos)

# 改造后
self.entity = backend.load_mjcf(file="...", pos=pos)
```

#### Step 3: Genesis 后端下沉 (3-4 周)

将现有 Genesis 调用封装为 `GenesisBackend`、`GenesisSceneBackend`、`GenesisEntityBackend`：

```python
# backends/genesis_backend.py
import genesis as gs
from cloud_robotics_sim.backend.base import SimulatorBackend, SceneBackend, EntityBackend

class GenesisBackend(SimulatorBackend):
    def initialize(self, use_gpu=True, **kwargs):
        backend = gs.backends.CUDA if use_gpu else gs.backends.CPU
        gs.init(backend=backend)
    
    def create_scene(self, dt, substeps, headless=False, **kwargs):
        viewer_options = None if headless else gs.options.ViewerOptions(...)
        gs_scene = gs.Scene(
            viewer_options=viewer_options,
            sim_options=gs.options.SimOptions(dt=dt, substeps=substeps),
            show_viewer=not headless,
        )
        return GenesisSceneBackend(gs_scene)
    
    def create_box(self, size, pos, quat=None, color=None, static=True):
        morph = gs.morphs.Box(size=size, pos=pos, quat=quat)
        surface = gs.surfaces.Default(color=color) if color else None
        return GenesisEntityBackend(morph, surface, static)
    
    def load_mjcf(self, file, pos, **kwargs):
        morph = gs.morphs.MJCF(file=file, pos=pos, **kwargs)
        return GenesisEntityBackend(morph)
    
    def load_urdf(self, file, pos, **kwargs):
        morph = gs.morphs.URDF(file=file, pos=pos, **kwargs)
        return GenesisEntityBackend(morph)
```

#### Step 4: MT Lambda 后端实现 (6-8 周)

基于 MuJoCo-Warp-musa 和 MT Photon 实现：

```python
# backends/mt_lambda_backend.py
import mujoco
from mujoco import mjx  # MuJoCo JAX (或 MuJoCo-Warp-musa 等效接口)
from mt_lambda import mt_photon  # MT Lambda 渲染 SDK
from cloud_robotics_sim.backend.base import SimulatorBackend

class MTLambdaBackend(SimulatorBackend):
    """摩尔线程 MT Lambda 后端实现.
    
    物理: MuJoCo-Warp-musa (GPU加速, MUSA架构)
    渲染: MT Photon / 3DGS
    AI: Torch-MUSA
    """
    
    def __init__(self):
        self.model = None
        self.data = None
        self.renderer = None
    
    def initialize(self, use_gpu=True, **kwargs):
        # MuJoCo-Warp-musa 初始化
        if use_gpu:
            # 检测 MUSA 设备
            pass
    
    def create_scene(self, dt, substeps, headless=False, **kwargs):
        # 构建 MuJoCo XML 场景描述
        # 或使用 MuJoCo 场景图 API
        return MTLambdaSceneBackend(self, dt, substeps, headless)
    
    def create_box(self, size, pos, quat=None, color=None, static=True):
        # 创建 MuJoCo body + geom
        return MTLambdaEntityBackend("box", size=size, pos=pos, quat=quat)
    
    def load_urdf(self, file, pos, **kwargs):
        # MuJoCo 通过 mj_loadXML / URDF→MJCF 转换
        # 或使用 dm_control/mujoco 的 URDF 加载
        return MTLambdaEntityBackend("urdf", file=file, pos=pos)
    
    def load_mjcf(self, file, pos, **kwargs):
        # 直接加载 MJCF
        return MTLambdaEntityBackend("mjcf", file=file, pos=pos)
```

#### Step 5: 向量化环境重构 (3-4 周)

当前 `GenesisVectorizedEnv` 是占位实现。迁移到 MT Lambda 后，基于 **MJX (MuJoCo JAX)** 或 **MuJoCo-Warp** 实现真正的 GPU 批量并行：

```python
# MJX 天然支持 JAX vmap，可实现千级并行
import jax
from mujoco import mjx

class MTLambdaVectorizedEnv(VectorizedEnvironment):
    def __init__(self, config, scene_fn, robot_fn, task_fn):
        super().__init__(config)
        # 使用 MJX 的批量 API
        # self.batch_data = mjx.put_batch(self.data, self.num_envs)
    
    def step(self, actions):
        # JAX vmap 批量步进
        # obs, reward, done = jax.vmap(self._step_single)(actions)
        pass
```

**优势**：MJX 的 JAX-native 设计使得向量化比 Genesis 更自然，理论上可达 4096+ 并行。

#### Step 6: 大模型训练链路适配 (4-6 周)

| 环节 | Genesis (当前) | MT Lambda (目标) | 改造点 |
|------|---------------|-----------------|--------|
| 训练框架 | PyTorch (CUDA) | Torch-MUSA | `import torch_musa` 自动替换，模型代码基本不动 |
| 数据加载 | Genesis 仿真生成 | MuJoCo-Warp 生成 | 通过 Backend 抽象，数据格式统一 |
| VLA 模型 | PyTorch + Transformers | Torch-MUSA + Transformers | 注意力算子需验证 MUSA 兼容性 |
| RL 训练 | PPO/SAC (PyTorch) | PPO/SAC (Torch-MUSA) | 算法代码不动，device="musa:0" |
| Sim2Real | Genesis → 真机 | MuJoCo → 真机 |  domain randomization 参数需重标定 |

**关键风险点**：
- Torch-MUSA 对某些复杂 CUDA kernel 的兼容性需验证
- VLA 模型的自定义算子（如旋转位置编码）可能需要手写 MUSA kernel
- BFM (Behavior Foundation Model) 的大规模分布式训练需验证 MUSA 通信后端

---

## 六、三种迁移策略对比

| 策略 | 描述 | 工作量 | 风险 | 推荐度 |
|------|------|--------|------|--------|
| **A. 全量重写** | 弃用 genesis-cloud-sim，在 MT Lambda 上重新开发 | 60+ 人月 | 中 | ⭐⭐ |
| **B. 抽象层迁移** (推荐) | 保留 core/ 业务逻辑，新增 Backend 抽象层，实现 MT Lambda Backend | 33-47 人月 | 低 | ⭐⭐⭐⭐⭐ |
| **C. 兼容层桥接** | 在 MT Lambda 上实现 Genesis API 兼容层 (`gs.morphs.*` 等) | 40-55 人月 | 极高 | ⭐ |

### 策略 B（推荐）详细阶段规划

```
Month 1-2:   架构设计 + Backend ABC 定义 + 接口评审
Month 3-4:   GenesisBackend 实现 + core/ 重构 + 回归测试
Month 5-6:   MTLambdaBackend 骨架 + MuJoCo-Warp 集成 + 单场景验证
Month 7-8:   场景系统完善 + MT Photon 渲染集成 + 机器人控制映射
Month 9-10:  向量化实现 + 插件评估 + 高价值插件迁移
Month 11-12: VLA训练链路验证 + BFM微调验证 + 性能调优
Month 13-14: 行业解决方案封装 + 文档 + 开源/交付
```

---

## 七、面向三个应用方向的专项评估

### 7.1 方向一：MT Lambda 全栈具身智能仿真平台底座移植

| 评估项 | 结论 |
|--------|------|
| **可行性** | ✅ 可行，但需架构重构 |
| **核心工作** | Backend 抽象层 + MTLambdaBackend 实现 |
| **技术难点** | MuJoCo-Warp-musa 的 Python API 成熟度；MT Photon 与 MuJoCo 的相机数据对接 |
| **底座价值** | 一次投入后，所有上层应用自动获得 MUSA 国产化算力支持 |

### 7.2 方向二：基于 MT Lambda 的具身行业解决方案

| 评估项 | 结论 |
|--------|------|
| **可行性** | ✅ 可行，行业方案层改动最小 |
| **核心工作** | 行业方案主要调用 `EnvironmentComposer` + `Task` 接口，Backend 替换对其透明 |
| **技术难点** | 行业场景的高保真重建（从 Genesis 场景 → MuJoCo XML）；Domain Randomization 参数重标定 |
| **复用度** | Scene/Robot/Task 的业务逻辑可 90%+ 复用 |

### 7.3 方向三：VLA / WAM / BFM 类具身大模型训练/微调/移植

| 评估项 | 结论 |
|--------|------|
| **可行性** | ✅ 可行，且 MT Lambda 的 Torch-MUSA + MJX 对大模型训练更友好 |
| **核心工作** | ① 仿真环境 Backend 替换；② 训练脚本 device 迁移 CUDA→MUSA |
| **技术难点** | ① 大规模并行仿真（MJX vmap）的稳定性；② VLA 模型特定算子的 MUSA 兼容性；③ 训练 checkpoint 的跨平台兼容 |
| **优势** | MJX 的 JAX-native 设计天然支持大规模并行，比 Genesis 更适合大模型预训练的数据生成 |

---

## 八、关键风险与应对

| 风险 | 概率 | 影响 | 应对措施 |
|------|------|------|----------|
| MuJoCo-Warp-musa Python API 不成熟 | 中 | 高 | 预留标准 MuJoCo CPU backend 作为 fallback；与摩尔线程共建 |
| MT Photon 与 MuJoCo 相机数据格式不匹配 | 中 | 高 | 在 Backend 层做数据格式转换；使用 numpy bridge |
| Torch-MUSA 对某些模型算子不支持 | 中 | 高 | 提前跑通 VLA 模型算子清单；准备 custom kernel 手写方案 |
| 插件生态迁移工作量超预期 | 高 | 中 | 制定插件优先级矩阵，先迁移 controllers/predictors，datasets 延后 |
| 物理一致性差异导致策略失效 | 中 | 高 | 利用现有 ABTestFramework 做物理对齐验证；Domain Randomization 重标定 |
| 大规模并行性能不及预期 | 低 | 中 | MJX benchmark 先行；与 NVIDIA Isaac Sim / Genesis 做性能对标 |

---

## 九、总结与建议

### 核心结论

1. **不能直接"只替换底层"**，因为当前代码没有物理后端抽象层，Genesis API 深度嵌入业务逻辑。
2. **可以通过"加一层"实现等效效果**：新增 Backend 抽象层，将 Genesis 调用下沉，上层业务逻辑保持不动。
3. **迁移工作量约 33-47 人月**（14个月周期，3-4人团队），推荐策略 **B（抽象层迁移）**。
4. **三个应用方向均可行**，其中大模型训练方向在 MT Lambda 上反而可能获得更好的并行性能。

### 下一步行动建议

| 优先级 | 行动 | 负责人 | 时间 |
|--------|------|--------|------|
| P0 | 与摩尔线程确认 MT Lambda Python SDK / API 文档可用性 | 架构师 | 1周内 |
| P0 | 获取 MuJoCo-Warp-musa 早期版本，跑通 Hello World | 引擎组 | 2周内 |
| P1 | 设计 Backend ABC 接口，内部评审 | 架构师 | 3周内 |
| P1 | 选取 1个核心插件（如 hugwbc）做 PoC 验证 | 插件组 | 4周内 |
| P2 | 启动 core/ 重构，同步下沉 GenesisBackend | 核心组 | Month 2 开始 |
| P2 | VLA 模型（如 OpenVLA/π0）在 Torch-MUSA 上跑通推理 | 算法组 | Month 2-3 |

---

*本评估基于 `genesis-cloud-sim` 代码库（截至 2026-06）和公开 MT Lambda 技术资料编制。实际工作量取决于 MT Lambda SDK 的成熟度和摩尔线程的技术支持深度。*
