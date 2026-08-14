# Backend ABC 接口设计方案：genesis-cloud-sim → MT Lambda

> **选定策略：路径 A — 保留 Genesis 物理引擎，通过 Quadrants/MUSA backend 适配 MT Lambda。**
>
> 基于 P0 调研结果（torch_musa / mujoco_warp_musa）与当前代码库耦合分析。

## 一、目标与范围

### 1.1 目标
为 `genesis-cloud-sim` 引入**物理/渲染后端抽象层（Backend ABC）**，实现：
1. **解耦**：`core/` 业务逻辑与 Genesis API 解耦。
2. **可替换**：同一套 Scene/Robot/Task 代码可同时跑在 `GenesisBackend`（CUDA）和 `MTLambdaBackend`（MUSA）上。
3. **性能提升**：MT Lambda 版本在关键指标上**比当前 CUDA/Genesis 版本提升 ≥ 10%**。
4. **可验证**：复用现有 `ABTestFramework`，对新旧后端做物理一致性与性能回归测试。

### 1.2 范围
- 新增 `src/cloud_robotics_sim/backend/` 抽象层。
- 重构 `core/composer.py`、`core/scene.py`、`core/embodiment.py` 的引擎相关代码。
- 新增 `backends/genesis_backend.py`（保留现有 CUDA 能力）。
- 新增 `backends/mt_lambda_backend.py`（Genesis/Quadrants MUSA backend 适配层 + `torch_musa`）。
- 适配 `core/vectorized.py`，基于 Genesis/Quadrants 的并行能力实现 GPU 批量并行。
- 本次**不涉及**全部 13 个插件迁移，仅对 `core/` 解耦；插件后续通过统一的 `Backend` API 适配。

---

## 二、上游技术栈调研

### 2.1 torch_musa（https://github.com/MooreThreads/torch_musa）

| 维度 | 结论 |
|------|------|
| API 兼容性 | 与 PyTorch API 格式一致，只需将 `cuda` 替换为 `musa` |
| 典型用法 | `torch.tensor(..., device='musa')`、`torch.musa.is_available()` |
| 分布式 | 使用 `mccl` backend（S4000），S80/S3000 不支持 MCCL |
| 依赖 | MUSA-SDK、muDNN、MCCL（S4000）、muThrust、muAlg |
| 生态 | 已 musified：torchvision、torchaudio、pytorch3d、Transformers、Accelerate 等 |
| Docker | 提供 `registry.mthreads.com/.../musa-pytorch-release-public:latest` |
| 风险 | 复杂自定义 CUDA kernel、自定义算子可能需要手写 MUSA kernel |

**对项目影响**：训练/推理代码改动很小，主要是 `device="musa:0"` 和 `dist.init_process_group("mccl", ...)`。

### 2.2 Genesis/Quadrants 技术栈澄清

**Genesis 物理引擎不是基于 PyTorch，而是基于 Taichi/Quadrants。**

- **Quadrants** 是 Genesis AI 从 Taichi fork 出来的 Python-to-GPU 编译器，负责把 Python kernel JIT 编译到 GPU。
- Quadrants 当前公开支持的后端包括：**CUDA、ROCm、Metal、Vulkan、x86/ARM64 CPU**。
- 截至 `genesis-world 1.2.2`，**Quadrants 尚未公开支持 MUSA**；backend 枚举只有 `cpu/cuda/gpu/metal/amdgpu`。

这意味着：
- `torch_musa` 只能覆盖**训练/推理链路**，无法替代物理引擎。
- 要让 Genesis 跑在 MUSA 上，必须有 **Quadrants MUSA backend**（路径 A），或切换物理引擎（路径 B）。
- 路径 A 的优势：保留 Genesis 自研物理引擎、场景 API、渲染管线和现有资产，迁移成本最低。

### 2.3 mujoco_warp_musa（备选方案）

| 维度 | 结论 |
|------|------|
| 来源 | Fork 自 Google DeepMind `mujoco_warp`，添加 MUSA 计算后端 |
| API | 与 `mujoco_warp` 保持一致；导入为 `import mujoco_warp.musa_api as mjw` |
| 精度 | 与 MuJoCo Warp CPU 版本单步 qpos 相对误差 ≤ 1e-5 |
| 支持 GPU | MTT S4000 / S5000 |
| 依赖 | MUSA SDK 4.3.4、muThrust、Python ≥ 3.10、CMake ≥ 3.18 |
| 构建 | 支持 `uv sync` / `uv run`；可增量编译 `python build_lib.py` |
| 可视化 | 可用 `viewer_musa.py` 或 Mesa 软渲染 |
| 测试 | 提供 `mujoco_warp/_src/mujoco_musa/tests/all_test.py` |

**作为备选方案**：
- 如果 Quadrants 长期无法支持 MUSA，或 MT Lambda 官方更推荐 MuJoCo-Warp-MUSA，可切换到路径 B。
- 路径 B 需要重写场景构建、机器人控制和向量化逻辑，工作量大。

---

## 三、当前代码库耦合分析

### 3.1 Genesis 依赖热力图

| 模块 | 严重度 | 主要 Genesis 依赖 |
|------|--------|-------------------|
| `core/composer.py` | 🔴 极高 | `gs.Scene`, `gs.options.*`, `scene.build()`, `scene.step()` |
| `core/scene.py` | 🔴 极高 | `gs.morphs.Box/Sphere/Cylinder/Mesh`, `gs.surfaces.Default`, `scene.add_entity()`, `gs.lights.*` |
| `core/embodiment.py` | 🔴 极高 | `gs.morphs.MJCF/URDF`, `entity.set_qpos`, `entity.get_qpos`, `entity.control_dofs_position` |
| `utils/genesis_compat.py` | 🔴 极高 | `gs.init`, backend 选择、`gs.materials.Rigid`、`gs.lights` |
| `utils/rendering.py` | 🟠 高 | `gs.RenderBodyComponent`, `gs.render.*`, `gs.Texture` |
| `utils/camera.py` | 🟠 高 | `gs.Viewer`, `gs.render.*` |
| `core/task.py` | 🟡 中 | 仅通过 `entity.get_pos()` 弱耦合 |
| `core/vectorized.py` | 🟡 中 | 目前为 stub，依赖 `ensure_genesis_initialized` |

### 3.2 关键抽象缺失
- 当前 `Scene(ABC)` / `RobotEmbodiment(ABC)` / `Task(ABC)` 只定义了**业务层**抽象，没有定义**仿真原语**抽象。
- 没有 `SimulatorBackend` / `EntityBackend` / `RendererBackend` 概念。
- 类型注解写死 `gs.Scene`，导致无法注入其他后端。

---

## 四、Backend ABC 设计方案

### 4.1 架构分层

```
┌─────────────────────────────────────────────────────────────┐
│  Application / Learning / Plugins                           │
├─────────────────────────────────────────────────────────────┤
│  Core Layer (cloud_robotics_sim.core)                       │
│  EnvironmentComposer · Scene · RobotEmbodiment · Task        │
│  ── 业务逻辑保持不动，只调用 Backend 接口 ──                   │
├─────────────────────────────────────────────────────────────┤
│  Backend Abstract Layer (cloud_robotics_sim.backend)         │
│  SimulatorBackend · SceneBackend · EntityBackend             │
│  ArticulationBackend · RendererBackend · SensorBackend       │
├─────────────────────────────────────────────────────────────┤
│  Backend Implementations                                     │
│  ├── backends/genesis_backend.py     (Quadrants/CUDA)       │
│  └── backends/mt_lambda_backend.py   (Quadrants/MUSA)       │
├─────────────────────────────────────────────────────────────┤
│  Physics Engine + Compiler                                   │
│  Genesis Engine + Quadrants   →   MUSA backend (目标)        │
│  (当前：CUDA backend)                                        │
└─────────────────────────────────────────────────────────────┘
```

### 4.2 模块结构

```
src/cloud_robotics_sim/
├── backend/                         # 抽象层
│   ├── __init__.py                  # get_backend(), register_backend()
│   ├── base.py                      # ABC 定义
│   ├── factory.py                   # 后端工厂与全局注册表
│   └── types.py                     # 共享类型/枚举
└── backends/                        # 具体实现
    ├── __init__.py
    ├── genesis_backend.py           # GenesisBackend (Quadrants/CUDA)
    └── mt_lambda_backend.py         # MTLambdaBackend (Quadrants/MUSA，继承 GenesisBackend)
```

### 4.3 核心 ABC 接口

#### 4.3.1 `SimulatorBackend`

```python
class SimulatorBackend(ABC):
    """物理仿真后端抽象接口。"""

    @property
    @abstractmethod
    def name(self) -> BackendName: ...

    @abstractmethod
    def initialize(self, *, headless: bool = True, device: str = "musa", **kwargs) -> None: ...

    @abstractmethod
    def create_scene(
        self,
        *,
        dt: float,
        substeps: int,
        headless: bool = True,
        viewer_options: ViewerOptions | None = None,
    ) -> SceneBackend: ...

    @abstractmethod
    def create_box(
        self,
        size: tuple[float, float, float],
        pos: tuple[float, float, float],
        quat: tuple[float, float, float, float] | None = None,
        color: tuple[float, float, float, float] | None = None,
        static: bool = True,
        friction: float = 0.5,
        density: float | None = None,
        name: str | None = None,
    ) -> EntityBackend: ...

    @abstractmethod
    def create_sphere(...) -> EntityBackend: ...

    @abstractmethod
    def create_cylinder(...) -> EntityBackend: ...

    @abstractmethod
    def create_mesh(
        self,
        file: str,
        pos: tuple[float, float, float],
        quat: tuple[float, float, float, float] | None = None,
        scale: tuple[float, float, float] | None = None,
        static: bool = True,
        name: str | None = None,
    ) -> EntityBackend: ...

    @abstractmethod
    def load_mjcf(
        self,
        file: str,
        pos: tuple[float, float, float],
        **kwargs,
    ) -> ArticulationBackend: ...

    @abstractmethod
    def load_urdf(
        self,
        file: str,
        pos: tuple[float, float, float],
        **kwargs,
    ) -> ArticulationBackend: ...

    @abstractmethod
    def create_light(
        self,
        light_type: LightType,
        **kwargs,
    ) -> LightBackend: ...
```

#### 4.3.2 `SceneBackend`

```python
class SceneBackend(ABC):
    """场景后端抽象接口。"""

    @abstractmethod
    def add_entity(self, entity: EntityBackend) -> None: ...

    @abstractmethod
    def add_articulation(self, articulation: ArticulationBackend) -> None: ...

    @abstractmethod
    def add_light(self, light: LightBackend) -> None: ...

    @abstractmethod
    def build(self) -> None: ...

    @abstractmethod
    def step(self) -> None: ...

    @abstractmethod
    def reset(self) -> None: ...

    @abstractmethod
    def get_physics_state(self) -> PhysicsState: ...

    @abstractmethod
    def set_physics_state(self, state: PhysicsState) -> None: ...
```

#### 4.3.3 `EntityBackend`

```python
class EntityBackend(ABC):
    """普通刚体实体抽象接口（静态/动态物体）。"""

    @property
    @abstractmethod
    def name(self) -> str | None: ...

    @abstractmethod
    def get_pos(self) -> np.ndarray: ...

    @abstractmethod
    def set_pos(self, pos: np.ndarray) -> None: ...

    @abstractmethod
    def get_quat(self) -> np.ndarray: ...

    @abstractmethod
    def set_quat(self, quat: np.ndarray) -> None: ...

    @abstractmethod
    def set_color(self, color: tuple[float, float, float, float]) -> None: ...

    @abstractmethod
    def apply_force(self, force: np.ndarray, pos: np.ndarray | None = None) -> None: ...
```

#### 4.3.4 `ArticulationBackend`

```python
class ArticulationBackend(ABC):
    """关节机器人实体抽象接口。继承 EntityBackend 的位姿能力。"""

    @property
    @abstractmethod
    def n_dofs(self) -> int: ...

    @property
    @abstractmethod
    def n_qs(self) -> int: ...

    @abstractmethod
    def get_qpos(self) -> np.ndarray: ...

    @abstractmethod
    def set_qpos(self, qpos: np.ndarray) -> None: ...

    @abstractmethod
    def get_qvel(self) -> np.ndarray: ...

    @abstractmethod
    def set_qvel(self, qvel: np.ndarray) -> None: ...

    @abstractmethod
    def control_dofs_position(
        self,
        targets: np.ndarray,
        stiffness: np.ndarray | None = None,
        damping: np.ndarray | None = None,
    ) -> None: ...

    @abstractmethod
    def control_dofs_velocity(self, targets: np.ndarray) -> None: ...

    @abstractmethod
    def control_dofs_force(self, targets: np.ndarray) -> None: ...

    @abstractmethod
    def get_end_effector_pose(self) -> Pose: ...
```

#### 4.3.5 `RendererBackend` / `SensorBackend`

```python
class RendererBackend(ABC):
    """渲染后端抽象接口。"""

    @abstractmethod
    def render(
        self,
        camera_name: str | None = None,
        *,
        rgb: bool = True,
        depth: bool = False,
        segmentation: bool = False,
    ) -> RenderOutput: ...

    @abstractmethod
    def add_camera(
        self,
        name: str,
        pos: tuple[float, float, float],
        lookat: tuple[float, float, float],
        resolution: tuple[int, int],
        fov: float = 60.0,
    ) -> CameraBackend: ...

class CameraBackend(ABC):
    @abstractmethod
    def render(self, *, rgb: bool = True, depth: bool = False) -> np.ndarray | tuple[np.ndarray, ...]: ...
```

### 4.4 `core/` 改造示例

#### `EnvironmentComposer.compose()` 改造前/后

```python
# 改造前
import genesis as gs
ensure_genesis_initialized(headless=self.config.headless)
gs_scene = gs.Scene(
    viewer_options=gs.options.ViewerOptions(...),
    sim_options=gs.options.SimOptions(dt=..., substeps=...),
    show_viewer=not self.config.headless,
)
scene.build(gs_scene)
robot.spawn(gs_scene, position=spawn_pos)
gs_scene.build()

# 改造后
from cloud_robotics_sim.backend import get_backend

backend = get_backend("mt_lambda")  # or "genesis"
backend.initialize(headless=self.config.headless, device="musa")
scene_backend = backend.create_scene(
    dt=self.config.dt,
    substeps=self.config.substeps,
    headless=self.config.headless,
    viewer_options=ViewerOptions(...),
)
scene.build(scene_backend)
robot.spawn(scene_backend, position=spawn_pos)
scene_backend.build()
```

#### `Scene._build_room_structure()` 改造前/后

```python
# 改造前
floor = self.scene.add_entity(
    morph=gs.morphs.Box(size=(width, depth, thickness), pos=..., fixed=True),
    surface=gs.surfaces.Default(color=...),
)

# 改造后
floor = self.backend.create_box(
    size=(width, depth, thickness),
    pos=(0.0, 0.0, -thickness / 2),
    color=(0.9, 0.9, 0.9, 1.0),
    static=True,
    name="floor",
)
self.scene.add_entity(floor)
```

#### `FrankaPanda.spawn()` 改造前/后

```python
# 改造前
self.entity = scene.add_entity(morph=gs.morphs.MJCF(file=model_path, pos=pos))

# 改造后
self.articulation = scene.backend.load_mjcf(file=model_path, pos=pos)
scene.add_articulation(self.articulation)
```

---

## 五、性能优化设计（目标：比 CUDA/Genesis 版本提升 ≥ 10%）

### 5.1 优化维度矩阵（路径 A：Genesis/Quadrants → MUSA）

| 维度 | Genesis/CUDA 现状 | MT Lambda 优化点 | 预期提升 |
|------|-------------------|------------------|----------|
| 物理步进 | Genesis 自研引擎，Quadrants/CUDA | Quadrants/MUSA backend，kernel 针对 MUSA 优化 | 10-20% |
| 状态读写 | 每次 `get_qpos()` 都做 Python ↔ GPU 拷贝 | 批量预取 + DLPack 零拷贝（Quadrants ↔ torch_musa） | 15-25% |
| 向量化 | 当前 `GenesisVectorizedEnv` 为 stub | Genesis/Quadrants GPU batch / 多场景并行 | 3-5× |
| 渲染 | Genesis OpenGL / LuisaRender 串行 | MT Photon / 3DGS 异步渲染，按需 | 20-40% |
| 训练链路 | PyTorch-CUDA | torch_musa，MCCL 分布式 | 持平或略优 |
| 内存 | CUDA 显存池 | MUSA Unified Memory | 5-10% |

### 5.2 关键优化策略

#### 策略 1：批量状态读写接口（Batch State API）

当前代码中每个机器人每步调用多次 `get_qpos()` / `get_qvel()` / `control_dofs_position()`，每次调用都产生 Python ↔ GPU 往返。

```python
# 新增 Backend 批量接口
class ArticulationBackend(ABC):
    @abstractmethod
    def get_state_batch(self, env_ids: list[int] | None = None) -> ArticulationState:
        """一次性返回 qpos/qvel/pos/quat，减少 Python-GPU 往返。"""
        ...

    @abstractmethod
    def set_state_batch(self, state: ArticulationState, env_ids: list[int] | None = None) -> None: ...

    @abstractmethod
    def control_position_batch(
        self,
        targets: np.ndarray,
        stiffness: np.ndarray | None = None,
        damping: np.ndarray | None = None,
    ) -> None:
        """批量位置控制， targets shape: (num_envs, n_dofs)。"""
        ...
```

**预期收益**：单步 observation 收集延迟降低 10-20%。

#### 策略 2：零拷贝 DLPack 桥接（Quadrants ↔ torch_musa）

Genesis/Quadrants 已经支持 DLPack 与 PyTorch tensor 共享设备内存。在 MUSA 路径上，确保 `gs.from_torch` / `gs.to_torch` 使用 MUSA 设备 tensor，避免 host-device 拷贝。

```python
class GenesisArticulationBackend(ArticulationBackend):
    def get_qpos(self) -> np.ndarray:
        # 返回的 ndarray 最好直接是 DLPack view，避免 cudaMemcpy
        return np.asarray(self._entity.get_qpos())
```

**预期收益**：observation 生产 → PyTorch tensor 路径延迟降低 15-30%。

#### 策略 3：Genesis/Quadrants GPU 批量并行

将 `GenesisVectorizedEnv` 从 stub 改为真正的 GPU batch。Genesis/Quadrants 支持在一个 GPU 上下文中运行多个并行场景；利用 `gs.Scene` 的 batch 能力（或未来 Quadrants 提供的 multi-env API）：

```python
class MTLambdaVectorizedEnv(VectorizedEnvironment):
    def __init__(...):
        backend = get_backend("mt_lambda")
        backend.initialize(device="musa")
        self._scene_backend = backend.create_scene(...)
        # 利用 Quadrants 的 batch 能力创建 num_envs 个副本

    def step(self, actions: np.ndarray):
        # actions: (num_envs, action_dim)
        self._scene_backend.step_batch(actions)
        obs = self._scene_backend.get_state_batch()
        return obs, rewards, terminated, truncated, infos
```

**预期收益**：相比当前 Genesis stub，批量环境吞吐量提升 3-5×。

#### 策略 4：渲染与物理解耦 + 异步渲染

在 `GenesisRendererBackend` 中，物理 step 与相机渲染分离：

```python
class GenesisRendererBackend(RendererBackend):
    def step_physics_only(self) -> None:
        """不触发渲染管线，纯物理步进。"""
        ...

    def render_async(self, camera_names: list[str]) -> dict[str, Future[np.ndarray]]:
        """异步提交渲染任务，训练 loop 不必等待。"""
        ...
```

**预期收益**：headless 训练场景下，去除不必要渲染开销，step 延迟降低 20-40%。

#### 策略 5：Quadrants JIT 编译缓存与场景构建缓存

- 对 Quadrants kernel 编译结果做磁盘缓存（Genesis/Quadrants 已支持 warm-cache）。
- 对 `backend.create_scene()` 中重复的 geom/body 构建做内存缓存。

**预期收益**：大场景首次构建时间降低 30-50%（Quadrants warm-cache 已从 7.2s 降到 0.3s）。

#### 策略 6：MUSA 特化算子

在训练链路中，对 VLA/WAM 模型的常见算子（如旋转位置编码、注意力）使用 `torch_musa` 的 fused kernel，必要时手写 MUSA kernel。

**预期收益**：训练 throughput 提升 10-15%。

### 5.3 10% 提升目标拆解

以单环境 `Franka PickPlace` 为例，基准为当前 Genesis/CUDA 实现：

| 指标 | Genesis/CUDA 基准 | MT Lambda 目标 | 提升 |
|------|-------------------|----------------|------|
| step 延迟（无渲染） | 2.50 ms | ≤ 2.25 ms | ≥ 10% |
| observation 收集延迟 | 0.80 ms | ≤ 0.60 ms | ≥ 25% |
| 重置稳定步数 | 10 steps | ≤ 10 steps | 持平 |
| 成功率（100 eps） | baseline | ≥ baseline - 2% | 物理一致 |
| 内存占用 | baseline | ≤ baseline + 10% | 可接受 |

---

## 六、物理一致性与 AB 测试方案

### 6.1 对比维度

复用 `ABTestFramework`：

```python
runner = ABTestRunner(
    variant_a_name="genesis_cuda",
    variant_a_fn=make_genesis_env,
    variant_b_name="mt_lambda_musa",
    variant_b_fn=make_mt_lambda_env,
    output_dir="ab_test_results/backend_migration",
    warmup_steps=10,
)
```

| 指标 | 目标 |
|------|------|
| qpos 单步相对误差 | ≤ 1e-4（同一物理引擎，backend 切换应无显著差异） |
| 成功率差异 | ≤ 2% |
| 延迟降低 | ≥ 10% |
| 内存差异 | ≤ 10% |

### 6.2 渐进式迁移

复用 `GradualMigration`，按流量比例从 Genesis CUDA 切换到 MT Lambda MUSA。

---

## 七、风险与回退策略

| 风险 | 概率 | 影响 | 应对措施 |
|------|------|------|----------|
| Quadrants 尚未公开支持 MUSA | 高 | 高 | 与 MT Lambda 团队共建 MUSA backend；开发期 fallback 到 CUDA |
| MT Photon 相机数据格式不匹配 | 中 | 高 | Backend 层做 numpy bridge 转换；保留 Genesis 渲染 fallback |
| `torch_musa` 算子不兼容 | 中 | 高 | 提前跑通 VLA 算子清单；准备 custom kernel |
| 物理一致性差异导致策略失效 | 中 | 高 | ABTestFramework 验证；Domain Randomization 重标定 |
| 插件直接调用 Genesis API 导致 core/ 抽象被破坏 | 高 | 中 | 新增插件适配规范，先迁移 controllers/predictors |

---

## 八、实施路线图

### Phase 1：抽象层基础建设（Week 1-3）
1. 创建 `backend/base.py`、`backend/types.py`、`backend/factory.py`。
2. 定义 `SimulatorBackend`、`SceneBackend`、`EntityBackend`、`ArticulationBackend`、`RendererBackend`。
3. 单元测试：用 mock backend 验证 core/ 能正确注入。

### Phase 2：Genesis 后端下沉（Week 4-6）
1. 实现 `backends/genesis_backend.py`，将现有 `gs.*` 调用封装。
2. 重构 `core/composer.py`、`core/scene.py`、`core/embodiment.py` 使用 Backend 接口。
3. 保证现有测试全部通过，功能无损。

### Phase 3：MT Lambda 后端骨架（Week 7-10）
1. 实现 `backends/mt_lambda_backend.py`，继承 `GenesisBackend` 并选择 MUSA arch。
2. 与 MT Lambda 团队确认 Quadrants MUSA backend 的 API 和初始化方式。
3. 跑通 `Franka PickPlace` 单场景（开发期可 fallback 到 CUDA）。

### Phase 4：性能优化与向量化（Week 11-14）
1. 实现批量状态读写、DLPack 零拷贝桥接。
2. 实现 `MTLambdaVectorizedEnv` 真正的 Quadrants GPU batch。
3. 接入 MT Photon / 3DGS 渲染（可选，先保证 headless）。

### Phase 5：AB 测试与对标（Week 15-16）
1. 用 `ABTestFramework` 对比 Genesis CUDA vs MT Lambda MUSA。
2. 调优至延迟降低 ≥ 10%、物理一致性达标。
3. 输出迁移评估报告。

---

## 九、交付物

1. **设计文档**：`docs/backend_abc_design.md`（本计划获批后写入项目目录）。
2. **代码**：
   - `src/cloud_robotics_sim/backend/` 抽象层
   - `src/cloud_robotics_sim/backends/genesis_backend.py`
   - `src/cloud_robotics_sim/backends/mt_lambda_backend.py`
   - 重构后的 `core/composer.py`、`core/scene.py`、`core/embodiment.py`
3. **测试**：
   - `tests/backend/test_backend_abc.py`
   - `tests/backend/test_genesis_backend.py`
   - `tests/backend/test_mt_lambda_backend.py`
4. **AB 测试报告**：`ab_test_results/backend_migration/`。

---

## 十、推荐策略

**推荐采用“抽象层迁移”方案（策略 B）**：
- 保留 `core/` 业务逻辑，新增 Backend 抽象层。
- 先实现 `GenesisBackend` 保证回归无损，再实现 `MTLambdaBackend`（复用 GenesisBackend，选择 MUSA arch）。
- 通过批量状态读写、DLPack 零拷贝、Quadrants batch、异步渲染等优化，目标在单环境 step 延迟上比 Genesis/CUDA 版本降低 ≥ 10%。

此方案风险可控，复用度高，且为后续插件迁移和国产化替代奠定统一接口基础。
