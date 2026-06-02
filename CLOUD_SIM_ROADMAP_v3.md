# Genesis Cloud Sim 升级路线图 v3.0

> 基于当前版本 (2026-03-13, commit 81509da) + 未提交增量
> 目标：从"插件仓库"进化为"云原生机器人仿真操作系统"

---

## 一、现状诊断

### 1.1 已具备的能力

```
┌─────────────────────────────────────────────────────────────┐
│  基础设施层                                                  │
│  ├── PluginManager (动态发现/加载)                          │
│  ├── A/B Testing Framework (渐进迁移)                       │
│  ├── Composer (Scene+Robot+Task 组合)                       │
│  ├── Registry (组件注册表)                                   │
│  └── VectorizedEnv (GPU 并行, 4096 envs)                   │
├─────────────────────────────────────────────────────────────┤
│  插件生态层 (15个插件)                                       │
│  ├── 控制器: MPC-WBC, Residual-RL, HugWBC, SLAC, WBM-Embrace, OpenLoong
│  ├── 环境: Badminton, Table-Tennis, Humanoid-Falling, ManiSkill, Sky
│  ├── 预测器: BFM-Zero, Scene-Language
│  ├── Sim2Real: Dexterous-Manipulation
│  ├── 数据集: DreamDojo
│  └── 场景: Art-Scenes (春节主题)
├─────────────────────────────────────────────────────────────┤
│  工具链层                                                    │
│  ├── Genesis 兼容工具 (camera, rendering, genesis_compat)   │
│  ├── 迁移工具 (migrate_project.py)                          │
│  └── CI/CD (GitHub Actions, Sphinx 文档)                    │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 关键缺口

| 缺口 | 严重程度 | 影响 |
|------|---------|------|
| 测试覆盖率极低 | 🔴 高 | 插件质量无法保证，回归风险大 |
| 插件间缺乏联动 | 🔴 高 | 各自为战，无法形成组合能力 |
| 无云原生部署实现 | 🔴 高 | 号称 cloud-native 但只有概念 |
| 缺少数据流水线 | 🟡 中 | 仿真→训练→部署 链路断裂 |
| 无可视化/监控 | 🟡 中 | 黑盒运行，难以调试 |
| 文档与代码不同步 | 🟡 中 | 部分文档描述的是计划而非现状 |
| 未提交文件未整合 | 🟡 中 | openloong/dreamdojo/art_scenes/utils 游离在外 |

---

## 二、升级目标

### 2.1 愿景

> **"让机器人仿真像调用 API 一样简单"**

```python
# 最终形态愿景代码
from cloud_robotics_sim import CloudSim

sim = CloudSim()

# 1. 一键部署仿真集群
cluster = sim.deploy_cluster(
    name="h1-walking-cluster",
    num_envs=4096,
    plugins=["hugwbc", "humanoid_falling"],
    cloud="auto"  # 本地/云端自动选择
)

# 2. 配置驱动训练
job = cluster.train(
    config="configs/h1_walking.yaml",
    algorithm="ppo",
    checkpoint_every=1000
)

# 3. 实时监控
sim.dashboard(job.id)  # 打开浏览器看训练曲线

# 4. Sim2Real 一键导出
policy = job.export(format="onnx", quantize=True)
policy.deploy(target="unitree_h1")  # 直接烧录到真机
```

### 2.2 分阶段目标

| 阶段 | 版本 | 主题 | 时间 |
|------|------|------|------|
| Phase 1 | v2.1 | **夯实基础** | 2-3 周 |
| Phase 2 | v2.2 | **插件联动** | 3-4 周 |
| Phase 3 | v2.3 | **云原生落地** | 4-6 周 |
| Phase 4 | v3.0 | **生态闭环** | 2-3 月 |

---

## 三、Phase 1: 夯实基础 (v2.1)

### 3.1 测试体系建设

```
tests/
├── conftest.py                    # 共享 fixture
├── core/
│   ├── test_plugin_manager.py     # 插件发现/加载/卸载
│   ├── test_ab_test_framework.py  # A/B 测试正确性
│   ├── test_composer.py           # 环境组合 (已有，扩展)
│   ├── test_registry.py           # 注册表 CRUD
│   └── test_vectorized.py         # 并行环境一致性
├── plugins/
│   ├── test_mpc_wbc.py            # 控制器输出合理性
│   ├── test_hugwbc.py
│   ├── test_badminton.py          # 环境 reset/step 稳定
│   ├── test_table_tennis.py
│   └── test_sim2real_dexterous.py
├── integration/
│   ├── test_plugin_composition.py # 插件组合运行
│   └── test_end_to_end.py         # 完整训练流程
└── performance/
    ├── test_benchmark.py          # 性能基准
    └── test_memory_leak.py        # 内存泄漏检测
```

**关键测试策略：**
- 每个 plugin 必须有 `tests/test_<plugin>.py`，覆盖率 > 60%
- 集成测试：随机组合 2 个插件运行 100 步不崩溃
- 性能测试：4096 envs 下 step() 延迟 < 50ms

### 3.2 未提交文件整合

**当前未提交的文件需要决策：**

| 文件/目录 | 来源 | 建议 |
|-----------|------|------|
| `plugins/controllers/openloong/` | 手动添加 | ✅ 提交，但需补充 tests/ 和 examples/ |
| `plugins/datasets/dreamdojo/` | Phase 2 迁移 | ✅ 提交，整合到 datasets 类别 |
| `plugins/scenes/art_scenes/` | 手动添加 | ✅ 提交，作为场景模板示例 |
| `src/cloud_robotics_sim/utils/` | ManiSkill 迁移 | ✅ 提交，但需补充测试 |
| `examples/migration/` | 迁移示例 | ✅ 提交 |
| `src/__init__.py` 修改 | 兼容层 | ✅ 提交，解决循环导入问题 |

**整合后的新结构：**

```
plugins/
├── controllers/        (6个)
├── envs/               (5个)
├── predictors/         (2个)
├── sim2real/           (1个)
├── datasets/           (1个)  ← dreamdojo
├── scenes/             (1个)  ← art_scenes
└── templates/          (1个)  ← 新增：插件开发模板
```

### 3.3 Plugin 规范升级

**当前问题：** plugin.yaml 格式不统一（有的有 `type`，有的没有；有的有 `entry_points`，有的没有）

**统一规范 v2：**

```yaml
# plugin.yaml v2 标准格式
name: "mpc_wbc"
version: "0.2.0"
type: "controller"           # 新增：明确类型
api_version: "2.1"           # 新增：兼容版本

metadata:
  description: "MPC + WBC controller for humanoid locomotion"
  author: "Genesis Cloud Sim Team"
  source_project: "openloong-dyn-control"
  tags: ["humanoid", "locomotion", "mpc", "wbc"]
  
requirements:
  genesis_world: ">=0.4.0"   # 明确 Genesis 版本要求
  python: ">=3.10"
  dependencies:
    - numpy>=1.20
    - scipy>=1.7
  optional:
    - tensorboard>=2.13

exports:
  classes:
    - MPCWBCController
    - GaitScheduler
  functions:
    - create_default_config
  
config:
  schema: "configs/schema.json"   # 新增：配置校验 schema
  defaults: "configs/default.yaml"

hooks:
  on_load: "hooks.on_load"        # 新增：生命周期钩子
  on_unload: "hooks.on_unload"

compatibility:
  tested_with:
    - plugin: "hugwbc"
      version: ">=0.1.0"
    - plugin: "humanoid_falling"
      version: ">=0.1.0"
  conflicts:
    - plugin: "legacy_controller"
      reason: "功能重叠"
```

### 3.4 交付物

- [ ] 测试覆盖率从 ~5% 提升到 40%
- [ ] 所有未提交文件清理并提交
- [ ] Plugin 规范升级到 v2
- [ ] CI 增加测试门禁（PR 不通过测试不能合并）

---

## 四、Phase 2: 插件联动 (v2.2)

### 4.1 核心问题

当前插件是孤岛：

```python
# 现在：各自为战
from plugins.controllers.mpc_wbc import MPCWBCController
from plugins.envs.badminton import BadmintonEnv

controller = MPCWBCController()  # 不知道 BadmintonEnv 的存在
env = BadmintonEnv()             # 不知道 MPCWBCController 的存在
# 用户自己硬编码对接
```

### 4.2 解决方案：Plugin Composition DSL

```python
# 目标：声明式组合
from cloud_robotics_sim import Simulation, Plugin

sim = Simulation()

# 方式 1：声明式组合
sim.compose({
    "scene": "art_scenes/spring_festival",
    "robot": "maniskill/franka_panda",
    "controller": "mpc_wbc",
    "task": "pick_place",
    "dataset": "dreamdojo/online"
})

# 方式 2：能力匹配自动组合
sim.compose({
    "robot": "unitree_h1",
    "task": "walking",
    "constraints": {
        "controller": {"tags": ["humanoid", "locomotion"]},
        "min_fps": 1000
    }
})
# 自动选择 hugwbc 或 mpc_wbc，取决于性能要求

# 方式 3：训练流水线组合
pipeline = sim.pipeline()
pipeline.add_step("pretrain", plugin="slac", epochs=1000)
pipeline.add_step("finetune", plugin="residual_rl", epochs=500)
pipeline.add_step("sim2real", plugin="sim2real_dexterous")
```

### 4.3 能力图谱系统

```python
# cloud_robotics_sim/core/capability_graph.py

class CapabilityGraph:
    """
    插件能力图谱：自动发现插件间的兼容性和组合方式
    """
    
    def __init__(self):
        self._graph = nx.DiGraph()  # 能力依赖图
        
    def register_plugin(self, plugin_info: PluginInfo):
        """注册插件及其能力"""
        # 提取能力标签
        capabilities = self._extract_capabilities(plugin_info)
        self._graph.add_node(plugin_info.name, 
                            capabilities=capabilities,
                            metadata=plugin_info)
        
    def find_compositions(self, requirements: dict) -> list[Composition]:
        """
        根据需求自动寻找可行的插件组合
        
        Example:
            requirements = {
                "robot_type": "humanoid",
                "task": "locomotion",
                "needs_sim2real": True
            }
            
        Returns:
            [
                Composition([hugwbc, humanoid_falling, sim2real_dexterous]),
                Composition([mpc_wbc, sim2real_dexterous]),
            ]
        """
        pass
        
    def validate_composition(self, composition: list[str]) -> ValidationResult:
        """验证组合是否可行"""
        # 检查版本兼容性
        # 检查资源冲突
        # 检查循环依赖
        pass
```

### 4.4 跨插件数据流

```python
# cloud_robotics_sim/core/data_bus.py

class PluginDataBus:
    """
    插件间数据总线：标准化传感器/控制信号传输
    
    解决：A 插件的观测如何被 B 插件使用
    """
    
    # 标准数据类型注册
    STANDARD_TYPES = {
        "proprioception": ProprioceptionMsg,      # 本体感知
        "visual_observation": VisualObsMsg,       # 视觉观测
        "contact_force": ContactForceMsg,         # 接触力
        "target_pose": TargetPoseMsg,             # 目标位姿
        "action": ActionMsg,                      # 动作指令
        "reward": RewardMsg,                      # 奖励信号
        "termination": TerminationMsg,            # 终止信号
    }
    
    def publish(self, plugin_id: str, topic: str, data: Any):
        """发布数据到总线"""
        
    def subscribe(self, plugin_id: str, topic: str, callback: Callable):
        """订阅数据"""
        
    def connect(self, source: str, target: str, 
                source_topic: str, target_topic: str):
        """建立插件间数据连接"""
```

### 4.5 预设组合模板

```yaml
# configs/compositions/

# 人形行走模板
humanoid_locomotion.yaml:
  name: "Humanoid Locomotion Stack"
  description: "标准人形机器人行走控制栈"
  
  plugins:
    physics:
      plugin: sky
      version: ">=0.3.0"
      
    controller:
      plugin: hugwbc
      version: ">=0.1.0"
      config: "configs/hugwbc_walking.yaml"
      
    safety:
      plugin: humanoid_falling
      version: ">=0.1.0"
      mode: "monitor"  # 监控模式，不干预正常控制
      
  connections:
    - from: "physics/joint_states"
      to: "controller/proprioception"
    - from: "controller/actions"
      to: "physics/actuator_commands"
    - from: "physics/imu_data"
      to: "safety/imu_input"
      
  fallback:
    if: "safety.detect_fall_risk > 0.8"
    then:
      - activate: "safety/protective_posture"
      - reduce: "controller/action_scale"
        to: 0.3

# 球类运动模板
ball_sports.yaml:
  name: "Ball Sports Stack"
  
  variants:
    badminton:
      env: badminton
      controller: mpc_wbc  # 需要全身协调
      
    table_tennis:
      env: table_tennis
      controller: hugwbc
      predictor: bfm_zero  # 预测对手意图
```

### 4.6 交付物

- [ ] CapabilityGraph 系统上线
- [ ] PluginDataBus 跨插件通信
- [ ] 5 个预设组合模板
- [ ] 组合自动验证和推荐

---

## 五、Phase 3: 云原生落地 (v2.3)

### 5.1 当前状态 vs 目标

```
当前: "Cloud-native deployment support (Kubernetes, Docker)" 
      → 只是 pyproject.toml 里的一句话

目标: 一条命令部署仿真集群
```

### 5.2 容器化

```dockerfile
# Dockerfile
FROM nvidia/cuda:12.1-devel-ubuntu22.04

# 安装 Genesis + Cloud Sim
RUN pip install genesis-world>=0.4.0 cloud-robotics-sim>=2.3.0

# 安装插件（按需）
ARG PLUGINS="hugwbc,badminton"
RUN cloud-robotics-sim install-plugins $PLUGINS

# 启动仿真服务
EXPOSE 8080
CMD ["cloud-robotics-sim", "serve", "--port", "8080"]
```

```yaml
# docker-compose.yml (本地开发)
version: '3.8'
services:
  sim-master:
    build: .
    ports:
      - "8080:8080"
    environment:
      - CUDA_VISIBLE_DEVICES=0
      - PLUGINS=hugwbc,badminton,table_tennis
    volumes:
      - ./configs:/app/configs
      - ./checkpoints:/app/checkpoints
      
  sim-worker:
    build: .
    deploy:
      replicas: 4
    environment:
      - MASTER_URL=http://sim-master:8080
```

### 5.3 K8s 部署

```yaml
# k8s/simulation-cluster.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: genesis-sim-cluster
spec:
  replicas: 1
  selector:
    matchLabels:
      app: genesis-sim
  template:
    metadata:
      labels:
        app: genesis-sim
    spec:
      containers:
      - name: sim
        image: genesis-cloud-sim:v2.3
        resources:
          limits:
            nvidia.com/gpu: 1
            memory: "32Gi"
            cpu: "16"
        env:
        - name: CLOUDSIM_MODE
          value: "cluster"
        - name: PLUGINS
          value: "hugwbc,mpc_wbc,badminton,table_tennis"
        ports:
        - containerPort: 8080
        
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: sim-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: genesis-sim-cluster
  minReplicas: 1
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: nvidia.com/gpu
      target:
        type: Utilization
        averageUtilization: 80
```

### 5.4 仿真即服务 API

```python
# cloud_robotics_sim/server/api.py

from fastapi import FastAPI
from cloud_robotics_sim import CloudSim

app = FastAPI()
sim = CloudSim()

@app.post("/v1/simulations")
async def create_simulation(config: SimulationConfig):
    """创建仿真环境"""
    sim_id = sim.create(config)
    return {"id": sim_id, "status": "running"}

@app.post("/v1/simulations/{sim_id}/step")
async def step_simulation(sim_id: str, actions: list[float]):
    """执行一步仿真"""
    obs, reward, done, info = sim.step(sim_id, actions)
    return {"observation": obs, "reward": reward, "done": done}

@app.post("/v1/train")
async def start_training(job: TrainingJob):
    """启动训练任务"""
    job_id = sim.train(
        config=job.config,
        num_envs=job.num_envs,
        plugins=job.plugins
    )
    return {"job_id": job_id}

@app.get("/v1/train/{job_id}/metrics")
async def get_metrics(job_id: str):
    """获取训练指标"""
    return sim.get_metrics(job_id)

@app.websocket("/v1/stream/{sim_id}")
async def stream_simulation(websocket: WebSocket, sim_id: str):
    """实时仿真数据流（用于可视化）"""
    await websocket.accept()
    async for frame in sim.stream(sim_id):
        await websocket.send_json(frame)
```

### 5.5 训练任务调度器

```python
# cloud_robotics_sim/server/scheduler.py

class TrainingScheduler:
    """
    训练任务调度器：管理多个并行训练任务
    """
    
    def submit(self, job: TrainingJob) -> str:
        """
        提交训练任务，自动选择资源
        
        策略：
        1. 本地 GPU 足够 → 本地运行
        2. 本地不足 → 自动扩展到云端
        3. 多任务排队 → 优先级调度
        """
        
    def scale(self, job_id: str, num_envs: int):
        """动态扩缩容"""
        
    def checkpoint(self, job_id: str) -> str:
        """生成检查点"""
        
    def export(self, job_id: str, format: str) -> bytes:
        """导出训练好的策略"""
```

### 5.6 交付物

- [ ] Dockerfile + docker-compose 配置
- [ ] K8s deployment manifests
- [ ] REST API + WebSocket 实时流
- [ ] 训练任务调度器
- [ ] 自动扩缩容

---

## 六、Phase 4: 生态闭环 (v3.0)

### 6.1 数据飞轮

```
┌─────────────────────────────────────────────────────────────┐
│                      数据飞轮                                │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   ┌──────────┐    仿真数据    ┌──────────┐                │
│   │  Cloud   │ ─────────────→│  训练    │                │
│   │   Sim    │               │ Cluster  │                │
│   └──────────┘               └────┬─────┘                │
│        ↑                            │                      │
│        │                            ↓                      │
│   ┌────┴─────┐               ┌──────────┐                │
│   │ Real     │ ←─────────────│  Policy  │                │
│   │ World    │   真机反馈     │  Export  │                │
│   └──────────┘               └──────────┘                │
│                                                             │
│   循环: 仿真 → 训练 → 部署 → 真机反馈 → 改进仿真              │
└─────────────────────────────────────────────────────────────┘
```

### 6.2  marketplace 概念

```python
# cloud-robotics_sim marketplace

# 浏览可用插件
market = sim.marketplace()
plugins = market.search(
    tags=["humanoid", "locomotion"],
    min_rating=4.0,
    tested_with=["genesis-world>=0.4.0"]
)

# 一键安装
market.install("hugwbc", version="latest")

# 分享自定义插件
market.publish(
    path="./my_custom_controller",
    metadata={
        "name": "my_controller",
        "description": "...",
        "license": "MIT"
    }
)
```

### 6.3 可视化平台

```
Dashboard 功能：
├── 实时监控
│   ├── 训练曲线 (reward, success rate)
│   ├── 仿真可视化 (3D 场景渲染)
│   ├── 资源使用 (GPU/CPU/内存)
│   └── 插件状态
│
├── 调试工具
│   ├── 策略可视化 (attention heatmap)
│   ├── 物理检查 (碰撞检测, 接触力)
│   └── 对比工具 (A/B 策略并排对比)
│
└── 部署管理
    ├── 版本控制 (策略版本历史)
    ├── 灰度发布 (10% → 50% → 100%)
    └── 回滚机制
```

### 6.4 交付物

- [ ] 数据飞轮闭环
- [ ] Plugin Marketplace (基础版)
- [ ] Web Dashboard
- [ ] 策略版本管理和灰度发布

---

## 七、实施优先级矩阵

| 任务 | 价值 | 成本 | 优先级 |
|------|------|------|--------|
| 测试体系建设 | 极高 | 中 | **P0** |
| 未提交文件整合 | 高 | 低 | **P0** |
| Plugin 规范升级 | 高 | 低 | **P0** |
| 插件组合系统 | 极高 | 高 | **P1** |
| 容器化 | 高 | 中 | **P1** |
| K8s 部署 | 高 | 高 | **P2** |
| 仿真即服务 API | 极高 | 高 | **P2** |
| 可视化 Dashboard | 中 | 高 | **P3** |
| Marketplace | 中 | 极高 | **P3** |

---

## 八、风险与对策

| 风险 | 概率 | 影响 | 对策 |
|------|------|------|------|
| Genesis 版本升级破坏插件 | 中 | 高 | 建立 Genesis 兼容性测试矩阵 |
| 插件数量膨胀导致维护困难 | 高 | 中 | 引入插件分级 (core/community/experimental) |
| 云原生性能不达预期 | 中 | 高 | 先本地验证，再逐步上云 |
| 社区贡献质量参差 | 中 | 中 | 严格的 PR review + 自动化测试门禁 |

---

## 九、立即行动项

### 本周 (Week 1)

1. [ ] 提交所有未提交文件（openloong, dreamdojo, art_scenes, utils）
2. [ ] 为每个未提交插件补充 tests/ 和 examples/
3. [ ] 统一 plugin.yaml 格式到 v2 规范
4. [ ] 运行现有测试，建立 baseline

### 下周 (Week 2)

1. [ ] 为核心插件（mpc_wbc, hugwbc, badminton）编写完整测试
2. [ ] 实现 PluginDataBus 原型
3. [ ] 设计 CapabilityGraph 数据结构
4. [ ] 创建第一个组合模板（humanoid_locomotion）

---

*路线图版本: v3.0*
*最后更新: 2026-06-02*
*维护者: Cloud Robotics Sim Team*
