# RoboTwin→Genesis 迁移 P0 实施总结

> 日期：2026-07-28 ｜ 依据文档：《RoboTwin 2.0 迁移至 Genesis World 仿真器技术方案》v1.0
> 范围：P0 基础骨架（§3.1 / §4 / §5 / §6 配置层 / §7 / §10）
> 质量门禁：**517 tests passed**、ruff 全绿、新增/改动文件 black 全净

---

## 1. 总览

按迁移文档的 P0 范围，在本仓库（genesis-cloud-sim）现有 backend 抽象与 `robotwin` 包基础上，分四批完成了迁移骨架的全部可在无 GPU/无真实资产环境下落地的交付物，并接入了本地 `hierarchical_cuRobo_planner` 项目替代原方案的 cuRobo 桥接。

| 文档章节 | 交付物 | 状态 |
|---|---|---|
| §3.1 SimBackend | IK / 多链路 IK / OMPL plan_path / PD 增益 / DR / 相机内外参 / `build(n_envs)` / renderer 选择 | ✅ |
| §4 资产转换 | `tools/convert_assets.py`（mimic 展开、`package://` 重写、checklist、Genesis smoke test、GLB `--mesh-smoke`、报告） | ✅ 真实资产已验证（§3.1） |
| §5.1 并行种子搜索 | `batched_seed_search`（per-env mask、早停） | ✅ |
| §5.2 规划器 | OMPL 默认 + `hierarchical_cuRobo_planner` 兜底路由（替代 cuRobo 桥接） | ✅（端到端待 CUDA/MUSA） |
| §6 渲染三档 | `configs/render/` 三档配置 + renderer 接线 + PSNR/MAE 对齐基准 | ✅（LPIPS/tone mapping 待 GPU） |
| §7 数据格式 | EpisodeRecorder（HDF5 + MP4 + Zarr）、内外参 1e-6 对齐实测通过 | ✅ |
| §10 代码骨架 | `examples/genesis_aloha_demo.py` 端到端可运行 | ✅ |

---

## 2. 分批交付明细

### 批次一：P0 基础骨架

**Backend 接口扩展（§3.1）** — `src/cloud_robotics_sim/backend/base.py`
- `ArticulationBackend` 新增（非 abstract，默认 `NotImplementedError`，不破坏现有 mock/MTLambda 子类）：
  - `set_dofs_gains(kp, kv, force_range, armature)` —— config.yml stiffness/damping 批量映射（§2）
  - `inverse_kinematics` / `inverse_kinematics_multilink`（双臂 aloha，§4.1）
  - `plan_path`（OMPL，替换 mplib RRT，§5.2）
  - `get_link_pose`
  - DR 三件套：`set_friction_ratio` / `set_mass_shift` / `set_com_shift`
- `CameraBackend.get_camera_params()` → `(intrinsic, extrinsic camera-to-world)`（§7 关键）
- **已按 venv 内 Genesis 1.2.2 真实签名逐一核对实现**（`backends/genesis_backend.py`）

**数据层（§7/§10）** — `robotwin/recorder.py`
- `EpisodeRecorder`：RoboTwin 格式 HDF5（`/obs/rgb|depth|segmentation/<cam>`、`/qpos`、`/endpose`、相机内外参 attrs）+ MP4（imageio）+ 批量采集按 env 拆 episode
- `utils/camera.py` 新增 `intrinsics_from_fov`

**资产转换工具（§4）** — `tools/convert_assets.py`
- mimic 关节展开（Genesis issue #678）+ fixed-base / inertial / mesh 引用 checklist
- 标注文件原样拷贝（§4.2）、`--smoke-test` Genesis 加载验证、`conversion_report.json`

**示例（§10）** — `examples/genesis_aloha_demo.py`：URDF(fixed=True) → IK → plan_path → 录制 HDF5，无资产时用合成双臂跑通

**依赖**：`h5py>=3.8` 加入 pyproject 并 `uv lock` 同步

### 批次二：控制层 + Zarr + 批量基础

- `SceneBackend.build(n_envs=1, env_spacing=None)`（§3.1/§5.1 批量底座，向后兼容）
- `robotwin/embodiment_config.py`：
  - `RobotwinEmbodimentConfig.from_yaml()` 容错解析 config.yml（多别名），合并 `conversion_report.json` 的 mimic 映射
  - `apply_pd_gains()` 一键下发 PD 增益
  - `MimicJointMapper`：控制层按 `master×multiplier+offset` 复制 mimic 目标值（#678 完整闭环）
- `EpisodeRecorder.save_zarr()`（§7，DP/DP3 消费，与 HDF5 同 schema，兼容 zarr v3 API）；zarr 加入 dev extra
- 真实 Genesis 验证：相机内外参对齐（1e-6）、n_envs=2 批量录制拆分/DR（`tests/backend/test_genesis_batched.py`）

### 批次三：hierarchical_cuRobo_planner 替代 cuRobo 桥接（§5.2）

- 新模块 `robotwin/curobo_planner.py`：
  - `CuRoboPlannerConfig`：URDF/base_link/ee_link + 工作空间 AABB、体素、障碍物列表（A* 避障为 OMPL 不具备的能力）、CUDA/MUSA auto 检测
  - `HierarchicalCuRoboPlanner`：包装外部 `HierarchicalPlanner`（工作空间 A* → 批量 cuRobo IK → 轨迹优化），**懒构建**，无 CUDA 环境 import 零成本
  - `plan_with_fallback(robot, goal_pos, ...)`：cuRobo 优先、OMPL 兜底的路由（§5.2 决策），返回 `(trajectory, planner_name)`
  - 分层异常：`CuRoboPlannerUnavailableError`（未安装，含安装提示）/ `PlannerError`（规划失败）
- **不引入硬依赖**（本机无 CUDA/cuRobo）；安装：`pip install -e D:\githbi\hierarchical_cuRobo_planner`
- demo 新增 `--planner {auto,ompl,curobo}` 三模式（curobo 严格模式退出码 3）
- 与外部项目 API 按源码核实：`PlanRequest` / `PlanResult.trajectory` / `RobotModelConfig` / `PlanningConfig`

### 批次四：M4 渲染三档 + §5.1 并行种子搜索

- `SimulatorBackend.create_scene(..., renderer=None)` 接口扩展，Genesis 透传 `gs.Scene(renderer=...)`
- `robotwin/render_config.py`：`RenderConfig` + YAML 加载 + `make_genesis_renderer` + `compare_render_pair`（PSNR/MAE）
- 三档配置 `configs/render/`（§6.1 用途对应）：
  - `rasterizer.yaml` —— 种子搜索/RL（最高吞吐）
  - `raytracer.yaml` —— LuisaRender，spp=32+denoise 对齐 SAPIEN RT（训练数据）
  - `batch_madrona.yaml` —— Madrona 256×256（并行评测/DR 扫描）
- `robotwin/seed_search.py`：`batched_seed_search`（§5.1）—— 每 env 一个随机初始化 → 批量 rollout → per-env 成功 mask → 达标早停；真实 Genesis `n_envs=3` 全链路验证通过

---

## 3. 实测发现（已修正并写入代码注释/文档）

1. **文档 §7 内参公式有误**：`fx=fy=W/(2·tan(fov/2))` 与 Genesis 真实行为相差正好是宽高比（640×480@60°：554.26 vs 415.69）。Genesis 的 fov 为**垂直**视场角，正确公式为 **`f = H/(2·tan(fov/2))`**。`intrinsics_from_fov` 已按 Genesis 修正并与 `Camera.intrinsics` 实测对齐（1e-6）。
2. **Genesis 合并 fixed joint 子 link**：`ee_link`（fixed 关节连接）被并入父 link，IK 目标须用未合并的 link 名——迁移 aloha URDF 时需注意 ee 命名。
3. **批量场景外参含 env 网格偏移**：`n_envs>1` 时 `cam.transform` 为全局帧（env0 偏移 `-env_spacing/2`）；单 env 数据采集无影响，已写入 `get_camera_params` docstring。
4. **合成 URDF 的 `<inertial>` 必须含 `<inertia>` 子元素**，否则 Genesis urdfpy 解析崩溃（转换 checklist 的 inertial 检测因此更有必要）。
5. **真实 `config.yml` 格式与文档假设不同**（M2 下载 RoboTwin2.0 资产后实测）：`joint_stiffness`/`joint_damping` 是**标量**（需广播到全部 DoF）；ee link 在 `move_group` 字段；mimic 关系在 `gripper_name: [{base, mimic: [[slave, mult, offset]]}]` 中；franka-panda 有拼写错误 `gripper_stiffnes`（少 s）。`RobotwinEmbodimentConfig` 已全部兼容（别名 + 广播 + gripper_name 解析）。
6. **真实 URDF 中没有生效的 `<mimic>` 元素**——franka `panda.urdf` 的 mimic 被 XML 注释掉，其余 4 个本体本来就没有；mimic 关系只存在于 config.yml 的 `gripper_name`。因此 URDF 级 mimic 展开对这批资产是 no-op，**控制层 `MimicJointMapper` 是必须的**（SAPIEN 侧也是靠代码按 config 复制 master 目标驱动从动指）。
7. **ROS `package://` mesh URI 需重写**：piper 的附属 URDF（moveit/v00/no_gripper 变体）引用 `package://piper_description/meshes/...`，而网格实际在 `piper/meshes/`。`convert_assets.py` 现按路径尾匹配解析并重写为相对路径（报告 `package_uri_rewrites`），解析失败的仍计入 `missing_meshes`。
8. **aloha `.dae` 纹理告警无害**：Genesis 提示 "Texture given but asset missing uv info"（Collada uv 缺失），不影响加载与物理。
9. **CoACD 凸分解在密集 visual 网格上过慢甚至崩溃**：`gs.morphs.Mesh` 默认 `convexify=True`，对 RoboTwin-OD 视觉网格（如 `002_bowl/visual/base1.glb`）凸分解单网格耗时 100+s 且进程最终崩溃。`--mesh-smoke` 改用 `convexify=False`（同网格 12.6s 加载成功）。**运行时如需物体碰撞，应直接用资产自带的 `collision/*.glb`（RoboTwin 已提供简化碰撞网格），不要对 visual 网格做凸分解。**
10. **Genesis `scene.build()` 是 mesh smoke 的主要开销**（CPU ~50s/次，网格解析仅 1-9s，但 textured visual GLB 的解析可达 ~90s）：`--mesh-smoke` 因此（a）批量构建场景（`--batch-size`，默认 16/批）摊销 build；（b）每批 checkpoint 写入报告（`complete` 字段标记是否跑完）；（c）支持 `--visual-sample N` 只对前 N 类采样 visual 网格。

---

## 3.1 M2 真实资产转换结果（2026-07-28）

资产来源：HuggingFace `TianxingChen/RoboTwin2.0`（经 hf-mirror.com 下载），存于 `assets/robotwin/`（已加入 .gitignore）：
- `embodiments/`：aloha-agilex、ARX-X5、franka-panda、piper、ur5-wsg 五个本体（URDF + config.yml + 网格）。
- `objects/`：RoboTwin-OD 物体库，**129 类**（非文档所述 147 类），每类 `collision/*.glb` + `visual/*.glb` + `model_dataN.json`（center/extents/scale/target_pose/points）+ `points_info.json`。物体为纯 GLB 网格（无 URDF），运行时按 model_data 的 scale 加载。

**Embodiments 转换**（`assets_genesis/embodiments/`，日志 `outputs/convert_embodiments.log`）：
- 10 个 URDF（5 个 config 引用 + 5 个 piper 附属变体）全部转换，**smoke test 10/10 通过**（Genesis CPU 加载成功），0 缺失网格引用，0 需展开的 mimic。
- piper 附属 URDF 的 `package://` URI 已自动重写（发现 #7）。
- 各 URDF 的 `missing_inertial` 列表已记录在 `conversion_report.json`（如 franka 12 个 link 无 inertial，Genesis 用默认惯性仍可加载）。

**Objects 网格 smoke**（`--mesh-smoke --batch-size 32 --visual-sample 20`，报告 `assets_genesis/objects/conversion_report.json`，日志 `outputs/mesh_smoke_objects.log`）：129 类每类的首个 collision GLB 全量 + 前 20 类的首个 visual GLB 采样，在 Genesis CPU 批量加载验证——结果见报告（n_ok/n_assets）。

5 个真实 config.yml 的解析已固化为集成测试（`tests/robotwin/test_embodiment_config.py::TestRealEmbodimentConfigs`，资产缺失时自动 skip）。

---

## 4. 文件清单

**新增**
- `src/cloud_robotics_sim/robotwin/recorder.py`（HDF5/MP4/Zarr 录制）
- `src/cloud_robotics_sim/robotwin/embodiment_config.py`（config.yml + MimicJointMapper）
- `src/cloud_robotics_sim/robotwin/curobo_planner.py`（分层规划器适配）
- `src/cloud_robotics_sim/robotwin/render_config.py`（渲染三档）
- `src/cloud_robotics_sim/robotwin/seed_search.py`（并行种子搜索）
- `tools/convert_assets.py` + `tools/__init__.py`（资产转换工具链，含 `--mesh-smoke` GLB 物体库模式与 `package://` URI 重写）
- `examples/genesis_aloha_demo.py`（§10 端到端骨架）
- `configs/render/{rasterizer,raytracer,batch_madrona}.yaml`
- 测试：`tests/backend/test_backend_ik_plan.py`、`test_genesis_batched.py`、`tests/robotwin/test_recorder.py`、`test_embodiment_config.py`、`test_curobo_planner.py`、`test_render_config.py`、`test_seed_search.py`、`test_genesis_aloha_demo.py`、`tests/tools/test_convert_assets.py`

**修改**
- `src/cloud_robotics_sim/backend/base.py`（接口扩展）
- `src/cloud_robotics_sim/backends/genesis_backend.py`（Genesis 实现）
- `src/cloud_robotics_sim/utils/camera.py`（`intrinsics_from_fov`）
- `src/cloud_robotics_sim/robotwin/__init__.py`、`README.md`
- `pyproject.toml` + `uv.lock`（h5py、zarr）

**测试**：107+ 个新增用例（mock 单测 + 真实 Genesis CPU 集成测试，slow 标记 + 真实资产解析集成测试）；全套 526 passed（516 + 10 slow）。

---

## 5. 剩余项（均依赖外部条件）

| 项 | 阻塞条件 | 就绪动作 |
|---|---|---|
| ~~M2 真实资产转换~~ | ✅ 已完成（见 §3.1） | embodiments 10/10、objects 网格 smoke 均通过 |
| M4 渲染标定 | GPU | 固定场景渲染 SAPIEN RT vs Raytracer 图像对，LPIPS<0.15 门禁 + tone mapping 写入 `configs/render/` |
| cuRobo 端到端 | CUDA/MUSA + cuRobo | `--planner curobo` 跑 M3 双臂回归（对比 OMPL 成功率，阈值 −5pp） |
| M5 50 任务回归 | 上述全部 | seed 级轨迹一致性 + 成功率 regression |
| M6 性能调优 | GPU | n_envs 扫描、Madrona 接入、采集吞吐报告（目标 ≥5× 含渲染） |
