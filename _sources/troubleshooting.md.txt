# 排障与工程记录

本文保存从 `AGENTS.md` 移出的历史排障、性能测量和版本适配记录。路径均相对于仓库根目录；具体实现以当前代码为准。

## Genesis 1.4.0

- 后端 wrapper 暴露 `set_dofs_gains`，不暴露 `set_dofs_kp/kv`。实体在 spawn 时尚未 build，PD gains 和资产默认 qpos 必须在首次 `reset()` 时设置，不能在 `spawn()` 时设置。
- OMPL `plan_path` 在 RRT 静默失败时可能返回全零轨迹，使用 `plan_with_fallback` 防护。
- GPU 场景的 `get_pos/get_quat` 返回 CUDA tensor，通过 `backends/genesis_backend.py` 的 `_to_numpy_f64` 转换。
- Genesis build 时会合并 fixed-joint 链。Panda 的 `panda_hand` 会消失，EE link 是 `panda_link7`；合并后的手指碰撞体延伸到 link7 原点下方数厘米，descend goal 需要额外间隙。
- Rigid body 使用 `control_dofs_force([Fx,Fy,Fz,Tx,Ty,Tz])`（世界坐标系）；外力通过兼容层的 `apply_entity_force`，底层是 `RigidLink.apply_external_force/apply_external_wrench`。`gs.materials.Rigid` 使用 `rho=` 而非 `density=`，camera 必须在 `scene.build()` 前添加。
- 网格刚体质量来自凸分解后的碰撞体积；非水密扫描且 `collision=False` 时来自凸包。Wright Flyer 扫描曾因此从 340 kg 膨胀到 8.4–17.4 t，需在 `scene.build()` 后调用 `entity.set_mass()`，其惯性张量会按相同比例缩放。
- 297k-face Flyer STL 的 CoACD 分解约需 2.5 分钟，之后缓存于 `~/.cache/genesis/cvx/`。
- 角速度使用 `entity.get_ang()`（世界坐标系）；`get_angular_velocity()` 不存在。对该方法做 `hasattr` 防护会静默禁用依赖它的阻尼。

## 向量化吞吐与性能测量

### 实现注意事项

`core/vec_tasks.py::FrankaPickCubeVecTask` 使用 `core/vectorized.py` 的 batched-scene 路径。关键约束：

- PD gains 和资产默认 qpos 在首次 `reset()` 设置。
- 原始 Genesis entity 暴露 `set_dofs_kp/set_dofs_kv`，项目后端 wrapper 则暴露 `set_dofs_gains`，注意所在层级。
- Batched getter 可能返回 subset-width tensor，reshape 使用 `-1`，不要直接用 `num_envs`。
- `gs.init(backend=cuda)` 会把 torch 默认设备进程级设为 CUDA；CPU generator 的 `torch.rand/randn` 必须显式指定 `device="cpu"`。

### 2026-10-02 基线

本机 GTX 1650 4GB，默认配置从 1 到 256 envs 约 50 steps/s（20 ms/step）。`scene.step()` 占约 17 ms，obs/control/reward tensor 操作合计约 1.1 ms。

保持 solver rate 不变，将 dt 0.02×2 substeps 调为 0.01×1，并启用 `gs.init(performance_mode=True)`（`VecEnvConfig.performance_mode`），在 256 envs 达到约 187 steps/s、0.7 GB VRAM，约 3.7×。静态数组模式下无运行时场景编辑，确定性不受影响。

测量工具为 `tools/bench_vec_throughput.py`，完整数据在 `outputs/benchmarks/vec_throughput_20261002/`。

### 2026-10-04 ManiSkill3 对比

同机 ManiSkill3 3.0.1 / SAPIEN 3.0.3 / PickCube-v1 为 13–24 steps/s；本项目 batched pick-cube 为 88–119 steps/s。ManiSkill3 从 1 到 256 envs 时单步耗时由 45 增至 77 ms。

该机 Windows 环境下，ManiSkill3 sim+render 在 Vulkan GPU 和 CPU render backend 下均出现段错误。其公开 `peak_vram` 只统计 torch allocation，不含 PhysX/renderer pool，因此公开高端 GPU/Linux 指标不能直接外推到本机。数据位于 `outputs/benchmarks/ms3_rebench_20261004/`。

仍存在的结构性差距：Genesis single-scene batching 是同质的，异构场景需要多进程 sharding；batched rendering 仍需 Madrona。

### 2026-10-04 profiler 与 A/B

`torch.profiler` 显示该 GPU 上是 launch/host-bound：wall time 约 7–12 ms/step，CUDA busy 约 2 ms。项目当时默认 `noslip_iterations=5` 时，`kernel_noslip` 占 CUDA busy 约 80%，但 GPU 在 launch 间空闲，因此减少 kernel busy 对 wall-clock 帮助很小。

将默认改为 Genesis 自身的 `noslip_iterations=0` 后，在 1/16/64/256 envs 的配对 A/B 中吞吐变化处于 ±10% 噪声范围；scene build/init 则提速 1.5–3.5×，因为每个进程需要 JIT 的 solver kernel 更少。

`Newton iterations=15`、`enable_self_collision=False`、`use_hibernation=True` 未带来可靠 wall-clock 收益，保留原默认值和可调参数。数据中心 GPU 可能转为 kernel-bound，需要重新测量。

该桌面 GPU 相隔 30 分钟的批次可有 ±60% 漂移，只信任背靠背配对比较；跨天绝对值仅作数量级参考。数据位于 `outputs/benchmarks/vec_throughput_20261004_ab/` 和 `outputs/benchmarks/profile_vec_step_20261004/`。

## 本机 CUDA overlay

本机执行 `uv sync` 或未加 `--no-sync` 的 `uv run` 后，需要重新应用 CUDA wheel overlay，否则 lockfile 中的 CPU torch 会覆盖 CUDA 版本。

当前可用方式是直接安装 Aliyun wheel URL：

```bash
uv pip install \
  "https://mirrors.aliyun.com/pytorch-wheels/cu128/torch-2.11.0%2Bcu128-cp313-cp313-win_amd64.whl" \
  "https://mirrors.aliyun.com/pytorch-wheels/cu128/torchvision-0.26.0%2Bcu128-cp313-cp313-win_amd64.whl"
```

该 mirror 的 cu128 index 没有 torch 2.13.0，按 index 解析会回落到 `+cpu`。`tools/install_torch.py` 在该 venv 上受无 pip module 和依赖混淆保护影响而不可用。安装后用 `torch.cuda.is_available()` 验证。

xuvdb 自 1.3.3 起直接从 PyPI 安装，不再需要 editable `../xuvdb` overlay。

## WFC 场景

- 坐标旋转必须一致：tile 的 `rotated_sockets` / `ROT_ARROWS` 为顺时针，`PrimBuilder._to_world` 也必须顺时针。曾错误翻转后者，导致 rot 90/270 时家具与 socket 不对齐；per-tile fit tests 用于回归。
- `counters_ok` 和 entry-flow presets 依赖 `neighbors_inside` 返回 `(direction, cell)` pairs，修改时保持该契约。
- `assets/furniture_cn.py::prim_mesh` 曾未应用 `prim.pos`，导致 GLB 和 `bake_layout` 的网格全部堆在原点；仅做数量测试无法发现。修复位置后同时覆盖 bake_layout 和 GLB。
- 小尺寸 cell 重装扮时，低于 `TileSpec.min_edge` 应告警；未知 tile 应报错，不能静默接受。

## ReplicaCAD / ManiSkill

### Batching 与资源限制

`ReplicaCADEnv` 的 `num_envs=1` 是保守限制。`build()` 本身未使用 num_envs，entity 也无 `envs_idx`；batched scene 会复制整套公寓。32 个同质 env 曾在 4GB GPU 上 OOM，单个 apt_0 的凸分解构建曾超过 15 分钟。解除限制之外，还需要构建缓存和每 GPU env 数上限。

Genesis single-scene batch 无法承载每个 env 不同的场景，异构需求使用多进程 sharding。插件 `datasets` 子包采用绝对导入，只在 `plugins/envs/maniskill/core` 加入 `sys.path` 后可按顶层 `genesis_maniskill` 导入，因此 `scenes/replica_cad_scene.py` 使用延迟导入。

### SDF 模式

`ReplicaCADSceneBuilder(config={...})` 支持 `static_mode="mesh"|"sdf"`、`sdf_voxel_size`、`sdf_band`、`sdf_cache_dir`、`sdf_ground="box"|"none"` 和 `sdf_visual`。

SDF 模式中 stage 和 STATIC mesh 离开碰撞场景并 bake 到 query-only `builder.static_sdf`，绝不进入 Newton contact path；DYNAMIC 对象和关节家具不变。地面 box 是 30×30×0.1 m、顶面 z=0 的 slab，不能复现楼梯，楼梯任务必须使用 mesh 模式。示例为 `plugins/envs/maniskill/examples/replica_cad_scene_sdf.py`。

### 2026-10-16 验收

apt_0、CPU、dt=0.01×1 substep：

- Mesh 冷构建 1251.7 s，主要耗时为 stage/statics 的 CoACD。
- Mesh 热构建 96–99 s，使用 `~/.cache/genesis/cvx/` 缓存。
- SDF 构建 92.3 s，包含 0.05 m SDF bake，约 830k active voxels。

当时 mesh 模式支撑异常：510 个 object-episodes 中 345 个穿过 non-convex stage 自由下落，final z 均值 −22.1，最差 −49.7。SDF 模式恢复到 slab 上的合理支撑，final z 均值 +0.16，5/510 次穿透。

### 2026-10-18 双重旋转根因

穿透不是缺 contact，而是坐标系双重旋转：

- Genesis glTF import 后的 mesh geometry 已经是 Z-up，GLB node transform 包含 habitat→Z-up 归一化。
- `habitat_pose_to_genesis` 当时从 ManiSkill 移植了左乘 RotX(+90°)，即 `S⊗q`。
- 该姿态与已旋转几何组合后，整套公寓再绕世界 x 轴倾斜 90°，13 m 轴变为竖直方向，3 m 高度轴变为水平方向。
- 对象按共轭平移生成，却处于倾斜壳体内；重力沿水平 habitat 轴作用，对象从 stage 末端滑出。Broadphase AABB 不重叠，因此 literally 0 contacts；曾探到 box-bottom SDF 穿透 −5 cm 仍无 contact。

修复：Genesis 加载实体使用共轭 `S⊗q⊗S⁻¹`，平移仍为 `S·t`。SDF bake 路径处理的是 RAW trimesh habitat-frame vertices，继续通过 `habitat_pose_to_bake` 左乘，原路径一直正确。

修复后，box 从 apt_0 地板上方 0.3 m 落下，稳定于 z=0.030，有 4 个向上法向的稳定 contact。

### 修复后验收与残留问题

相同 harness，broad-phase `multiplier_collision_broad_phase=64`、`max_collision_pairs=1024`：

- Mesh @ substeps=1：两阶段分别 5/510、6/510 穿透，final z 均值 −0.21/−0.05。
- 剩余 faller 是约 1 cm 厚的 `frl_apartment_choppingboard_02-54`。下落约 0.25 m 后以约 2 m/s 到达，每个 0.01 s step 走 2 cm；impact contacts 会触发（探到 8 contacts、penetration 约 1.5 cm、stored normal `[0,0,−1]`），但未使其减速。这是 impact-handling 边界情况，不是 missing-contact。
- 新增 `ReplicaCADEnv(sim_substeps=N)`，不改变 control dt。
- Mesh @ substeps=4：两个阶段均 0/510 穿透，final z 均值 +0.36/+0.38，无 NaN/runaway。
- SDF @ substeps=4 曾在每回合约产生 1 对 spawn-overlap movable pair，导致 xy_runaways 5/5、z 最高 11 m；与坐标修复无关，SDF 模式保持 substeps=1。
- SDF 的 slab 语义与 mesh 的家具支撑语义不同，final z 不应直接视为等价。

诊断脚本：`outputs/repro_v3.py`、`outputs/probe_board.py`、`outputs/probe_board_s4.py`、`outputs/which_fall.py`。约 120-entity 公寓会超过默认 broad/narrow phase 预算；全强度随机动作即使提高预算也会淹没 broad-phase buffer，扰动应保持温和。

`PickPlaceTask` 指向 `env.objects[0]`，而该对象是 STATIC furniture，因此 ReplicaCAD 上其 success flag 无意义；抓取成功率仍需 OMPL/RL 策略评估。

## Sparse VDB / xuvdb

- torch↔xuvdb 转换经过 dense numpy。xuvdb `to_dense` 返回 tight ACTIVE bbox `(dense, ijk_min)`，原点可能为负；`from_xuvdb` 将 torch domain 向上取整到 block 倍数。
- `xuvdb.read_vdb` 返回 grid list，不是单个 grid。
- 薄墙只有 1–2 voxel 厚时，trilinear sampling 可能把负内部插值掉；符号检查应 census leaf values，不能依赖随机线性采样。
- xuvdb host `VdbGrid.sample_linear/nearest/quadratic` 曾将 batched input 截断到第一个点。项目的 `XuvdbVolume.sample_*` 已改为 batch-aware：单点走 host path，批量走 kernel-side `GpuVolume.sample`，无法 pack 的 grid（例如 f64 ShapeKernel bridge）使用 per-point host fallback。不要直接用原始 host API 处理 batch。
- `RandomizedRoomSDF` 通过同一 kernel-side 路径查询；回归测试位于 `tests/vdb/test_xuvdb_interop.py`。
- ShapeKernel 在 mm world 中构造，union 后 `rescale_grid(1e-3)`；values、voxel_size、origin、background 均线性缩放。

## Grid solver / CFD / nozzle

### Quadrants 1.3

- Kernel 内不接受 field alias，例如先 `x = self._field` 再 `x[i]`；直接引用 field。
- 不能 autodiff host-precomputed constant fields，材料参数需 inline 到 kernel，参考 `joule_heating._heat_step_*`。
- Runtime `if` 中不要使用 return。
- Source-coupling backward path 仍受上游 quadrants AD 问题阻塞，见 `scripts/quadrants_issue_thermal_source_autodiff.md`。

### CFD coupling

- 压力方程使用比压 `p/ρ`，压头是 `p/g`，不是 `p/(ρg)`。
- 两个 backend 的 autodiff hooks 都是 no-op。
- Immersed obstacle 必须避开 `x=0/x=Lx` 面。
- `add_obstacle(..., track=pose_fn)` 每个 substep 重新 rasterize，属于单向耦合。
- Genesis 测试保持每进程一次 `gs.init`；同一文件中的 Genesis 测试应放在同模块运行。

### Nozzle coupling

准稳态 nozzle 在 `p_back>=p_c` 时 mdot 可突降到零，接口必须双向 slew-limit：`backpressure_max_rate_pa_s` 和 `inlet_mdot_slew_kg_s2`，否则会在 blocked/choked 间 limit-cycle。

高温超声速射流由动量主导且出口压力匹配，冷流或低压力比配置更容易产生明显 3D→1D backpressure 信号。3D step 受出口射流速度 CFL 限制，通常要求 `u_e*dt/h ≲ 0.5`。

## 模拟设备与资产

- SharpaHand 的 vendored URDF 需要先将 `package://` mesh path 改写为绝对路径。
- Genesis 合并后，指尖 link 位于 `*_DP` distal-phalanx link，不再使用原独立 fingertip 名称。
- Articulation 自动标注中，低置信度、验证失败和 verifier 异常分别进入 review/quarantine，不能静默接受。Genesis 相邻 link 默认无 self-collision，因此镜像 hinge-side 错误需要显式几何穿透检查才能发现。
- `result_to_urdf` 输出 placeholder masses，需在 `scene.build()` 后调用 `set_mass()`。
