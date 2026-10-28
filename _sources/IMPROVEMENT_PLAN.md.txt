# AnyTask 最小能力集改善方案

> 制定日期: 2026-09-14
> 上游依据: `docs/ROBOTWIN_TO_ROBODOJO_GAP_ANALYSIS.md`(gap 表自估追平 RoboDojo 全集需 55–95 人周)
> 总体思路: **不追平 RoboDojo 全集,只收敛到 AnyTask 最小能力集**;四个阶段串行推进,接触保真度单独走"基准驱动 + 止损判据"的旁路线,GPU 环境作为第一优先级前置解决。
> 总量: 约 8–12 人周。
> 状态(2026-09-14): Phase 1 第 1 条已完成——`GenesisVectorizedEnv` 已重写为真正的批量入口(新增 `VecTask` 批量任务接口、掩码式 `reset_idx`、物理参数透出到 `VecEnvConfig`),占位 `np.zeros` 实现已删除;GPU 上的 4096-envs 实测与 PPO 收敛验收仍待 GPU 环境到位。

---

## 与 gap 文档的对齐关系

| 本方案阶段 | 对应 gap 表条目 |
|---|---|
| Phase 0/1(数据流水线) | #1 配置驱动任务系统(部分)、#9 前置解锁(GPU) |
| Phase 2(抓取管线) | #4 技能库 + demo 合成(收敛为 grasp 单技能) |
| Phase 3(规划与渲染) | #4 cuRobo 端到端、#7 策略接口(蒸馏 student 雏形) |
| Phase 4(接触保真度) | #9 M6 性能/保真调优(以基准套件替代全量回归) |
| 非目标 | 显式排除 #2(42 任务)、#3(资产标注)、#6(VR 遥操作)、#10(RealEval) |

依赖关系: `Phase 1 → Phase 2 → (Phase 3 ‖ Phase 4)`;Phase 0 的 GPU 前置贯穿始终。

---

## Phase 0 — 前置与度量(约 1 周,与其他阶段部分并行)

| 事项 | 说明 |
|---|---|
| 落实 GPU/CUDA 环境 | 单卡 A100/H100 即可。它是三个 blocked 项(cuRobo 端到端、批量渲染、任务回归)的共同前置,不解决则 Phase 2/3 全部空转 |
| 建 contact-rich 微基准套件 | 立方体抓取静置、叠方块、抽屉开合三个任务,固定 seed、输出成功率统计。这是 Phase 4 调优的反馈环和止损判据的度量工具;工作流与 `examples/robotwin/`、`outputs/*_probe.txt` 这类探针同构,顺手 |

---

## Phase 1 — 数据流水线(P0,约 2–3 周)

目标: 把 `GenesisVectorizedEnv`(`src/cloud_robotics_sim/core/vectorized.py`,现为返回 `np.zeros((n_envs, 23))` 的占位壳)变成真正的批量训练入口,这是其余四项中三项的地基。

1. **重写 `GenesisVectorizedEnv`**: 删掉占位实现,以 Genesis 官方 `examples/manipulation/grasp_env.py` 为模板——`scene.build(n_envs=...)`、`_reset_idx(envs_idx)` 掩码式 reset、obs/reward 全部 `(n_envs, ...)` torch tensor 常驻 GPU。
2. **接 rsl-rl**: 照 Genesis 官方 `examples/locomotion/go2_train.py` 的 `OnPolicyRunner` + PPO 范式,不自建 gymnasium 封装。
3. **数据落盘**: 用 LeRobot 格式 writer 替换 `CRS_DATASET_PIPELINE` 现在的缓存清理占位(`cache_cleanup` 目前只 stage 目录、无数据集写出),打通"仿真→数据集→训练消费"。
4. **接触参数暴露**: 把 `RigidOptions` 的关键面(`noslip_iterations=5`、`integrator=implicitfast`、dt/substeps)透出为 cloud-sim 任务配置项。

**验收**: 4096 envs 下 step time 有实测数字;PPO 在抓取任务上收敛;与 IsaacLab 同任务吞吐比 ≥ 1/5(达不到就先做 profiler 再谈下一步)。

---

## Phase 2 — 抓取管线(约 3–4 周,依赖 Phase 1)

目标: 把"单点天顶吸盘重放"升级为"候选生成 → 过滤 → 批量校验"的完整管线。

1. **候选生成**: trimesh 上 antipodal 采样(夹爪)+ 表面法线聚类(吸盘),CPU 多进程,不进仿真热路径。
2. **静力学过滤**: 批量 IK 可达性(`src/cloud_robotics_sim/robotwin/curobo_planner.py` 中 `inverse_kinematics` 现成,支持 `envs_idx` 批量)+ RRT 碰撞(`plugins/envs/sky/core/genesis/utils/path_planning.py` 的 GPU RRTConnect 现成)。
3. **动力学校验**: 候选塞进批量 scene 逐个跑 settle→lift→hold,success mask 汇总——骨架直接复用 `src/cloud_robotics_sim/robotwin/seed_search.py` 的 `batched_seed_search`。
4. **suction 模型升级**: 给 `src/cloud_robotics_sim/robotwin/suction_grasp.py` 加接触力阈值判据(`get_contacts` 现成),替代纯 `set_qpos` 钉住,否则 sim2real 会被骗。

**验收**: 单物体千级候选分钟级筛完;静力学过滤 top-k 与动力学校验结果的一致率有统计数字。

---

## Phase 3 — 规划与渲染(约 2–4 周,依赖 GPU)

1. **cuRobo 端到端打通**: `curobo_planner.py` 已有分层(cuRobo 优先、OMPL 兜底),补的是端到端验证 + 产出轨迹在 Genesis 前向仿真中的执行校验闭环。
2. **collision mask API**: 在 fork 的 broadphase 加逐实体 bitmask(参考 MuJoCo 的 contype/conaffinity),这是五项里唯一的内核级新增;改动尽量做成可推上游的形态,控制 fork 维护面。
3. **DP3 观测**: `render_all_cameras`(返回 `(n_envs,H,W,3)`,见 `plugins/envs/sky/examples/rigid/single_franka_batch_render.py` 用法)+ torch batched unproject 出点云;训练策略走 **privileged distillation**——teacher 纯 state 零渲染,student 低频(5–10Hz)渲染蒸馏,把在线渲染需求降一个量级。

**验收**: cuRobo→Genesis 轨迹执行成功率;student 与 teacher 成功率差 ≤ 5 个百分点。

---

## Phase 4 — 接触保真度(持续,与 Phase 2/3 并行)

用 Phase 0 的基准套件驱动参数扫描: `implicitfast` + `noslip=5` 起步,扫 dt/substeps、摩擦锥、coacd 凸分解、质量比约束;任务侧用 action 低通兜底。

**止损判据(预先定死,不事后挪)**: 调参后基准成功率仍 < IsaacLab 同任务的 70% → 停止单引擎执念,转双引擎架构: IsaacLab 跑接触重的 RL,Genesis 保留可微物理(fork 的 PBD/grad 主线)与渲染卖点。

---

## 非目标(明确不做)

- 不自建 gymnasium/skrl 封装层——官方范式就是手写 env + rsl-rl;
- 不把 Genesis 补成 PhysX——那是上游数年的路;
- 不追 50 任务全量回归——先在 3 个基准任务上做到可信。

## 主要风险

| 风险 | 缓解 |
|---|---|
| GPU 环境持续缺位 | Phase 0 单列;云上单卡成本可控,属组织决策不属技术问题 |
| 接触保真度天花板 | 止损判据已预定义(70% 线) |
| fork 与上游演进冲突 | 内核改动只留 collision mask 一项,做上游可推形态 |

## 关键代码锚点(已核实,2026-09-14)

| 引用 | 实际位置 |
|---|---|
| 占位 vectorized env | `src/cloud_robotics_sim/core/vectorized.py:88`(`np.zeros((n_envs, 23))` 于 :139/:153) |
| grasp_env / go2_train 模板 | Genesis 上游官方 examples(本仓库未 vendored) |
| GPU RRTConnect | `plugins/envs/sky/core/genesis/utils/path_planning.py:650` |
| cuRobo 分层规划器 | `src/cloud_robotics_sim/robotwin/curobo_planner.py` |
| 批量种子搜索 | `src/cloud_robotics_sim/robotwin/seed_search.py:69` `batched_seed_search` |
| 吸盘抓取 | `src/cloud_robotics_sim/robotwin/suction_grasp.py` |
| 批量渲染 | `plugins/envs/sky/core/genesis/engine/scene.py` `render_all_cameras` |
