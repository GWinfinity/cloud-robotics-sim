# RoboDojo 对标 P0 优先级计划

> 来源:`docs/ROBOTWIN_TO_ROBODOJO_GAP_ANALYSIS.md`(2026-08-26)
> P0 定义:**解锁后续一切工作、且能端到端验证流水线**的最小切片。
> 原则:先做骨架和闭环,不做全量;全部建立在现有 `core/`(Registry / Composer / Task)与 `robotwin/`(recorder / seed_search / grasp_report)设施之上,不新造轮子。
>
> 定稿核实:2026-09-26 对全部代码锚点逐条核实(记录见附录 A)。结论:P0 各工作项均未动工、设施类断言全部成立,唯一过时断言(franka_pickplace.yaml 无人消费)已在 W1 修正。
>
> **与 AnyTask 最小能力集方案的关系**:`docs/IMPROVEMENT_PLAN.md`(2026-09-14,更新)把范围收敛到 AnyTask 最小能力集(约 8–12 人周),显式排除 gap 表 #2(42 任务全量)、#3(资产标注)、#6(VR 遥操作)、#10(RealEval)。两份方案重叠项裁决如下:
>
> | 本计划项 | AnyTask 方案对应 | 裁决 |
> |---|---|---|
> | W0 GPU 解锁 | Phase 0 前置 | 同一事项,谁先行谁落地,不重复投入 |
> | W1 任务 YAML 框架 | Phase 0/1 部分(gap #1) | 不冲突:W1 要的是任务级 schema,AnyTask 未覆盖(见 W1 修订说明) |
> | W3 资产标注 | 显式非目标(gap #3) | **挂起**:仅当路线决策转回“全量对标 RoboDojo”时恢复 |
> | W4 cuRobo 端到端 + grasp/place | Phase 2/3(gap #4) | 重叠,视为同一件事,不做两遍 |
> | W8 评测协议 + 聚合 | Phase 0 基准套件 | 重叠:AnyTask 的 3 任务基准套件可视为 W8 的最小试点,向上兼容 |
> | W2 pilot 任务 5–8 | 非目标(gap #2) | **收敛**:AnyTask 路线下以 Phase 0 的 3 个 contact-rich 基准任务代替(见 W2 修订说明) |

## 0. P0 范围划定

| 纳入 P0 | 排除出 P0(留给 P1/P2) |
|---|---|
| W0 GPU/CUDA 环境解锁(硬前置) | 42 个任务全量移植(P1,流水线验证后批量铺开) |
| W1 配置驱动任务系统(loader + reset 采样 + 成功判定抽象) | 进程内异构并行架构改造(P1 先用多进程分片) |
| W3 资产标注 **schema + 工具 + 10 类试点**(非全量 129 类) | 全量资产标注(P1 数据工) |
| W4 cuRobo 端到端打通 + grasp/place 两个技能原语 | 其余 6 个技能(handover/insert/open/close/stack/push_up,P1) |
| W8 评测协议 + 结果聚合(复用 grasp_report 格式) | XPolicyLab 完整策略服务器(P1) |
| W2 **5–8 个 pilot 任务**端到端验证 | VR 遥操作(P2)、RealEval 真机平台(P2) |

## 1. 工作流分解(按依赖排序)

### W0 — GPU/CUDA 环境解锁(第 0 周,阻塞项)

- 内容:CUDA 或 MUSA 环境到位;`tools/install_torch.py` 选对 wheel;cuRobo v2 安装;LuisaRender / Madrona 后端冒烟。
- 验收(2026-09-26 修订):原验收“`-k curobo` 不再全 skip”已失效——`tests/robotwin/test_curobo_planner.py` 现采用 stub 注入的 CUDA-free 设计(14 个用例无 CUDA 也照跑)。修订后验收 = **真实 CUDA/MUSA 环境端到端**:`examples/robotwin/aloha_demo.py --planner curobo` 跑通(退出码 0,非 3),并在至少一个真实抓取场景输出 cuRobo 优先、OMPL 兜底的成功率对比数据。
- 工作量:0.5–1 人周(不含采购)。**没有它,W4/W6 全部停摆。**

### W1 — 配置驱动任务系统(第 1–3 周,P0 核心)

> **状态(2026-09-26):已落地。** 交付物 1–5 全部实现:`core/task_loader.py`(任务级 schema v1 + loader + 确定性 reset 采样 + `ConfigurableTask` 包装)、`core/success.py`(distance_threshold / pose_window / staged 三种判定器)、`configs/tasks/pick_place_cube.yaml`(schema v1 参考实现)。测试 `tests/core/test_success.py` + `tests/core/test_task_loader.py` 共 56 用例(Genesis-free)。偏差记录:schema 在计划字段之外新增了 `task.scene`/`task.type`(注册表工厂名,沿用 config_loader 惯例);`cameras`/`randomization` v1 仅解析携带,由 W8 消费;reset 采样与 `batched_seed_search` 的对接通过 `make_apply_seed_fn(spec, scenes)` 回调适配器完成。

现状缺口(2026-09-26 核实修正):`core/registry.py`(装饰器注册)+ `core/composer.py`(`compose_from_registry`)+ `core/task.py`(gym 风格 Task ABC)已齐,**任务级 YAML→loader 仍缺**。

> ⚠️ 原断言“`configs/franka_pickplace.yaml` 无任何代码消费”**已过时**:该 YAML 现已被 `core/config_loader.py::load_sim_config` 消费(improvement loop 经 `runtime/main.py::make_env_from_config` 调用,含 `environment.scene/robot/task/simulation` 结构校验),硬编码绝对路径问题亦已由 `core/robot_assets.py` 自动解析(`robot.urdf_path` 可为 null)解决。但现有 loader 是**环境级**配置(单实验视角),与本工作项要的**任务级** schema(`task/assets/init_distribution/randomization/success/evaluation`,批量评测视角)不同层。因此 W1 从“从零建 loader”收窄为“在 `config_loader.py` 旁新增任务级 schema 与 loader,两者字段不合并、各管一层”。

交付物:

1. **任务 YAML schema v1**(`configs/tasks/*.yaml`),对齐 RoboDojo 字段:
   ```yaml
   task: {name, robot, max_episode_steps}
   assets: {task_relevant: [...], distractors: [...]}   # 指向 object_library + 标注
   init_distribution: {object_poses, articulation_states, clutter_layout}
   randomization: {friction, mass, com, lighting, background_texture}
   cameras: [...]              # 复用 robotwin CameraConfig
   success: {type: ..., params: {...}}   # 见交付物 3
   evaluation: {seeds, episodes_per_seed}
   ```
2. **Loader**:`src/cloud_robotics_sim/core/task_loader.py` — YAML → `ComposerConfig` + Scene + Robot + Task,走现有 `AssetRegistry`/`compose_from_registry`(环境级路径解析复用 `core/config_loader.py`/`core/robot_assets.py`,不再重复解决)。明确**选用装饰器 Registry 一套**,plugin.yaml 体系不做桥接(P0 不碰)。
3. **可配置成功判定器**:`core/success.py` — 抽象 `SuccessCondition`(distance_threshold / pose_window / staged),staged 模式直接复用 `robotwin/grasp_report.py` 的 `FAIL_STAGES` 七级(现为 7 元素 tuple 而非 enum.Enum,`grasp_report.py:14-22`,可直接引用其字符串常量);替代各 Task 子类里硬编码的阈值逻辑(保持旧子类行为不变,仅新增)。
4. **确定性 reset 采样器**:按 `evaluation.seeds` 采样布局,复用 `batched_seed_search` 的回调接口;同一 seed 必现同一场景。
5. 测试:loader round-trip、seed 确定性(同 seed 两次 reset 状态全等)、每种 success type 各一个单测。

工作量:**3–4 人周**。验收:一个 YAML 从零定义一个新任务,不写 Python;`pytest tests/core` 全绿 + ruff/black/mypy 过门禁。

### W3 — 资产标注 schema + 工具 + 试点(第 2–4 周,与 W1 并行)

> **状态(2026-09-26):挂起。** AnyTask 最小能力集方案(`docs/IMPROVEMENT_PLAN.md`)已将资产标注(gap #3)列为显式非目标;本工作项仅在路线决策转回“全量对标 RoboDojo”时恢复。核实确认:仓库内无任何先行标注工作(仅 `tools/convert_assets.py` 透传 RoboTwin 源树关键点文件,非本项目标注 schema)。

- 标注 schema(`model_dataN.json` 扩展或旁挂 `annotations.yaml`):`graspable_regions / placement_regions / functional_parts / success_annotations`,字段对齐 RoboDojo affordance 层。
- 工具:`tools/annotate_asset.py` — Genesis 可视化点击/框选标注,写回 JSON。
- 试点:pilot 任务涉及的 **~10 个物体类**完成标注,验证 schema 够不够用。
- 工作量:2–3 人周(schema+工具 1.5,试点标注 0.5–1)。全量 129 类标注是 P1 数据工,不在此列。

### W8 — 评测协议 + 结果聚合(第 3–4 周)

> **状态(2026-09-26):已落地并端到端验证。** `core/benchmark.py`(EpisodeRecord 泛化记录、suite YAML v1 加载器、`run_episode`/`run_benchmark` 执行循环、leaderboard `summarize` + `report.json`/`report.md`/`episodes.jsonl` 三件套)+ `tools/run_benchmark.py` CLI + `data/suites/pick_place_smoke.yaml` 示例套件。策略可插拔,v1 内置 `zero`/种子化 `random`。关键设计:为满足"两次同 seed 运行 report.json 逐字节一致",`report.json` 只含确定性字段(状态/步数/reward),墙钟耗时与轨迹路径只进 `episodes.jsonl`——**已用真实 Genesis CPU 运行两次实测字节一致**。测试 `tests/core/test_benchmark.py` 22 用例(fake-env 端到端含字节复现验证)。
>
> **顺带修复的预存阻塞**(原改进回路同样 crash,与 W1/W8 无关):`core/embodiment.py` 三处——(1) franka/UR5 URDF 以浮置底座加载(`fixed=True` 缺失),机器人掉落触发约束求解器 NaN;(2) `reset()` 全零 qpos 是 franka 自碰撞奇异位形,改为恢复 spawn 时捕获的资源默认位形;(3) spawn 后未下发 PD 增益,手臂重力下失稳;(4) `get_qvel` 在 genesis 1.4 已改名,新增 `_dof_velocity()` 兼容层。真实烟跑:`run_benchmark.py --suite pick_place_smoke --episodes-limit 2` 跑通 2 episodes x 200 步,place_fail 经 ConfigurableTask 正确传递。轨迹录制已补(`--save-trajectories`,per-episode actions/rewards JSON 落盘 `trajectories/` 并记入 episodes.jsonl,路径用 POSIX 分隔符保证跨平台一致)。剩余待办:按维度聚合(更多任务维度落地后)、真实机器人策略接入(依赖 W4)。

- 评测 runner:`tools/run_benchmark.py` — 读 suite YAML(参照 `data/suites/home_5s_suite.yaml` 的 recipes/assets/validation 三段式),按 `seeds × episodes` 跑任务,逐条写 `GraspRecord` 泛化版(任务名/seed/状态/耗时/轨迹路径)。
- 聚合:`summarize` 扩展为 leaderboard 表(任务 × 维度 × 成功率 + failures_by_stage),输出 `report.json` + `report.md`,一键复现。
- 工作量:1.5–2 人周。验收:两次同 seed 运行 report.json 逐字节一致。

> 现状核实(2026-09-26):`tools/run_benchmark.py` 仍不存在;`grasp_report.summarize` 目前仅有按物体类别的单次抓取聚合(total/success_rate/failures_by_stage),无任务 × 维度 leaderboard——本工作项保持空缺、未动工。AnyTask 方案的 Phase 0 基准套件(固定 seed 成功率统计)若先落地,可直接作为本 runner 的第一个 suite 输入。

### W4 — cuRobo 端到端 + 两个技能原语(第 4–6 周,依赖 W0/W1/W3)

- `HierarchicalCuRoboPlanner.plan_with_fallback` 端到端回归:cuRobo 优先、OMPL 兜底,对比成功率(阈值 −5pp,沿用迁移文档 M3 标准)。
- 技能层 `core/skills/`:`grasp` + `place` 两个原语,从 W3 标注读 graspable/placement regions,技能编排器(有序技能序列 → 轨迹)先做最简版。注意命名冲突:仓库已有 `src/cloud_robotics_sim/runtime/skills.py`(agent 技能层,专利/仿真任务调度),与本处的机器人技能原语无关——新建目录建议用 `core/robot_skills/` 避免混淆。
- 工作量:3–5 人周(含 CUDA 环境调试余量)。

### W2 — Pilot 任务 5–8 个(第 5–8 周,依赖 W1/W3/W4)

> **状态(2026-09-26):AnyTask 路线下收敛。** AnyTask 方案以 3 个 contact-rich 基准任务(立方体抓取静置/叠方块/抽屉开合)作为 Phase 0 反馈环,与本工作项重叠;AnyTask 路线下 W2 收敛为这 3 个基准任务,5–8 个 YAML-only pilot 任务仅在转回全量对标路线时恢复(且仍需 W1 框架先行)。
>
> **落地进展(2026-09-26):** 其中 2 个已用 W1 schema 零 Python 落地并被 W8 runner 端到端验证:`configs/tasks/pick_place_cube.yaml`(抓取放置,distance_threshold)+ `configs/tasks/stack_cubes.yaml`(双物堆叠,staged + pose_window 组合,成功验证 staged 判定的失败阶段归因:stack 未达成时报最早失败 stage grasp_fail)。套件 `data/suites/contact_rich_smoke.yaml`,真实 CPU 双跑 report.json 字节一致。**抽屉开合阻塞**:schema v1 的 `articulation_states` 已能采样并随布局携带,但 ObjectSpawn v1 无关节资产类型(PartNet-Mobility 类资产需要),留待 articulated spawn 支持后补。真实机器人策略(非 zero/random)仍依赖 W4。

- 选型原则:覆盖 RoboDojo 五维能力中的三维(Generalization / Precision / Long-Horizon),全部用已转换的 5 本体 + 已标注物体,如:单物 pick-place、杂乱场景 pick-place、双物 stack、双臂 handover(ALOHA)、抽屉开合(PartNet-Mobility 关节资产)。
- 每个任务 = 1 个 YAML + 标注 + 成功判定配置,**不允许写任务专用 Python**(倒逼 W1 框架完备)。
- 每个任务配回归测试(seed 级成功率门禁)。
- 工作量:4–6 人周。这是 P0 的"验收仪式":流水线能不能批量产任务,P1 能否直接铺 42 个,全看这一步。

## 2. 里程碑与时间线(2 名工程师基线)

```
周:     0    1    2    3    4    5    6    7    8
W0 GPU  ██
W1 框架      ████████
W3 标注         ████████
W8 评测              ████
W4 技能                     ████████
W2 Pilot                       ████████████
              ↑              ↑              ↑
            M-a            M-b            M-c(P0 完成)
```

- **M-a(第 2 周末)**:任务 YAML schema v1 + loader 单测通过 —— 后续一切的接口冻结点。
- **M-b(第 4 周末)**:标注工具可用 + 评测 runner 能出 report —— 数据工和批量评测可开工。
- **M-c(第 8 周末)**:5–8 个 pilot 任务在评测 runner 上跑出 leaderboard,同 seed 完全可复现。
- **总计:约 13–20 人周;2 人 8 周 / 3 人 5–6 周。**

## 3. 风险与对策

| 风险 | 对策 |
|---|---|
| GPU 不到位,W0 滑期 | W1/W3/W8 全部不依赖 GPU,可先行;W4 是唯一被卡的,必要时用 OMPL 兜底先跑 pilot |
| YAML schema 设计不足,任务写到一半要加字段 | M-a 后 schema 冻结但留 `extra:` 扩展字段;pilot 任务故意选 3 个维度倒逼 schema 完备 |
| 标注工具没人愿意用(体验差) | 工具本身预算 1 人周封顶;试点 10 类验证够用即可,全量标注 P1 再议(可考虑外包/半自动) |
| Genesis n_envs 同构限制,pilot 批量评测慢 | P0 接受慢;多进程分片(每进程一任务)是 P1 第一项 |
| 与 RoboDojo 语义对齐过度 | 字段名对齐其论文附录 F.1 即可,不追求配置互导——目标是对标能力,不是兼容格式 |

## 4. P0 完成后(P1 入口条件)

- 任务流水线已验证 → P1 批量移植剩余 ~35 任务(流水线作业,2–5 人天/个)
- 全量资产标注(数据工,可外包)
- 多进程任务分片 + 剩余 6 技能原语
- 差异化投入启动:柔体任务维度(Genesis MPM/SPH,对标 RoboDojo 结构性弱点)

## 附录 A — 2026-09-26 定稿核实记录

| 计划断言 | 核实结论 |
|---|---|
| franka_pickplace.yaml 无人消费、有硬编码绝对路径 | ❌ 已过时:已被 `core/config_loader.py::load_sim_config` 消费(improvement loop 经 `runtime/main.py::make_env_from_config`);路径已由 `core/robot_assets.py` 自动解析。W1 已改写 |
| Registry / Composer / Task ABC 已齐 | ✅ `core/registry.py`(装饰器注册)、`core/composer.py:338` `compose_from_registry`、`core/task.py:41` `Task(ABC)` |
| 任务级 loader、`core/success.py`、`tools/annotate_asset.py`、`tools/run_benchmark.py`、`configs/tasks/`、`core/skills/` | ✅ 均不存在——W1/W3/W8/W4 技能层未动工 |
| `FAIL_STAGES` 七级枚举 | ✅ 七级成立;类型为 7 元素 tuple 而非 enum.Enum(`grasp_report.py:14-22`) |
| `data/suites/home_5s_suite.yaml` recipes/assets/validation 三段式 | ✅ 成立 |
| `HierarchicalCuRoboPlanner.plan_with_fallback` cuRobo 优先/OMPL 兜底 | ✅ `curobo_planner.py:247`;测试已改为 stub 注入 CUDA-free(14 用例),故 W0 验收标准已修订 |
| `aloha_demo.py --planner {auto,ompl,curobo}` | ✅ 成立 |
| `batched_seed_search` 回调接口可复用 | ✅ `seed_search.py:69`(apply_seed/rollout/evaluate 回调) |
| 资产标注先行工作 | ❌ 无(仅 convert_assets 透传 RoboTwin 关键点文件,非本项目标注 schema) |
| 测试规模 | ✅ 全仓约 1050 个测试函数(tests/core ~187、tests/robotwin ~145) |
| `core/vectorized.py` | ⚠️ 已被 AnyTask Phase 1 重写为真实批量入口(VecTask/`reset_idx`/物理参数透出,占位 `np.zeros` 已删);rsl-rl/LeRobot 接入未做(与 AnyTask 方案文档自述一致) |
