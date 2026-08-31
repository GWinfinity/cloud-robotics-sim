# RoboDojo 对标 P0 优先级计划

> 来源:`docs/ROBOTWIN_TO_ROBODOJO_GAP_ANALYSIS.md`(2026-08-26)
> P0 定义:**解锁后续一切工作、且能端到端验证流水线**的最小切片。
> 原则:先做骨架和闭环,不做全量;全部建立在现有 `core/`(Registry / Composer / Task)与 `robotwin/`(recorder / seed_search / grasp_report)设施之上,不新造轮子。

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
- 验收:`uv run python -m pytest tests/robotwin -k curobo` 不再全 skip;`examples/robotwin/aloha_demo.py --planner curobo` 跑通一次。
- 工作量:0.5–1 人周(不含采购)。**没有它,W4/W6 全部停摆。**

### W1 — 配置驱动任务系统(第 1–3 周,P0 核心)

现状缺口:`configs/franka_pickplace.yaml` 存在但**无任何代码消费它**;`core/registry.py`(装饰器注册)+ `core/composer.py`(`compose_from_registry`)+ `core/task.py`(gym 风格 Task ABC)已齐,缺一个 YAML→loader。

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
2. **Loader**:`src/cloud_robotics_sim/core/task_loader.py` — YAML → `ComposerConfig` + Scene + Robot + Task,走现有 `AssetRegistry`/`compose_from_registry`;修掉 franka_pickplace.yaml 的硬编码绝对路径。明确**选用装饰器 Registry 一套**,plugin.yaml 体系不做桥接(P0 不碰)。
3. **可配置成功判定器**:`core/success.py` — 抽象 `SuccessCondition`(distance_threshold / pose_window / staged),staged 模式直接复用 `robotwin/grasp_report.py` 的 `FAIL_STAGES` 七级枚举;替代各 Task 子类里硬编码的阈值逻辑(保持旧子类行为不变,仅新增)。
4. **确定性 reset 采样器**:按 `evaluation.seeds` 采样布局,复用 `batched_seed_search` 的回调接口;同一 seed 必现同一场景。
5. 测试:loader round-trip、seed 确定性(同 seed 两次 reset 状态全等)、每种 success type 各一个单测。

工作量:**3–4 人周**。验收:一个 YAML 从零定义一个新任务,不写 Python;`pytest tests/core` 全绿 + ruff/black/mypy 过门禁。

### W3 — 资产标注 schema + 工具 + 试点(第 2–4 周,与 W1 并行)

- 标注 schema(`model_dataN.json` 扩展或旁挂 `annotations.yaml`):`graspable_regions / placement_regions / functional_parts / success_annotations`,字段对齐 RoboDojo affordance 层。
- 工具:`tools/annotate_asset.py` — Genesis 可视化点击/框选标注,写回 JSON。
- 试点:pilot 任务涉及的 **~10 个物体类**完成标注,验证 schema 够不够用。
- 工作量:2–3 人周(schema+工具 1.5,试点标注 0.5–1)。全量 129 类标注是 P1 数据工,不在此列。

### W8 — 评测协议 + 结果聚合(第 3–4 周)

- 评测 runner:`tools/run_benchmark.py` — 读 suite YAML(参照 `data/suites/home_5s_suite.yaml` 的 recipes/assets/validation 三段式),按 `seeds × episodes` 跑任务,逐条写 `GraspRecord` 泛化版(任务名/seed/状态/耗时/轨迹路径)。
- 聚合:`summarize` 扩展为 leaderboard 表(任务 × 维度 × 成功率 + failures_by_stage),输出 `report.json` + `report.md`,一键复现。
- 工作量:1.5–2 人周。验收:两次同 seed 运行 report.json 逐字节一致。

### W4 — cuRobo 端到端 + 两个技能原语(第 4–6 周,依赖 W0/W1/W3)

- `HierarchicalCuRoboPlanner.plan_with_fallback` 端到端回归:cuRobo 优先、OMPL 兜底,对比成功率(阈值 −5pp,沿用迁移文档 M3 标准)。
- 技能层 `core/skills/`:`grasp` + `place` 两个原语,从 W3 标注读 graspable/placement regions,技能编排器(有序技能序列 → 轨迹)先做最简版。
- 工作量:3–5 人周(含 CUDA 环境调试余量)。

### W2 — Pilot 任务 5–8 个(第 5–8 周,依赖 W1/W3/W4)

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
