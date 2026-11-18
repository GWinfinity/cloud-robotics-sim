# EFE（Embodied First Exam）全量迁移梳理

> 对标对象：[Embodied First Exam](https://arxiv.org/abs/2611.12177)
>（arXiv:2611.12177，2026-11-09；代码与任务包
> [mll-lab-nu/embodied-first-exam](https://github.com/mll-lab-nu/embodied-first-exam)，MIT）。
> 本文档把 EFE 的 **35 套件 / 47 机器人 / 1,156 任务 / 2,288 测试（456 核心）**
> 全部登记入册，并给出逐套件、逐机器人的迁移映射与分批路线。
> 机器可读登记册：`data/efe/`（生成物 `data/efe/registry.json`，CI 门
> `uv run python tools/build_efe_registry.py --check`）。

## 1. EFE 是什么、迁移什么

EFE 是面向 coding agent 的统一机器人评测：每个任务里 agent 写程序控制
冻结场景中的机器人，三种评测设定——**Privileged**（全状态+无限重置，交
轨迹/程序）、**Standard**（仅传感器观测、默认不可重置，仿真作为服务）、
**Realtime**（Standard + 时钟不停）。任务包为 Harbor 格式：
`instruction.md`（目标/约定）+ `task.toml`（预算/元数据）+
`environment/instance.json`（哈希冻结的实例）+ `tests/`（隔离判分器）。
上游逐套件 README 精确记录了动作布局、成功判据与世界坐标系——这是移植时
最重要的规格来源。

迁移到本项目 = 在 Genesis 1.4.6 上重建每个套件的**场景、机器人、成功
判据**，接入现有 `core/`（Registry / Composer / Task）执行栈与
`configs/tasks/*.yaml` + `data/suites/*.yaml` 评测协议，使 EFE 任务可以
在本仓库复跑、复评。

## 2. 规模对账（与论文 Table 1/2 逐条对齐）

| 项 | 论文 | 本仓库登记 | 对账 |
|---|---|---|---|
| 套件 | 35（9 新建 + 26 改造） | 36（另登记 humanclaw，论文不计） | `--check` 强校验 |
| 机器人 | 47 | 47（+4 component_only 部件机） | 并集计数强校验 |
| 任务 | 1,156 | 1,156（@all 含 humanclaw 为 1,211） | 逐套件强校验 |
| 测试 | 2,288 | 2,288（上游 HEAD 2,398，见漂移节） | 论文值强校验 |
| 核心任务 | 456 | 456 | 强校验 |
| 设定分布 | privileged / standard / realtime | 1,108 / 1,210 / 86（@all 2,404 dirs） | 数据驱动 |

已知漂移：上游 HEAD 的 robodojo 为 54 任务 × 2 模式 = 108 dirs，论文
Table 2 记 102 tests（论文快照后上游加了模式）。登记册以论文为基准定义，
漂移记 warning 不阻断 CI。

## 3. `data/efe/` 布局与工具链

```
data/efe/
  suites/<suite>.yaml   # 36 条策划记录：efe.*（论文事实）+ project.*（迁移映射）
  tasks/<suite>.json    # 36 份上游任务清单（task.toml 解析物，生成物）
  robots.yaml           # 47+4 机器人资产来源对账表
  registry.json         # 生成物：合并视图 + totals + warnings（build 工具产出）
tools/fetch_efe_inventory.py    # 快照 -> tasks/*.json（手动、离线；不走网络）
tools/build_efe_registry.py     # 合并 + 校验；--check 为 CI 门
```

三层数据相互独立校验：策划 YAML（论文 Table 2 人工登记）↔ 任务清单 JSON
（上游 task.toml 解析）↔ 论文总数常量（内嵌 build 工具）。任何一层漂移
`--check` 即失败；机器人必须登记 `robots.yaml` 来源才能通过结构校验。

## 4. 分层迁移策略

按**源后端 → Genesis 可行路径**分五层（记录在每条 `project.tier`）：

| 层 | 判定 | 套件数 | 任务数 | 迁移路径 |
|---|---|---|---|---|
| T0 | EFE 内即 Genesis | 1 | 15 | 直接对齐（RoboWits） |
| T1 | MuJoCo / robosuite / dm_control | 20 | 673 | MJCF 导入 + 场景/判据重建 |
| T2 | SAPIEN / ManiSkill | 3 | 115 | 项目已有 maniskill 插件与 RoboTwin 资产 |
| T3 | Isaac Sim / Isaac Lab / OmniGibson | 9 | 301 | 需 USD/资产管线；部分有许可门与大数据集 |
| TX | 特殊/受阻 | 2(+1) | 52(+55) | MyoSuite（肌肉骨骼）、RoboEngineer（隐藏动力学）、HumanCLAW（论文外） |

- **T1 是主干（58% 任务）**：Genesis 支持 MJCF；本体对标 Menagerie
  （`assets_genesis/embodiments/`）已建转换与认证门；多数套件机器人已在库
  或 Menagerie 可转。rope/cloth 类可变形体（RoboARC）需 Genesis 等效方案
  或刚体链降级，逐套件决策。
- **T2 借力现有设施**：`plugins/envs/maniskill`、`src/cloud_robotics_sim/
  robotwin`、RoboTwin 资产包是 Robotwin-2 / RoboPaint / RoboDynamicExam 的
  直接底座。
- **T3 的关键前置是资产管线**而非物理：BEHAVIOR-1K（32GB 数据集+许可门）、
  RoboDojo（41GB 镜像）等大头都在这层；套件内的机器人生物（如 ARX-X5、
  franka-robotiq）多数反而已就绪。
- **TX 需先做路线决策**：MyoSuite 依赖 MuJoCo 肌腱/肌肉致动器，Genesis
  1.4.6 无等效（自研 solver 插件或显式跳过）；RoboEngineer 的考点是
  sim-to-sim 隐藏动力学闭环，可移植公开半段；HumanCLAW 不在论文计数内。

## 5. 套件全景（35+1）

| 套件（上游目录） | 显示名 | 后端 | 本体类 | 机器人 | 任务/测试/核心 | 层 | 状态 |
|---|---|---|---|---|---|---|---|
| robowits | RoboWits | Genesis | 双臂 | marvin-bimanual | 15/30/6 | T0 | partial |
| robotwin-2 | RoboTwin 2.0 | SAPIEN | 双臂 | aloha-agilex | 47/94/3 | T2 | partial |
| metaworldplus | MetaWorld+ | MuJoCo | 单臂 | sawyer | 50/100/2 | T1 | planned |
| libero-max | LIBERO-MAX | robosuite | 单臂 | franka-panda | 64/128/2 | T1 | planned |
| vlabench | VLABench | dm_control | 单臂 | franka-panda | 36/72/2 | T1 | planned |
| robopaint | RoboPaint | ManiSkill | 单臂 | franka-panda | 43/86/22 | T2 | planned |
| kinder | KinDER | MuJoCo | 移动操作 | tidybot-plus-plus | 5/10/4 | T1 | planned |
| roboarc | RoboARC | MuJoCo | 单臂/双臂 | aloha-2, franka-panda-bimanual, franka-robotiq-2f85 | 106/212/58 | T1 | planned |
| dexhand | DexHandExam | MuJoCo | 灵巧手 | sharpa-wave×3 | 30/60/23 | T1 | planned |
| dexjoco | DexJoCo | MuJoCo | 灵巧手 | franka-allegro×2 | 11/22/2 | T1 | planned |
| dextoolbench | DexToolBench | MuJoCo | 灵巧手 | kuka-iiwa14-sharpa-ha4 | 21/42/20 | T1 | planned |
| abc-sim | ABC Sim | MuJoCo | 双臂 | i2rt-yam-bimanual | 29/60/0 | T1 | planned |
| yamlab | YAMLab | Isaac Lab | 双臂 | i2rt-yam-bimanual | 10/20/2 | T3 | planned |
| embodied-safety-exam | Embodied Safety Exam | MuJoCo | 单臂/双臂/移动 | franka-robotiq, ur5e-robotiq, tidybot++, yam | 72/76/50 | T1 | planned |
| autobio | AutoBio | MuJoCo | 单臂/双臂 | aloha-2, ur5e 组合 | 9/18/0 | T1 | planned |
| molmospaces | MolmoSpaces | MuJoCo | 单臂/移动 | franka-robotiq, rainbow-rby1 | 8/16/2 | T1 | planned |
| mujoco-playground | MuJoCo Playground | MuJoCo | 单臂/双臂/四足 | aloha-2, spot, franka, barkour, go1 | 13/27/7 | T1 | planned |
| bigym2 | BiGym 2.0 | MuJoCo | 人形 | unitree-g1-dex1 | 13/39/10 | T1 | planned |
| humanoidbench | HumanoidBench | MuJoCo | 人形 | unitree-h1, h1hand | 10/30/8 | T1 | planned |
| robocasa-gr1 | RoboCasa-GR1 | robosuite | 人形 | fourier-gr1 | 24/48/1 | T1 | planned |
| robocasa | RoboCasa | robosuite | 移动操作 | franka-panda-omron | 29/58/2 | T1 | planned |
| robocasa365 | RoboCasa365 | robosuite | 移动操作 | franka-panda-omron | 50/101/5 | T1 | planned |
| olympics-fun | Embodied Olympics - Fun | MuJoCo | 人形 | unitree-g1, h1-2 | 79/179/56 | T1 | planned |
| olympics-match | Embodied Olympics - Match | MuJoCo | 单臂/人形/移动 | booster-t1, foosball-rods, franka, ssl-robot, g1 | 14/41/13 | T1 | partial |
| pandadyn | RoboDynamicExam | ManiSkill | 单臂/双臂 | franka×2 | 25/69/10 | T2 | planned |
| roboengineer | RoboEngineer | MuJoCo+learned | 空中/车/人形/四足 | anycar, aero, neural-fly, g1, go2 | 30/30/15 | TX | planned |
| embodiedswe | EmbodiedSWE | Isaac Lab | 单臂/双臂/人形 | franka, franka-bimanual, g1-dex3 | 24/48/12 | T3 | planned |
| behavior-1k | BEHAVIOR-1K | OmniGibson | 移动操作 | r1pro | 100/200/71 | T3 | planned |
| ispatialgym | iSpatialGym | OmniGibson | 移动操作 | r1pro | 12/24/8 | T3 | planned |
| maniguard | ManiGuard | OmniGibson | 单臂 | franka-panda-longfinger | 30/60/7 | T3 | planned |
| robodojo | RoboDojo | Isaac Lab | 双臂 | arx-x5-bimanual | 54/102/12 | T3 | planned |
| robolab | RoboLab | Isaac Lab | 单臂 | franka-robotiq-2f85 | 53/106/6 | T3 | planned |
| labutopia | LabUtopia | Isaac Sim | 单臂 | franka-panda | 9/18/0 | T3 | planned |
| ambench | AM-Bench | Isaac Lab | 空中 | ambench-fahexa | 9/17/2 | T3 | planned |
| myosuite | MyoSuite | MuJoCo | 肌肉骨骼 | myo*×6 | 22/45/13 | TX | blocked |
| humanclaw | HumanCLAW | humanclaw | 人形 | smplx-human | 55/110/— | TX | excluded（论文外） |

> 完整字段（source 论文引用、target 落点、note 风险）见
> `data/efe/suites/*.yaml`；逐任务清单（名称/模式/oracle/预算）见
> `data/efe/tasks/<suite>.json`。

## 6. 本体分布（论文 9 类，按用户口径归并为 5+4）

| 本体类 | 任务数* | 套件数 | 代表套件 |
|---|---|---|---|
| 单臂 arm | 556 | 15 | MetaWorld+、LIBERO-MAX、VLABench、RoboPaint |
| 双臂 two_arms | 404 | 11 | RoboTwin 2.0、ABC Sim、RoboARC、RoboDojo |
| 移动操作 mobile_manip | 290 | 8 | BEHAVIOR-1K、RoboCasa×3、KinDER、iSpatialGym |
| 人形 humanoid | 194 | 7 | Olympics-Fun、BiGym 2.0、HumanoidBench、RoboCasa-GR1 |
| 灵巧手 hand | 62 | 3 | DexHandExam、DexJoCo、DexToolBench |
| 四足 quadruped | 43 | 2 | MuJoCo Playground、RoboEngineer |
| 空中 aerial | 39 | 2 | AM-Bench、RoboEngineer |
| 车 car | 30 | 1 | RoboEngineer |
| 肌肉骨骼 musculoskeletal | 22 | 1 | MyoSuite（blocked） |

\* 多本体套件在其每个 kind 下重复计数，合计 >1,156。

## 7. 47 机器人资产对账（`data/efe/robots.yaml`）

| 来源 | 台数 | 说明 |
|---|---|---|
| have（已在库） | 2 | franka-panda(gold)、aloha-agilex(silver)；ARX-X5 以部件身份在库 |
| compose（组件合成） | 8 | 双臂装配、franka+robotiq/allegro、ur5e+robotiq、kuka+sharpa 等 |
| menagerie（Menagerie 可转） | 10 | unitree g1/h1/h1-2/go1/go2、spot、barkour 等，走 `tools/convert_assets.py` |
| public（上游公开） | 7 | sawyer(MetaWorld MJCF)、aloha-2、GR1、RB-Y1、TidyBot++、Booster T1 |
| efe_custom（需抽取/重建） | 14 | marvin、sharpa-wave、i2rt-yam、r1pro、omron 底座、RoboEngineer 平台等 |
| blocked | 6 | myo*（Genesis 无肌肉肌腱模型） |
| out_of_paper | 1 | smplx-human（HumanCLAW） |

组件机（robotiq-2f85、ur5e、kuka-iiwa14、sharpa-ha4、allegro-hand、ARX-X5）
登记为 `component_only: true`，不计入 47 台。

## 8. 分批路线

**P0 — 对齐与冒烟（1–2 周）**
- T0：RoboWits 与 EFE 任务清单对账（marvin-bimanual ↔ 现有 port 的命名与
  判据），补 `data/suites/efe_smoke.yaml`。
- 登记册本身就是 P0 交付：`--check` 进 CI，后续任何套件迁移必须同步改
  `data/efe/suites/*.yaml` 的 `project.status`。

**P1 — T1 单臂先遣（约 200 任务）**

> **状态（2026-11-17 更新）：MetaWorld+ 首个垂直切片引擎级通过。**
> 已交付：`core/success.py` 判据扩展（distance_threshold 轴加权 /
> axis_window 目标锚点相对模式 / in_region / count_in_region /
> body_point_near，TDD 44 过）；`tools/compile_efe_suite.py` 编译器
> （MJCF include 解析 + 世界坐标合成 + 判据表→知识库→schema v1 配置，
> `--check` CI 门）；`runtime/efe.py` 注册（sawyer MJCF 机器人 /
> efe_tabletop 场景 / efe_static 基任务）；`tests/efe/` 双层门
> （离线 nop-fail 契约 + Genesis 引擎冒烟，slow）。
> **MetaWorld+ 50/50 全部编译并通过双层门（100%，2026-11-17）**——上游 wheel 镜像
> 后近目标族批量解锁：判据句逐字转写自 instruction.md，目标语义取自
> env .py 常量（goal_space/self.goal/TARGET_RADIUS），代表点注明"spec 级、
> 非冻结实例值"；nop 门累计拦截 3 处 XML 声明位姿≠任务初始位姿
> （plate-slide-back×2/push-back/button-press-topdown，均以 init_override
> 或退回 pending 修正）。登记册状态 `partial`。其余 24 个：机构类
> （door/window/drawer/faucet/dial/lever/button-topdown）需逐 env 翻译
> reset 语义（door 为平移语义已核实）；reach×2 需 ee 谓词；stick×2 需
> 接触谓词；lever-pull/dial-turn 需角度/旋转目标。
>
> **保真度分级**（写入每个编译配置的 `fidelity` 注释）：
> `spec` = 判据与参数按上游 README/instruction 的规格逐条转写、初始布局
> 按上游随机化范围重建（非位精确）；`frozen` = 应用上游 instance.json
> 冻结位姿。MetaWorld+ v1 为 `spec` 级；判据句一律从上游
> instruction.md 的 "## Success"/"## Scoring" 节摘编进
> `data/efe/compile/<suite>.specs.md` 后人工转写，禁止凭记忆写阈值。
>
> **两个套件的路线修正**（读上游 README 后的发现）：
> - RoboPaint 的成功判据是纸面墨迹的图像指标（IoU/boundary-F/覆盖/
>    stray）+ 笔画过程规则，本质是 2D 光栅评分（libmypaint 体系），
>   不是 Genesis 物理谓词——需要独立的评分器移植（raster scorer +
>   笔画样本记录），保留在 P1 但单独立项。
> - LIBERO-MAX 依赖 robosuite 位精确状态哈希（AVX-512 数值核）与
>   LIBERO-PRO 资产（CC-BY-4.0）及 8 类事件运行时——Genesis 重建属于
>   `spec` 级降级（事件语义可移植，位精确不可承诺），排在 P1 末位。

- 机器人资产 100% 就绪的套件优先：MetaWorld+(sawyer 公开 MJCF)、
  LIBERO-MAX / VLABench / RoboPaint / LabUtopia(franka 在库)——这四个就
  是 193 任务；加 KinDER(5)、Safety-Exam 的 franka/ur5e 子集。
- 每个套件按 §9 的 DoD 落地；P1 完成时 `ported` 任务数应 ≥ 200。

**P2 — T1 双臂/灵巧/人形 + T2（约 480 任务）**
- RoboTwin 2.0（资产已备）→ ABC Sim / RoboARC / AutoBio / MolmoSpaces /
  RoboCasa 系 / MuJoCo Playground / Olympics-Fun / HumanoidBench / BiGym 2.0
  / Dex 三件套（借 dexterous_gnn_qp 设施）→ Olympics-Match（复用
  table_tennis/badminton 对手设施）。
- 机器人资产波次：W1 unitree 系+spot+barkour（Menagerie 批量转换）→
  W2 GR1/RBY1/TidyBot++/aloha-2/allegro（公开）→ W3 marvin/sharpa/yam/
  omron 等 EFE 自有资产抽取。

**P3 — T3 Isaac 系（301 任务）**
- 前置：USD→Genesis 资产管线与许可/数据集获取（BEHAVIOR 32GB、RoboDojo
  41GB 镜像条款）。RoboDojo 因 ARX-X5 本体已入库 + 项目已有 P0 对标文档，
  作为 T3 首套件。

**P× — 决策项**
- MyoSuite：决策「自研肌肉肌腱 solver 插件」或「永久跳过并记录理由」。
- RoboEngineer：只移植公开半段需 EFE 协议许可评估；建议随 sim2real 插件
  路线一起走。
- HumanCLAW：论文外，登记在册不投入。

## 9. 套件级 DoD（"ported" 的判定标准）

一个套件标记 `status: ported` 需全部满足：

1. 套件内全部任务以 schema v1 配置落地 `configs/tasks/efe/<suite>/*.yaml`，
   并由 `data/suites/efe_<suite>.yaml` 汇总；
2. 机器人资产过 embodiment 认证门（gold/silver，或登记过的 compose 装配）；
3. 成功判据按上游 README 的规格逐条实现为 verifier（可执行，非注释）；
4. nop 必失败、参考解/规则解可通过（对齐 EFE 的入册自检）；
5. 三设定映射明确：Privileged ↔ 本仓库 evaluation runner；Standard /
   Realtime ↔ 仿真即服务（复用 robowits_ported 的 service 经验）；
6. `data/efe/suites/<suite>.yaml` 的 `project.status` 改为 `ported`，
   `--check` 通过，套件文档链入本文档。

## 10. 约束与风险（对齐 AGENTS.md）

- **测试零网络**：登记册工具链全部离线（`fetch_efe_inventory.py` 为手动
  步骤，CI 不跑）；端口判分不得访问模型 API。
- **资产许可**：BEHAVIOR 数据包 / HSSD / BlenderKit 等有许可门或 HF gated
  数据集，只能以「显式、可禁用、可 mock」的方式接入；Menagerie/公开 URDF
  转换须记录出处（沿用 `conversion_report.json` 机制）。
- **不重判安全语义**：Safety-Exam / ManiGuard 的拒止判定是规则 verifier，
  与项目 SafetyViolationError 边界一致，不得用元数据替代运行时保护。
- **性能现实**：AutoBio 500Hz、VLABench 单控步最多 100 物理步等预算约束
  要在任务配置里显式声明，Genesis 步进成本需逐套件实测（记
  `docs/troubleshooting.md`）。
- **上游漂移**：上游仓库持续演进（robodojo +6 dirs 已发生）；刷新
  `tasks/*.json` 必须重跑 `--check` 并人工复核 diff。

## 11. 与现有对标文档的关系

| 文档 | 与 EFE 的关系 |
|---|---|
| `docs/ROBOWITS_BENCHMARK.md` / `ROBOWITS_GAP_ANALYSIS.md` | RoboWits = EFE 套件 #35（T0） |
| `docs/ROBODOJO_P0_PLAN.md` / `ROBOTWIN_TO_ROBODOJO_GAP_ANALYSIS.md` | RoboDojo = EFE 套件 #30（T3 首套件） |
| `docs/ROBOTWIN_GENESIS_MIGRATION_P0.md` | RoboTwin 2.0 = EFE 套件 #32（T2） |
| `docs/VLABENCH_PORT_PLAN.md` | VLABench = EFE 套件 vlabench（T1，dm_control→quadrants 可行性与路线） |
| `plugins/envs/table_tennis` / `badminton` | Olympics-Match 的子设施 |
| `plugins/manipulation/dexterous_gnn_qp` | Dex 三件套（T1）的灵巧操作底座 |
| `data/embodiments/README.md` | 本体库质量条 = 47 机器人资产线的认证门 |

本文档是上述各线的**总索引**：所有套件迁移的状态真相以
`data/efe/registry.json` 与 `--check` 门为准。
