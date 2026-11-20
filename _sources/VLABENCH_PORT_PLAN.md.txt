# VLABench 迁移方案与可行性（dm_control → Genesis 1.4.6 / quadrants）

> 套件登记：`data/efe/suites/vlabench.yaml`（T1，36 任务 / 72 测试 / 2 核心）。
> 本文回答一个问题：**dm_control 任务能否迁移到基于 quadrants 的 Genesis**，
> 并给出迁移路径、保真度分级与缺口清单。状态真相以
> `data/efe/registry.json` 与 `tools/build_efe_registry.py --check` 为准。

## 0. 结论

**可行，按 spec 级保真落地。** dm_control 在 VLABench 中只是 **Python 组合层**
（composer 拼装场景、可观测封装、`should_terminate_episode` 判据）；其产出的
任务 XML 全部是标准 MJCF，由 MuJoCo 编译。Genesis 1.4.6 的 MJCF 导入器本身
**委托官方 `mujoco` 包完成编译**（`genesis/utils/mjcf.py::build_model` →
`mujoco.MjModel.from_xml_string`），再把编译产物转成 Genesis 刚体实体交给
quadrants 求解器。因此：

- MuJoCo 能编译的 MJCF（含 dm_control 生成物）→ Genesis 能读；
- quadrants 是仿真后端，与导入路径正交；同一导入栈已在 MetaWorld+
  （sawyer MJCF）全量通过双层门，本仓库有先例。

## 1. 依赖面调查（2026-11-17，上游 cf588fe / 资产 08d7a44）

| 层 | 事实 | 迁移含义 |
|---|---|---|
| 任务 XML | `obj/meshes/**` 下 36 实例引用的组件 XML（counter/shaker/billiards…）全是标准 MJCF：mesh geom + 材质贴图 + visual/collision 分组；billiards 等是空 worldbody 的**资产库** | 直接 MJCF 导入；资产库式组件需像 VLABench composer 那样以自由体实例化 |
| 场景 XML | `scenes/<name>/`（18 个房间，studyroom.xml 等）：贴图 + 房间网格 + 地板 | 同上；实例 `scene.name` 指向目录，XML 文件名 ≠ 场景名 |
| 机器人 | `franka_emika_panda/panda.xml`（GitHub，MIT）：7 臂关节 `<general>` 执行器（`dyntype=none biastype=affine`，gainprm/biasprm = P 到位置目标）+ 2 指 `<position>` 执行器；**无关节限位**（文档值见 instruction）；肌腱/耦合全注释掉 | 执行器模型与导入器原生支持的分式 `force = gain·ctrl + bias₀ + bias₁·pos + bias₂·vel` 完全一致；限位需在控制器层用文档值钳制 |
| 引擎选项 | instance.json 的 `integrator=implicitfast`、`impratio=10`、`multiccd=enable`、流体 density/viscosity | MuJoCo 专属，**不映射**；Genesis 用自家求解器参数，保真度记 spec 级 |
| 任务逻辑 | dm_control `composer` + `should_terminate_episode`（contain/above/pour/press 等谓词，部分带朝向语义如 "tip past horizontal"） | Python 层，必须在 Genesis 侧重实现为 `core/success.py` 条件；判据句逐字转写自 instruction.md / grade.py（DoD §9.3）。谓词语义源（上游 MIT，已镜像至 `.efe_stage/OpenMOSS_VLABench-git` 稀疏检出与 `data/efe/upstream/images-vlabench/`）：`VLABench/tasks/condition.py` 谓词库 + `hierarchical_tasks/**` 逐任务 `should_terminate_episode` + EFE `grade.py`/`vlb_tool.py`（GHCR `efe:vlabench-files-0.5.2` 层） |
| 动作接口 | 9 维绝对位置目标 @10Hz（0.1s = 100 物理步，dt=0.001） | 映射到 Genesis 关节位置控制（`control_dofs_position`） |
| 冻结机制 | `model_sha256` / `initial_state_sha256`（MuJoCo 3.2.2 位精确） | Genesis 重建不可复现 MuJoCo 哈希；冻结改为 Genesis 侧实例钉住（init 位姿来自 instance.json，等价 compile_efe_suite 的 init_override） |

## 2. 导入器支持矩阵（genesis-world 1.4.6 源码核对）

| 构造 | 支持 | 证据 / 备注 |
|---|---|---|
| MJCF 编译 | ✅ 原生 | 委托官方 `mujoco` 包；本 venv = 3.10.0（≥ 上游 3.2.2，`refquat` 等 3.2 构造可编译；版本差记入保真度说明） |
| mesh geom（obj/stl）+ 贴图 | ✅ | `parse_geom`；相对路径由导入器改写 meshdir/assetdir 为绝对路径 |
| `<general>` 执行器 FIXED gain / AFFINE bias | ✅ 原生 | `mjcf.py:412-467`，与 panda 的 P 执行器公式一致 |
| `<position>` 执行器 | ✅ | 同上（biastype=affine 特例） |
| 肌腱 | ⚠️ 近似 | 转为关节执行器并告警；panda 肌腱已注释，VLABench 36 任务未依赖 |
| 球关节限位 / 执行器内部动力学 / 非常规 gain-bias | ⚠️ 告警降级 | VLABench 未使用 |
| `option`（integrator/impratio/multiccd…） | ❌ 按设计丢弃 | MuJoCo 求解器语义，Genesis 无对应物 |

## 3. 资产镜像（"镜像其任务 XML"的落点）

- `tools/fetch_vlabench_assets.py`：幂等镜像，三源——
  EFE 任务文件（instance.json + instruction.md + grade.py）→
  `data/efe/upstream/tasks/vlabench/`；组件/场景 XML + 引用资产（HF
  `lerobot/vlabench-assets@08d7a44`）→ `assets/vlabench/`；panda 机器人
  （GitHub MIT）→ `assets/vlabench/robots/franka_emika_panda/`。
- **许可**：VLABench 资产集不允许再分发 → 镜像只留本地并 gitignore
  （`assets/vlabench/`），入仓库的只有工具和 PROVENANCE 记录；HF 端点可配
  （`HF_ENDPOINT`，默认 huggingface.co 不通时走 hf-mirror.com）。
- 手动步骤（与 `fetch_upstream_assets.py` 同级），CI/测试不得调用。

## 4. 缺口与对策

1. **位精确不可承诺**：`frozen` 保真（上游哈希冻结）不可达，v1 一律 `spec`
   级——判据按上游逐字、初始布局用 instance.json 冻结位姿。与 MetaWorld+ v1
   同一分级。
2. **谓词覆盖面**：`core/success.py` 现有 Distance/Pose/AxisWindow、in_region、
   count_in_region、body_point_near、ee_near、staged。**已补 VLABench 首批两个**
   （2026-11-17，TDD 65 过）：`above_region`（VLABench `AboveCondition`：z ≥ 平台
   原点 + 足迹圆/盒）与 `tipped`（`PourCondition`：实体局部 top/bottom 站点
   世界 z 差 > 阈值，"tip past horizontal"）。后续按需增补：`contain` 的
   足迹+高度带语义（`FlatContainer.contain`：根点落于足迹且 0 < z−center.z <
   z_threshold，如 select_fruit 的 0–0.1 m 带）、`IsGraspedCondition`、
   `ButtonPressedCondition`、`OnCondition`（接触对+z 上限）。
3. **关节限位**：XML 无限位 → 在 VLABench 任务控制器Wrapper 按 instruction
   文档值钳制 9 维动作（Panda 限位逐关节列出）。
4. **保活/语义漂移**：36 个冻结实例逐个跑 nop 门（起始状态不命中判据）+
   引擎冒烟（Genesis 上实例化合、落体稳定、位置控制可动）。

## 5. 路线（接 EFE_MIGRATION_PLAN P1）

1. ✅ 可行性钉死（本文 §6）。
2. ✅ 镜像管线（tools/fetch_vlabench_assets.py；组件 58/58 + 任务文件 144/144
   + panda + 判据源码；场景大网格随镜像轮次补齐）。
3. 🚧 垂直切片 `add-condiment`（2026-11-18）：
   - ✅ `core/success.py` 谓词首批（above_region/tipped，TDD 65 过）；
   - ✅ schema v1 资产 `shape_type: mjcf` 全链路（task_loader → ObjectSpawn →
     GenesisSceneBackend.create_mjcf；动态抓取物经内容哈希 stage 副本注入
     freejoint，MuJoCo 3.10 拒收的薄壳网格自动补 `inertia="shell"`——
     镜像保持字节级原样）；
   - ✅ `runtime/efe.py` 注册 `franka_panda`（VLABench panda.xml）；
   - ✅ compile_efe_suite.py `resolve_vlabench`（instance.json 驱动：
     条件名→success 类型分发，足迹半径/站点偏移编译期从镜像 MJCF 解析）；
   - ✅ 编译产物 + 离线 nop 门（tests/efe/test_compiled_configs.py 自动覆盖
     efe_vlabench 聚合）。
   - ⏳ 引擎冒烟（tests/efe/test_vlabench_smoke.py，slow）。
     ✅ 通过（2026-11-18，110 s）：franka + 8 个 MJCF 组件（3 动态含 freejoint
     stage 注入 + 1 薄壳 shell-inertia 补丁）实例化合、reset、起始态判据 False。
     双层门齐活——VLABench 第一条任务端到端可跑。
4. 批量 36 任务：按判据族扩 condition_map（contain 足迹+高度带、press、
   grasped、on），逐族解锁 → `data/suites/efe_vlabench.yaml` 全量 →
   `--check` → `project.status: partial→ported`（DoD §9 全满足才改 ported）。
   **进展（2026-11-18 批产）**：谓词第二批落地（`contains` 圆足迹+z带/箱袋
   bbox/any-of/negate、`lift`、`ordered`，TDD 98 过）；condition_map 扩至
   above/pour/contain/not_contain/lift/order；**18/36 编译落地**
   （offline nop 门 + compile --check 全绿；引擎冒烟 17 任务，select_drink
   被 nop 门拦截退回 pending——冻结初始态果汁已在冰箱 bbox 外，记入判据表）。
   剩余 18 个 pending 的全部理由已写进 `vlabench.predicates.yaml`：
   - 接触/关节类谓词待接入（8）：press_button×4（QA 三件套+select_painting）、
     is_grasped×4（poker 系）、joint_in_range（hammer-nail）；
   - 条件实体内嵌容器 MJCF 需子体抽取（6）：book×2、chemistry KMnO4×3、
     find-unseen-object、take-out-cool-drink（实际 7 项，含 heat-food 附带的
     门/按钮语义）；
   - heat_food 需门关闭+启动钮附加语义；texas_holdem 判据在代码层需人工转写；
   - select_drink 见上（nop 拦截）。
   **当日二轮（31/36）**：`in_contact` 谓词（press_button 力带按终态接触转写
   spec 注；grasp 用整机接触近似夹爪）+ 类级几何合成（BilliardBall/Poker/
   NumberCube 资产库注入；Button/CardHolder 纯代码类转写）+ **subentities
   递归展开**（书/试管/苹果/饮料经父链位姿合成）+ RandomGeom assets_override
   + 镜像工具子实体递归。引擎冒烟 31 任务。剩 4 个 pending：hammer-nail
   （joint_in_range+关节化 spawn）、heat-food（门/钮附加语义）、texas-holdem
   （代码层判据）、select-drink/take-out-cool-drink（nop 拦截的"出生即成功"
   同型疑问，待专查上游冻结语义）。
   **当日三轮（34/36）**：`joint_window` 谓词（joint_in_range + 门状态
   |qpos|<π/10）；**articulated spawn**（关节化组件保关节无 freejoint；
   活动体网格惯性自动补 shell——MuJoCo 3.10 对活动 mesh geom 的强制）；
   heat_food/texas_holdem 经 extra_conditions 转写（判据表声明式附加条件，
   contains 型同样编译期解析容器几何）。剩余 2 个 pending 为 nop 门拦截的
   饮料同型疑问（冻结态即在容器外，待专查上游语义）。
   **+100 路线评估（2026-11-18）**：VLABench 已接近收官；再 +100 需新开
   套件，固定成本排序——KinDER 5（seed-only 实例需 kindergarden 代码栈重建
   场景 + TidyBot++ 移动本体，性价比低）、LIBERO-MAX 64（最大单体，但需
   LIBERO-PRO 场景栈与 robosuite 位精确语义降级，多日工程）、RoboPaint 43
   （需独立光栅评分器，单独立项）、Olympics-Fun 79 / Dex 系 62（Menagerie
   本体转换波次 + 逐场景重建）。建议下一波从 LIBERO-MAX 垂直切片立项。

## 6. 冒烟证据（panda+counter+shaker，CUDA/quadrants，2026-11-17）

场景 = `add-condiment` 冻结实例的三件组合（panda @ (0,-0.4,0.78) + counter +
shaker 注入 freejoint），全部使用 VLABench 上游原始 XML、未改动物理参数。
冒烟脚本为一次性脚手架；入库版测试将由 `tests/efe/test_vlabench_smoke.py`
承担（同 metaworldplus 双层门模式）。

| 检查 | 结果 |
|---|---|
| MuJoCo 3.10.0 编译 panda.xml | ✅ nq=9 / nactuator=9 |
| Genesis 场景合成（plane+counter+shaker+panda） | ✅ |
| DOF 布局 | ✅ 9 关节 = joint1..7 + finger_joint1/2，与上游一致 |
| shaker 落体稳定 | ✅ z 0.799→0.780 落定柜面，末 100 步稳 |
| 位置控制 | ✅ joint2/4 目标 +0.15，qpos 有限响应（超调 0.77/0.16——控制标定列入端口工作，见 §4.5） |
| 离屏渲染 | ✅ 台面木纹贴图、shaker、panda 网格均正确 |

导入告警（预期内，不阻断）：torsional friction（condim≥4）声明但未启用；
指尖 authored 惯量与几何估算不符的提示；上游 solref 0.001 被求解器抬到
0.002；qpos0 自碰撞几何对过滤。CoACD 凸分解 shaker 的 32 片碰撞网格耗时
~250 s（一次性；genesis 缓存后整场重建 ~75 s）。

### 端口工作明细（§4 增补）

5. **控制标定**：MJCF 的 MuJoCo 级增益（gainprm 4500/3500/2000、solref/solimp）
   按公式原样进 Genesis 执行器模型（`dofs_act_gain/bias`），但 Genesis 的
   `control_dofs_position` 默认值（速度限幅等）与 MuJoCo 位置执行器行为
   有差；垂直切片阶段需逐关节对照 MuJoCo 参考响应标定，记入任务配置。
