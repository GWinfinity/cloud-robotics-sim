# LIBERO-MAX 移植记录（EFE 套件 21/35）

日期：2026-11-18 ｜ 状态：**64/64 编译落地，offline nop 门 + Genesis 引擎冒烟（64 任务 EXIT=0）全部通过**

## 上游事实（三条 pin 住的源）

| 部件 | 来源 | 许可 |
|---|---|---|
| EFE 任务目录（instance/instruction/oracle） | mll-lab-nu/embodied-first-exam `tasks/libero-max` | EFE 仓库 |
| LIBERO-MAX 事件运行时 | liberomax/LIBERO-MAX @ `a1e3cef`（`src/libero_max` + overlay） | 作者书面授权再分发 |
| LIBERO 包 + 对象/场景资产 | Zxy-MLlab/LIBERO-PRO @ `2b910b5` | MIT |
| 任务数据（54 BDDL + 49 pruned_init + metadata） | HF zhouxueyang/LIBERO-Pro @ `c86fc3b` | CC-BY-4.0 |
| 仿真栈 | robosuite 1.4.0 / MuJoCo 3.2.6（EFE 镜像 pin） | — |

instance.json 只给 `init_state_index + policy_seed + scenario`：场景由 LIBERO-MAX
代码从 bddl 重建，`.pruned_init` 是**整条仿真状态的 torch pickle**
（time+qpos+qvel+act 扁平向量，经 `ProSubstrateEnv.set_init_state` 应用，PRO
变体拓扑不同时按关节名适配），因此"初始物体位姿"没有声明式来源，必须跑
上游栈取——这是与 VLABench（instance 自含位姿）的本质区别。

## 工件链

1. `tools/fetch_liberomax_assets.py` — 稀疏克隆三源 + HF 按需拉取
   （sha256 校验，断点续传）；资产落 `assets/liberomax/`（gitignore，本地）。
2. `tools/setup_lmx_probe_venv.py` + `.lmxvenv`（py3.10）— 编译期探测 venv：
   robosuite 的 Windows `mujoco.dll` 拷贝 + 关 GPU 渲染宏（纯物理探测）。
3. `tools/liberomax_probe.py` — 用 pin 住的栈重建每个冻结 case
   （env build → reset → 冻结态 + PRO substrate 扰动），dump：
   物体/夹具世界位姿 + MJCF 相对路径 + 关节 qpos + articulation 量程、
   BDDL 区域站点（世界盒）、goal_state、场景 setup/trigger/change、
   arena 台面盒（body "table" 的 box geom）、`success_at_init`（64/64 False）。
   dump 在 `data/efe/upstream/liberomax-dumps/`（本地）。
4. `tools/compile_efe_suite.py --suite libero-max --resolve` — resolver 从
   dump 生成 KB → 64 份 schema v1 配置 + `data/suites/efe_libero-max.yaml`。

## 判词映射（逐字转写自 LIBERO-PRO 源码）

| BDDL | 语义（源码依据） | 本仓类型 |
|---|---|---|
| `on`（物体支撑） | 支撑 z≤物体 z ∧ 接触 ∧ 水平原点距 <0.03（严格） | `libero_on` support 模式 |
| `on`（区域站点） | 站点本地 dz∈[hz−0.005, hz+0.10] ∧ |dx|,|dy|<hxy；父夹具存在时加接触 | `libero_on` site 模式（父为 fixtures 才加 `site_contact`） |
| `in`（区域站点） | 物体原点严格在 |xmat@halfsize| 盒内，z 下界 −0.01 | `libero_in` |
| `close`（站点） | 父夹具量程 `default_close_ranges`：全部关节 qpos > min | `libero_joint_state` close |
| `turnon/turnoff` | `default_turnon/off_ranges`：任一关节 ≥ min / 全部关节 < max | `libero_joint_state` |
| 多谓词 | `And` 树 | `all` |

## 运行时

- `efe_liberomax` 任务（runtime/efe.py）：`on_proximity` 触发（EE 距实体
  <0.18m，逐步检查，同 LIBERO-MAX runtime）；物理操作 move_object /
  insert_obstacle（含 eef-path 放置）/ insert_distractors / remove_object
  （setup 阶段出库停放到 offworld）经 entity backend `set_pos` 传送；
  视觉类操作（theme/lighting/sensor_noise/camera_shift）只记入 info——
  Genesis 配置评最终物理态，观测域变更不影响判分（spec 注记）。
- 冻结关节初值（开着的抽屉 qpos −0.159 等）走
  `init_distribution.articulation_states`；`apply_layout` 已扩展为按关节名
  应用（`get_joint_names`/`get_joint_qs_idx_local`/`set_qpos`）。
- 机器人 franka_panda 基座 (−0.66, 0, 0.912)（LIBERO MountedPanda 世界系）。

## 降级与限制（诚实清单）

- 观测域事件不改物理（记录为已应用语义=观测）；若后续接观测渲染需补。
- 台面以 arena box geom 复刻（body "table"）；Floor 场景无台面，用
  efe_tabletop 地板（物体 z≈0.035 一致）。
- LIBERO 视觉装饰（墙/背景网格）未移植——不影响判词。
- EFE 的 bit-exact 哈希链（OpenBLAS/numpy AVX-512 内核依赖）不追求：本仓
  spec 级保真，物理初始态语义一致即可。
- 上游 stove XML（flat/yellow）有两个同名 `tex-stove_knob` 纹理及重名
  body/site（`button`/`base`/`burner`；MuJoCo 3.10 与 quadrants 编译器均
  拒绝任何类别的重名）：staging 副本按文档序把后发生者重命名为
  `__dupN`、随后的引用重绑（按钮材质绑回深色贴图，符合上游意图）；镜像
  字节不变，修复只在 content-hash 暂存副本（`_stage_mjcf_component`
  scheme v5，tests/backend/test_mjcf_staging.py）。关节名匹配对实例前缀
  （probe 侧 `flat_stove_1_button`）与裸名+修复后缀（spawn 侧
  `button__dup2`）做去后缀的唯一后缀匹配（success.py `_joint_qpos`）。

## 复跑

```bash
uv run python tools/fetch_liberomax_assets.py            # 镜像（幂等）
# .lmxvenv 一次性搭建见 tools/setup_lmx_probe_venv.py 头注释
.lmxvenv/Scripts/python.exe tools/liberomax_probe.py --all
uv run python tools/compile_efe_suite.py --suite libero-max --resolve
uv run python tools/compile_efe_suite.py --suite libero-max --check
uv run python -m pytest tests/efe/test_compiled_configs.py tests/efe/test_liberomax_event.py
uv run python -m pytest tests/efe/test_liberomax_smoke.py  # slow，需要资产镜像
```
