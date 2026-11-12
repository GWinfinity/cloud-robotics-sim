# 绳缆仿真后续尝试计划（IPCCoupler / PBD 基线 / 上游 solver 跟踪）

日期：2026-11-10 · 状态：**未实施**，仅供评审
现状：`XpbdRope` + xuvdb/解析 SDF 碰撞已落地并验证（commit `82eb99d`，
`tests/vdb/` 45 全过，demo 见 `examples/cable_sim/`）。本文规划三个互不阻塞的
尝试方向，每轨给出前置条件、步骤、验证标准与风险，实施顺序建议 1 → 2 → 3。

## 环境验证记录（2026-11-10 已在 WSL2 实测）

轨道 1 的关键前置已实际验证，安装与兼容性结论如下：

| 项目 | 结论 |
|---|---|
| WSL | WSL2 Ubuntu 24.04（kernel 6.6.87.2-microsoft），`wsl -l -v` 确认 |
| GPU 透传 | OK，WSL 内 `nvidia-smi` 看到 GTX 1650（sm_75，4GB 显存） |
| venv | `~/.venv-uipc`（Ubuntu 24.04 系统 Python 3.12.3，PEP 668 需 venv） |
| 安装 | **PyPI 包名是 `pyuipc`（不是 `uipc`）**；阿里云镜像无此包，必须 `-i https://pypi.org/simple` |
| 版本 | pyuipc **0.0.29** + polyscope 2.6.1，Windows 侧仍未安装（Windows 无官方 wheel） |
| coupler 符号兼容 | genesis 1.4.6 `ipc_coupler/coupler.py` 的全部 TYPE_CHECKING import（`SceneVisitor`、9 个 constitution 含 `ExternalArticulationConstraint`、6 个 core、3 个 geometry、`SceneGUI`）在 pyuipc 0.0.29 **逐一存在，无缺失** |
| CUDA 后端 | `Engine('cuda')` 在 WSL2 初始化成功；空场景 `init→advance→retrieve` 一帧跑通（Newton 求解器日志正常） |
| pyuipc API 差异 | 0.0.29 无 `scene.gravity`/`world.gravity`（`scene.config` 是方法不是 dict）；实现时以 genesis coupler 内部用法或 libuipc 仓库 python examples 为准 |

由此轨道 1 的前置条件从"二选一"收窄为：**整条 demo 进程跑在 WSL2 venv**
（IPCCoupler 需要 genesis 进程内 `import uipc` 成功），Windows 侧保持不装 uipc。

## 背景结论（已确认的事实）

- genesis-world 1.4.6 是当前 PyPI 最新，**无原生绳缆实体**；上游
  [discussion #2980](https://github.com/Genesis-Embodied-AI/genesis-world/discussions/2980)
  称绳/缆在 roadmap（约 3 个月量级），并计划开放外部 solver 插件系统。
- `genesis/engine/couplers/ipc_coupler/coupler.py` 已随 1.4.6 发行：libuipc
  后端，含 `ExternalArticulationConstraint`（关节空间动力学嵌入 IPC 优化）。
  `uipc` 为懒导入，缺包时 `UIPC_AVAILABLE=False`，不影响基础包。
- genesis PBD solver（`genesis/options/solvers.py:774`）有
  `max_stretch_solver_iterations` / `max_bending_solver_iterations`，
  与自写 XPBD 同族（距离约束 + 弯曲约束），可做基线对照。
- 测试纪律：genesis 插件遵守每进程一次 `gs.init`；xuvdb 相关测试由
  `tests/optional_deps.py` 的 `xuvdb_only` 守护。

## 轨道 1：IPCCoupler 刚体-柔体耦合验证（FEM 绳近似）

目标：用 libuipc IPC 求解器复现 demo 的披覆场景，与 XPBD 结果对照，
评估"绳 = FEM 柔体 + 外部关节约束"路线的精度与性能。

前置条件（部分已完成，见环境验证记录）：
- [x] WSL2 + venv + pyuipc 0.0.29 + polyscope，CUDA 后端可用
- [ ] WSL venv 补齐 genesis 侧依赖：`pip install genesis-world==1.4.6` +
  torch cu128（Windows 版 pyproject 的 index 配置在 WSL 内重新落地一份，
  torch 走 `download.pytorch.org/whl/cu128`）；**Windows 侧不装 uipc**
- [ ] 用项目代码：WSL 内 `pip install -e .`（editable，demo 直接改直接跑）
  或挂载本仓库路径后 `PYTHONPATH=src`

步骤：
1. `pyproject.toml` 新增 extra `ipc = ["pyuipc", "polyscope"]` 并注明
   Linux-only（提交时 `uv lock` 会解析 pyuipc——Windows 开发者需
   `--extra ipc` 仅在 WSL/CI-Linux 使用；lock 兼容性在实现时验证，
   必要时该 extra 只进文档不进 lock）。
2. 读通 1.4.6 的 `IPCCouplerOptions`（`genesis/options/solvers.py`）与
   coupler 的 coupling type 语义（`external_articulation` 要求实体带关节
   DOF 且基座固定；与 `Mesh.get_sdf_grid` 的 genesis_contact 桥是否
   冲突需实测确认）。
3. 新建 `examples/cable_sim/ipc_cable_demo.py`（进程整体在 WSL 跑）：
   - 场景与 XPBD demo 完全对齐（同球/地面几何、同钉轨迹），便于逐项对照；
   - 绳用 FEM 柔体近似：沿曲线扫掠细四面体管（trimesh 生成，半径 ~3mm，
     单元尺寸 ~6mm），材质 StableNeoHookean（coupler 已 import 该 constitution）；
   - 钉：固定基座的单link夹爪，`external_articulation` 耦合，按
     smoothstep 轨迹拖动（复用 demo 的 `pin_x`）。
4. 输出与 XPBD demo 相同的诊断量（最大穿透、等效"拉伸"、逐步耗时），
   另存 GIF 对照；4GB 显存是硬约束（FEM 管 + IPC 的 CCD 活跃集都吃显存），
   场景规模先从小做起。
5. 若通过，再考虑 `tests/backend/` 增加 `genesis_only` 守护的回归测试
   （遵守每进程一次 `gs.init`；uipc 可用性用 `tests/optional_deps.py`
   新增 `HAS_UIPC` 守护，Windows CI 自动 skip）。

验证标准：
- 披覆终态与 XPBD 版形态偏差 < 5cm（视觉 + tip 位置数值）；
- 全程无穿透（IPC 保证，重点核对与 genesis 刚体侧的接触一致性）；
- 记录 ms/step 与显存峰值，写入 `docs/troubleshooting.md` 性能表。

风险（已按实测更新）：
- ~~libuipc 与 1.4.6 的 API/版本错配~~ → **已排除**：pyuipc 0.0.29
  coupler import 符号全齐；
- pyuipc 0.0.29 的 Scene/World 属性与部分网络教程不同（无 `gravity` 直挂），
  实现时以 coupler 源码与 libuipc 仓库 examples 为准；
- WSL2 下 GTX 1650 仅 4GB 显存，FEM 场景规模受限，性能结论只作参考；
- `polyscope` GUI 在 WSL 无显示环境会失败（coupler 的 `_init_ipc_gui`
  确认开关，headless 下不开）；
- `uv lock` 在 Windows 解析 pyuipc 的行为待验证（必要时 ipc extra 不进 lock）。

## 轨道 2：genesis 原生 PBD 基线

目标：给 `XpbdRope` 一个同场景、同指标的基线，量化自写 XPBD 的精度/性能
位置，并验证"SdfField 碰撞层"在 genesis 原生求解器下的复用边界。

步骤：
1. `examples/cable_sim/pbd_rod_demo.py`（独立进程，避开 `gs.init` 测试约束）：
   - `gs.init(backend=gs.gpu)` 或 cpu（`utils/genesis_compat.py` 兼容接口）；
   - 细杆网格（圆柱分段，PBD ELASTIC 材质）替代绳，一端 fixed；
   - 同一球/地面场景：地面用解析 plane（genesis 原生），球用刚体 mesh——
     注意这里**不走** xuvdb SDF（PBD 刚体-柔体接触走 genesis 自带 narrowphase），
     目的正是对照"SDF 投影 vs 原生接触"；
   - 钉轨迹复用 `pin_x`。
2. 记录相同诊断量 + 原生求解器迭代参数扫描（stretch/bend iterations 4/8/16）。
3. 结论写入 `docs/troubleshooting.md`：PBD 基线 vs XPBD+xuvdb 的精度、速度、
   适用场景区分（可编辑 SDF 场景选后者，原生刚体交互选前者）。

验证标准：demo 可复现披覆；参数扫描表落盘；与轨道 1/XPBD 的三方对照图。

风险：PBD 杆的 pinned 端在 1.4.6 的设置方式需查（`PBD3DEntity` 的 fix 语义）；
demo 进程级 `gs.init` 与 CUDA 上下文在 WSL/Windows 的差异。

## 轨道 3：上游原生绳缆 solver 跟踪与接入预案

目标：上游 rope/cable solver（或外部 solver 插件系统）发布后，绳芯可平替，
`SdfField` 碰撞层与 demo 指标不变。

常态化（无需环境前置）：
1. 跟踪 genesis-world release notes 与 discussion #2980；季度检查一次。
2. 维护接入检查单（新 solver 落地时逐项打勾）：
   - [ ] 原生 `Cable/Rope` entity 或插件注册点存在，API 形态（spawn 参数、
     材料、compliance 语义与 XPBD 一致还是 MuJoCo 系）
   - [ ] 与刚体 link 的附着/约束（是否覆盖 `ExternalArticulationConstraint` 能力）
   - [ ] 是否可微（对照 `LaunchRecorder` 框架）、是否 batch 化
   - [ ] 接触路径：原生 narrowphase vs 可挂外部 SDF（决定 xuvdb 层保留范围）
   - [ ] 升级代价：`pyproject.toml` 钉版本、CI Python 3.12/3.14 复跑
3. 若外部 solver 插件系统先于 rope solver 发布：评估把 `XpbdRope` 包成
   genesis 插件 solver（参考 `plugins/solvers/` 的 LaunchRecorder 移植模式），
   换取场景统一 stepping 与渲染。

验证标准：检查单更新；升级 PR 中 demo 对照 GIF 无回退。

## 三方对照与决策矩阵（实施 1、2 后填写实测值）

| 场景特征 | 推荐路线 | 理由（待实测确认） |
|---|---|---|
| 静态/编辑 SDF 场景、绳索数量多、需 CSG 编辑场驱动接触 | XPBD + xuvdb | 碰撞层与 genesis 解耦，bake 管线现成 |
| 绳与刚体机器人强耦合、无穿透要求极高、绳数少 | IPCCoupler（轨道 1） | 关节空间与接触联合求解 |
| 快速基线、已在 genesis 场景内、无 SDF 编辑需求 | 原生 PBD（轨道 2） | 零额外依赖 |
| 上游 rope solver 发布后 | 原生优先，XPBD 退居对照 | 维护成本最低 |

## 不做的事

- 不在 Windows 原生环境硬装 libuipc（无官方 wheel）；
- 不为基线测试在 `tests/` 引入多次 `gs.init`（demo 脚本独立进程解决）；
- 不提前抽象"通用绳芯接口"——等轨道 1/2 实测后再定（避免过早设计）。
