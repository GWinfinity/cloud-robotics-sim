# MetaWorld+ `button-press-topdown` 迁移攻坚实录

> 状态：**跳过（pending，结构性疑问待与 EFE 作者口径对齐后再做）**
> 记录时间：2026-11-19 ｜ 套件：metaworldplus（49/50 已迁移，本任务为最后 1 个）
> 证据强度：探针 venv（mujoco 3.4.0）对上游原 XML 实测，数据可复现

## 1. 任务与判据

| 项 | 内容 |
|---|---|
| EFE 任务 | `metaworldplus-button-press-topdown-i00`（标准/特权双模式） |
| 上游 | Meta-World `button-press-topdown-v3`（`sawyer_button_press_topdown.xml`） |
| EFE 判据（instruction.md 逐字） | "The button's top is within 2.4 cm of the target point in height; **only z is compared** (the button is pushed down). The target point is `meta['task']['target_pos']`." |

## 2. 上游机制（逐字核对）

- `box` 主体（含 buttonbox）固接于世界，XML 位姿 `(0, 0.85, 0.12)`，`euler="-1.57 0 0"`；
- 按钮 = 套内**滑轨关节** `btnbox_joint`（slide，axis 局部 `-y`，`range [-0.06, 0]`，spring stiffness 0.5 springref 0.5 damping 1）；rest = qpos 0（弹簧顶在高位）；
- socket = 4 个碰撞盒围成的框（真机物理，非装饰）；
- reset：`target_pos` = `hole` 站点世界位置。

## 3. 实测数据（MuJoCo 3.4 mj_forward，原 XML）

| 滑轨 qpos | 按钮体原点 z | 按钮顶 z | hole z | \|顶 − hole\| |
|---|---|---|---|---|
| 0（rest） | 0.1200 | 0.3140 | 0.22 | 0.0940 |
| −0.03（半程） | 0.0900 | 0.2840 | 0.22 | 0.0640 |
| −0.06（全行程） | 0.0600 | 0.2540 | 0.22 | **0.0340** |

**关键结论：全行程 |顶−hole| 恒为 0.034 m > EFE 判据 0.024 m，与 box 位置无关（刚体几何不变量）。**
即按 EFE instruction 逐字口径，该冻结实例物理上不可达成功——并非迁移手法问题。

## 4. 历史 pending 的两个原因（均已被本轮解释）

1. **自由球体建模 nop 失败**：MW+ 通用做法把被控物建成自由球体；本任务是**竖直向下按**，
   成功轴 z 恰是重力方向——悬空球体复位后落体穿过成功窗口（z≈0.13 ∈ [0.096, 0.144]），
   nop 门拦截。sibling `button-press`（水平按 y）不受重力影响，故同建模可通过。
2. **真机制建模也够不到**：见第 3 节实测。

## 5. 待查口径（恢复此任务时按序核对，预计 10–30 分钟）

1. **EFE 冻结实例的 `meta['task']['target_pos']` 实际取值**：`hole` 站点（0.22）还是
   `self.goal`（[0, 0.88, 0.1]）？——拉 EFE 镜像内 `rcb_mw`/`standard-files` 的 grader 源码
   （本仓库 checkout 只有 instruction.md，无 grader）。
2. **grader 的 "button top" 帧定义**：是否含 box 下沉、是否用 MW reward-v2 的
   `obj = body_com + [0,0,0.193]` 口径（该口径 rest=0.5065/压到底=0.4465，亦不可达，
   说明 EFE 应未采用）。
3. **上游可达性反证**：Meta-World 官方将 button-press-topdown 列为可解任务，说明
   官方 reward 口径下可达；EFE 的 0.024 阈值大概率译自官方 reward 容差(0.005)放宽版，
   但换算链条有一处对不上实例几何——找到那处即可定正确窗口。
4. 若最终确认 EFE 实例判据与几何冲突：**按 EFE 的 pending 惯例保留并上报作者**，
   不本地放宽阈值（违反"判据逐字"原则）。

## 6. 若恢复任务，推荐实现路径（已验证可行）

- 建 `assets/metaworld/worlds/sawyer_button_press_topdown_mech.xml` 包装件
  （`<include>` buttonbox_dependencies + `box` 主体按 XML 位姿），相对路径引用；
- `axis_window` 支持 `asset::link` 语法（约 6 行：split 后 `entity.get_link_pos`），
  判据对象 `button_mech::button`；
- 滑轨弹簧 + 限位使复位态天然 nop-safe（rest 顶 0.314 远在窗口外）。

## 7. 决议

**跳过**：不计入已迁移数（维持 49/50），predicates 表保持 `pending: true` 与注记，
本文件存档全部证据。批次 5 按 `.efe_stage/batch5_notes.md` 继续（下一步 Playground 13）。
