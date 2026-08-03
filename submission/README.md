# mt-embodied-sim 预赛提交资料

> 整理日期：2026-08-02
> 项目：面向家庭 5S 隐形家务的国产具身智能仿真平台（基于摩尔线程 MT Lambda 算力底座）

## 资料清单

| 序号 | 文件 | 内容 | 本次更新 |
|---|---|---|---|
| 01 | `01_项目申报.md` | 项目名称、一句话简介、项目概述、核心技术与创新点 | ✅ 新增创新点 7「分层运动规划轨迹生成」 |
| 02 | `02_商业计划书.md` | 完整商业计划书（市场痛点、产品方案、商业模式等） | ✅ 升级 v1.2：新增「模块五：动作规划引擎」与技术创新点 5 |
| 03 | `03_个人简介.md` | 创始人/技术负责人个人简介（双版本） | 无改动 |
| 04 | `04_运动规划轨迹生成方案.md` | **新增**：使用 hierarchical_cuRobo_planner 生成机器人动作规划轨迹的技术方案 | ✅ 新增 |

## 附件（`attachments/`）

- `BUSINESS_PLAN_mt-embodied-sim_v1.2.pdf` — 商业计划书 PDF 版（由 `md_to_pdf.py` 生成）
- `trajectories/` — 机器人动作规划轨迹产物（由 `D:\githbi\hierarchical_cuRobo_planner` 生成）：
  - `home_tidy_astar_path.npz` / `home_tidy_astar_path.png` — 家庭台面整理场景的 A* 避障轨迹（本次生成，CPU 可复现）
  - `curobo_full_pipeline_result.npz` / `curobo_full_pipeline_demo.png` — A* → IK → TrajOpt 完整管线输出（GPU + cuRobo 运行）
  - `generate_home_trajectory.py` — 家居场景轨迹一键复现脚本

## 近期工作进展摘要（本次整理依据）

1. **Backend 抽象层与 MUSA 适配完善**（2026-07）：genesis_compat 后端注入加固、CI 无头环境（xvfb/mesa）跑通、MUSA 设备抽象与全量测试；
2. **插件生态修复**：15+ 插件（controllers/envs/sim2real/predictors/datasets/scenes）API 对齐与测试全绿，do-as-i-do 复现插件、CoStream Genesis 复现、RoboTwin 轨迹回放落地；
3. **动作规划能力补齐**（本次新增）：自研 `hierarchical_cuRobo_planner`（PyTorch + cuRobo，CUDA/MUSA 双后端），三层分层规划（GPU A* → 批量 IK → 轨迹优化）+ 长程子目标规划 + PACE 式执行器动力学标定，已生成家居场景轨迹产物（见附件）。

## 复现轨迹生成

```bash
# 家居台面场景 A* 轨迹（无需 GPU）
.venv/Scripts/python.exe submission/attachments/trajectories/generate_home_trajectory.py
```

完整 IK/TO 管线需 CUDA/MUSA + cuRobo，参见 `04_运动规划轨迹生成方案.md` 第 4 节。
