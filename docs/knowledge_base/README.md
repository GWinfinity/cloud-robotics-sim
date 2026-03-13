# Genesis Cloud Sim 知识库

本知识库沉淀了从多个 Genesis 项目中提取的核心算法和经典实现。

## 文档列表

### 1. MPC + WBC 全身控制器
**文件**: `01_mpc_wbc_controller.md`

**来源项目**: `openloong-dyn-control`, `hugwbc-genesis`

**核心内容**:
- MPC (模型预测控制) QP求解实现
- WBC (全身控制) 零空间投影
- 步态调度器设计
- 人形机器人平地/楼梯/地形行走控制

**适用场景**: 人形机器人全身控制、复杂地形行走、跳跃

---

### 2. 残差强化学习 (Residual RL)
**文件**: `02_residual_rl.md`

**来源项目**: `genesis-residual-rl`

**核心内容**:
- 残差网络架构设计 (轻量级 + 输出受限)
- 组合策略 (BC + Residual)
- Residual SAC 训练 (只更新残差网络)
- Sim-to-Real 微调策略

**适用场景**: 行为克隆后微调、操作任务优化、安全探索

---

### 3. 球类运动控制
**文件**: `03_ball_sports_control.md`

**来源项目**: `genesis-humanoid-badminton`, `genesis-table-tennis`

**核心内容**:
- 羽毛球/乒乓球物理建模 (空气动力学)
- 三阶段课程学习 (步法→挥拍→协调)
- EKF 轨迹预测器
- 全身协调控制 (23-29 DOF)

**适用场景**: 人形机器人球类运动、高速物体追踪

---

### 4. 通用操作奖励函数
**文件**: `04_generalized_manipulation_reward.md`

**来源项目**: `genesis-sim2real-dexterous`

**核心内容**:
- 接触奖励设计 (手掌 + 手指)
- 物体目标奖励 (任务特定)
- 手部引导奖励
- 多任务通用框架

**适用场景**: 抓取放置、双手交接、箱体提升、插销装配

---

### 5. 双预测器架构
**文件**: `05_dual_predictor_architecture.md`

**来源项目**: `genesis-table-tennis`

**核心内容**:
- 学习预测器 (LSTM/MLP，快速鲁棒)
- 物理预测器 (数值积分，精确一致)
- 预测增强奖励设计
- 观测增强策略输入

**适用场景**: 高速球类运动、动态避障、多智能体预测

---

## 快速参考

### 按任务类型选择文档

| 任务类型 | 推荐文档 |
|---------|---------|
| 人形机器人行走 | 01_mpc_wbc_controller.md |
| 操作任务微调 | 02_residual_rl.md |
| 球类运动 | 03_ball_sports_control.md + 05_dual_predictor_architecture.md |
| 抓取放置 | 04_generalized_manipulation_reward.md |
| Sim-to-Real | 02_residual_rl.md + 04_generalized_manipulation_reward.md |
| 高速预测 | 05_dual_predictor_architecture.md |

### 关键技术栈

```
控制器层:
  ├── MPC (模型预测控制)
  ├── WBC (全身控制)
  └── 残差 RL (安全微调)

感知层:
  ├── EKF 预测
  └── 双预测器 (学习 + 物理)

学习层:
  ├── 课程学习
  ├── 奖励塑形
  └── Sim-to-Real

奖励层:
  ├── 接触奖励
  ├── 目标奖励
  └── 预测增强奖励
```

## 使用建议

1. **阅读顺序**: 
   - 新手: 04 → 02 → 01
   - 球类运动: 03 → 05
   - 全身控制: 01 → 02

2. **代码迁移**:
   - 每个文档包含完整可运行的代码示例
   - 可以直接复制到 genesis-cloud-sim 项目中使用
   - 注意调整超参数以适应具体任务

3. **调试技巧**:
   - 每个文档包含故障排查表格
   - 按现象查找可能原因和解决方案

---

## 贡献指南

如需添加新的知识模块:

1. 创建新的 markdown 文件 (`06_xxx.md`)
2. 遵循统一的文档结构:
   - 算法原理 (论文引用 + 核心思想)
   - 使用场景 (适用任务 + 限制)
   - 代码示例 (最小可运行)
   - 超参数指南 (调优建议)
3. 更新本 README 的索引

---

## 相关资源

- [Genesis 官方文档](https://genesis-world.readthedocs.io/)
- [OpenLoong 控制器](https://github.com/OpenLoongRobot)
- [ManiSkill 文档](https://maniskill.readthedocs.io/)

---

*最后更新: 2026-03-13*
