# Genesis 仿真持续改进 Agent Loop 设计

## 目标

在 `genesis-cloud-sim` 上构建一个 **持续改进闭环**：让 agent 自动发现仿真中的缺陷、生成改进假设、用 A/B 实验验证、合并通过的改进，并持续监控回归。

## 核心循环

```
┌─────────┐   ┌──────────┐   ┌──────────┐   ┌─────────────┐   ┌─────────┐   ┌────────┐
│ Observe │ → │ Diagnose │ → │ Propose  │ → │   Validate    │ → │  Adopt  │ → │  Learn │
│  观察    │   │  诊断    │   │  提案    │   │  A/B 验证     │   │  采纳   │   │  学习   │
└─────────┘   └──────────┘   └──────────┘   └─────────────┘   └─────────┘   └────────┘
     ↑                                                                                │
     └────────────────────────────────────────────────────────────────────────────────┘
```

### 阶段说明

1. **Observe（观察）**
   - 运行大量仿真 episode
   - 收集：成功率、延迟、物理稳定性、奖励曲线、碰撞次数、关节饱和等

2. **Diagnose（诊断）**
   - 对比当前指标与目标阈值
   - 识别问题类型：
     - `success_rate_low`: 成功率低
     - `physics_unstable`: 物理震荡/穿模
     - `reward_shaping_bad`: 奖励 shaping 不佳
     - `sim_slow`: 仿真 FPS 低
     - `generalization_poor`: 泛化能力差

3. **Propose（提案）**
   - 针对诊断结果生成候选改进：
     - 调整物理参数（dt, substeps, joint stiffness/damping）
     - 修改奖励函数（success_threshold, step_penalty, shaping weight）
     - 调整场景配置（spawn 随机化、目标位置分布）
     - 切换控制器/策略超参数

4. **Validate（验证）**
   - 使用现有的 `ABTestRunner` 对比 baseline vs proposal
   - 运行 N 个 episode，计算成功率、延迟、稳定性
   - 使用 `GradualMigration` 做渐进式切换

5. **Adopt（采纳）**
   - 若改进显著且通过置信度检验，写入新的配置
   - 保存 checkpoint 与实验报告

6. **Learn（学习）**
   - 更新问题-改进知识库
   - 记录失败提案，避免重复尝试
   - 用于后续诊断的参考

## 架构设计

```
cloud_robotics_sim/
├── runtime/
│   ├── agent_loop.py          # 主循环入口
│   ├── metrics.py             # 指标收集与存储
│   ├── diagnostics.py         # 诊断引擎
│   ├── proposals.py           # 改进提案生成器
│   ├── experiments.py         # 实验运行器（包装 ABTest）
│   └── knowledge_base.py      # 问题-改进知识库
├── configs/
│   └── improvement_loop.yaml  # 循环配置
└── outputs/improvement_loop/  # 实验结果与报告
```

## 关键抽象

- `ImprovementLoop`: 主循环，控制观察-诊断-提案-验证-采纳-学习流程
- `MetricCollector`: 统一收集仿真运行指标，支持 episode 级和 step 级
- `DiagnosticEngine`: 根据指标和阈值诊断问题，输出 `Diagnosis` 列表
- `ProposalGenerator`: 根据诊断生成候选改进 `ImprovementProposal`
- `ExperimentValidator`: 负责 baseline vs proposal 的对比实验
- `KnowledgeBase`: 记录历史改进、成功/失败模式

## 与现有项目集成

- 复用 `EnvironmentComposer` + `ComposedEnvironment` 运行仿真
- 复用 `ABTestRunner` / `GradualMigration` 做 A/B 验证
- 复用 `registry` 注册新场景/机器人/任务变体
- 配置使用现有 `configs/*.yaml` 格式

## 最小可运行流程

1. 加载 baseline 配置
2. 运行 `n_baseline_episodes` 收集指标
3. 诊断问题
4. 生成 3 个候选改进
5. 对每个候选运行 A/B 测试
6. 选择最佳改进并更新配置
7. 循环 2-6 直到收敛或达到最大迭代次数

## 输出

- 每个 iteration 的 JSON 报告
- 对比图表（成功率、延迟、奖励）
- 最终采纳的配置文件
- Markdown 总结报告

## 扩展方向

- 引入 LLM 做自然语言诊断和提案生成
- 连接 VLA 模型做视觉-语言任务改进
- 多目标优化（Pareto 前沿：成功率 vs 速度）
- 自动化回归测试（CI 触发 loop）
- 分布式并行实验（多 GPU）
