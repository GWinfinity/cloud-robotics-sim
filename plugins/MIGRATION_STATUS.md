# Plugin Migration Status

## 已迁移项目 (3/13)

| # | 项目 | 来源 | Plugin 位置 | 类别 | 状态 |
|---|------|------|------------|------|------|
| 1 | **mpc_wbc** | openloong-dyn-control | `plugins/controllers/mpc_wbc/` | controllers | ✅ 完成 |
| 2 | **badminton** | genesis-humanoid-badminton | `plugins/envs/badminton/` | envs | ✅ 完成 |
| 3 | **residual_rl** | genesis-residual-rl | `plugins/controllers/residual_rl/` | controllers | ✅ 完成 |

---

## 待迁移项目 (10个)

### 高优先级 (P0) - 建议先迁移

| # | 项目 | 类别 | 核心功能 | 预计工作量 | 依赖 |
|---|------|------|---------|-----------|------|
| 4 | **table_tennis** | envs | 乒乓球环境 + 双预测器架构 | 中等 | 类似 badminton |
| 5 | **hugwbc** | controllers | HugWBC 全身控制器 | 中等 | 类似 mpc_wbc |
| 6 | **sim2real_dexterous** | sim2real | Sim2Real 灵巧操作 | 大 | 通用奖励函数 |

### 中优先级 (P1)

| # | 项目 | 类别 | 核心功能 | 预计工作量 |
|---|------|------|---------|-----------|
| 7 | **humanoid_falling** | envs | 跌倒保护策略 | 小 |
| 8 | **maniskill** | envs | 操作环境框架 | 大 |
| 9 | **slac** | algorithms | 潜在动作空间 | 中等 |
| 10 | **bfm_zero** | algorithms | BFM-Zero 行为基础模型 | 大 |

### 低优先级 (P2)

| # | 项目 | 类别 | 核心功能 | 预计工作量 |
|---|------|------|---------|-----------|
| 11 | **wbm_embrace** | envs | 全身操作拥抱 | 中等 |
| 12 | **scene_language** | envs | 场景语言 (CVPR) | 大 |

### 不需要迁移

| 项目 | 原因 |
|------|------|
| genesis-cloud-sim | 目标项目本身 |
| genesis-sky | Genesis 引擎官方仓库，作为依赖使用 |

---

## 项目依赖关系

```
                  genesis-sky (引擎)
                       ↑
        ┌──────────────┼──────────────┐
        │              │              │
   controllers      envs          algorithms
        │              │              │
   ┌────┴────┐    ┌───┴───┐      ┌───┴───┐
   │         │    │       │      │       │
 mpc_wbc  hugwbc  │   badminton  │  residual_rl  bfm_zero
   │         │    │   table_tennis│   slac
   └────┬────┘    │   maniskill  │
        │         │   humanoid_* │
        │         │   scene_lang │
        │         │   wbm_embrace│
        │         │              │
        └─────────┴──────────────┘
                      ↑
                 sim2real
                      │
              sim2real_dexterous
```

---

## 推荐迁移顺序

### Phase 1: 核心能力完善 (Week 1-2)
目标: 完善控制器和环境能力

```
1. table_tennis (envs)
   - 来源: genesis-table-tennis
   - 核心: 乒乓球 + 双预测器
   - 收益: 球类运动能力 + 预测器架构

2. hugwbc (controllers)
   - 来源: hugwbc-genesis
   - 核心: 人形全身控制
   - 收益: 与 mpc_wbc 互补
```

### Phase 2: Sim2Real 能力 (Week 3-4)
目标: 建立 Sim2Real 管道

```
3. sim2real_dexterous (sim2real)
   - 来源: genesis-sim2real-dexterous
   - 核心: 域随机化 + Real2Sim
   - 收益: Sim-to-Real 迁移能力

4. humanoid_falling (envs)
   - 来源: genesis-humanoid-falling
   - 核心: 跌倒保护
   - 收益: 安全性 + 课程学习
```

### Phase 3: 高级算法 (Week 5-6)
目标: 前沿算法实现

```
5. slac (algorithms)
   - 来源: genesis-slac
   - 核心: 潜在动作空间
   - 收益: 高效探索

6. bfm_zero (algorithms)
   - 来源: genesis-bfm-zero
   - 核心: 行为基础模型
   - 收益: Zero-shot 能力
```

### Phase 4: 扩展 (Week 7+)
目标: 多样化任务

```
7. maniskill (envs)
   - 来源: genesis-maniskill
   - 核心: 操作环境框架
   - 收益: 丰富操作任务

8. wbm_embrace (envs)
   - 来源: genesis-wbm-embrace
   - 核心: 全身操作
   - 收益: 复杂交互

9. scene_language (envs)
   - 来源: genesis-scene-language
   - 核心: 场景语言 (CVPR)
   - 收益: 语言-场景理解
```

---

## 每个项目的核心可沉淀知识

### 1. table_tennis ⭐ 推荐优先
```
核心实现:
- 双预测器架构 (学习 + 物理)
- 乒乓球物理模型 (马格努斯效应)
- EKF 轨迹预测
- 全身协调控制

沉淀价值:
- 高速运动预测框架
- 双预测器设计模式
- 球类运动通用组件
```

### 2. hugwbc ⭐ 推荐优先
```
核心实现:
- HugWBC 控制器
- RSL-RL 风格 PPO
- 非对称 Actor-Critic
- 命令跟踪

沉淀价值:
- 另一种全身控制方案
- RSL-RL 集成模式
- 步态控制策略
```

### 3. sim2real_dexterous ⭐ 推荐优先
```
核心实现:
- 通用奖励函数 (接触 + 目标)
- Real2Sim 自动调参
- 域随机化
- 策略蒸馏

沉淀价值:
- Sim-to-Real 最佳实践
- 通用奖励设计模式
- 自动调参工具
```

### 4. slac
```
核心实现:
- 潜在动作空间
- DIAYN 风格技能发现
- 低保真预训练

沉淀价值:
- 潜在表示学习
- 技能发现框架
```

### 5. bfm_zero
```
核心实现:
- Forward-Backward 模型
- 无监督预训练
- Zero-shot 任务适应

沉淀价值:
- 基础模型架构
- Zero-shot 迁移模式
```

### 6. maniskill
```
核心实现:
- ManiSkill API 适配
- 数据集转换器
- 多种机器人支持

沉淀价值:
- 操作环境框架
- 数据集格式转换
```

---

## 迁移工作量估算

| 项目 | 核心文件数 | 预计工作量 | 难度 |
|------|-----------|-----------|------|
| table_tennis | 5-6 | 1天 | ⭐⭐ |
| hugwbc | 4-5 | 1天 | ⭐⭐ |
| sim2real_dexterous | 6-8 | 2天 | ⭐⭐⭐ |
| humanoid_falling | 3-4 | 0.5天 | ⭐ |
| slac | 4-5 | 1天 | ⭐⭐ |
| bfm_zero | 6-8 | 2天 | ⭐⭐⭐ |
| maniskill | 8-10 | 2天 | ⭐⭐⭐ |
| wbm_embrace | 4-5 | 1天 | ⭐⭐ |
| scene_language | 10+ | 3天 | ⭐⭐⭐⭐ |

**总计: 约 14 天工作量 (按优先级逐步完成)**

---

## 下一步建议

### 立即可做
```bash
# 迁移 table_tennis (与 badminton 相似)
python plugins/migrate_project.py \
    --source ../../genesis-table-tennis \
    --name table_tennis \
    --category envs
```

### 本周完成
- [ ] table_tennis (envs) - 双预测器架构
- [ ] hugwbc (controllers) - 全身控制

### 下周完成
- [ ] sim2real_dexterous (sim2real) - Sim2Real 管道
- [ ] humanoid_falling (envs) - 跌倒保护

---

*当前进度: 3/13 (23%)*
*下一目标: 6/13 (46%)*
