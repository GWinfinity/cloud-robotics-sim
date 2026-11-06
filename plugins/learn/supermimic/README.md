# supermimic

SuperMimic 风格的**可微世界模型 + 参考动作模仿**学习算法核心,从
[supermimic-genesis](https://github.com/Motphys/Uni-SuperMimic) 路线
(算法源自 Motphys/Uni-SuperMimic,Apache-2.0)移植并**解耦为任务无关
形式**。

## 为什么它比传统 RL 高效

PPO 等 on-policy 方法每个环境步只贡献一次梯度更新,样本效率低。本插件
的核心循环是 Dreamer 式的:

1. **世界模型 (WorldModel)** 在真实转移上训练(帧局部增量 + 解析积分,
   RunningNorm 归一化),学会"context 依赖的机器人动力学";
2. **策略**通过 `horizon` 步**想象 rollout 做 BPTT**——折扣回报 + critic
   bootstrap 直接反传回策略,UTD(更新/数据比)可远大于 1;
3. 仿真器只在采集真实转移和评估时出现,梯度全在 PyTorch 里。

结果是样本效率比 PPO 高一个量级(参考上游 supermimic-genesis 在 G1
全身动作跟踪上的报告)。

## 结构

```
core/supermimic/
  contracts.py   EnvProtocol + SuperMimicMdpProtocol + MdpMirror(解耦边界)
  geometry.py    SO(3) 批量几何(quaternion wxyz,State 展开/折叠)
  networks.py    WorldModel / Policy / Critic / RunningNorm
  optim.py       MuonAdam(隐藏矩阵用 Muon,其余参数用 Adam)
  replay.py      (time, env) 布局的设备常驻 replay buffer
  learner.py     SuperMimicLearner 训练循环(采集/WM 更新/想象 BPTT)
  logging.py     Rich 实时面板 + TensorBoard + CSV(rich 缺失时降级)
  toys/          最小参考任务:双关节正弦跟踪(ToyMdp + ToyEnv,纯 PyTorch)
examples/
  toy_train.py   CPU 端到端训练示例(分钟级)
tests/           52 个用例:几何/网络/replay/learner 循环/玩具任务
```

## 任务解耦边界

learner 不 import 任何机器人相关内容。接入新任务(如 G1 人形、Franka
臂)只需实现:

- **MDP**(`SuperMimicMdpProtocol`):观测、目标帧、奖励、软终止、
  critic 特征、上下文归一化、想象起点校验;可选 `mirror`
  (左右镜像增强)、`goal_q_offset`/`default_q`/`action_scale`
  (启用 `pi_ref_skip` 参考残差策略时必需);
- **Env**(`EnvProtocol`):批量环境,在 `info` 里按
  `SUPERMIMIC_STATE_KEY` / `SUPERMIMIC_TRANSITION_KEY` 提供约定字段
  (见 `toys/toy_env.py`)。

State 布局约定:`(pos(3), rot(3x3), lin_vel(3), ang_vel(3), q(J), qd(J))`,
折叠为 `13 + 2J` 维(quaternion wxyz)。

## 稳定性开关(全部默认关闭,保持上游 SuperMimic 行为)

| 配置 | 作用 | 出处 |
|---|---|---|
| `ema_decay` | 采集用 Polyak 平均策略 | InstantTrack |
| `value_clip` | critic TD 目标钳制到可达回报区间 | allmimic |
| `noise_free_frac` | 部分环境无探索噪声采集 | allmimic |
| `position_obs` | 把机体坐标系 root 位置误差附加到 actor 观测(MDP 需一致) | BlinkMimic |
| `contact_reward` | 可微足部接触匹配/防滑奖励项(MDP 需一致) | BlinkMimic-plus |
| `pi_ref_skip` | 在下一帧参考关节姿态上学习动作残差 | BlinkMimic-plus |
| `wm_structured` | 半隐式(漂移项)世界模型 | BlinkMimic-plus |
| `wm_mirror` | WM 批数据左右镜像增强(需 MDP 提供 `mirror`) | BlinkMimic-plus |
| `pi_opt`/`wm_opt="muon"` | 隐藏矩阵用 Muon 更新 | BlinkMimic-plus |
| `eprobe_*` | 按部署策略失败位置聚焦 reset(需任务提供 hooks,见 contracts) | BlinkMimic-plus |

## 上游差异说明

- 移除 G1/LAFAN1 任务实现与 genesis 后端(本插件核心零 genesis 依赖);
- `GOAL_Q_OFFSET` → `mdp.goal_q_offset`(MDP 属性);`G1Mirror` →
  `mdp.mirror`(MDP 可选属性);
- learner 配置去掉 `fail_reset`(它是环境侧选项,属于任务实现,如
  genesis 后端的 env cfg,而非算法核心);
- 新增 `wm_dt` 配置(世界模型积分步长,默认 0.02,须匹配任务控制 dt)。

## 快速开始

```bash
# CPU 端到端示例(分钟级)
python plugins/learn/supermimic/examples/toy_train.py

# 测试
pytest plugins/learn/supermimic/tests -q -m "not slow"
```

依赖:torch>=2.9(用到 `torch.optim.Muon`)、numpy。rich/tensorboard 可选
(缺失时实时面板禁用,CSV 日志不受影响)。
