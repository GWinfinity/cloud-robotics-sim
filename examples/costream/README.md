# CoStream Genesis 仿真复现

本目录在 **Genesis** 物理引擎中对论文 **CoStream: Composing Simple Behaviors for Generalizable Complex Manipulation**（arXiv:2606.26423）的核心思想做最小可行仿真复现：

- 语义行为（Semantic）→ 提供任务坐标系锚点 `WTI`
- 预测行为（Predictive）→ 在任务坐标系下生成名义轨迹 `I T_traj(t)`
- 反应行为（Reactive）→ 基于触觉/力反馈输出残差 `T_tac(t)`
- 行为组合器 → 通过 SE(3) 右乘合成最终末端位姿命令
- 编译型柔顺控制器 → 将命令映射为关节目标并加入力闭环

> 注意：这是**算法骨架/概念验证**，不包含真实 LLM/VLM/视频世界模型/GelSight 硬件。真实系统中的 LLM/VLM 被替换为基于仿真真值的场景解析器；视频世界模型被替换为参数化运动先验；GelSight 被替换为 Genesis 接触力与对象滑移估计。

## 目录结构

```
examples/costream/
├── costream/
│   ├── math_utils.py      # SE(3) 工具函数
│   ├── specs.py           # StageSpec / ComposeSpec / 场景摘要
│   ├── behaviors.py       # 语义/预测/反应行为
│   ├── composer.py        # 多速率 SE(3) 行为组合器
│   ├── controller.py      # 编译型柔顺控制器
│   ├── sim_robot.py       # Franka + 运动学 Peg 封装
│   ├── scene_builder.py   # Genesis 场景搭建
│   └── runtime.py         # 阶段监督器与演示循环
├── tests/test_costream.py # 单元测试
└── demo_insertion.py      # 可运行的插孔演示
```

## 运行环境

使用 `genesis-cloud-sim` 仓库已经配置好的 Python 环境（`genesis-world>=0.4.0`）。

```bash
cd D:\githbi\genesis-cloud-sim
.venv\Scripts\python examples\costream\demo_insertion.py
```

运行一次约 1.5 分钟（CPU），典型输出：

```
[CoStream] Stage: approach (2.0s)
[CoStream] Stage: insert (4.0s)
[CoStream] Stage: home (2.0s)

============================================================
Result Summary
============================================================
  approach  : OK
  insert    : OK
  home      : OK

Final TCP pos:  [ 3.99917019e-01 -4.82661812e-04  9.70341014e-01]
Max contact force observed: 0.980 N
```

### 鲁棒性测试（有语义偏置）

```bash
.venv\Scripts\python examples\costream\demo_insertion_perturbed.py
```

这个版本把“孔”在场景摘要里故意向 +y 方向偏移 6 mm（等于单侧间隙的一半），模拟感知/语义误差。CoStream 仍然完成插入，三阶段均 OK，最终 TCP 回到接近真实孔中心的位置。

## 运行测试

```bash
.venv\Scripts\python -m pytest examples\costream\tests\test_costream.py -v
```

## 演示任务

`demo_insertion.py` 搭建一个**窄缝插孔**任务：

- Franka Panda 手臂末端携带一个圆柱形工具（运动学跟随）。
- 两块挡板形成一条宽约 12 mm 的缝隙，工具半径 5 mm，单侧间隙约 1 mm。
- 三阶段：接近（approach）→ 插入（insert）→ 收回（home）。
- 反应行为读取工具与挡板的接触力，输出横向修正，演示接触柔顺。

## 扩展路线

1. 在 `SemanticBehavior` 中接入 LLM/VLM，把自然语言指令解析为 `StageSpec`。
2. 在 `PredictiveBehavior` 中接入视频生成模型，替代参数化轨迹。
3. 在 `ReactiveBehavior` 中接入真实 GelSight Mini 法向流/力矩传感器。
4. 把 `FrankaSim` 替换为真实机器人 SDK（Franka Robotics `libfranka` / ROS2）。
