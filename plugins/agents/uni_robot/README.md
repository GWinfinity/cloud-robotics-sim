# uni_robot 插件：通用 Genesis 机器人智能体适配器

把**任意 Genesis 场景**接入 [uni-robot-agent](https://pypi.org/project/uni-robot-agent/)
（LLM 操作智能体 harness），让任意 Responses 兼容的多模态大模型
（GPT 系列、Qwen 系列、DeepSeek 等）通过共享工具层
（`move` / `grasp` / `release` / `capture_scene` / `locate` / 技能库）
直接操作仿真机器人。

## 通用而非测评专用

本插件**不绑定任何基准任务**：

- **任务 = 自由文本指令**（`SceneSpec.instruction`）："把红方块放到目标区"
  或任意长程组合任务，由模型自行分解、执行、根据执行回执纠错。
- **成功判据可插拔且可选**（`JudgeSpec`）：声明式位姿目标（物体到达目标点
  容差内）。缺省为**通用模式**——`finish()` 的 `task_success` 恒为
  `None`，由调用方（人或下游流程）评估结果，适合数据采集、遥操作回放、
  自定义评测等场景。协议规定环境公共元数据不得携带成败/分值字段，
  适配器严格遵守（无 privileged 信息）。
- **场景完全由 SceneSpec 描述**：单臂/双臂、任意 URDF、相机布置、
  桌面/物体/障碍，均为配置而非代码。

## 架构

```
SceneSpec (YAML) ──> builder.build_scene ──> BuiltScene (gs.Scene + 句柄)
                                                    │
                              GenesisEnvironmentBackend  <-  协议: read_state /
                                                    │        step_pose / step_joints /
                    EnvironmentService (官方通用)    │        capture / finish
                                                    │
                          HTTP (build_router/create_app/HttpServer)
                                                    │
              uni-robot-agent agent 侧 (PreparedEpisodeSession + 工具层)
                  `--integration genesis --env-url http://127.0.0.1:18760`
```

- 宿主端把每个协议方法调用收敛到一条 owner 线程执行（与官方
  Isaac/RoboDojo 宿主同款模式），Genesis/Taichi 的线程敏感性由此隔离。
- 坐标系约定：世界米制、四元数 wxyz、控制点 = 夹爪中心（grasp site）；
  相机外参按协议光学约定（z 前）输出，从 Genesis OpenGL 约定显式转换。

## 安装

```bash
uv sync --extra agent        # 安装 uni-robot-agent 及插件依赖
```

插件位于 `plugins/agents/uni_robot/core/uni_robot_genesis`，测试/示例通过
把 `core/` 加入 `sys.path` 导入（与 maniskill 插件同一约定）。

## 快速开始

终端 1（宿主端，需要 GPU 的 Linux 主机或本机 CUDA）：

```bash
cd plugins/agents/uni_robot/core
python -m uni_robot_genesis serve --spec ../../examples/franka_table.yaml --port 18760
```

终端 2（agent 侧）：

```bash
export UNI_ROBOT_AGENT_API_KEY=...
python -m uni_robot_genesis agent run --integration genesis \
    --env-url http://127.0.0.1:18760 --task franka-table-demo \
    --model YOUR_MODEL_ID --provider-url https://provider.example/v1 \
    --no-sam3 --output-dir ./runs
```

编程方式（内嵌到自己的流程）：

```python
from uni_robot_genesis import GenesisEpisodeHost, SceneSpec

spec = SceneSpec.from_yaml("examples/franka_table.yaml")
host = GenesisEpisodeHost(spec, session_dir="runs/demo")
url = host.start()
# ... 交给 agent（uni_robot_agent SDK 的 run(SessionConfig(integration="genesis", env_url=url, ...))）
host.stop()
```

## 已知限制

- 驱动增益：Genesis 对 URDF 加载的机器人使用默认 PD 增益（kp=100/kv=40），
  跟踪偏软但稳定；工具层以小步进下发目标，通常够用。需要更快响应时在
  `ArmSpec.kp` / `kv` 按 dof 显式配置（构建时经 `set_dofs_kp/kv` 应用）。
- `--episode-timeout-s` 监督模式会在子进程重放官方 CLI 入口，子进程
  丢失本插件的集成注册；需要超时控制时请用外部 `timeout` 命令包裹
  `python -m uni_robot_genesis agent ...`。
- `--deploy local` / `setup` 是官方内置集成（RoboDojo/RoboUse/RoboLab）
  的进程编排路径，本插件不走该路径；环境服务由本插件宿主端托管。
- 混合动作（hybrid VLA + reviewer）依赖官方 action binding 注册表，
  本插件仅提供 direct 动作模式。
- `preview_joints` 返回限位/跳变诊断而非假想位姿（Genesis 不暴露
  免状态切换的单臂 FK 预览）。
- 单环境（非向量化）；`step_native` 一律拒绝（不透明原生动作与通用
  适配器无关）。多臂 = 多台独立机械臂实体（如双臂场景配两个 ArmSpec）。

## 测试

```bash
# 协议/注册测试：fake 场景，无 genesis、无网络，进插件 CI 白名单
pytest plugins/agents/uni_robot/tests -q -m "not slow"

# 真实 Genesis 冒烟（CPU，需 genesis-world；验证 builder/backend 对接真实 API）
pytest plugins/agents/uni_robot/tests/test_genesis_smoke.py -q -m slow
```
