# Agent 接口指南

本文介绍如何让 agent（LLM 或其他自动化程序）驱动仿真平台：
技能注册表（Skill Registry）、任务执行器（Task Executor）、回放缓冲
（Replay Buffer），以及 MCP 工具面。

## 三层结构

```
agent (LLM / CLI / 脚本)
   │
   ├─ MCP 工具面  runtime/agent_hub.py   ← list_tools / call_tool / serve_stdio
   ├─ 任务执行器  runtime/skills.py      ← TaskExecutor（goal 解析 + 执行记录）
   └─ 技能注册表  runtime/skills.py      ← Skill（name + 描述 + 参数 schema + handler）
                     │
        内置技能：run_patent / list_patents / list_scenes /
                 list_robots / list_tasks / submit_sim_task
```

## CLI 用法

```bash
# 列出全部技能
cloud-robotics-sim agent --list-skills

# 用自然语言目标（确定性匹配到最相关的技能并执行）
cloud-robotics-sim agent --goal "run patent US821393 headlessly"

# 直接调用指定技能（参数自动解析 int/float/bool/JSON）
cloud-robotics-sim agent --skill run_patent --param run=US821393 --param steps=500
```

`--goal` 的解析是确定性的 token 匹配（技能名权重高于描述），不依赖
LLM；`run_patent` 会从目标文本中正则提取专利号（如 `US821393`）。
需要精确控制参数时请使用 `--skill` + `--param`。

## Python API

```python
from cloud_robotics_sim.runtime.skills import (
    TaskExecutor, SkillRegistry, ReplayBuffer, register_skill,
)

executor = TaskExecutor()                      # 内置技能注册表
record = executor.execute_goal("run patent US821393")
print(record.status, record.result)            # ok {...}
print(executor.replay.records())               # 执行回放

# 注册自定义技能（追加到默认注册表）
@register_skill(
    "my_skill",
    "描述：做什么、什么时候用。",
    parameters={
        "type": "object",
        "properties": {"x": {"type": "integer"}},
        "required": ["x"],
    },
)
def _my_skill(params):
    return {"x_squared": params["x"] ** 2}
```

## MCP 工具面

`runtime/agent_hub.py` 提供与 `devices/mcp_adapter.py` 同风格的零依赖
MCP 适配器：

```python
from cloud_robotics_sim.runtime.agent_hub import SimHub, list_tools, call_tool

hub = SimHub()
print(list_tools(hub))
result = call_tool(hub, "sim.goal.run", {"goal": "list the patents"})
```

工具一览：

| 工具 | 说明 |
|------|------|
| `sim.skills.list` | 发现可调用的仿真技能 |
| `sim.skill.describe` | 查看单个技能的参数 schema |
| `sim.skill.run` | 按名执行技能（`{"name", "params"}`） |
| `sim.goal.run` | 自然语言目标 → 最佳匹配技能并执行 |
| `sim.replay.list` | 查看执行回放记录 |
| `sim.components.list` | 列出已注册的场景/机器人/任务（`{"kind": "scenes"\|"robots"\|"tasks"}`） |
| `sim.queue.length` | KEDA 部署中某 Redis 队列的积压任务数 |

安装可选依赖 `mcp` 包后，可直接作为 MCP server 挂在任何支持 MCP 的
agent 框架上：

```bash
uv add mcp
python -m cloud_robotics_sim.runtime.agent_hub   # stdio 模式
```

## 与 Kubernetes 部署的关系

`submit_sim_task` 技能把任务推入 Redis 队列，由 KEDA 自动扩缩容的
worker 池异步执行（见 [Kubernetes 部署指南](kubernetes.md)）；同步执行
则直接由 `run_*` 技能在本地完成。两者共享 `run_patent_task` 实现。
