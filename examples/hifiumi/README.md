# HiFi-UMI-2K 轨迹回放验证原型

把 [HiFi-UMI-2K](https://huggingface.co/datasets/simple-world-lab/HiFi-UMI-2K)
（arXiv:2607.25895，LeRobot v3 格式）的双手末端轨迹在 Genesis 中用 Franka Panda 做
**运动学重定向回放**，输出 3mm 精度对标偏差报告。

## 数据准备

下载一个 part 的元数据 + 帧表（约 82 MB，无需下载视频）：

```bash
# 国内可用 hf-mirror 镜像
base="https://hf-mirror.com/datasets/simple-world-lab/HiFi-UMI-2K/resolve/main/chunk-0000/part-0000"
mkdir -p data/hifiumi/chunk-0000/part-0000/{data/chunk-000,meta/episodes/chunk-000}
curl -L "$base/data/chunk-000/file-000.parquet"            -o data/hifiumi/chunk-0000/part-0000/data/chunk-000/file-000.parquet
curl -L "$base/meta/info.json"                             -o data/hifiumi/chunk-0000/part-0000/meta/info.json
curl -L "$base/meta/tasks.parquet"                         -o data/hifiumi/chunk-0000/part-0000/meta/tasks.parquet
curl -L "$base/meta/episodes/chunk-000/file-000.parquet"   -o data/hifiumi/chunk-0000/part-0000/meta/episodes/chunk-000/file-000.parquet
```

`data/hifiumi/` 已加入 `.gitignore`。

## 运行

```bash
# 右手，125 Hz 重采样回放（默认 CPU；有 GPU 可 --device cuda）
uv run python -m examples.hifiumi.replay_hifiumi --episode 0 --hand right --rate 125

# 左手 / 限定帧数冒烟 / 锚定模式
uv run python -m examples.hifiumi.replay_hifiumi --episode 7 --hand left --max-frames 500
```

输出到 `outputs/hifiumi_replay/episodeXXXX_{hand}/`：`report.json`、`report.md`、`deviation_report.png`。

## 管线说明

1. **加载**（`hifiumi_loader.py`）：LeRobot v3 chunked parquet → 20 维 state 拆成双手
   `[xyz + rot6d(first_two_rows) + gripper_rad]`，rot6d → wxyz 四元数；
2. **重采样**：25 fps → 125 Hz（位置/夹爪线性插值，姿态 slerp），对齐论文真机部署的 125 Hz IK 流；
3. **重定位**：轨迹质心锚定到机械臂工作空间甜区（episode 世界原点任意，只保留相对运动；
   姿态经首帧常值偏移对齐 Franka `hand` 连杆系）；
4. **回放**：逐帧 Genesis 内置 IK 解算 + `set_qpos` 运动学回放（偏差反映纯重定向保真度，
   不含 PD 控制滞后；动力学跟踪另测）；
5. **报告**：位置/姿态偏差统计、≤3mm 帧占比、工作空间可达率（>20mm 判不可达）、吞吐与实时倍率。

## 实测结果（CPU，episode 0 / 7）

| Episode | 手 | 位置误差 mean/max | ≤3mm 占比 | 可达率 | 实时倍率 |
|---|---|---|---|---|---|
| 0（微波炉取餐盒） | right | 0.46 / 0.94 mm | 100% | 100% | 0.78× |
| 0 | left | 0.52 / 0.98 mm | 100% | 100% | 0.76× |
| 7（换纸巾盒） | right | 5.35 / 119 mm（含不可达段） | 88.6% | 91.4% | 0.71× |

结论：在机械臂可达范围内，重定向偏差 < 1mm，远优于 HiFi-UMI 的 3mm 采集精度；
误差主要来自人体大幅度动作超出固定基座机械臂工作空间（可达率指标即为该场景设计）。

两条工程经验（工作空间可达率优先于跟踪精度、偏差度量须用运动学重定向）的详细记录见
[LESSONS.md](./LESSONS.md)。
