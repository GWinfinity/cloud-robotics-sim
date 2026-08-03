# vr_bridge — VR 遥操作桥接插件

把 VR 头显（PICO / 任何 OpenXR 设备）的手柄信号接入 genesis-cloud-sim
仿真，用于遥操作与模仿学习数据采集。头显客户端是独立仓库（Unity 工程），
本插件是 Genesis 侧桥接，两端通过 `protocol/v1/` 的版本化协议解耦。

## 架构

```
headset client                    vr_bridge (this plugin)
┌──────────────┐  UDP 90Hz  ┌─────────────────────────────────────┐
│ 6DoF / 扳机  │ ─────────▶ │ L1 transport   (UDP+TCP, 后台线程)   │
│ 按键事件     │  TCP/NDJSON│ L2 sync_buffer (时钟同步/信箱/事件队列)│
└──────────────┘ ─────────▶ │ L3 filters     (One Euro 滤波)       │
                            │ L4 mapping     (YAML 语义映射)       │
                            │    retargeting (离合增量)            │
                            │ L5 safety      (限幅/看门狗/急停)    │
                            │    IK → control_dofs_position        │
                            │    TeleopRecorder (数据采集)         │
                            └─────────────────────────────────────┘
```

关键设计：

- **双通道**：高频位姿走 UDP（旧包即废），事件/急停走 TCP（可靠送达）。
- **最新值信箱**：网络线程覆盖式写入，控制循环按仿真步频读取，输入与
  仿真频率解耦，抖动不积累延迟。
- **离合增量映射**：按住 grip 记录手柄与末端锚点，之后手柄 delta 按
  `pos_scale` 缩放映射到末端 delta；松开即冻结。
- **YAML 映射表**：换机器人只改 `configs/mappings/*.yaml`，不改代码。
- **安全层独立**：工作空间球限幅（投影而非丢弃）、末端速度限幅、
  100ms 数据龄期冻结、1s 断连缓停、菜单键锁存急停。

## 快速开始（无头显）

```bash
# 终端 1：启动桥接 + 仿真（--mock 模式不需要 genesis）
uv run python plugins/teleop/vr_bridge/examples/run_teleop.py --mock

# 终端 2：键盘模拟客户端（W/S/A/D/R/F 移动，空格=离合，T=扳机，B=录制，M=急停，ESC=退出）
uv run python plugins/teleop/vr_bridge/examples/mock_client.py --mode keyboard
```

`mock_client.py` 还支持 `--mode script`（内置轨迹）、`--mode replay FILE`
（回放录制的输入流）、`--record FILE`（录制输入流用于复现问题）。

### Genesis 真机模式（M1 前置验证）

去掉 `--mock` 即为真机路径：headless Genesis 场景 + Franka（内置 MJCF）
+ 真 IK + 离屏相机。这是 PICO 客户端接入前的完整彩排——控制链路与
真机一致，只是输入来自 mock 客户端：

```bash
# 遥操作 + 采数据：录制 20s episode，存为 dreamdojo 兼容 HDF5
uv run python plugins/teleop/vr_bridge/examples/run_teleop.py \
    --record data/trajectories/teleop_franka.h5 --duration 20
```

注意：Genesis IK 偶发不收敛时会返回 best-effort 解。适配器以当前
`qpos` 热启动并以 `return_error` 校验（位置误差 >1cm 视为 IK 失败，
保持上一帧目标）——切勿直接命令未校验的 IK 输出，那是机械臂跳变的
经典根因。该路径由 `tests/test_e2e_genesis.py` 覆盖（真 Genesis +
真 socket + 真相机渲染 + dreamdojo 加载验证）。

## 接入真实 PICO 客户端

1. 客户端实现 `protocol/v1/README.md`（TCP 握手 → UDP 状态流）。
2. 坐标系：发送前转到 Genesis 世界系（右手系 Z-up），四元数 wxyz。
3. 修改 `configs/vr_bridge.yaml` 的 `network.host` 为局域网地址。

## 数据录制（M2 数据闭环）

- 按 **B 键**（`record_toggle`）开始/停止录制；停止时若
  `configs/vr_bridge.yaml` 的 `recording.output` 已配置，这一段 episode
  会**自动追加写入**该 HDF5 文件并清空缓冲（多次录制按
  `episode_0/1/2...` 累加，适合采 50 条轨迹进同一文件）。
- HDF5 布局与 `plugins/datasets/dreamdojo` 完全兼容：
  `episode_N/observations`（T,H,W,3 uint8 视频）+
  `episode_N/actions`（T,D float32 关节目标），每帧的
  ee_targets/grippers/engaged 以 JSON 存在 group attr `teleop_meta`。
- 视频帧来自控制循环侧：`VRBridge.step(dt, rgb=frame)` 把相机帧透传给
  recorder。**没有 rgb 时 `save_hdf5` 会报 ValueError**（dreamdojo 布局
  必须有视频），此时可手动 `bridge.recorder.save_npz(path)` 退化导出。

加载示例：

```python
import sys
sys.path.insert(0, "plugins/datasets")
from dreamdojo.core.dataset import GenesisDataset

ds = GenesisDataset(
    pre_generated_path="data/trajectories/teleop_franka.h5",
    num_frames=8,          # 小于每条 episode 的帧数
    robot_type="franka",
    device="cpu",
)
sample = ds[0]             # video: (8,3,H,W), action: (8,D)
```

## 配置

- `configs/vr_bridge.yaml`：网络、滤波、安全限幅、映射表选择、录制输出。
- `configs/mappings/`：`franka_single` / `franka_dual` /
  `humanoid_dexhand` 三个示例。

## 测试

```bash
uv run python -m pytest plugins/teleop/vr_bridge/tests -q
```

全部测试使用假机器人与本地回环网络，不依赖 genesis 与头显硬件。

## 路线图

- M0：全链路骨架 + mock 客户端 + 测试。
- M1：PICO Unity 客户端真机联调（协议 v1 已冻结）。
- M2（当前）：`TeleopRecorder` 自动保存 dreamdojo 兼容 HDF5 进
  `data/trajectories`，跨插件加载验证已覆盖（`tests/test_e2e_recording.py`）。
- M3：仿真画面/点云回传头显（引入 WebRTC）、灵巧手手势 retargeting
  （协议 v1.1 `hand_joints`）。
