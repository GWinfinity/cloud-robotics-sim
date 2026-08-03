# VR Bridge Protocol v1

VR 头显客户端（PICO / Quest / 任何 OpenXR 设备）与 genesis-cloud-sim
`vr_bridge` 插件之间的通信协议。本目录是协议的**权威定义**，两端实现以
`messages.json` 为机读 schema。

## 1. 传输通道

| 通道 | 协议 | 方向 | 内容 | 默认端口 |
|---|---|---|---|---|
| 状态流 | UDP | 客户端 → 服务端 | `ControllerState` @ 60–90Hz | 5555 |
| 控制通道 | TCP（换行分隔 JSON） | 双向 | 握手 / 事件 / 心跳 / 急停 | 5556 |

设计理由：

- 状态流走 UDP：高频位姿"旧包即废"，TCP 队头阻塞会放大延迟；丢包由
  服务端插值与看门狗兜底。
- 控制通道走 TCP：急停、reset、录制开关等事件必须可靠送达。
- 所有消息体为 UTF-8 JSON。v1 刻意不用 protobuf，降低客户端首版接入
  成本；性能不足时在 v2 引入二进制编码（见 CHANGELOG 升级策略）。

## 2. 坐标系约定（强制）

- **客户端发送前必须转换到 Genesis 世界系**：右手系，**Z-up**，单位米。
- 四元数顺序为 **w, x, y, z**（与 Genesis / 本项目 `backend.types.Pose` 一致）。
- OpenXR 原生坐标是右手系 Y-up，转换在客户端完成；服务端不做基变换，
  只做平移 / 缩放 / 限幅。
- 位姿基准：客户端应相对于"标定原点"（通常是头显 Guardian 原点）发送。

## 3. 消息目录

### 3.1 控制通道（TCP，每行一个 JSON 对象）

客户端 → 服务端：

| type | 字段 | 说明 |
|---|---|---|
| `hello` | `protocol_version`, `device`, `client_time_ms` | 连接后第一条消息 |
| `ping` | `client_time_ms` | 心跳，建议 1Hz |
| `event` | `event`, `pressed`, `client_time_ms` | 按键事件：`a`/`b`/`x`/`y`/`menu`/`grip_left`/`grip_right` 等。注意 `menu`（急停）是**按下翻转**语义，见 §4 |
| `bye` | — | 主动断开 |

服务端 → 客户端：

| type | 字段 | 说明 |
|---|---|---|
| `welcome` | `protocol_version`, `session_id`, `server_time_ms` | 握手成功 |
| `error` | `code`, `message` | 见错误码表 |
| `pong` | `client_time_ms`, `server_time_ms` | 回显 ping，用于客户端测 RTT |
| `estop_ack` | `server_time_ms` | 急停已生效 |

### 3.2 状态流（UDP，每个数据报一个 JSON 对象）

`ControllerState`：

```json
{
  "type": "state",
  "seq": 12345,
  "client_time_ms": 812345,
  "head":  {"pos": [0, 0, 1.6], "quat": [1, 0, 0, 0]},
  "left":  {"pos": [0.2, 0.3, 1.0], "quat": [1, 0, 0, 0],
            "trigger": 0.0, "grip": 1.0, "thumbstick": [0.0, 0.0]},
  "right": {"pos": [0.2, -0.3, 1.0], "quat": [1, 0, 0, 0],
            "trigger": 0.8, "grip": 1.0, "thumbstick": [0.0, 0.5]}
}
```

- `seq` 单调递增，服务端据此检测丢包 / 乱序。
- `client_time_ms` 为客户端单调时钟（如 OpenXR predicted display time
  或 steady clock），服务端用它估算时钟偏移与数据龄期。
- `trigger` / `grip` ∈ [0, 1]；`thumbstick` 两个分量 ∈ [-1, 1]。
- `head` 可省略（v1 服务端不消费，预留给 M3 画面回传）。

## 4. 会话时序

```
client                                server
  │ ─── TCP connect ──────────────────▶ │
  │ ─── hello ────────────────────────▶ │
  │ ◀── welcome / error ─────────────── │
  │ ─── UDP state @90Hz ──────────────▶ │  （握手成功后才接受状态流）
  │ ─── ping (1Hz) ───────────────────▶ │
  │ ◀── pong ────────────────────────── │
  │ ─── event(a, pressed) ────────────▶ │
  │ ─── event(menu, pressed=true) [急停挂起] ▶ │
  │ ◀── estop_ack ──────────────────────────── │
  │ ─── event(menu, pressed=true) [急停解除] ▶ │
  │ ◀── estop_ack ──────────────────────────── │
  │ ─── bye ────────────────────────────────▶ │
```

约束：

- 同一时刻只允许一个控制会话；第二个连接收到 `error(SESSION_BUSY)`。
- 状态包龄期 > `freeze_timeout_ms`（默认 100ms）服务端冻结末端目标；
  > `disconnect_timeout_ms`（默认 1000ms）判定断连，机械臂缓停。**会话不
  因此结束**：状态流恢复后控制自动继续，无需重新握手；只有 TCP 控制通道
  断开（或 `bye`）才结束会话。
- 急停（`menu` 键）是锁存的，**按下翻转**：`event(menu, pressed=true)`
  在 ESTOP 与正常状态间切换；`pressed=false` 被忽略。这样物理点按产生的
  press+release 事件对不会造成“挂上又瞬间解除”。客户端不需要自行实现
  切换逻辑，透传原始按键事件即可。
- `estop_ack` 表示“急停事件已被服务端处理”，**不表示当前处于 ESTOP
  状态**（v1 无服务端状态推送，HUD 状态回报见 CHANGELOG v1.2 候选）。
- `menu` 在协议层保留给急停：服务端对每个 menu 按下都会回 `estop_ack`，
  且所有官方 mapping 都将 menu 映射为 `emergency_stop`。客户端不应把
  menu 挪作他用。

## 5. 错误码

| code | 含义 |
|---|---|
| `PROTOCOL_MISMATCH` | 客户端协议版本不兼容 |
| `SESSION_BUSY` | 已有活动会话 |
| `BAD_MESSAGE` | JSON 解析失败或字段缺失 |
| `INTERNAL` | 服务端内部错误 |

## 6. 客户端实现注意事项

- **UDP 源绑定**：服务端只接受与 TCP 控制通道**同一源 IP** 的状态包。
  两端必须直连或同网段；NAT 导致两通道源 IP 不一致的场景 v1 不支持。
- **心跳不强制**：`ping`/`pong` 仅用于客户端测 RTT，服务端不做应用层
  心跳超时。TCP 活性由 socket 断连检测；状态流活性由服务端看门狗检测
  （冻结/缓停，见 §4）。
- **容错规则**（向后兼容的基础，两端都必须遵守）：
  - 接收方必须**忽略未知字段**，不得因多出字段拒绝消息。
  - 客户端必须**忽略未知的服务端消息类型**（为未来版本的新消息预留）。
  - 服务端对未知的客户端消息类型回 `error(BAD_MESSAGE)` 但**保持连接**。
  - 状态流（UDP）中的坏包被**静默丢弃**（仅服务端日志），无任何反馈；
    客户端应依据 `messages.json` 在发送前自校验。
- **四元数**：客户端应发送单位四元数；服务端做防御性归一化，但客户端
  不应依赖此行为。
- **时钟**：`client_time_ms` 必须单调不减（如 steady clock / monotonic）。
  服务端用它估算时钟偏移与数据龄期，回拨会破坏看门狗。

## 7. 版本协商与升级策略

- `hello.protocol_version` 为主版本号整数（当前为 `1`）。服务端按**精确
  匹配**校验：不等于当前主版本直接拒绝（`PROTOCOL_MISMATCH`）。
- v1.x 生命周期内 `protocol_version` 恒为 `1`：新增消息类型 / 可选字段
  属于次版本演进，依赖 §6 的容错规则实现向后兼容，不需要版本号变化。
- 修改既有字段语义 / 删除字段 = 主版本 +1，两端同步切换。
