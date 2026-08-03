# Protocol Changelog

## v1.0 (2026-07-30)

- 初版：UDP 状态流 + TCP 控制通道，JSON 编码。
- 消息：`hello` / `welcome` / `error` / `ping` / `pong` / `event` /
  `estop_ack` / `bye` / `state`。
- 坐标系约定：Genesis 世界系（右手系 Z-up），四元数 wxyz，客户端负责
  基变换。
- 冻结评审修订（同日）：急停改为**按下翻转**语义（`pressed=false` 被忽
  略），适配真实 OpenXR 客户端的点按事件对；状态超时不再宣称结束会话
  （会话保持，流恢复自动继续）；补充 UDP 源 IP 绑定、容错规则、时钟与
  四元数要求等客户端实现注意事项（README §6）。

## 候选升级（未实现，仅记录方向）

- v1.1：状态流增加手势追踪关节（`hand_joints`，26×2，可选字段）。
- v1.2：服务端 → 客户端的状态回报（EE 实际位姿、录制状态、安全状态），
  用于头显端 HUD。
- v2.0：二进制编码（protobuf / flatbuffers）与 WebRTC DataChannel 承载
  （配合 M3 视频回传）。主版本升级，两端同步切换。
