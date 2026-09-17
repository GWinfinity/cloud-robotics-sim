# 渲染阴影伪影排查：MuJoCo 阴影贴图 bug 与 Genesis 的差异

**日期**: 2026-09-16
**来源**: 外部项目（WHAM/GMR 人形重映射管线，MuJoCo 渲染）发现并坐实的渲染器 bug；本文排查本仓库（genesis-cloud-sim，genesis-world 1.4.0）是否暴露同类问题。
**结论**: Genesis 不存在同款 bug（机制上不成立）；仅 patents 长跑渲染存在表观相似的次级风险。

---

## 1. 背景：MuJoCo 已坐实的阴影伪影 bug

外部项目渲染人形机器人重放视频（同进程顺序渲染 2680 帧）时发现幻影黑色楔形/三角伪影，三组对照实验闭合了证据链：

| 实验 | 结果 |
|---|---|
| 全新进程单帧渲染（3.12） | 干净（暗像素 0） |
| 同进程顺序渲染到第 262 帧（3.12） | 伪影（区域 min=23，745 暗像素） |
| 最新 MuJoCo 3.13.0 干净 venv 顺序渲染 | **伪影原样复现**（逐位相同） |
| 3.12 + 整条轨迹平移到世界原点 | **伪影完全消失** |

- **根因机制**：平行光的阴影正交相机**锚定在光源 `pos`（原点）**，视锥固定。机器人根轨迹活动范围离锚点 3m+ 时，阴影贴图采样产生位置漂移的幻影黑斑。
- **排除的嫌疑**：反射通道（开关反射都在）；阴影贴图缓存未失效（"关→开阴影"强制刷新无效，min=23/745 不变）；环境问题（最新版 3.13.0 复现）。
- **上游记录**：MuJoCo changelog 有 "Fixed a bug that caused shadow rendering to flicker on platforms that do not support ARB_clip_control" 及 issue #1831（离屏多次渲染状态泄漏），同属阴影/状态问题家族，但本案例（EGL/NVIDIA）在含修复版本上仍存在，属未修复变体。
- **已验证的修法**：交付前把根 xy 平移到质心≈世界原点（刚体平移，动作不变，棋盘地面无限延伸视觉上无差）。备选：关阴影（丢影子）或光源 pos 跟随机器人。

## 2. Genesis 1.4.0 的机制差异（逐行核实，非同款 bug）

Genesis 的阴影相机**不锚定光源位置，而是锚定场景 AABB 质心，且每帧强制重算**：

```python
# .venv/Lib/site-packages/genesis/ext/pyrender/renderer.py:786-789 (_get_light_cam_matrices)
if isinstance(light, DirectionalLight):
    direction = -pose[:3, 2]
    c = scene.centroid
    loc = c - direction * scene.scale        # 阴影相机位置 = AABB 质心 沿光源反方向外推一个场景尺度
    pose[:3, 3] = loc

# .venv/Lib/site-packages/genesis/ext/pyrender/light.py:164-166 (_get_shadow_camera)
return OrthographicCamera(znear=0.01 * scene_scale, zfar= 10 * scene_scale,
                          xmag=scene_scale, ymag=scene_scale)   # scene_scale = 场景 AABB 对角线

# .venv/Lib/site-packages/genesis/vis/rasterizer_context.py:1152 (每次 update_scene)
self._scene._bounds = None   # 重置包围盒触发重算 → 阴影相机矩阵与 shadow map 每帧重建
```

| | MuJoCo（出问题的） | Genesis 1.4.0（本仓库） |
|---|---|---|
| 平行光阴影相机锚点 | 光源 pos 原点，固定 | 场景 AABB 质心（`renderer.py:786-789`） |
| 正交视锥范围 | 固定 | 跟随 AABB 对角线（`light.py:164-166`） |
| 帧间更新 | 状态可残留 | 每帧 `_bounds=None` 强制重算，阴影相机矩阵/shadow map 重建 |

推论：

1. **"轨迹离光源锚点远"这一触发条件在 Genesis 不成立**——视锥围着场景质心画，机器人在哪都在视锥内。MuJoCo 的"平移到原点"修复对本仓库**不是必需的**。
2. **同进程顺序渲染的状态残留也不存在**——每帧阴影相机矩阵重算、shadow map 重渲，无缓存可残留。
3. `gs.morphs.Plane` 地板特殊处理：只取质心参与包围盒（`ext/pyrender/scene.py:236-237`），无限地板不会撑大阴影视锥。

## 3. 本仓库渲染入口盘点

全部为**同进程顺序渲染多帧**（与 MuJoCo 侧相同的使用模式），无每帧新进程路径：

| 入口 | 位置 | 相机 | 场景规模 | 风险 |
|---|---|---|---|---|
| patents runner / `run_patent`（skills.py 同步与 K8s 异步共用） | `src/cloud_robotics_sim/patents/runner.py:135-144`，`base.py:193-208` | 固定或 `follow_entity` | 1–15 m，长跑轨迹可达数十米 | **中**（见 §4） |
| home_demo | `examples/robotwin/home_demo.py:823-890` | OrbitCamera 移动 | 4.2 m 房间 | 低 |
| composer 环境 | `src/.../core/composer.py:192-208` | 固定 head_cam | 10×10×3 m 房间 | 低 |
| do_as_i_do | `plugins/do_as_i_do/core/env.py:134-144,242-245` | 固定 | 桌面 ~1 m | 低 |
| aloha_demo / replay.py | `examples/robotwin/aloha_demo.py`、`replay.py:199-246` | 固定 | 桌面 ~1 m | 低 |
| dengyu grasp | `examples/grasp/dengyu_multiscale_soft_grasp.py:225` | 固定 | 桌面 | 低 |
| hifiumi | `examples/hifiumi/replay_hifiumi.py` | **不做 Genesis 渲染**（仅 matplotlib 偏差图）；loader 已做 `recenter_to_workspace` | — | 无 |
| GPU batch renderer | `batch_renderer.py:333-340`（Madrona） | — | — | 未知（阴影逻辑在黑盒内；当前入口均不走此路径） |
| vendored sky（0.3.11） | `plugins/envs/sky/core/genesis/` | — | — | 中（旧栈，AGENTS.md 禁止 core 引用） |

灯光侧：阴影默认开启（`VisOptions.shadow=True`，默认 DirectionalLight）。阴影贴图 8192²（`ext/pyrender/constants.py:11`），正交参数全部 genesis 内硬编码，项目代码零暴露。

## 4. 次级风险：patents 长跑渲染（非同款 bug，表观可能相似）

Wright Flyer / Goddard Rocket 等长跑专利（轨迹数十米 + `follow_entity`）可能出现：

1. **阴影分辨率稀释**：动态实体拉出长尾 AABB → 8192² 贴图被整个航迹稀释 → 阴影块状/低清；
2. **深度精度变粗**：`znear/zfar` 按场景 scale 同比放大 → z-fighting / 阴影边缘 shimmer；
3. **帧间跳变**：每帧重算 bounds → 阴影边缘和分辨率帧间变化（外观像 flicker）。

**规避办法**（二选一）：

- 长跑专利渲染时保持场景包围盒紧致（避免高速实体拉出长尾 AABB）；
- 在 `PatentSimConfig` 层关闭阴影（`VisOptions(shadow=False)`）。

其余入口场景小、AABB 稳定，风险低，无需处理。

## 5. 附带发现：composer 灯光配置静默失效

`src/cloud_robotics_sim/backends/genesis_backend.py:915-917` 的 `add_light` 调用 `genesis_compat.get_genesis_lights()`，而 genesis-world ≥1.2 移除了 `gs.lights`（helper 恒返回 None），于是 **debug 日志后静默跳过**。结果是 composer 路径实际吃 `VisOptions` 默认灯，与 `core/scene.py:371-393` 的显式灯光配置意图（ambient + 主平行光 `cast_shadow=True`，默认 pos `(5,-5,8)`）不符。

- 影响：不影响阴影正确性（默认灯也带阴影），但显式配置的灯位/强度/环境光都不生效。
- 建议：把 `SceneConfig.main_light` 注入 `VisOptions`（`gs.options.VisOptions` 的灯光字段），并删除/告警 `add_light` 死路径。

## 6. 排查方法备忘（复用于其他渲染问题）

1. **对照实验矩阵**：全新进程单帧 vs 同进程顺序渲染（隔离状态残留）；干净 venv 最新版交叉验证（隔离环境问题）；平移轨迹到原点（隔离锚点/距离因素）。
2. **定量检测**：锁定固定屏幕区域统计 min 灰度 + 暗像素数，逐位对比，避免"看起来没问题"式目检。
3. **直接读渲染器源码**定位阴影相机锚定逻辑（`renderer.py` / `light.py`），比搜 issue 更快定性。
4. **注意 /tmp 丢失**（WSL 回收）：探针脚本放 home 目录。
