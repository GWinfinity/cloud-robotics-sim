# Cable / 绳缆仿真示例

`xpbd_rope_demo.py` —— XPBD 粒子链绳缆（`cloud_robotics_sim.vdb.rope_xpbd.XpbdRope`）
+ 稀疏 SDF 场碰撞演示：绳竖直悬挂于球障碍物左侧，钉以 smoothstep 轨迹向右拖动，
绳被拉过球顶形成披覆。

- `--backend xuvdb`：trimesh 网格（地面 + 球）经 `meshes_to_sdf` 烘成窄带 level
  set（voxel 4mm），碰撞查询走 xuvdb kernel（`GpuVolume.sample` 三线性 +
  `sdf_normal` 法向）。
- `--backend analytic`：纯 torch 解析场（地面 ∪ 球），无 xuvdb 依赖。
- 输出：轨迹 `trajectory.npz`、侧视动画 `rope_drape.gif`、逐步耗时与最终诊断
  （最大穿透 / 拉伸率 / 动能）。

```bash
uv run python examples/cable_sim/xpbd_rope_demo.py --backend auto
```

实现要点与已知边界（详见 `src/cloud_robotics_sim/vdb/rope_xpbd.py` 模块 docstring）：

- small-steps XPBD：8 substeps × 4 Jacobi 迭代，拉伸硬约束（compliance 0），
  位置级库仑摩擦；初始形状不得与场相交（深埋粒子会被投影弹出）。
- xuvdb 采样是 host API（每子步一次 host 往返）。50 粒子规模 CPU 更快
  （本机实测：CPU ~12ms/step、CUDA ~45ms/step，小张量 launch 开销主导）；
  更大规模/批量绳应上 CUDA 或把采样融进 quadrants kernel。
- 钉可每步更新（`rope.pin(0, pos)`），即单向运动学耦合刚体端点；
  静态场景可用 `bake_static_scene` 直接从网格文件烘场。
